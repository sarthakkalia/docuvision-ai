import os
import json
import numpy as np
import faiss

from app.core.embedders import get_embedder, faiss_dir_for


class FaissStore:
    def __init__(self, dim: int, index_dir: str):
        self.dim = dim
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "metadata.json")
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        faiss.normalize_L2(vectors)
        return vectors

    def add(self, embeddings, metadatas):
        vecs = self._normalize(np.array(embeddings, dtype="float32"))
        if vecs.shape[1] != self.dim:
            raise ValueError(f"Embedding dim {vecs.shape[1]} != index dim {self.dim}")
        self.index.add(vecs)
        self.metadata.extend(metadatas)

    def query(self, query_embedding, top_k: int = 20):
        if self.index.ntotal == 0:
            return []
        vec = self._normalize(np.array(query_embedding, dtype="float32"))
        scores, indices = self.index.search(vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            md = self.metadata[idx]
            results.append({
                "id": md.get("id", str(idx)),
                "score": float(score),
                "text": md.get("text", ""),
                "content_type": md.get("content_type", "text"),
                "source": md.get("source", "unknown"),
                "metadata": md,
            })
        return results

    def remove_by_source_prefix(self, prefix: str) -> int:
        """Rebuild the index excluding vectors whose metadata['source'] starts with prefix."""
        keep = [i for i, m in enumerate(self.metadata) if not m.get("source", "").startswith(prefix)]
        removed = len(self.metadata) - len(keep)
        if removed == 0:
            return 0
        vectors = np.zeros((len(keep), self.dim), dtype="float32")
        for new_i, old_i in enumerate(keep):
            vectors[new_i] = self.index.reconstruct(old_i)
        self.metadata = [self.metadata[i] for i in keep]
        self.index = faiss.IndexFlatIP(self.dim)
        if keep:
            self.index.add(vectors)
        return removed

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)
        print(f"  Saved FAISS index ({self.index.ntotal} vectors) → {self.index_dir}")

    def load(self):
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"No FAISS index at {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        print(f"  Loaded FAISS index ({self.index.ntotal} vectors) from {self.index_dir}")
        return self

    @property
    def size(self):
        return self.index.ntotal


_STORE_CACHE = {}


def get_store(embedder_kind: str, create_if_missing: bool = False) -> FaissStore:
    if embedder_kind in _STORE_CACHE:
        return _STORE_CACHE[embedder_kind]
    emb = get_embedder(embedder_kind)
    store = FaissStore(dim=emb.dim, index_dir=faiss_dir_for(embedder_kind))
    if os.path.exists(store.index_path):
        store.load()
    elif not create_if_missing:
        raise FileNotFoundError(
            f"No index for embedder '{embedder_kind}'. Run ingestion with this embedder first."
        )
    _STORE_CACHE[embedder_kind] = store
    return store


def invalidate_store(embedder_kind: str):
    _STORE_CACHE.pop(embedder_kind, None)