import os
from app.config import (
    HF_EMBEDDING_MODEL, OPENAI_EMBEDDING_MODEL,
    FAISS_BASE_DIR, MODELS_DIR, get_openai_client,
)

class HFEmbedder:
    dim = 384
    def __init__(self, model_name=HF_EMBEDDING_MODEL):
        from langchain_huggingface import HuggingFaceEmbeddings
        local_path = os.path.join(MODELS_DIR, "all-MiniLM-L6-v2")
        if os.path.exists(local_path):
            self.model = HuggingFaceEmbeddings(model_name=local_path)
        else:
            self.model = HuggingFaceEmbeddings(model_name=model_name)
            try:
                self.model._client.save(local_path)
            except Exception:
                pass
 
    def embed_documents(self, texts):
        return self.model.embed_documents(texts)
 
    def embed_query(self, text):
        return self.model.embed_query(text)
    
class OpenAIEmbedder:
    dim = 1536
    def __init__(self, model_name=OPENAI_EMBEDDING_MODEL):
        self.client = get_openai_client()
        self.model_name = model_name
 
    def embed_documents(self, texts, batch_size=100):
        out = []
        for i in range(0, len(texts), batch_size):
            resp = self.client.embeddings.create(
                model=self.model_name, input=texts[i:i + batch_size]
            )
            out.extend([d.embedding for d in resp.data])
        return out
 
    def embed_query(self, text):
        return self.embed_documents([text])[0]
    

_EMBEDDER_CACHE = {}
_REGISTRY = {"hf": HFEmbedder, "openai": OpenAIEmbedder}

def get_embedder(kind: str):
    if kind not in _REGISTRY:
        raise ValueError(f"Unknown embedder '{kind}'. Options: {list(_REGISTRY)}")
    if kind not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[kind] = _REGISTRY[kind]()
    return _EMBEDDER_CACHE[kind]
 
 
def faiss_dir_for(kind: str) -> str:
    return os.path.join(FAISS_BASE_DIR, kind)