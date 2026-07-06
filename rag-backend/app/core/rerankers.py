import os
from app.config import COHERE_API_KEY, COHERE_RERANK_MODEL, LOCAL_RERANKER_MODEL, MODELS_DIR
 
_cohere_client = None
_local_reranker = None


def _get_cohere():
    global _cohere_client
    if _cohere_client is None:
        import cohere
        _cohere_client = cohere.Client(COHERE_API_KEY)
    return _cohere_client

def _get_local():
    global _local_reranker
    if _local_reranker is None:
        from sentence_transformers import CrossEncoder
        local_path = os.path.join(MODELS_DIR, "ms-marco-MiniLM-L-6-v2")
        if os.path.exists(local_path):
            _local_reranker = CrossEncoder(local_path)
        else:
            _local_reranker = CrossEncoder(LOCAL_RERANKER_MODEL)
            _local_reranker.save(local_path)
    return _local_reranker

def rerank(kind: str, query: str, texts: list, top_n: int):
    top_n = min(top_n, len(texts))
    if kind == "cohere":
        resp = _get_cohere().rerank(
            model=COHERE_RERANK_MODEL, query=query, documents=texts, top_n=top_n
        )
        return [(r.index, float(r.relevance_score)) for r in resp.results]
    elif kind == "local":
        scores = _get_local().predict([(query, t) for t in texts])
        order = sorted(range(len(texts)), key=lambda i: scores[i], reverse=True)[:top_n]
        return [(i, float(scores[i])) for i in order]
    raise ValueError(f"Unknown reranker '{kind}'. Options: ['cohere', 'local']")