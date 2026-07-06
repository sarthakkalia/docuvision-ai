# settings, defaults, paths

import os
from dotenv import load_dotenv
 
load_dotenv()
 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# ---------- Paths ----------
DATA_DIR = os.getenv("DATA_DIR", "./data")
DOCS_DIR = os.path.join(DATA_DIR, "docs")   
PDF_CHUNKS_DIR = os.path.join(DATA_DIR, "artifacts", "pdf_chunks")
SARVAM_OUTPUT_DIR = os.path.join(DATA_DIR, "artifacts", "sarvam_outputs")
EXTRACT_DIR = os.path.join(DATA_DIR, "artifacts", "extracted_content")
IMAGES_DIR = os.path.join(DATA_DIR, "artifacts", "extracted_images")
FAISS_BASE_DIR = os.path.join(DATA_DIR, "artifacts", "faiss_index")
MODELS_DIR = os.path.join(DATA_DIR, "models")
CHAT_HISTORY_DIR = os.path.join(DATA_DIR, "chat_history")

for d in [DOCS_DIR, PDF_CHUNKS_DIR, SARVAM_OUTPUT_DIR, EXTRACT_DIR,
          IMAGES_DIR, FAISS_BASE_DIR, MODELS_DIR, CHAT_HISTORY_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------- Model names ----------
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
LOCAL_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COHERE_RERANK_MODEL = "rerank-v3.5"
OPENAI_MODEL = "gpt-4o-mini"


DEFAULT_EMBEDDER = "hf"          # "hf" | "openai"
DEFAULT_RERANKER = "cohere"      # "cohere" | "local"
DEFAULT_MAX_LOOPS = 5
DEFAULT_TOP_K_RETRIEVE = 20
DEFAULT_TOP_N_RERANK = 6

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


AVAILABLE_EMBEDDERS = {
    "hf": {"label": "HuggingFace (all-MiniLM-L6-v2)", "dim": 384},
    "openai": {"label": "OpenAI (text-embedding-3-small)", "dim": 1536},
}
AVAILABLE_RERANKERS = {
    "cohere": {"label": "Cohere rerank-v3.5 (API)"},
    "local": {"label": "Local CrossEncoder (ms-marco-MiniLM-L-6-v2)"},
}


_openai_client = None

def get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client