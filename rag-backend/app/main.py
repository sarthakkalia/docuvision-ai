import os
import json
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import (
    CHAT_HISTORY_DIR, IMAGES_DIR,
    AVAILABLE_EMBEDDERS, AVAILABLE_RERANKERS,
    DEFAULT_EMBEDDER, DEFAULT_RERANKER,
    DEFAULT_MAX_LOOPS, DEFAULT_TOP_K_RETRIEVE, DEFAULT_TOP_N_RERANK,
)
from app.agent.query import run_agentic_rag_query
from app.api.routes_ingest import router as ingest_router

app = FastAPI(title="Agentic Multimodal RAG API", version="1.0.0")

app.include_router(ingest_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


# ==================== Schemas ====================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    embedder: str = Field(DEFAULT_EMBEDDER, description="hf | openai")
    reranker: str = Field(DEFAULT_RERANKER, description="cohere | local")
    max_loops: int = Field(DEFAULT_MAX_LOOPS, ge=1, le=20)
    top_k_retrieve: int = Field(DEFAULT_TOP_K_RETRIEVE, ge=1, le=100)
    top_n_rerank: int = Field(DEFAULT_TOP_N_RERANK, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    conversation_id: str
    time_seconds: float
    loops: int
    retrieved: int
    ranked: int
    sources: List[str]
    images: List[str]
    settings_used: Dict


# ==================== Chat history helpers ====================

def _history_path(conv_id: str) -> str:
    safe = "".join(c for c in conv_id if c.isalnum() or c in "_-")
    return os.path.join(CHAT_HISTORY_DIR, f"{safe}.json")


def load_history(conv_id: str) -> List[Dict]:
    path = _history_path(conv_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_history(conv_id: str, history: List[Dict]):
    with open(_history_path(conv_id), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ==================== Endpoints ====================

@app.get("/config")
def get_config():
    return {
        "embedders": AVAILABLE_EMBEDDERS,
        "rerankers": AVAILABLE_RERANKERS,
        "defaults": {
            "embedder": DEFAULT_EMBEDDER,
            "reranker": DEFAULT_RERANKER,
            "max_loops": DEFAULT_MAX_LOOPS,
            "top_k_retrieve": DEFAULT_TOP_K_RETRIEVE,
            "top_n_rerank": DEFAULT_TOP_N_RERANK,
        },
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if req.embedder not in AVAILABLE_EMBEDDERS:
        raise HTTPException(400, f"Unknown embedder '{req.embedder}'")
    if req.reranker not in AVAILABLE_RERANKERS:
        raise HTTPException(400, f"Unknown reranker '{req.reranker}'")

    conv_id = req.conversation_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    history = load_history(conv_id)

    try:
        answer, final_state, elapsed = run_agentic_rag_query(
            req.query,
            chat_history=history,
            embedder=req.embedder,
            reranker=req.reranker,
            max_loops=req.max_loops,
            top_k_retrieve=req.top_k_retrieve,
            top_n_rerank=req.top_n_rerank,
        )
    except FileNotFoundError as e:
        raise HTTPException(400, f"{e} — run POST /ingest with embedder='{req.embedder}' first.")
    except Exception as e:
        raise HTTPException(500, f"Query error: {e}")

    images = []
    for doc in final_state.get("image_docs", []):
        p = doc.get("metadata", {}).get("image_path", "")
        if p and os.path.exists(p):
            images.append(f"/images/{os.path.basename(p)}")

    sources = sorted({d.get("source", "unknown") for d in final_state.get("ranked_docs", [])})

    now = datetime.now().isoformat()
    history.append({"role": "user", "content": req.query, "timestamp": now})
    history.append({
        "role": "assistant", "content": answer, "timestamp": now,
        "metadata": {
            "time": elapsed, "loops": final_state.get("loop_count", 0),
            "retrieved": len(final_state.get("retrieved_docs", [])),
            "ranked": len(final_state.get("ranked_docs", [])),
            "sources": sources, "images": images,
            "embedder": req.embedder, "reranker": req.reranker,
        },
    })
    save_history(conv_id, history)

    return QueryResponse(
        answer=answer,
        conversation_id=conv_id,
        time_seconds=round(elapsed, 2),
        loops=final_state.get("loop_count", 0),
        retrieved=len(final_state.get("retrieved_docs", [])),
        ranked=len(final_state.get("ranked_docs", [])),
        sources=sources,
        images=images,
        settings_used={
            "embedder": req.embedder, "reranker": req.reranker,
            "max_loops": req.max_loops, "top_k_retrieve": req.top_k_retrieve,
            "top_n_rerank": req.top_n_rerank,
        },
    )


@app.get("/conversations")
def list_conversations():
    convs = [f[:-5] for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".json")]
    return {"conversations": sorted(convs, reverse=True)}


@app.get("/conversations/{conv_id}")
def get_conversation(conv_id: str):
    history = load_history(conv_id)
    if not history:
        raise HTTPException(404, "Conversation not found")
    return {"conversation_id": conv_id, "messages": history}


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    path = _history_path(conv_id)
    if not os.path.exists(path):
        raise HTTPException(404, "Conversation not found")
    os.remove(path)
    return {"status": "deleted", "conversation_id": conv_id}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)