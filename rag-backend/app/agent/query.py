import time
from typing import List, Dict, Optional

from app.agent.state import create_initial_state
from app.agent.graph import get_graph
from app.config import (
    DEFAULT_EMBEDDER, DEFAULT_RERANKER,
    DEFAULT_MAX_LOOPS, DEFAULT_TOP_K_RETRIEVE, DEFAULT_TOP_N_RERANK,
)


def run_agentic_rag_query(
    user_query: str,
    chat_history: Optional[List[Dict]] = None,
    embedder: str = DEFAULT_EMBEDDER,
    reranker: str = DEFAULT_RERANKER,
    max_loops: int = DEFAULT_MAX_LOOPS,
    top_k_retrieve: int = DEFAULT_TOP_K_RETRIEVE,
    top_n_rerank: int = DEFAULT_TOP_N_RERANK,
):
    start_time = time.time()

    context_messages = ""
    if chat_history:
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            context_messages += f"{role}: {msg['content']}\n\n"

    enriched_query = user_query
    if context_messages:
        enriched_query = f"""Previous conversation:
{context_messages}

Current question: {user_query}

Answer the current question considering the conversation above."""

    initial_state = create_initial_state(
        enriched_query,
        embedder=embedder,
        reranker=reranker,
        max_loops=max_loops,
        top_k_retrieve=top_k_retrieve,
        top_n_rerank=top_n_rerank,
    )

    final_state = get_graph().invoke(initial_state)
    elapsed = time.time() - start_time
    return final_state["final_answer"], final_state, elapsed


__all__ = ["run_agentic_rag_query"]