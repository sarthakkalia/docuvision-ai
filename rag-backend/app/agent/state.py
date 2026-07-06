from typing import List, Dict, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from app.config import (
    DEFAULT_EMBEDDER, DEFAULT_RERANKER,
    DEFAULT_MAX_LOOPS, DEFAULT_TOP_K_RETRIEVE, DEFAULT_TOP_N_RERANK,
)


class AgentState(TypedDict):
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    loop_count: int
    retrieved_docs: List[Dict]
    ranked_docs: List[Dict]
    agent_analysis: str
    agent_intent: str
    text_docs: List[Dict]
    image_docs: List[Dict]
    table_docs: List[Dict]
    final_answer: str

    embedder: str
    reranker: str
    max_loops: int
    top_k_retrieve: int
    top_n_rerank: int


def create_initial_state(
    query: str,
    embedder: str = DEFAULT_EMBEDDER,
    reranker: str = DEFAULT_RERANKER,
    max_loops: int = DEFAULT_MAX_LOOPS,
    top_k_retrieve: int = DEFAULT_TOP_K_RETRIEVE,
    top_n_rerank: int = DEFAULT_TOP_N_RERANK,
) -> AgentState:
    return {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "loop_count": 0,
        "retrieved_docs": [],
        "ranked_docs": [],
        "agent_analysis": "",
        "agent_intent": "",
        "text_docs": [],
        "image_docs": [],
        "table_docs": [],
        "final_answer": "",
        "embedder": embedder,
        "reranker": reranker,
        "max_loops": max_loops,
        "top_k_retrieve": top_k_retrieve,
        "top_n_rerank": top_n_rerank,
    }


__all__ = ["AgentState", "create_initial_state"]