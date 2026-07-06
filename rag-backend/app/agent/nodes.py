import os
import base64
from langchain_core.messages import AIMessage

from app.config import OPENAI_MODEL, get_openai_client
from app.core.embedders import get_embedder
from app.core.faiss_store import get_store
from app.core.rerankers import rerank
from app.agent.state import AgentState


def agent_node(state: AgentState):
    print("\n[AGENT] Analyzing situation...")

    if state["loop_count"] >= state["max_loops"]:
        print("  Max loops reached, forcing answer generation")
        state["agent_intent"] = "answer"
        return state

    if not state["retrieved_docs"]:
        state["agent_intent"] = "retrieve"
        state["agent_analysis"] = "No documents retrieved yet."
        print("  Decision: RETRIEVE")
        return state

    if not state["ranked_docs"]:
        state["agent_intent"] = "rerank"
        state["agent_analysis"] = "Need to rerank retrieved documents."
        print("  Decision: RERANK")
        return state

    if not state["text_docs"] and not state["image_docs"] and not state["table_docs"]:
        state["agent_intent"] = "route"
        state["agent_analysis"] = "Need to separate documents by modality."
        print("  Decision: ROUTE")
        return state

    state["agent_intent"] = "answer"
    state["agent_analysis"] = "All processing complete. Generate answer."
    print("  Decision: ANSWER")
    return state


def retrieve_node(state: AgentState):
    print(f"\n[RETRIEVE] Searching FAISS (embedder={state['embedder']})...")
    try:
        embedder = get_embedder(state["embedder"])
        store = get_store(state["embedder"])
        query_embedding = embedder.embed_query(state["query"])
        state["retrieved_docs"] = store.query(query_embedding, top_k=state["top_k_retrieve"])
        print(f"  Retrieved {len(state['retrieved_docs'])} documents")
    except Exception as e:
        print(f"  FAISS error: {e}")
        state["retrieved_docs"] = []
        # nothing to work with — jump straight to answer instead of looping
        state["agent_analysis"] = f"Retrieval failed: {e}"
    state["loop_count"] += 1
    return state


def rerank_node(state: AgentState):
    print(f"\n[RERANK] Scoring documents (reranker={state['reranker']})...")
    docs = state["retrieved_docs"]
    if not docs:
        state["ranked_docs"] = []
        state["loop_count"] += 1
        return state

    texts = [d["text"] if d["text"] else "No content" for d in docs]
    try:
        ranked = []
        for idx, score in rerank(state["reranker"], state["query"], texts, state["top_n_rerank"]):
            doc = docs[idx]
            doc["rerank_score"] = score
            ranked.append(doc)
        state["ranked_docs"] = ranked
        for i, d in enumerate(ranked, 1):
            print(f"  {i}. [{d['content_type']}] score={d['rerank_score']:.3f}")
    except Exception as e:
        print(f"  Reranking error: {e} — falling back to vector-score order")
        state["ranked_docs"] = docs[:state["top_n_rerank"]]

    state["loop_count"] += 1
    return state


def route_node(state: AgentState):
    print("\n[ROUTE] Separating by content type...")
    state["text_docs"] = [d for d in state["ranked_docs"] if d["content_type"] == "text"]
    state["image_docs"] = [d for d in state["ranked_docs"] if d["content_type"] == "image"]
    state["table_docs"] = [d for d in state["ranked_docs"] if d["content_type"] == "table"]
    print(f"  Text: {len(state['text_docs'])}, Images: {len(state['image_docs'])}, Tables: {len(state['table_docs'])}")
    state["loop_count"] += 1
    return state


def generate_answer_node(state: AgentState):
    print("\n[GENERATE] Creating answer...")

    context_parts = [f"[Text from {d['source']}]\n{d['text']}" for d in state["text_docs"]]
    for d in state["table_docs"]:
        context_parts.append(
            f"[Table from {d['source']}]\n{d['metadata'].get('raw_content', d['text'])}"
        )
    text_context = "\n\n".join(context_parts)

    image_context_parts = []
    for d in state["image_docs"]:
        md = d["metadata"]
        part = f"[Image from {d['source']}]\n"
        if md.get("sarvam_description"):
            part += f"Description: {md['sarvam_description']}\n"
        if md.get("caption"):
            part += f"Caption: {md['caption']}\n"
        image_context_parts.append(part)
    image_context = "\n".join(image_context_parts)

    prompt = f"""You are an expert analyst answering questions based on documents.

Question:
{state['query']}

Document Context:
{text_context if text_context else "No text documents provided."}

Image Context:
{image_context if image_context else "No images provided."}

Instructions:
1. Analyze text, tables, and images.
2. Use image captions and context for understanding.
3. Combine insights from all modalities.
4. Provide a comprehensive answer.
5. Cite sources when possible.
6. Be specific and data-driven.
7. If there is no image, do not reference figure numbers.
"""

    content = [{"type": "text", "text": prompt}]
    image_loaded_count = 0
    for d in state["image_docs"]:
        image_path = d["metadata"].get("image_path")
        if image_path and os.path.exists(image_path):
            try:
                with open(image_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"}
                })
                image_loaded_count += 1
            except Exception as e:
                print(f"  Error loading image {image_path}: {e}")

    try:
        print(f"  → {OPENAI_MODEL} ({image_loaded_count} images)")
        response = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.3,
            max_tokens=1024,
        )
        final_answer = response.choices[0].message.content
        print(f"  Answer generated ({len(final_answer)} characters)")
        return {**state, "final_answer": final_answer, "messages": [AIMessage(content=final_answer)]}
    except Exception as e:
        error = f"Error generating answer: {e}"
        return {**state, "final_answer": error, "messages": [AIMessage(content=error)]}


__all__ = ["agent_node", "retrieve_node", "rerank_node", "route_node", "generate_answer_node"]