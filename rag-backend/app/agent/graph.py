from langgraph.graph import StateGraph, END
 
from app.agent.state import AgentState
from app.agent.nodes import agent_node, retrieve_node, rerank_node, route_node, generate_answer_node
 
_GRAPH = None

def route_agent_decision(state: AgentState):
    if state.get("loop_count", 0) >= state.get("max_loops", 5):
        print("\n  Loop limit reached, forcing answer")
        return "generate_answer"
 
    routing_map = {
        "retrieve": "retrieve",
        "rerank": "rerank",
        "route": "route",
        "answer": "generate_answer",
    }
    return routing_map.get(state.get("agent_intent", "answer"), "generate_answer")

def build_agentic_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("route", route_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", route_agent_decision, {
        "retrieve": "retrieve",
        "rerank": "rerank",
        "route": "route",
        "generate_answer": "generate_answer",
    })
    workflow.add_edge("retrieve", "agent")
    workflow.add_edge("rerank", "agent")
    workflow.add_edge("route", "agent")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agentic_graph()
    return _GRAPH

__all__ = ["build_agentic_graph", "get_graph", "route_agent_decision"]