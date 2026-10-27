"""The supervisor-worker graph.

    supervisor ──► retriever ──► grader ──┬─(relevance too low)─► supervisor
                                          └─(good enough)────────► synthesizer
                                                                        │
                                                                    citation ──► END

Five agents, one conditional edge. `src/core/graph.py` is a dependency-free
StateGraph with the LangGraph API, so swapping in real LangGraph only touches
this file.
"""

from src.agents import citation, grader, retriever, supervisor, synthesizer
from src.core.graph import END, StateGraph


def build_graph():
    graph = StateGraph(dict)
    graph.add_node("supervisor", supervisor.run)
    graph.add_node("retriever", retriever.run)
    graph.add_node("grader", grader.run)
    graph.add_node("synthesizer", synthesizer.run)
    graph.add_node("citation", citation.run)

    graph.set_entry_point("supervisor")
    graph.add_edge("supervisor", "retriever")
    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges(
        "grader",
        lambda state: state.get("next_action", "write"),
        {"retrieve": "supervisor", "write": "synthesizer"},
    )
    graph.add_edge("synthesizer", "citation")
    graph.add_edge("citation", END)
    return graph.compile()


def ask(question: str) -> dict:
    """Question in, answer + citations + trace out."""
    final_state = build_graph().invoke({
        "question": question, "loops": 0, "trace": [], "meta": {},
    })
    return {
        "question": question,
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "meta": final_state.get("meta", {}),
        "trace": final_state.get("trace", []),
    }
