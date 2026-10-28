"""The supervisor-worker graph, on either engine.

    supervisor ──► retriever ──► grader ──┬─(relevance too low)─► supervisor
                                          └─(good enough)────────► synthesizer
                                                                        │
                                                                    citation ──► END

`USE_LANGGRAPH=1` runs this on the real LangGraph engine; otherwise the
dependency-free shim in `src/core/graph.py` is used. `tests/test_engine.py`
runs both and asserts they agree.
"""

from src.agents import citation, grader, retriever, supervisor, synthesizer
from src.core import tracing
from src.core.engine import build_state_graph
from src.core.state import ResearchState


def _build():
    graph, end, engine = build_state_graph(ResearchState)

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
    graph.add_edge("citation", end)
    return graph.compile(), engine


def build_graph():
    """The compiled graph (used by tests and /health)."""
    return _build()[0]


def ask(question: str, trace: bool = True) -> dict:
    """Question in, answer + citations + trace out."""
    compiled, engine = _build()
    tracer = tracing.Tracer("querymesh.ask", {"question": question, "engine": engine})

    final_state = compiled.invoke({
        "question": question, "loops": 0, "trace": [], "meta": {},
    })

    for step in final_state.get("trace", []):
        tracer.event(step["agent"], {"message": step["message"]})
    result = {
        "question": question,
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "meta": final_state.get("meta", {}),
        "trace": final_state.get("trace", []),
    }
    result["meta"]["engine"] = engine
    if trace:
        tracer.end({"answer_chars": len(result["answer"]),
                    "citations": len(result["citations"]),
                    "grounding": result["meta"].get("grounding", 0.0)})
    return result
