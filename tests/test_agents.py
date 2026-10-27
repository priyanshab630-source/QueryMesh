"""The five-agent graph: retrieval loop, citations, coverage gate."""

from src.agents.workflow import ask, build_graph


def test_pipeline_answers_with_citations():
    result = ask("What did the graph RAG paper find about global questions?")
    assert result["answer"]
    assert result["citations"]
    assert result["citations"][0]["paper_id"]
    assert result["meta"]["grounding"] > 0


def test_every_agent_ran():
    state = build_graph().invoke({"question": "How does ReAct interleave reasoning "
                                             "and acting?", "loops": 0,
                                  "trace": [], "meta": {}})
    agents = [step["agent"] for step in state["trace"]]
    for expected in ("supervisor", "retriever", "grader", "synthesizer", "citation"):
        assert expected in agents


def test_off_corpus_question_is_refused():
    result = ask("What is the airspeed velocity of an unladen swallow?")
    assert "does not directly answer" in result["answer"] or not result["citations"]


def test_loop_counter_is_bounded():
    state = build_graph().invoke({"question": "quantum chromodynamics lattice gauge",
                                  "loops": 0, "trace": [], "meta": {}})
    assert state["loops"] <= 3


def test_no_unsupported_citations_survive():
    result = ask("How do agentic RAG systems decide when to retrieve again?")
    for citation in result["citations"]:
        assert citation["paper_id"]
