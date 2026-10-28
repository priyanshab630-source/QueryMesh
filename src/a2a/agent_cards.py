"""A2A agent cards.

Each agent publishes what it can do and where to reach it, so an external
orchestrator can discover and call the five agents without knowing this codebase.
Card shape follows the A2A specification (name, description, url, capabilities,
skills).
"""

import config

PROTOCOL_VERSION = "0.2.0"


def _card(name: str, description: str, skills: list) -> dict:
    return {
        "name": name,
        "description": description,
        "url": f"{config.A2A_PUBLIC_URL}/a2a",
        "version": "0.1.0",
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills,
    }


AGENT_CARDS = {
    "querymesh": _card(
        "QueryMesh orchestrator",
        "Five-agent arXiv research assistant: hybrid retrieval, graded evidence, "
        "cited answers.",
        [
            {"id": "ask", "name": "Ask a research question",
             "description": "Full pipeline: retrieve, grade, synthesise, verify citations.",
             "tags": ["rag", "research", "citations"]},
            {"id": "search", "name": "Search the corpus",
             "description": "Hybrid BM25 + dense search over indexed arXiv papers.",
             "tags": ["retrieval"]},
        ],
    ),
    "supervisor": _card(
        "Supervisor agent",
        "Decomposes a question into focused retrieval queries and rewrites the query "
        "when the evidence comes back weak.",
        [{"id": "decompose", "name": "Decompose a question",
          "description": "Returns up to three retrieval sub-questions.",
          "tags": ["planning"]}],
    ),
    "retriever": _card(
        "Retriever agent",
        "Hybrid search over the corpus: BM25 plus dense vectors, fused with "
        "Reciprocal Rank Fusion.",
        [{"id": "retrieve", "name": "Retrieve passages",
          "description": "Returns ranked chunks with paper ids.",
          "tags": ["retrieval", "bm25", "embeddings"]}],
    ),
    "grader": _card(
        "Grader agent",
        "Scores retrieved chunks for relevance and decides whether another "
        "retrieval round is needed.",
        [{"id": "grade", "name": "Grade evidence",
          "description": "Returns kept/rejected counts and relevance.",
          "tags": ["evaluation"]}],
    ),
    "synthesizer": _card(
        "Synthesizer agent",
        "Writes a cited answer from graded chunks only, and refuses when corpus "
        "coverage is too low.",
        [{"id": "synthesize", "name": "Synthesise an answer",
          "description": "Returns an answer with [n] markers.",
          "tags": ["generation"]}],
    ),
    "citation": _card(
        "Citation agent",
        "Verifies every [n] marker against the chunk it cites and removes the ones "
        "that are not supported.",
        [{"id": "verify", "name": "Verify citations",
          "description": "Returns verified sources and a grounding score.",
          "tags": ["grounding", "citations"]}],
    ),
}


def get(name: str) -> dict:
    return AGENT_CARDS[name]
