"""QueryMesh as an MCP server.

Three tools are exposed, so any MCP client (Claude Desktop, an agent framework,
another assistant) can search the corpus, ask a question and ingest new papers.

Run with FastMCP installed:

    pip install fastmcp
    python mcp_server.py            # stdio transport

Without fastmcp the same functions are still callable as a plain registry, which
is what the tests and `--list` use.
"""

import json
import sys

from src.agents.workflow import ask
from src.search.index import get_index, rebuild
from src.tools import arxiv_client

TOOLS = {
    "search_papers": {
        "description": "Hybrid (BM25 + dense + RRF) search over the indexed arXiv corpus.",
        "parameters": {"query": "str", "top_k": "int (default 5)"},
        "function": lambda query, top_k=5: [
            {"paper_id": hit["chunk"]["paper_id"], "title": hit["chunk"]["title"],
             "score": hit["score"], "text": hit["chunk"]["text"][:300]}
            for hit in get_index().search(query, k=int(top_k))
        ],
    },
    "ask_research_question": {
        "description": "Run the five-agent RAG loop and return an answer with verified citations.",
        "parameters": {"question": "str"},
        "function": lambda question: ask(question),
    },
    "ingest_arxiv": {
        "description": "Fetch papers from arXiv for a query and add them to the local index.",
        "parameters": {"query": "str", "limit": "int (default 5)"},
        "function": lambda query, limit=5: {
            "ingested": len(arxiv_client.ingest(query, limit=int(limit))),
            "papers": len(rebuild().papers),
        },
    },
}


def call_tool(name: str, arguments: dict):
    if name not in TOOLS:
        raise KeyError(f"unknown tool: {name}")
    return TOOLS[name]["function"](**arguments)


def build_fastmcp():
    """Expose the same registry through FastMCP, if it is installed."""
    from fastmcp import FastMCP

    server = FastMCP("querymesh")

    @server.tool()
    def search_papers(query: str, top_k: int = 5):
        """Hybrid search over the indexed arXiv corpus."""
        return call_tool("search_papers", {"query": query, "top_k": top_k})

    @server.tool()
    def ask_research_question(question: str):
        """Answer a research question with verified arXiv citations."""
        return call_tool("ask_research_question", {"question": question})

    @server.tool()
    def ingest_arxiv(query: str, limit: int = 5):
        """Ingest arXiv papers for a query into the local index."""
        return call_tool("ingest_arxiv", {"query": query, "limit": limit})

    return server


def main() -> int:
    if "--list" in sys.argv:
        print(json.dumps({name: {"description": tool["description"],
                                 "parameters": tool["parameters"]}
                          for name, tool in TOOLS.items()}, indent=2))
        return 0

    try:
        build_fastmcp().run()
    except ImportError:
        print("fastmcp is not installed; showing the registry instead.\n"
              "pip install fastmcp  (then run `python mcp_server.py`)")
        print(json.dumps(list(TOOLS), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
