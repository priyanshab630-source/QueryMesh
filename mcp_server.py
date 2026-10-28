"""QueryMesh as an MCP server (FastMCP).

Three tools are exposed so any MCP client — Claude Desktop, an agent framework,
another assistant — can search the corpus, ask a question and ingest new papers:

    search_papers(query, top_k)      hybrid BM25 + dense search
    ask_research_question(question)  the five-agent pipeline, cited answer
    ingest_arxiv(query, limit)       pull new papers into the local index

    pip install fastmcp
    python mcp_server.py                 # stdio transport
    python mcp_server.py --http 8001     # streamable HTTP transport
    python mcp_server.py --list          # registry without starting a server
"""

import json
import sys

from src.agents.workflow import ask
from src.search.index import get_index, rebuild
from src.tools import arxiv_client

TOOLS = {
    "search_papers": "Hybrid (BM25 + dense + RRF) search over the indexed arXiv corpus.",
    "ask_research_question": "Run the five-agent RAG loop; returns a cited answer.",
    "ingest_arxiv": "Fetch papers from arXiv for a query and add them to the index.",
}


def search_papers(query: str, top_k: int = 5) -> list:
    return [
        {"paper_id": hit["chunk"]["paper_id"], "title": hit["chunk"]["title"],
         "score": hit["score"], "text": hit["chunk"]["text"][:300]}
        for hit in get_index().search(query, k=int(top_k))
    ]


def ask_research_question(question: str) -> dict:
    return ask(question)


def ingest_arxiv(query: str, limit: int = 5) -> dict:
    found = arxiv_client.ingest(query, limit=int(limit))
    index = rebuild() if found else get_index()
    return {"ingested": len(found), "papers": len(index.papers),
            "ids": [p["id"] for p in found]}


def build_server():
    """Build the FastMCP server. Imported lazily so the repo runs without it."""
    from fastmcp import FastMCP

    server = FastMCP("querymesh")

    @server.tool()
    def search_papers_tool(query: str, top_k: int = 5) -> list:
        """Hybrid search over the indexed arXiv corpus."""
        return search_papers(query, top_k)

    @server.tool()
    def ask_research_question_tool(question: str) -> dict:
        """Answer a research question with verified arXiv citations."""
        return ask_research_question(question)

    @server.tool()
    def ingest_arxiv_tool(query: str, limit: int = 5) -> dict:
        """Ingest arXiv papers for a query into the local index."""
        return ingest_arxiv(query, limit)

    return server


def main() -> int:
    if "--list" in sys.argv:
        print(json.dumps(TOOLS, indent=2))
        return 0

    try:
        server = build_server()
    except ImportError:
        print("fastmcp is not installed.\n  pip install fastmcp\n"
              "Meanwhile, the tool functions are importable directly:")
        print(json.dumps(TOOLS, indent=2))
        return 1

    if "--http" in sys.argv:
        port = sys.argv[sys.argv.index("--http") + 1] if len(sys.argv) > sys.argv.index("--http") + 1 else "8001"
        server.run(transport="http", port=int(port))
    else:
        server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
