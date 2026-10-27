"""FastAPI wrapper for QueryMesh.

    uvicorn api.app:app --reload --port 8000
    open http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import answer
from src.search import embeddings
from src.search.index import get_index, rebuild
from src.tools import arxiv_client, cache, llm

app = FastAPI(
    title="QueryMesh",
    description="Multi-agent arXiv research assistant: hybrid search + agentic RAG.",
    version="0.1.0",
)

STATIC_DIR = __file__.rsplit("/", 1)[0] + "/static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., examples=["How do agentic RAG systems decide when to retrieve?"])
    use_cache: bool = True


class IngestRequest(BaseModel):
    query: str = Field(..., examples=["retrieval augmented generation"])
    limit: int = 5


@app.get("/health")
def health():
    index = get_index()
    return {
        "status": "ok",
        "papers": len(index.papers),
        "chunks": len(index.chunks),
        "llm": "sarvam" if llm.available() else "rule-based fallback",
        "embeddings": embeddings.backend_name(),
        "cache": cache.backend_name(),
        "agents": 5,
    }


@app.get("/papers")
def papers():
    return [{"id": p.get("id"), "title": p.get("title"),
             "published": p.get("published"), "url": p.get("url")}
            for p in get_index().papers]


@app.post("/ask")
def ask_question(request: AskRequest):
    return answer(request.question, use_cache=request.use_cache)


@app.post("/ingest")
def ingest(request: IngestRequest):
    found = arxiv_client.ingest(request.query, limit=request.limit)
    if not found:
        return {"ingested": 0, "detail": "network unavailable or no results"}
    index = rebuild()
    return {"ingested": len(found), "papers": len(index.papers),
            "chunks": len(index.chunks), "ids": [p["id"] for p in found]}


@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR + "/index.html")
