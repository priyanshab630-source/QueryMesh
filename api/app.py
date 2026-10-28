"""FastAPI wrapper for QueryMesh.

    uvicorn api.app:app --reload --port 8000
    open http://localhost:8000

Routes
    GET  /health                  backends, engine, corpus size
    GET  /papers                  indexed papers
    POST /ask                     five-agent pipeline  (bearer token if AUTH_ENABLED)
    POST /ingest                  pull new arXiv papers (bearer token if AUTH_ENABLED)
    POST /token                   exchange client credentials for a JWT
    POST /a2a                     A2A JSON-RPC endpoint for all six agent cards
    GET  /.well-known/agent.json  orchestrator agent card
"""

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from api.auth import check_scope, create_token, require_token
from main import answer
from src.a2a import agent_cards
from src.a2a.server import handle as a2a_handle
from src.core import tracing
from src.core.engine import engine_name
from src.search import embeddings, opensearch_backend
from src.search.index import get_index, rebuild
from src.tools import arxiv_client, cache, llm

app = FastAPI(
    title="QueryMesh",
    description="Multi-agent arXiv research assistant: hybrid search + agentic RAG.",
    version="0.2.0",
)

STATIC_DIR = __file__.rsplit("/", 1)[0] + "/static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AskRequest(BaseModel):
    question: str = Field(..., examples=["How does Self-RAG decide when retrieval is needed?"])
    use_cache: bool = True


class IngestRequest(BaseModel):
    query: str = Field(..., examples=["retrieval augmented generation"])
    limit: int = 5


class TokenRequest(BaseModel):
    client_id: str = Field(..., examples=["demo-client"])
    client_secret: str = Field(..., examples=["demo-secret"])


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
        "engine": engine_name(),
        "search": opensearch_backend.backend_name(),
        "tracing": tracing.sinks(),
        "auth": "jwt" if config.AUTH_ENABLED else "open",
        "agents": 5,
        "a2a_cards": sorted(agent_cards.AGENT_CARDS),
    }


@app.get("/papers")
def papers():
    return [{"id": p.get("id"), "title": p.get("title"),
             "published": p.get("published"), "url": p.get("url")}
            for p in get_index().papers]


@app.post("/token")
def token(request: TokenRequest):
    """Issue a JWT for the demo client (swap for your identity provider)."""
    if (request.client_id != config.API_CLIENT_ID
            or request.client_secret != config.API_CLIENT_SECRET):
        raise HTTPException(status_code=401, detail="invalid client credentials")
    return {"access_token": create_token(request.client_id),
            "token_type": "bearer", "expires_in": config.JWT_TTL_SECONDS}


@app.post("/ask")
def ask_question(request: AskRequest, claims: dict = Depends(require_token)):
    check_scope(claims, "ask")
    return answer(request.question, use_cache=request.use_cache)


@app.post("/ingest")
def ingest(request: IngestRequest, claims: dict = Depends(require_token)):
    check_scope(claims, "ingest")
    found = arxiv_client.ingest(request.query, limit=request.limit)
    if not found:
        return {"ingested": 0, "detail": "network unavailable or no results"}
    index = rebuild()
    return {"ingested": len(found), "papers": len(index.papers),
            "chunks": len(index.chunks), "ids": [p["id"] for p in found]}


@app.post("/a2a")
def a2a(payload: dict):
    """A2A JSON-RPC: message/send, tasks/get, agent/getCard."""
    return a2a_handle(payload)


@app.get("/.well-known/agent.json")
def agent_card():
    return agent_cards.get("querymesh")


@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR + "/index.html")
