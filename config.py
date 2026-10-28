"""Central configuration for QueryMesh.

Everything is optional. With no keys at all the assistant runs on the bundled
sample corpus, a hashed-vector retriever and an extractive answer writer.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PAPERS_DIR = DATA_DIR / "sample_papers"

# ---------- LLM (Sarvam AI by default; any OpenAI-compatible endpoint) ----------
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.sarvam.ai/v1/chat/completions")
LLM_MODEL = os.getenv("LLM_MODEL", "sarvam-m")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "45"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "900"))

# ---------- Embeddings (Jina AI optional, hashed vectors otherwise) ----------
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))

# ---------- Cache (Upstash Redis optional, in-process otherwise) ----------
REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# ---------- Orchestration engine ----------
# 1 = real LangGraph StateGraph, 0 = the dependency-free shim in src/core/graph.py
USE_LANGGRAPH = os.getenv("USE_LANGGRAPH", "0") in ("1", "true", "yes")

# ---------- Observability ----------
# Local JSONL traces are always written; LangSmith / Langfuse are opt-in.
TRACE_DIR = Path(os.getenv("TRACE_DIR", str(ROOT / "traces")))
TRACE_ENABLED = os.getenv("TRACE_ENABLED", "1") in ("1", "true", "yes")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "querymesh")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ---------- Auth (JWT, HS256) ----------
# AUTH_ENABLED=1 protects /ask and /ingest with a bearer token issued by /token.
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "0") in ("1", "true", "yes")
JWT_SECRET = os.getenv("JWT_SECRET", "dev-secret-change-me")
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", "3600"))
API_CLIENT_ID = os.getenv("API_CLIENT_ID", "demo-client")
API_CLIENT_SECRET = os.getenv("API_CLIENT_SECRET", "demo-secret")

# ---------- A2A (agent-to-agent) ----------
A2A_PUBLIC_URL = os.getenv("A2A_PUBLIC_URL", "http://localhost:8000")

# ---------- Ingestion ----------
USE_DOCLING = os.getenv("USE_DOCLING", "0") in ("1", "true", "yes")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# ---------- Retrieval ----------
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "90"))
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_RETRIEVAL_LOOPS = int(os.getenv("MAX_RETRIEVAL_LOOPS", "2"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.35"))   # below this we say "not covered"
RRF_K = int(os.getenv("RRF_K", "60"))
