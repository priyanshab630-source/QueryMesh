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

# ---------- Retrieval ----------
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "90"))
TOP_K = int(os.getenv("TOP_K", "6"))
MAX_RETRIEVAL_LOOPS = int(os.getenv("MAX_RETRIEVAL_LOOPS", "2"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))
MIN_COVERAGE = float(os.getenv("MIN_COVERAGE", "0.35"))   # below this we say "not covered"
RRF_K = int(os.getenv("RRF_K", "60"))
