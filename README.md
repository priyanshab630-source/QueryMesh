# QueryMesh

**A multi-agent arXiv research assistant — hybrid retrieval, graded evidence, cited answers.**

Ask a research question, get an answer built only from papers in the corpus, with
every claim traced to a source. Five specialised agents run under one supervisor;
a retrieval loop retries with a rewritten query when the evidence is too weak, and
a citation agent deletes any reference it cannot verify.

Runs **with no API keys**: a bundled corpus, a hashed-vector retriever and an
extractive writer keep the whole pipeline functional offline.

Beyond the CLI it ships a FastAPI service, a Gradio UI, a Telegram bot, an
**MCP server** and an **A2A endpoint** so other agents can discover and call the
five workers directly. The graph runs on the **real LangGraph engine**
(`USE_LANGGRAPH=1`) or on a dependency-free shim with the same API.

---

## What it actually does

```
question
   │
   ▼
┌─────────────┐   decompose into ≤3 search queries (or rewrite on retry)
│ Supervisor  │
└──────┬──────┘
       ▼
┌─────────────┐   BM25 + dense vectors, fused with Reciprocal Rank Fusion
│  Retriever  │
└──────┬──────┘
       ▼
┌─────────────┐   relevance = 0.65 × term coverage + 0.35 × normalised rank
│   Grader    │──── too weak? ──► back to Supervisor with the missing terms
└──────┬──────┘
       │ good enough
       ▼
┌─────────────┐   answer from graded chunks only, each claim marked [n]
│ Synthesizer │   (refuses when corpus coverage < MIN_COVERAGE)
└──────┬──────┘
       ▼
┌─────────────┐   verifies every [n]; strips unsupported markers
│  Citation   │
└──────┬──────┘
       ▼
answer + sources + grounding score + full agent trace
```

---

## Quickstart

```bash
git clone https://github.com/priyanshab630-source/QueryMesh.git
cd QueryMesh

python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py --demo                                 # four questions, no keys needed
python main.py --ask "How does Self-RAG decide when retrieval is needed?"
python main.py --ask "What did the graph RAG paper find?" --json
python main.py --stats
```

Web UI and other interfaces:

```bash
uvicorn api.app:app --reload --port 8000     # http://localhost:8000
python ui/gradio_app.py                      # http://localhost:7860
TELEGRAM_BOT_TOKEN=... python ui/telegram_bot.py
python mcp_server.py                         # MCP over stdio (needs fastmcp)
```

Real LangGraph engine, JWT-protected API, and the A2A endpoint:

```bash
USE_LANGGRAPH=1 python main.py --ask "How does Self-RAG decide when to retrieve?"
AUTH_ENABLED=1 uvicorn api.app:app --port 8000
curl -X POST localhost:8000/token -H 'Content-Type: application/json' \
     -d '{"client_id":"demo-client","client_secret":"demo-secret"}'
curl localhost:8000/.well-known/agent.json
```

Add live papers from arXiv (needs network):

```bash
python main.py --ingest "graph retrieval augmented generation" --limit 5
```

### Optional keys

```bash
cp .env.example .env
# SARVAM_API_KEY=...   # LLM-written answers (Sarvam AI, OpenAI-compatible)
# JINA_API_KEY=...     # real semantic embeddings instead of hashed vectors
# REDIS_URL=...        # Upstash Redis cache instead of the in-process dict
```

`GET /health` reports which backends are active, so a demo never silently
pretends it used a model it did not have.

---

## Folder structure

```
querymesh/
├── main.py                     # CLI: --ask / --demo / --ingest / --stats
├── config.py                   # every setting, all env-optional
├── mcp_server.py               # the same tools exposed over MCP (FastMCP)
├── api/
│   ├── app.py                  # FastAPI: /health /papers /ask /ingest /token /a2a
│   ├── auth.py                 # HS256 JWT issue + verify (no PyJWT)
│   └── static/index.html       # single-page UI, no build step
├── src/a2a/
│   ├── agent_cards.py          # one A2A card per agent (6 total)
│   └── server.py               # JSON-RPC: message/send, tasks/get, agent/getCard
├── ui/
│   ├── gradio_app.py           # browser UI over the same answer()
│   └── telegram_bot.py         # long-polling Telegram interface
├── deploy/
│   ├── helm/querymesh/         # EKS chart: ALB ingress, HPA, external secrets
│   └── README.md               # deployment steps (template, not deployed)
├── dags/
│   ├── arxiv_ingest_dag.py     # Airflow DAG for scheduled ingestion
│   └── run_ingest.py           # the same task, runnable without Airflow
├── src/
│   ├── agents/
│   │   ├── supervisor.py       # decompose / rewrite the query
│   │   ├── retriever.py        # hybrid search per sub-question
│   │   ├── grader.py           # relevance gate + retry signal
│   │   ├── synthesizer.py      # cited answer, or an honest refusal
│   │   ├── citation.py         # verifies and prunes [n] markers
│   │   └── workflow.py         # the graph wiring
│   ├── search/
│   │   ├── bm25.py             # Okapi BM25, pure Python
│   │   ├── embeddings.py       # Jina API or hashed vectors
│   │   ├── rrf.py              # reciprocal rank fusion
│   │   └── index.py            # chunking + hybrid search
│   ├── tools/
│   │   ├── arxiv_client.py     # arXiv Atom API search + PDF download
│   │   ├── pdf_parser.py       # pypdf, or Docling when installed
│   │   ├── cache.py            # SHA256 exact-match cache (Redis or memory)
│   │   └── llm.py              # Sarvam AI client with silent fallback
│   └── core/
│       ├── state.py            # shared ResearchState + trace logger
│       ├── engine.py           # picks real LangGraph or the built-in shim
│       ├── tracing.py          # local JSONL + LangSmith/Langfuse sinks
│       └── graph.py            # LangGraph-shaped StateGraph (dependency-free)
├── data/sample_papers/         # 6-paper offline corpus
├── tests/                      # 64 pytest tests, run on both engines
├── .github/workflows/ci.yml    # tests on 3.11 + 3.12, both engines
├── .github/workflows/cd.yml    # image build -> GHCR, optional Helm rollout
└── Dockerfile
```

---

## Design decisions worth knowing

**Hybrid retrieval, fused by rank.** BM25 and cosine similarity live on
incomparable scales, so `src/search/rrf.py` fuses *ranks*
(`Σ w / (k + rank)`) instead of inventing a normalisation. Both rankers are
visible in every hit (`bm25_rank`, `dense_rank`), so you can see which one
found the chunk.

**The grader can send the graph backwards.** If the best relevance is below
`RELEVANCE_THRESHOLD`, the grader records which query terms were missing and
loops to the supervisor, which rewrites the query with them. The loop is capped
by `MAX_RETRIEVAL_LOOPS` so a run always terminates.

**Citations are verified, not decorative.** `src/agents/citation.py` re-reads
each sentence that carries a `[n]`, compares its content words against the chunk
it points at, and removes the marker when the support is not there. The
`grounding` score in `meta` is the share of markers that survived.

**It refuses when the corpus does not cover the question.** If term coverage
stays below `MIN_COVERAGE`, the writer says so and names the closest papers
instead of producing a plausible-sounding answer.

**Two engines, one graph.** `src/core/engine.py` returns either LangGraph's
`StateGraph` or the shim; `src/agents/workflow.py` wires nodes and edges once.
`tests/test_engine.py` runs the suite on both and asserts identical answers, so
the shim can never silently drift from the real thing.

**Agents are addressable, not just internal.** `src/a2a/agent_cards.py` publishes
a card per agent and `src/a2a/server.py` serves them over JSON-RPC, so an external
orchestrator can call the grader or the citation checker without importing this
codebase. Each sub-agent endpoint runs only its prerequisites, so it exposes that
agent's real step rather than a copy of the final answer.

**Auth is opt-in and small.** `api/auth.py` implements HS256 JWTs directly
(base64url + HMAC, `exp` enforced) with scope checks per route. `AUTH_ENABLED=0`
keeps local development open; the code path is tested either way.

**Deployment is a template.** `deploy/helm/querymesh/` describes the EKS shape —
ALB ingress, HPA, IRSA service account, secrets from AWS Secrets Manager via the
External Secrets Operator — and `cd.yml` builds the image on every push but only
rolls out when `ENABLE_DEPLOY=true` and a cluster is configured.

**Caching is exact-match.** `SHA256(model ‖ question ‖ corpus size)` as the key.
Identical repeated queries return from cache; anything else re-runs the graph.
Redis/Upstash when `REDIS_URL` is set, otherwise an in-process TTL dict.

---

## MCP server

`mcp_server.py` exposes three tools — `search_papers`, `ask_research_question`,
`ingest_arxiv — over the Model Context Protocol:

```bash
pip install fastmcp
python mcp_server.py            # stdio transport
python mcp_server.py --list     # inspect the registry without fastmcp
```

Any MCP client can then use the corpus and the agent loop as tools.

---

## Scheduled ingestion

`dags/arxiv_ingest_dag.py` is a daily Airflow DAG that pulls papers for a list
of topics into `data/papers/`, which the index picks up on the next rebuild.
Copy it into your Airflow `dags/` folder, or point Airflow at this repo.

---

## Honest notes (read this)

- **The bundled corpus is six papers with summarised abstracts**, enough to demo
  retrieval and citation, not enough to answer most research questions. Ingest
  real papers (`--ingest`) before drawing conclusions.
- **Nothing is deployed.** The Helm chart and `cd.yml` are written to be applied;
  no cluster, registry or Helm release is attached to this repo. Say "manifests
  and pipeline written", not "running on EKS".
- **Docling and Airflow are adapters, not dependencies.** `USE_DOCLING=1` switches
  the PDF parser when Docling is installed, and `dags/run_ingest.py` runs the
  ingestion task without Airflow. Neither was executed in this environment.
- **Telegram needs a bot token**, so that path is exercised with a stubbed HTTP
  layer in the tests rather than against Telegram itself.
- **Without `JINA_API_KEY` the dense half is a hashing trick** — deterministic,
  but not semantic. It contributes recall, not understanding.
- **Without `SARVAM_API_KEY` the writer is extractive**: it quotes the most
  relevant sentences verbatim. Good for grounding, weaker for synthesis.
- **PDF ingestion is wired but not automatic**: `src/tools/pdf_parser.py` parses
  a downloaded PDF, and `arxiv_client.download_pdf()` fetches it; the DAG shows
  where to join the two. The default index uses abstracts only.
- **Grounding ≠ correctness.** The citation agent checks that a sentence is
  supported by the chunk it cites, not that the chunk is right.

---

## Roadmap

- [ ] Full-text indexing: PDF → chunks instead of abstracts
- [ ] Cross-encoder reranker over the fused top-20
- [ ] Persistent vector store (pgvector / Qdrant) instead of in-process numpy-free lists
- [x] LangGraph engine (`USE_LANGGRAPH=1`) + LangSmith/Langfuse tracing
- [x] A2A cards + JSON-RPC endpoint, JWT auth, Gradio and Telegram UIs
- [ ] Answer-level eval harness (groundedness, coverage, refusal precision)
- [~] A2A discovery implemented as one endpoint; split into per-agent services next

---

## Tests

```bash
pytest -q                        # 64 tests, no network, no keys
USE_LANGGRAPH=1 pytest -q        # the same suite on the LangGraph engine
python main.py --demo
```

## License

MIT.
