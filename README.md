# QueryMesh

**A multi-agent arXiv research assistant — hybrid retrieval, graded evidence, cited answers.**

Ask a research question, get an answer built only from papers in the corpus, with
every claim traced to a source. Five specialised agents run under one supervisor;
a retrieval loop retries with a rewritten query when the evidence is too weak, and
a citation agent deletes any reference it cannot verify.

Runs **with no API keys**: a bundled corpus, a hashed-vector retriever and an
extractive writer keep the whole pipeline functional offline.

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

Web UI:

```bash
uvicorn api.app:app --reload --port 8000
# open http://localhost:8000
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
│   ├── app.py                  # FastAPI: /health /papers /ask /ingest
│   └── static/index.html       # single-page UI, no build step
├── dags/
│   └── arxiv_ingest_dag.py     # Airflow DAG for scheduled ingestion
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
│       └── graph.py            # LangGraph-shaped StateGraph (dependency-free)
├── data/sample_papers/         # 6-paper offline corpus
├── tests/                      # 24 pytest tests
├── .github/workflows/ci.yml    # tests on 3.11 + 3.12
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
- [ ] LangGraph engine swap + Langfuse tracing per agent hop
- [ ] Answer-level eval harness (groundedness, coverage, refusal precision)
- [ ] A2A agent discovery so the five agents can be deployed separately

---

## Tests

```bash
pytest -q            # 24 tests, no network, no keys
python main.py --demo
```

## License

MIT.
