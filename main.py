"""QueryMesh - command line entry point.

    python main.py --ask "how do agentic RAG systems decide when to retrieve again?"
    python main.py --demo
    python main.py --ingest "graph retrieval augmented generation" --limit 3
    python main.py --stats
"""

import argparse
import json
import sys
import time

import config
from src.agents.workflow import ask
from src.search.index import get_index, rebuild
from src.tools import arxiv_client, cache, llm
from src.search import embeddings


def _cache_key(question: str) -> str:
    return cache.key_for("ask", question.strip().lower(), config.LLM_MODEL,
                         llm.available(), len(get_index().chunks))


def answer(question: str, use_cache: bool = True) -> dict:
    """Cached end-to-end run."""
    key = _cache_key(question)
    if use_cache:
        hit = cache.get(key)
        if hit:
            hit["meta"]["cache"] = "hit"
            return hit

    started = time.time()
    result = ask(question)
    result["meta"]["elapsed_ms"] = int((time.time() - started) * 1000)
    result["meta"]["cache"] = "miss"
    if use_cache:
        cache.set(key, result)
    return result


def print_result(result: dict) -> None:
    print("=" * 78)
    print(f"Q: {result['question']}")
    print("=" * 78)
    print(result["answer"])

    if result["citations"]:
        print("\nSources")
        for citation in result["citations"]:
            print(f"  [{citation['number']}] {citation['title']} "
                  f"(arXiv {citation['paper_id']}, {citation['published']})")
            print(f"      {citation['url']}")

    meta = result["meta"]
    print(f"\nwriter={meta.get('writer')} grounding={meta.get('grounding', 0):.2f} "
          f"cache={meta.get('cache')} {meta.get('elapsed_ms', 0)} ms")

    print("\nAgent trace")
    for step in result["trace"]:
        print(f"  {step['agent']:<12} {step['message']}")
    print()


DEMO = [
    "How does Self-RAG decide when retrieval is needed?",
    "What did the graph RAG paper find about global questions?",
    "What is the cost of the graph RAG indexing stage?",
    "What is reciprocal rank fusion used for in hybrid search?",
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Multi-agent arXiv research assistant")
    parser.add_argument("--ask", help="question to answer from the corpus")
    parser.add_argument("--demo", action="store_true", help="run the built-in questions")
    parser.add_argument("--ingest", help="arXiv search query to add to the corpus")
    parser.add_argument("--limit", type=int, default=5, help="papers per ingest")
    parser.add_argument("--json", action="store_true", help="emit raw JSON")
    parser.add_argument("--no-cache", action="store_true", help="bypass the cache")
    parser.add_argument("--stats", action="store_true", help="show backends and corpus size")
    args = parser.parse_args(argv)

    if args.stats:
        index = get_index()
        print(json.dumps({
            "papers": len(index.papers),
            "chunks": len(index.chunks),
            "cache": cache.stats(),
            "embeddings": embeddings.backend_name(),
            "llm": "sarvam" if llm.available() else "rule-based fallback",
        }, indent=2))
        return 0

    if args.ingest:
        papers = arxiv_client.ingest(args.ingest, limit=args.limit)
        if not papers:
            print("Nothing ingested (network unavailable or no results).")
            return 1
        index = rebuild()
        print(f"Ingested {len(papers)} paper(s); index now has "
              f"{len(index.papers)} papers / {len(index.chunks)} chunks.")
        for paper in papers:
            print(f"  - {paper['id']}  {paper['title'][:70]}")
        return 0

    questions = DEMO if args.demo else ([args.ask] if args.ask else [])
    if not questions:
        parser.print_help()
        return 2

    for question in questions:
        result = answer(question, use_cache=not args.no_cache)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
