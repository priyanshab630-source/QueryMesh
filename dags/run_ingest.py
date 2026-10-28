"""Run the ingestion task locally, without Airflow.

Airflow is the scheduler, not the logic: `ingest_topics` is the same function the
DAG calls, so you can run the job on a laptop, in CI, or from cron:

    python dags/run_ingest.py --limit 5
    python dags/run_ingest.py --topic "graph retrieval" --limit 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TOPICS = [
    "retrieval augmented generation",
    "multi-agent large language models",
    "graph neural network drug discovery",
]


def ingest_topics(topics: list = None, limit: int = 10) -> dict:
    """Fetch papers per topic and rebuild the index. Returns a summary."""
    from src.search.index import rebuild
    from src.tools import arxiv_client

    topics = topics or TOPICS
    per_topic = {}
    for topic in topics:
        papers = arxiv_client.ingest(topic, limit=limit)
        per_topic[topic] = len(papers)

    total = sum(per_topic.values())
    index = rebuild() if total else None
    return {
        "topics": per_topic,
        "ingested": total,
        "papers": len(index.papers) if index else None,
        "chunks": len(index.chunks) if index else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Local arXiv ingestion run")
    parser.add_argument("--topic", action="append", help="repeatable; defaults to the DAG topics")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)

    summary = ingest_topics(args.topic, limit=args.limit)
    for topic, count in summary["topics"].items():
        print(f"  {count:>2} paper(s)  {topic}")
    print(f"ingested {summary['ingested']} paper(s); "
          f"corpus now {summary['papers']} papers / {summary['chunks']} chunks")
    return 0 if summary["ingested"] else 1


if __name__ == "__main__":
    sys.exit(main())
