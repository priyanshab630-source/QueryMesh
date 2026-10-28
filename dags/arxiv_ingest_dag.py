"""Airflow DAG: daily arXiv ingestion.

Drops one JSON file per paper into data/papers/, which the hybrid index picks up
on the next rebuild. Copy this file into your Airflow `dags/` folder, or point
Airflow at this repo's `dags/` directory.

    airflow dags trigger querymesh_arxiv_ingest
"""

from datetime import datetime, timedelta

try:
    from airflow import DAG

    try:                                        # Airflow 3
        from airflow.providers.standard.operators.python import PythonOperator
    except ImportError:                         # Airflow 2
        from airflow.operators.python import PythonOperator
except ImportError:                              # Airflow is optional
    DAG = None

TOPICS = [
    "retrieval augmented generation",
    "multi-agent large language models",
    "graph neural network drug discovery",
]

default_args = {
    "owner": "querymesh",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def ingest_topic(topic: str, limit: int = 10) -> int:
    from src.tools import arxiv_client

    papers = arxiv_client.ingest(topic, limit=limit)
    print(f"[ingest] {len(papers)} paper(s) for '{topic}'")
    return len(papers)


if DAG is not None:
    with DAG(
        dag_id="querymesh_arxiv_ingest",
        description="Fetch new arXiv papers into the QueryMesh corpus",
        default_args=default_args,
        start_date=datetime(2024, 1, 1),
        schedule="@daily",
        catchup=False,
        tags=["querymesh", "rag"],
    ) as dag:
        for topic in TOPICS:
            PythonOperator(
                task_id="ingest_" + topic.replace(" ", "_"),
                python_callable=ingest_topic,
                op_kwargs={"topic": topic},
            )
