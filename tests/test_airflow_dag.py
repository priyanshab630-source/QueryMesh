"""The Airflow DAG is parsed by Airflow itself (no scheduler, no database)."""

import pytest

pytest.importorskip("airflow", reason="apache-airflow is an optional extra")

try:                                            # Airflow 3
    from airflow.dag_processing.dagbag import DagBag
except ImportError:                             # Airflow 2
    from airflow.models.dagbag import DagBag

DAG_ID = "querymesh_arxiv_ingest"


def _bag() -> "DagBag":
    return DagBag(dag_folder="dags")


def test_dag_is_discovered_by_the_dagbag():
    bag = _bag()
    assert DAG_ID in bag.dag_ids, bag.import_errors


def test_dag_has_one_task_per_topic():
    # read the in-memory DAG: get_dag() would touch the metadata database,
    # which this test deliberately does not require
    dag = _bag().dags[DAG_ID]
    assert len(dag.tasks) == 3
    assert all("ingest_" in task.task_id for task in dag.tasks)
    assert dag.schedule is not None


def test_task_function_is_the_same_one_the_cli_runs():
    """The DAG and dags/run_ingest.py must not drift apart."""
    from dags import arxiv_ingest_dag, run_ingest

    assert arxiv_ingest_dag.TOPICS == run_ingest.TOPICS
