from __future__ import annotations

from pathlib import Path

from eval_harness.models import ModelConfig
from eval_harness.providers.mock import MockProvider
from eval_harness.runner import run_task
from eval_harness.runs import batch_path, load_records, new_batch_id, write_record
from eval_harness.tasks import load_tasks


def test_new_batch_id_is_unique() -> None:
    assert new_batch_id() != new_batch_id()


def test_write_and_load_record_round_trip(tmp_path: Path, example_tasks_dir: Path) -> None:
    task = load_tasks([example_tasks_dir])[0]
    config = ModelConfig(id="echo-baseline", provider="mock", provider_model="echo", seed=0)
    record = run_task(task, config, MockProvider(), "batch-1")

    path = batch_path(tmp_path, "batch-1")
    write_record(path, record)
    loaded = load_records(path)

    assert len(loaded) == 1
    assert loaded[0].run_id == record.run_id
    assert loaded[0].task_id == record.task_id
    assert loaded[0].parsed_answer == record.parsed_answer


def test_rerun_appends_without_overwriting(tmp_path: Path, example_tasks_dir: Path) -> None:
    task = load_tasks([example_tasks_dir])[0]
    config = ModelConfig(id="echo-baseline", provider="mock", provider_model="echo", seed=0)
    path = batch_path(tmp_path, "batch-1")

    record_a = run_task(task, config, MockProvider(), "batch-1")
    write_record(path, record_a)
    record_b = run_task(task, config, MockProvider(), "batch-1")
    write_record(path, record_b)

    loaded = load_records(path)
    assert len(loaded) == 2
    assert {r.run_id for r in loaded} == {record_a.run_id, record_b.run_id}


def test_new_invocation_gets_a_new_batch_file(tmp_path: Path) -> None:
    first = batch_path(tmp_path, new_batch_id())
    second = batch_path(tmp_path, new_batch_id())
    assert first != second
