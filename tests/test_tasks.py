from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eval_harness.tasks import TaskLoadError, load_task, load_tasks

VALID_TASK: dict = {
    "id": "t1",
    "track": "claims",
    "family": "classification",
    "title": "Title",
    "description": "Desc",
    "input": {"kind": "inline", "text": "some input text"},
    "answer_set": ["a", "b", "c"],
    "ground_truth": {"label": "a"},
    "scoring": {"metric": "classification"},
    "baseline": {"strategy": "uniform"},
    "oracle": {"kind": "ground_truth"},
}


def _write(tmp_path: Path, data: dict, name: str = "task.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data))
    return path


def test_load_valid_task(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID_TASK)
    task = load_task(path)
    assert task.id == "t1"
    assert task.answer_set == ["a", "b", "c"]
    assert task.ground_truth.true_label(task.answer_set) == "a"
    assert task.rendered_input() == "some input text"


def test_malformed_yaml_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("id: [unterminated\n  nope: :::")
    with pytest.raises(TaskLoadError):
        load_task(path)


def test_missing_required_field_fails(tmp_path: Path) -> None:
    data = dict(VALID_TASK)
    del data["answer_set"]
    path = _write(tmp_path, data)
    with pytest.raises(TaskLoadError):
        load_task(path)


def test_ground_truth_label_not_in_answer_set_fails(tmp_path: Path) -> None:
    data = {**VALID_TASK, "ground_truth": {"label": "zzz"}}
    path = _write(tmp_path, data)
    with pytest.raises(TaskLoadError):
        load_task(path)


def test_scoring_metric_must_match_family(tmp_path: Path) -> None:
    data = {**VALID_TASK, "scoring": {"metric": "ordered_band"}}
    path = _write(tmp_path, data)
    with pytest.raises(TaskLoadError):
        load_task(path)


def test_duplicate_answer_set_entries_fails(tmp_path: Path) -> None:
    data = {**VALID_TASK, "answer_set": ["a", "a", "b"]}
    path = _write(tmp_path, data)
    with pytest.raises(TaskLoadError):
        load_task(path)


def test_baseline_majority_class_requires_label(tmp_path: Path) -> None:
    data = {**VALID_TASK, "baseline": {"strategy": "majority_class"}}
    path = _write(tmp_path, data)
    with pytest.raises(TaskLoadError):
        load_task(path)


def test_baseline_label_must_be_in_answer_set(tmp_path: Path) -> None:
    data = {**VALID_TASK, "baseline": {"strategy": "majority_class", "label": "zzz"}}
    path = _write(tmp_path, data)
    with pytest.raises(TaskLoadError):
        load_task(path)


def test_ground_truth_probabilities_must_sum_to_one(tmp_path: Path) -> None:
    data = {**VALID_TASK, "ground_truth": {"probabilities": {"a": 0.5, "b": 0.5, "c": 0.5}}}
    path = _write(tmp_path, data)
    with pytest.raises(TaskLoadError):
        load_task(path)


def test_input_file_kind_requires_files(tmp_path: Path) -> None:
    data = {**VALID_TASK, "input": {"kind": "file"}}
    path = _write(tmp_path, data)
    with pytest.raises(TaskLoadError):
        load_task(path)


def test_input_file_kind_reads_referenced_file(tmp_path: Path) -> None:
    (tmp_path / "packet.txt").write_text("packet contents")
    data = {**VALID_TASK, "input": {"kind": "file", "files": ["packet.txt"]}}
    path = _write(tmp_path, data)
    task = load_task(path)
    assert task.rendered_input() == "--- file: packet.txt ---\npacket contents"


def test_load_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    _write(tmp_path, VALID_TASK, "a.yaml")
    _write(tmp_path, VALID_TASK, "b.yaml")
    with pytest.raises(TaskLoadError):
        load_tasks([tmp_path])


def test_load_tasks_missing_path_fails() -> None:
    with pytest.raises(TaskLoadError):
        load_tasks(["/nonexistent/path/for/sure"])


def test_example_tasks_load(example_tasks_dir: Path) -> None:
    tasks = load_tasks([example_tasks_dir])
    ids = {t.id for t in tasks}
    assert ids == {"claims_triage_3class", "reserve_adequacy_band"}
    by_id = {t.id: t for t in tasks}
    assert by_id["claims_triage_3class"].family == "classification"
    assert by_id["reserve_adequacy_band"].family == "ordered_band"
