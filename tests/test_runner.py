from __future__ import annotations

import pytest

from eval_harness.models import ModelConfig
from eval_harness.providers.mock import MockProvider
from eval_harness.runner import ParseError, build_messages, parse_answer, run_task
from eval_harness.tasks import load_tasks


@pytest.fixture
def classification_task(example_tasks_dir):
    tasks = load_tasks([example_tasks_dir])
    return next(t for t in tasks if t.family == "classification")


@pytest.fixture
def ordered_band_task(example_tasks_dir):
    tasks = load_tasks([example_tasks_dir])
    return next(t for t in tasks if t.family == "ordered_band")


def test_build_messages_contains_answer_set_marker(classification_task) -> None:
    messages = build_messages(classification_task)
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert "ANSWER_SET: [\"low\", \"medium\", \"high\"]" in messages[1].content


def test_parse_answer_happy_path() -> None:
    parsed = parse_answer('{"probabilities": {"a": 0.2, "b": 0.3, "c": 0.5}}', ["a", "b", "c"])
    assert parsed.probabilities == {"a": 0.2, "b": 0.3, "c": 0.5}


def test_parse_answer_strips_markdown_fences() -> None:
    content = '```json\n{"probabilities": {"a": 1.0, "b": 0.0}}\n```'
    parsed = parse_answer(content, ["a", "b"])
    assert parsed.probabilities == {"a": 1.0, "b": 0.0}


def test_parse_answer_renormalizes_near_one() -> None:
    parsed = parse_answer('{"probabilities": {"a": 0.5, "b": 0.52}}', ["a", "b"])
    assert abs(sum(parsed.probabilities.values()) - 1.0) < 1e-9


def test_parse_answer_rejects_not_json() -> None:
    with pytest.raises(ParseError):
        parse_answer("this is not json at all", ["a", "b"])


def test_parse_answer_rejects_missing_key() -> None:
    with pytest.raises(ParseError):
        parse_answer('{"answer": "a"}', ["a", "b"])


def test_parse_answer_rejects_wrong_classes() -> None:
    with pytest.raises(ParseError):
        parse_answer('{"probabilities": {"a": 1.0, "zzz": 0.0}}', ["a", "b"])


def test_parse_answer_rejects_negative_values() -> None:
    with pytest.raises(ParseError):
        parse_answer('{"probabilities": {"a": 1.5, "b": -0.5}}', ["a", "b"])


def test_parse_answer_rejects_bad_sum() -> None:
    with pytest.raises(ParseError):
        parse_answer('{"probabilities": {"a": 0.1, "b": 0.1}}', ["a", "b"])


def test_run_task_happy_path_uses_tool_and_parses(classification_task) -> None:
    config = ModelConfig(id="echo-baseline", provider="mock", provider_model="echo", seed=0)
    record = run_task(classification_task, config, MockProvider(), "batch-1")
    assert record.parse_ok
    assert not record.retried
    assert record.parsed_answer is not None
    assert set(record.parsed_answer.probabilities) == set(classification_task.answer_set)
    assert len(record.tool_trace) == 1
    assert record.tool_trace[0].result.exit_code == 0


def test_run_task_flaky_retries_then_succeeds(classification_task) -> None:
    config = ModelConfig(
        id="echo-flaky", provider="mock", provider_model="echo-flaky", seed=1, tool_use=False
    )
    record = run_task(classification_task, config, MockProvider(), "batch-1")
    assert record.parse_ok
    assert record.retried


def test_run_task_permanent_failure_is_recorded(classification_task) -> None:
    config = ModelConfig(
        id="echo-fail", provider="mock", provider_model="echo-fail", seed=2, tool_use=False
    )
    record = run_task(classification_task, config, MockProvider(), "batch-1")
    assert not record.parse_ok
    assert record.retried
    assert record.parsed_answer is None


def test_run_task_ordered_band_task_round_trips(ordered_band_task) -> None:
    config = ModelConfig(
        id="echo-baseline", provider="mock", provider_model="echo", seed=0, tool_use=False
    )
    record = run_task(ordered_band_task, config, MockProvider(), "batch-1")
    assert record.parse_ok
    assert set(record.parsed_answer.probabilities) == set(ordered_band_task.answer_set)
