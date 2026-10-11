from __future__ import annotations

import math

import pytest

from eval_harness.models import ModelConfig
from eval_harness.providers.mock import MockProvider
from eval_harness.runner import run_task
from eval_harness.runs import ParsedAnswer, RunRecord
from eval_harness.scoring import (
    bootstrap_ci,
    brier_score,
    log_loss_score,
    macro_f1,
    parse_failure_distribution,
    ranked_probability_score,
    score_batch,
    score_task_record,
)
from eval_harness.tasks import load_tasks


def test_brier_score_perfect_is_zero() -> None:
    assert brier_score({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}, ["a", "b"]) == 0.0


def test_brier_score_maximally_wrong_is_two() -> None:
    assert brier_score({"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 0.0}, ["a", "b"]) == pytest.approx(2.0)


def test_log_loss_perfect_prediction_is_near_zero() -> None:
    loss = log_loss_score({"a": 1.0 - 1e-9, "b": 1e-9}, {"a": 1.0, "b": 0.0}, ["a", "b"])
    assert loss < 1e-5


def test_log_loss_confident_wrong_prediction_is_large() -> None:
    loss = log_loss_score({"a": 1e-9, "b": 1.0 - 1e-9}, {"a": 1.0, "b": 0.0}, ["a", "b"])
    assert loss > 10


def test_ranked_probability_score_perfect_is_zero() -> None:
    classes = ["low", "mid", "high"]
    rps = ranked_probability_score(
        {"low": 1.0, "mid": 0.0, "high": 0.0}, {"low": 1.0, "mid": 0.0, "high": 0.0}, classes
    )
    assert rps == 0.0


def test_ranked_probability_score_penalizes_distance() -> None:
    classes = ["low", "mid", "high"]
    true = {"low": 1.0, "mid": 0.0, "high": 0.0}
    near = ranked_probability_score({"low": 0.0, "mid": 1.0, "high": 0.0}, true, classes)
    far = ranked_probability_score({"low": 0.0, "mid": 0.0, "high": 1.0}, true, classes)
    assert far > near > 0.0


def test_macro_f1_perfect_predictions() -> None:
    pairs = [("a", "a"), ("b", "b"), ("c", "c")]
    assert macro_f1(pairs) == pytest.approx(1.0)


def test_macro_f1_all_wrong() -> None:
    pairs = [("a", "b"), ("b", "a")]
    assert macro_f1(pairs) == pytest.approx(0.0)


def test_macro_f1_empty_is_zero() -> None:
    assert macro_f1([]) == 0.0


def test_bootstrap_ci_contains_mean_for_constant_values() -> None:
    lo, hi = bootstrap_ci([0.5, 0.5, 0.5, 0.5], n_resamples=200, seed=0)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    values = [0.1, 0.9, 0.4, 0.6, 0.2]
    a = bootstrap_ci(values, n_resamples=500, seed=42)
    b = bootstrap_ci(values, n_resamples=500, seed=42)
    assert a == b


def test_bootstrap_ci_single_value() -> None:
    assert bootstrap_ci([0.7]) == (0.7, 0.7)


def test_bootstrap_ci_empty() -> None:
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_parse_failure_distribution_puts_mass_on_least_true_class(example_tasks_dir) -> None:
    task = next(t for t in load_tasks([example_tasks_dir]) if t.family == "classification")
    dist = parse_failure_distribution(task)
    assert dist[task.ground_truth.true_label(task.answer_set)] == 0.0
    assert sum(dist.values()) == pytest.approx(1.0)


def _fake_record(
    task_id: str, model_id: str, probabilities: dict, parse_ok: bool = True
) -> RunRecord:
    return RunRecord(
        run_id="r1",
        batch_id="b1",
        task_id=task_id,
        model_id=model_id,
        provider="mock",
        config=ModelConfig(id=model_id, provider="mock", provider_model="echo"),
        prompt=[],
        raw_response="{}",
        parsed_answer=ParsedAnswer(probabilities=probabilities) if parse_ok else None,
        parse_ok=parse_ok,
        retried=False,
        tool_trace=[],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        latency_ms=1.0,
        timestamp="2026-01-01T00:00:00Z",
        seed=0,
    )


def test_score_task_record_perfect_prediction_has_skill_one(example_tasks_dir) -> None:
    task = next(t for t in load_tasks([example_tasks_dir]) if t.family == "classification")
    record = _fake_record(task.id, "perfect", {"low": 1.0, "medium": 0.0, "high": 0.0})
    score = score_task_record(task, record)
    assert score.correct
    assert score.brier == pytest.approx(0.0)
    assert score.skill == pytest.approx(1.0)
    assert score.regret == pytest.approx(0.0)


def test_score_task_record_parse_failure_counts_as_wrong(example_tasks_dir) -> None:
    task = next(t for t in load_tasks([example_tasks_dir]) if t.family == "classification")
    record = _fake_record(task.id, "broken", {}, parse_ok=False)
    score = score_task_record(task, record)
    assert not score.correct
    assert not score.parse_ok


def test_score_batch_end_to_end_with_mock_provider(example_tasks_dir) -> None:
    tasks = load_tasks([example_tasks_dir])
    config = ModelConfig(id="echo-baseline", provider="mock", provider_model="echo", seed=0)
    provider = MockProvider()
    records = [run_task(t, config, provider, "batch-x") for t in tasks]
    report = score_batch(tasks, records, n_resamples=100, seed=0)
    assert len(report.task_scores) == len(tasks)
    assert len(report.model_summaries) == 1
    summary = report.model_summaries[0]
    assert summary.n_tasks == len(tasks)
    assert 0.0 <= summary.parse_failure_rate <= 1.0


def test_score_batch_rejects_unknown_task_id(example_tasks_dir) -> None:
    tasks = load_tasks([example_tasks_dir])
    record = _fake_record("not-a-real-task", "m", {"low": 1.0, "medium": 0.0, "high": 0.0})
    with pytest.raises(ValueError, match="no matching task spec"):
        score_batch(tasks, [record])
