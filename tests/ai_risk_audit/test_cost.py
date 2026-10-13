from __future__ import annotations

import pytest

from ai_risk_audit.cost import BudgetExceededError, CostTracker


def test_record_and_totals() -> None:
    tracker = CostTracker(budget_warn_usd=5.0, budget_abort_usd=10.0)
    tracker.record("profiler", "model-a", 0.01, prompt_tokens=10, completion_tokens=5)
    tracker.record("planner", "model-a", 0.02, prompt_tokens=20, completion_tokens=10)

    assert tracker.total_usd == pytest.approx(0.03)
    assert tracker.by_stage() == {"profiler": pytest.approx(0.01), "planner": pytest.approx(0.02)}


def test_record_aborts_over_budget() -> None:
    tracker = CostTracker(budget_warn_usd=1.0, budget_abort_usd=2.0)
    tracker.record("execution", "model-a", 1.5)
    with pytest.raises(BudgetExceededError):
        tracker.record("execution", "model-a", 1.0)


def test_check_projection_aborts_before_spending() -> None:
    tracker = CostTracker(budget_warn_usd=1.0, budget_abort_usd=2.0)
    with pytest.raises(BudgetExceededError):
        tracker.check_projection(3.0, stage="execution")
    # nothing was actually recorded/spent
    assert tracker.total_usd == 0.0


def test_check_projection_warns_but_does_not_raise(capsys: pytest.CaptureFixture[str]) -> None:
    tracker = CostTracker(budget_warn_usd=1.0, budget_abort_usd=5.0)
    tracker.check_projection(2.0, stage="execution")
    captured = capsys.readouterr()
    assert "warning" in captured.out
