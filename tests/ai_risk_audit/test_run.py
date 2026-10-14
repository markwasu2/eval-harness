"""Offline tests for `execution/run.py`'s per-category epoch grouping.

Mocks `audit()`/`eval_async()` (both real `inspect_ai`/`inspect_petri` calls)
so this never makes a network call or needs a live Petri run — it only
checks that `run_plan` issues one `eval_async` call per distinct
`Scenario.repeats` value, each with the right seed files and `epochs=`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ai_risk_audit.config import AuditConfig
from ai_risk_audit.cost import CostTracker
from ai_risk_audit.execution import run as run_module
from ai_risk_audit.planner.types import AuditPlan, RiskCategory, Scenario


def _scenario(category: RiskCategory, idx: int, repeats: int) -> Scenario:
    return Scenario(
        id=f"{category.value}-{idx}",
        category=category,
        title="t",
        seed_instructions="i",
        pass_criteria="p",
        fail_criteria="f",
        repeats=repeats,
    )


def test_run_plan_issues_one_eval_async_call_per_distinct_repeats_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = AuditPlan(
        target_name="Acme Co",
        scenarios=[
            _scenario(RiskCategory.DATA_LEAKAGE, 1, repeats=4),
            _scenario(RiskCategory.HARMFUL_OUTPUT, 1, repeats=4),
            _scenario(RiskCategory.TOOL_CALL_ABUSE, 1, repeats=8),
        ],
    )
    config = AuditConfig()
    cost_tracker = CostTracker(budget_warn_usd=5.0, budget_abort_usd=10.0)

    audit_calls: list[dict] = []
    eval_calls: list[dict] = []

    def fake_audit(**kwargs):
        audit_calls.append(kwargs)
        return kwargs  # sentinel "task", only ever passed back to fake_eval_async

    async def fake_eval_async(task, **kwargs):
        eval_calls.append(kwargs)
        return []  # no samples: this test only checks call grouping, not scoring

    monkeypatch.setattr(run_module, "audit", fake_audit)
    monkeypatch.setattr(run_module, "eval_async", fake_eval_async)

    asyncio.run(run_module.run_plan(plan, config, tmp_path, cost_tracker))

    assert len(eval_calls) == 2
    assert {c["epochs"] for c in eval_calls} == {4, 8}

    assert len(audit_calls) == 2
    seed_dirs = [Path(c["seed_instructions"]) for c in audit_calls]
    seed_stems_by_group = [
        {p.stem for p in d.glob("*.md")} for d in seed_dirs
    ]
    assert {"data_leakage-1", "harmful_output-1"} in seed_stems_by_group
    assert {"tool_call_abuse-1"} in seed_stems_by_group


def test_run_plan_single_group_when_no_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = AuditPlan(
        target_name="Acme Co",
        scenarios=[
            _scenario(RiskCategory.DATA_LEAKAGE, 1, repeats=4),
            _scenario(RiskCategory.HARMFUL_OUTPUT, 1, repeats=4),
        ],
    )
    config = AuditConfig()
    cost_tracker = CostTracker(budget_warn_usd=5.0, budget_abort_usd=10.0)

    eval_calls: list[dict] = []

    def fake_audit(**kwargs):
        return kwargs

    async def fake_eval_async(task, **kwargs):
        eval_calls.append(kwargs)
        return []

    monkeypatch.setattr(run_module, "audit", fake_audit)
    monkeypatch.setattr(run_module, "eval_async", fake_eval_async)

    asyncio.run(run_module.run_plan(plan, config, tmp_path, cost_tracker))

    assert len(eval_calls) == 1
    assert eval_calls[0]["epochs"] == 4
