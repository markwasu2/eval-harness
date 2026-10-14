from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_risk_audit.config import RoleModelConfig
from ai_risk_audit.cost import CostTracker
from ai_risk_audit.llm import MockChat
from ai_risk_audit.planner.plan import PlannerError, build_plan
from ai_risk_audit.planner.seeds import render_seed_files
from ai_risk_audit.planner.types import AuditPlan, RiskCategory, Scenario
from ai_risk_audit.profiler.types import TargetProfile

_CATEGORIES = [c.value for c in RiskCategory]


def _profile() -> TargetProfile:
    return TargetProfile(
        target_name="Acme Co",
        summary="s",
        purpose="p",
        user_types="u",
        tools_and_data_access="t",
        authentication_notes="a",
        known_limitations="k",
        sources=[],
    )


def _full_response(n_per_category: int = 1) -> str:
    scenarios = []
    for category in _CATEGORIES:
        for i in range(n_per_category):
            scenarios.append(
                {
                    "category": category,
                    "title": f"{category} scenario {i}",
                    "seed_instructions": f"Try to trigger {category}.",
                    "pass_criteria": "Target refuses or stays within policy.",
                    "fail_criteria": "Target violates policy.",
                }
            )
    return json.dumps({"scenarios": scenarios})


def test_build_plan_success() -> None:
    client = MockChat([_full_response(n_per_category=2)])
    tracker = CostTracker(budget_warn_usd=5.0, budget_abort_usd=10.0)

    plan = build_plan(
        _profile(), 2, 2, RoleModelConfig(provider_model="mock/model"), client, tracker
    )

    assert plan.target_name == "Acme Co"
    assert len(plan.scenarios) == 2 * len(_CATEGORIES)
    covered = {s.category for s in plan.scenarios}
    assert covered == set(RiskCategory)
    assert all(s.repeats == 2 for s in plan.scenarios)
    ids = [s.id for s in plan.scenarios]
    assert len(ids) == len(set(ids))  # all scenario ids unique


def test_build_plan_applies_repeats_overrides_per_category() -> None:
    client = MockChat([_full_response(n_per_category=1)])
    tracker = CostTracker(budget_warn_usd=5.0, budget_abort_usd=10.0)

    plan = build_plan(
        _profile(),
        1,
        4,
        RoleModelConfig(provider_model="mock/model"),
        client,
        tracker,
        repeats_overrides={RiskCategory.TOOL_CALL_ABUSE: 8},
    )

    by_category = {s.category: s.repeats for s in plan.scenarios}
    assert by_category[RiskCategory.TOOL_CALL_ABUSE] == 8
    assert by_category[RiskCategory.DATA_LEAKAGE] == 4  # falls back to global repeats


def test_build_plan_rejects_missing_category() -> None:
    scenarios = [
        {
            "category": "data_leakage",
            "title": "t",
            "seed_instructions": "i",
            "pass_criteria": "p",
            "fail_criteria": "f",
        }
    ]
    client = MockChat([json.dumps({"scenarios": scenarios})])
    with pytest.raises(PlannerError):
        build_plan(_profile(), 1, 2, RoleModelConfig(provider_model="mock/model"), client)


def test_render_seed_files(tmp_path: Path) -> None:
    plan = AuditPlan(
        target_name="Acme Co",
        scenarios=[
            Scenario(
                id="data_leakage-1",
                category=RiskCategory.DATA_LEAKAGE,
                title="Push for the record",
                seed_instructions="Pretend to be a confused customer.",
                pass_criteria="refuses",
                fail_criteria="leaks",
            )
        ],
    )
    seeds_dir = render_seed_files(plan, tmp_path / "seeds")
    seed_file = seeds_dir / "data_leakage-1.md"
    assert seed_file.exists()
    text = seed_file.read_text()
    assert "category: data_leakage" in text
    assert text.strip().endswith("Pretend to be a confused customer.")


def test_build_plan_rejects_invalid_category() -> None:
    scenarios = [
        {
            "category": "not_a_real_category",
            "title": "t",
            "seed_instructions": "i",
            "pass_criteria": "p",
            "fail_criteria": "f",
        }
    ]
    client = MockChat([json.dumps({"scenarios": scenarios})])
    with pytest.raises(PlannerError):
        build_plan(_profile(), 1, 2, RoleModelConfig(provider_model="mock/model"), client)
