"""Import smoke tests against the real installed inspect_ai/inspect_petri
APIs, to catch signature/field-name drift that offline mocks can't see.
"""

from __future__ import annotations

from ai_risk_audit.config import AuditConfig
from ai_risk_audit.planner.types import AuditPlan, RiskCategory, Scenario


def test_target_agent_module_imports_against_real_petri_api() -> None:
    from ai_risk_audit.execution import target_agent

    assert callable(target_agent.acme_target_agent)
    assert len(target_agent._TOOL_INFOS) == 3


def test_run_module_imports_against_real_inspect_api() -> None:
    from ai_risk_audit.execution import run

    assert callable(run.run_plan)


def test_tau_target_agent_module_imports_against_real_tau2_and_petri_api() -> None:
    from ai_risk_audit.execution import tau_target_agent

    assert callable(tau_target_agent.make_tau_airline_target_agent)
    assert callable(tau_target_agent.run_agent_turn)


def test_estimate_projected_usd_is_pure_and_positive() -> None:
    from ai_risk_audit.execution.run import estimate_projected_usd

    plan = AuditPlan(
        target_name="Acme Co",
        scenarios=[
            Scenario(
                id="data_leakage-1",
                category=RiskCategory.DATA_LEAKAGE,
                title="t",
                seed_instructions="i",
                pass_criteria="p",
                fail_criteria="f",
            )
        ],
    )
    config = AuditConfig()
    projected = estimate_projected_usd(plan, config)
    assert projected > 0
    # scales linearly with repeats
    doubled_config = config.model_copy(update={"repeats": config.repeats * 2})
    doubled = estimate_projected_usd(plan, doubled_config)
    assert doubled == projected * 2
