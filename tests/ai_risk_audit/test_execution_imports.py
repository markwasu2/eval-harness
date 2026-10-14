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


def test_openai_cs_target_agent_module_imports_against_real_petri_api() -> None:
    from ai_risk_audit.execution import openai_cs_target_agent

    assert callable(openai_cs_target_agent.make_openai_cs_target_agent)
    assert callable(openai_cs_target_agent._input_items_to_dicts)


def test_estimate_projected_usd_is_pure_and_positive() -> None:
    from ai_risk_audit.execution.run import estimate_projected_usd

    def _plan(repeats: int) -> AuditPlan:
        return AuditPlan(
            target_name="Acme Co",
            scenarios=[
                Scenario(
                    id="data_leakage-1",
                    category=RiskCategory.DATA_LEAKAGE,
                    title="t",
                    seed_instructions="i",
                    pass_criteria="p",
                    fail_criteria="f",
                    repeats=repeats,
                )
            ],
        )

    config = AuditConfig()
    projected = estimate_projected_usd(_plan(2), config)
    assert projected > 0
    # scales linearly with each scenario's own repeats count
    doubled = estimate_projected_usd(_plan(4), config)
    assert doubled == projected * 2
