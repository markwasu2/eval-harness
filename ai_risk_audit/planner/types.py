"""Data types for the audit-planner stage."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RiskCategory(StrEnum):
    IP_INFRINGEMENT = "ip_infringement"
    HARMFUL_OUTPUT = "harmful_output"
    DATA_LEAKAGE = "data_leakage"
    TOOL_CALL_ABUSE = "tool_call_abuse"
    PROMPT_INJECTION = "prompt_injection"


class Scenario(BaseModel):
    """One adversarial test scenario mapped to a single risk category."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: RiskCategory
    title: str
    seed_instructions: str
    """Instructions handed to Petri's auditor model for this scenario."""

    pass_criteria: str
    fail_criteria: str
    repeats: int = Field(default=2, ge=1)


class AuditPlan(BaseModel):
    """Full test plan for one target: scenarios across all risk categories."""

    model_config = ConfigDict(extra="forbid")

    target_name: str
    scenarios: list[Scenario]
