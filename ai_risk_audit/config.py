"""Single config file controlling which model plays each role.

Every LLM call in the audit pipeline (profiler extraction, scenario
planning, Petri's auditor, judge scoring, and the mock target's own "brain")
is named here. Calls go directly through the Anthropic API (`ANTHROPIC_API_KEY`
in the environment; see `.env`) by default — there was no funded OpenRouter
account available, so `llm.py` and `execution/run.py` route "bare" model
ids (no provider prefix) straight to Anthropic. Defaults are cheap models for
every role except `judge_model`, which defaults to a stronger model since
score quality there is the thing the pipeline most needs to be trusted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ai_risk_audit.planner.types import RiskCategory

DEFAULT_CHEAP_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"


class RoleModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_model: str
    temperature: float = 0.0
    max_tokens: int = 2048


class AuditConfig(BaseModel):
    """Model-per-role config plus the budget knobs for one audit run."""

    model_config = ConfigDict(extra="forbid")

    profiler_model: RoleModelConfig = Field(
        default_factory=lambda: RoleModelConfig(provider_model=DEFAULT_CHEAP_MODEL)
    )
    planner_model: RoleModelConfig = Field(
        default_factory=lambda: RoleModelConfig(provider_model=DEFAULT_CHEAP_MODEL)
    )
    auditor_model: RoleModelConfig = Field(
        default_factory=lambda: RoleModelConfig(provider_model=DEFAULT_CHEAP_MODEL)
    )
    target_model: RoleModelConfig = Field(
        default_factory=lambda: RoleModelConfig(provider_model=DEFAULT_CHEAP_MODEL)
    )
    judge_model: RoleModelConfig = Field(
        default_factory=lambda: RoleModelConfig(provider_model=DEFAULT_JUDGE_MODEL)
    )
    petri_judge_model: RoleModelConfig = Field(
        default_factory=lambda: RoleModelConfig(provider_model=DEFAULT_CHEAP_MODEL)
    )
    """Model role fed to Petri's own built-in judge scorer. Petri's `audit()`
    task always attaches that scorer internally and this pipeline never reads
    its output (see `execution/run.py`); pointing it at the cheapest
    configured model keeps that otherwise-wasted call cheap."""

    target_kind: Literal["acme", "tau2_airline", "openai_cs", "minidesk"] = "acme"
    """Which target `pipeline.run_audit` drives: the built-in mock Acme Co.
    support agent, the tau2-bench airline agent (stock domain/policy,
    simulated user replaced by the auditor — see `execution/tau_target_agent.py`),
    the openai-cs-agents-demo airline multi-agent system (its own
    instructions/tools/handoffs/guardrails, unmodified, but with its backend
    LLM substituted from OpenAI to Claude via litellm since no
    `OPENAI_API_KEY` is available here — see
    `execution/openai_cs_target_agent.py`'s module docstring for the
    disclosed substitution), or MiniDesk, a minimal in-repo helpdesk agent
    scaffolded for this pipeline on the same Agents-SDK-via-litellm pattern
    (see `execution/minidesk_target_agent.py`'s module docstring).
    `target_model` is the model used for that target's own "brain" either
    way."""

    scenarios_per_category: int = 2
    repeats: int = 2
    repeats_overrides: dict[RiskCategory, int] = Field(default_factory=dict)
    """Per-category epoch-count overrides, e.g. {tool_call_abuse: 8}. Any
    category not listed here falls back to `repeats`. `planner/plan.py`'s
    `build_plan` sets each `Scenario.repeats` from this at plan-build time,
    and `execution/run.py`'s `run_plan` groups scenarios by that per-scenario
    `repeats` value into one independent `eval_async` call per distinct
    group — so a category with a higher override actually gets more epochs,
    not just a bigger shared pool (see that module's docstring)."""
    max_turns: int = 6
    budget_warn_usd: float = 3.0
    budget_abort_usd: float = 5.0
    reproducibility_tolerance: float = 10.0
    """+/- tolerance, in points on the 1-100 scale, for calling two runs of
    the same target "reproducible" per-category."""


class AuditConfigError(Exception):
    """Raised when an audit config file fails to parse or validate."""


def load_config(path: str | Path | None) -> AuditConfig:
    if path is None:
        return AuditConfig()
    path = Path(path)
    if not path.is_file():
        raise AuditConfigError(f"audit config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise AuditConfigError(f"{path}: could not parse YAML: {exc}") from exc
    try:
        return AuditConfig.model_validate(data)
    except Exception as exc:
        raise AuditConfigError(f"{path}: invalid audit config: {exc}") from exc
