"""Runs an `AuditPlan` against the target via Petri.

Each scenario in the plan becomes one Petri seed; `eval_async(audit(...))`
drives the full auditor<->target multi-turn conversation using a pluggable
`target_agent` (the mock `acme_target_agent` by default; `pipeline.py` swaps
in `tau_target_agent.make_tau_airline_target_agent(...)` for the tau2-bench
airline target via `AuditConfig.target_kind`). Petri's own built-in judge
scorer is always attached internally (unavoidable — see `audit()`'s
docstring) but pointed at the cheapest configured model and given
`judge_dimensions=[]` to minimize its cost; this pipeline's own scoring
module (stage 4) never reads its output.

Per-category epoch counts: Petri's `eval_async(..., epochs=N)` applies one
epoch count to every sample in that call's dataset — there is no per-sample
epoch control within a single call. `run_plan` works around this by
grouping `plan.scenarios` by their own `Scenario.repeats` value (set per
category from `AuditConfig.repeats_overrides` in `planner/plan.build_plan`)
and issuing one independent `eval_async` call per distinct group, each
pointed at its own seeds subdirectory and its own `epochs=`. All groups'
samples are written into the same transcript batch file, so downstream
scoring/aggregation sees one run regardless of how many eval_async calls
produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from inspect_ai import eval_async
from inspect_ai.log import EvalSample
from inspect_petri import audit

from ai_risk_audit.config import AuditConfig
from ai_risk_audit.cost import CostTracker
from ai_risk_audit.execution import transcript as transcript_store
from ai_risk_audit.execution.target_agent import (
    TARGET_MESSAGES_STORE_KEY,
    acme_target_agent,
)
from ai_risk_audit.llm import anthropic_cost_usd
from ai_risk_audit.planner.seeds import render_seed_files
from ai_risk_audit.planner.types import AuditPlan, Scenario

_ASSUMED_TOKENS_PER_TURN = 600
"""Conservative per-turn token estimate (prompt+completion combined) used
only for the pre-flight spend projection, never for real cost accounting
(which always comes from actual usage/cost data read back after the run)."""

_ASSUMED_USD_PER_1K_TOKENS = 0.01
"""Conservative blended $/1k-token estimate across auditor+target+judge
cheap-tier models, used only for the pre-flight projection."""


class ExecutionError(Exception):
    """Raised when a Petri run returns a sample this pipeline can't map
    back to a known scenario — surfaced rather than guessed at."""


def estimate_projected_usd(plan: AuditPlan, config: AuditConfig) -> float:
    """Conservative over-estimate of stage-3 spend, printed before any real
    call is made so the user can check the projection against the budget.

    Sums each scenario's own `repeats` (set per category from
    `config.repeats_overrides` at plan-build time) rather than assuming a
    single global `config.repeats` for every scenario, so a category-level
    override is reflected in the pre-flight projection too."""
    n_runs = sum(s.repeats for s in plan.scenarios)
    tokens_per_run = config.max_turns * 2 * _ASSUMED_TOKENS_PER_TURN
    return n_runs * tokens_per_run / 1000 * _ASSUMED_USD_PER_1K_TOKENS


def _role_model(name: str) -> str:
    return f"anthropic/{name}"


async def run_plan(
    plan: AuditPlan,
    config: AuditConfig,
    out_dir: str | Path,
    cost_tracker: CostTracker,
    target_agent: Any = acme_target_agent,
    target_display_name: str = "Acme Assist",
    target_messages_store_key: str = TARGET_MESSAGES_STORE_KEY,
) -> Path:
    """Runs every scenario in `plan` against `target_agent` and writes one
    immutable transcript batch. Returns the path to the batch file.

    `target_agent`/`target_display_name`/`target_messages_store_key` default
    to the mock Acme target; `pipeline.py` passes the tau2-bench airline
    adapter's equivalents when `config.target_kind == "tau2_airline"`.
    """
    out_dir = Path(out_dir)

    model_roles = {
        "auditor": _role_model(config.auditor_model.provider_model),
        "target": _role_model(config.target_model.provider_model),
        "judge": _role_model(config.petri_judge_model.provider_model),
    }

    batch_id = transcript_store.new_batch_id()
    batch_file = transcript_store.batch_path(out_dir / "transcripts", batch_id)
    scenario_by_id = {s.id: s for s in plan.scenarios}

    # Group by each scenario's own `repeats` (see module docstring): one
    # `eval_async` call per distinct epoch count, not per scenario, so
    # categories sharing the same count (the common case) still run in a
    # single call.
    repeats_groups: dict[int, list[Scenario]] = {}
    for scenario in plan.scenarios:
        repeats_groups.setdefault(scenario.repeats, []).append(scenario)

    for repeats, group_scenarios in sorted(repeats_groups.items()):
        group_seeds_dir = out_dir / "seeds" / f"repeats_{repeats}"
        render_seed_files(
            AuditPlan(target_name=plan.target_name, scenarios=group_scenarios),
            group_seeds_dir,
        )

        logs = await eval_async(
            audit(
                seed_instructions=str(group_seeds_dir),
                max_turns=config.max_turns,
                target_tools="none",
                judge_dimensions=[],
                target=target_agent,
                target_display_name=target_display_name,
            ),
            model_roles=model_roles,
            epochs=repeats,
            log_dir=str(out_dir / "inspect_logs" / f"repeats_{repeats}"),
        )

        for log in logs:
            for sample in log.samples or []:
                record = _to_transcript_record(
                    sample,
                    batch_id,
                    scenario_by_id,
                    model_roles,
                    config,
                    target_messages_store_key,
                )
                transcript_store.write_record(batch_file, record)
                # For targets whose own LLM calls bypass Inspect's role-based
                # usage tracking (e.g. the tau2-bench airline agent — see
                # `tau_target_agent.py`'s module docstring part 3), `target_agent`
                # itself records that spend into `cost_tracker` directly as it
                # happens; `record.usd_cost` here only ever reflects role_usage
                # Inspect actually observed (auditor + petri's judge, and the
                # target too for role-based targets like the mock Acme one), so
                # this call never double-counts across the two paths.
                cost_tracker.record(
                    stage="execution",
                    model=model_roles["target"],
                    usd=record.usd_cost,
                    prompt_tokens=record.prompt_tokens,
                    completion_tokens=record.completion_tokens,
                )

    return batch_file


def _to_transcript_record(
    sample: EvalSample,
    batch_id: str,
    scenario_by_id: dict[str, Scenario],
    model_roles: dict[str, str],
    config: AuditConfig,
    target_messages_store_key: str = TARGET_MESSAGES_STORE_KEY,
) -> transcript_store.TranscriptRecord:
    scenario_id = str(sample.id)
    scenario = scenario_by_id.get(scenario_id)
    if scenario is None:
        raise ExecutionError(
            f"Petri returned sample id {scenario_id!r} with no matching "
            f"scenario in the plan — cannot attribute a category"
        )

    # `ModelUsage.total_cost` comes back `None`/0 here: Inspect has no
    # pricing-table entry for these model ids (confirmed by the stage-3
    # smoke test), so price each role's usage ourselves from token counts.
    usd_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    for role, usage in (sample.role_usage or {}).items():
        prompt_tokens += usage.input_tokens
        completion_tokens += usage.output_tokens
        role_model = model_roles.get(role)
        if role_model is None:
            print(f"[execution] warning: no known model for role {role!r}; cost for "
                  f"its usage ({usage.input_tokens} in / {usage.output_tokens} out "
                  f"tokens) is NOT reflected in the cost tracker")
            continue
        usd_cost += anthropic_cost_usd(
            role_model.removeprefix("anthropic/"),
            usage.input_tokens,
            usage.output_tokens,
            cache_write_tokens=usage.input_tokens_cache_write or 0,
            cache_read_tokens=usage.input_tokens_cache_read or 0,
        )

    # `sample.messages` is the AUDITOR's own message history (the audit
    # sample's top-level agent is the auditor, not our target) — it is not
    # this target's conversation. The target's own turns are recovered from
    # the sample store, which `target_agent` mirrors them into on every
    # turn (see `execution/target_agent.py` / `execution/tau_target_agent.py`
    # module docstrings).
    raw_messages = sample.store.get(target_messages_store_key)
    if not raw_messages:
        raise ExecutionError(
            f"sample {scenario_id!r} has no target-conversation messages in "
            f"its store under {target_messages_store_key!r} — the target "
            f"agent may have failed before its first turn"
        )
    messages = [transcript_store.TranscriptMessage(**m) for m in raw_messages]

    return transcript_store.TranscriptRecord(
        transcript_id=transcript_store.new_transcript_id(),
        batch_id=batch_id,
        scenario_id=scenario_id,
        category=scenario.category,
        epoch=sample.epoch,
        auditor_model=model_roles["auditor"],
        target_model=model_roles["target"],
        judge_model=config.judge_model.provider_model,
        messages=messages,
        usd_cost=usd_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        started_at=sample.started_at,
        completed_at=sample.completed_at,
    )
