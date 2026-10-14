"""Top-level orchestrator wiring all five audit-pipeline stages together.

Stop conditions enforced here (per the pipeline's definition of done):
- Projected spend is estimated and checked against the budget BEFORE stage
  3 (execution) runs; if it would pass `budget_abort_usd`, the pipeline
  aborts with a written report documenting the reason and cost so far
  rather than spending the money.
- A score that can't be backed with cited evidence is never silently
  accepted: `scoring/judge.py` already enforces verbatim-quote evidence
  per call and raises `JudgeError` on repeated failure, which this module
  lets propagate rather than continuing with a partial/fabricated score.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ai_risk_audit.config import AuditConfig
from ai_risk_audit.cost import BudgetExceededError, CostTracker
from ai_risk_audit.execution.run import estimate_projected_usd, run_plan
from ai_risk_audit.execution.target_agent import (
    TARGET_MESSAGES_STORE_KEY,
    acme_target_agent,
)
from ai_risk_audit.execution.tau_target_agent import (
    TAU_TARGET_MESSAGES_STORE_KEY,
    make_tau_airline_target_agent,
)
from ai_risk_audit.execution.transcript import load_records
from ai_risk_audit.llm import ChatClient
from ai_risk_audit.planner.plan import build_plan
from ai_risk_audit.profiler.extract import profile_target
from ai_risk_audit.profiler.sources import save_sources
from ai_risk_audit.profiler.types import SourceDoc
from ai_risk_audit.report.build import build_report, write_report
from ai_risk_audit.scoring.aggregate import aggregate_scores
from ai_risk_audit.scoring.judge import score_transcript


class PipelineAbortedError(Exception):
    """Raised when a stop condition fires. A report documenting why has
    already been written to `out_dir` by the time this is raised."""


def run_audit(
    target_name: str,
    sources: list[SourceDoc],
    config: AuditConfig,
    client: ChatClient,
    out_dir: str | Path,
) -> Path:
    """Runs the full 5-stage pipeline synchronously. Returns `out_dir`."""
    out_dir = Path(out_dir)
    cost_tracker = CostTracker(
        budget_warn_usd=config.budget_warn_usd,
        budget_abort_usd=config.budget_abort_usd,
    )

    try:
        profile = profile_target(
            target_name, sources, config.profiler_model, client, cost_tracker
        )
        save_sources(sources, out_dir)

        plan = build_plan(
            profile,
            config.scenarios_per_category,
            config.repeats,
            config.planner_model,
            client,
            cost_tracker,
        )

        projected = estimate_projected_usd(plan, config)
        print(f"[pipeline] projected execution spend: ${projected:.4f}")
        cost_tracker.check_projection(projected, stage="execution")

        if config.target_kind == "tau2_airline":
            target_agent = make_tau_airline_target_agent(
                cost_tracker, config.target_model.provider_model
            )
            target_display_name = "tau2-bench Airline Agent"
            target_messages_store_key = TAU_TARGET_MESSAGES_STORE_KEY
            environment_note = (
                "tau2-bench (https://github.com/sierra-research/tau2-bench) airline "
                "domain agent, stock environment/policy/tools, running locally against "
                f"agent model {config.target_model.provider_model!r}. This is a "
                "benchmark harness's reference agent, not a deployed product."
            )
        else:
            target_agent = acme_target_agent
            target_display_name = "Acme Assist"
            target_messages_store_key = TARGET_MESSAGES_STORE_KEY
            environment_note = None

        batch_file = asyncio.run(
            run_plan(
                plan,
                config,
                out_dir,
                cost_tracker,
                target_agent=target_agent,
                target_display_name=target_display_name,
                target_messages_store_key=target_messages_store_key,
            )
        )

        records = load_records(batch_file)
        scenarios_by_id = {s.id: s for s in plan.scenarios}
        scores = [
            score_transcript(
                record,
                scenarios_by_id[record.scenario_id],
                config.judge_model,
                client,
                cost_tracker,
            )
            for record in records
        ]

        summary = aggregate_scores(scores)
        report = build_report(
            profile, plan, summary, cost_tracker.total_usd, environment_note=environment_note
        )
        write_report(report, out_dir)
    except BudgetExceededError as exc:
        _write_abort_report(out_dir, target_name, cost_tracker, reason=str(exc))
        raise PipelineAbortedError(str(exc)) from exc
    finally:
        cost_tracker.print_summary()

    return out_dir


def _write_abort_report(
    out_dir: Path, target_name: str, cost_tracker: CostTracker, reason: str
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Audit aborted: {target_name}\n",
        f"**Reason:** {reason}\n",
        "## Spend at time of abort\n",
    ]
    for stage, usd in cost_tracker.by_stage().items():
        lines.append(f"- {stage}: ${usd:.4f}")
    lines.append(f"- TOTAL: ${cost_tracker.total_usd:.4f}\n")
    (out_dir / "ABORTED.md").write_text("\n".join(lines) + "\n")
