"""Builds the final structured + readable audit report.

Writes `report.json` (machine-readable, mirrors `AuditReport` exactly) and
`report.md` (human-readable, with every category score backed by quoted
transcript excerpts) into the run's output directory.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ai_risk_audit.planner.types import AuditPlan, Scenario
from ai_risk_audit.profiler.types import TargetProfile
from ai_risk_audit.scoring.types import AuditScoreSummary, CategoryAggregate


class AuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_name: str
    profile: TargetProfile
    plan: AuditPlan
    score_summary: AuditScoreSummary
    total_usd_spent: float


def build_report(
    profile: TargetProfile,
    plan: AuditPlan,
    score_summary: AuditScoreSummary,
    total_usd_spent: float,
) -> AuditReport:
    return AuditReport(
        target_name=profile.target_name,
        profile=profile,
        plan=plan,
        score_summary=score_summary,
        total_usd_spent=total_usd_spent,
    )


def write_report(report: AuditReport, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "report.json"
    json_path.write_text(report.model_dump_json(indent=2))

    md_path = out_dir / "report.md"
    scenarios_by_id = {s.id: s for s in report.plan.scenarios}
    md_path.write_text(_render_markdown(report, scenarios_by_id))

    return json_path, md_path


def _render_markdown(report: AuditReport, scenarios_by_id: dict[str, Scenario]) -> str:
    lines: list[str] = []
    p = report.profile
    lines.append(f"# AI Risk Audit Report: {report.target_name}\n")
    lines.append(f"**Overall score: {report.score_summary.overall_score:.1f} / 100**\n")
    lines.append(f"Total model spend for this run: ${report.total_usd_spent:.4f}\n")

    lines.append("## Target profile\n")
    lines.append(f"- **Summary:** {p.summary}")
    lines.append(f"- **Purpose:** {p.purpose}")
    lines.append(f"- **User types:** {p.user_types}")
    lines.append(f"- **Tools and data access:** {p.tools_and_data_access}")
    lines.append(f"- **Authentication notes:** {p.authentication_notes}")
    lines.append(f"- **Known limitations:** {p.known_limitations}\n")

    lines.append("## Category scores\n")
    for cat_agg in sorted(report.score_summary.categories, key=lambda c: c.category.value):
        lines.extend(_render_category(cat_agg, scenarios_by_id))

    return "\n".join(lines) + "\n"


def _render_category(
    cat_agg: CategoryAggregate, scenarios_by_id: dict[str, Scenario]
) -> list[str]:
    lines = [
        f"### {cat_agg.category.value} — median {cat_agg.median_score:.1f} "
        f"/ 100 (weight {cat_agg.weight:.2f}, n={cat_agg.n_scores})\n"
    ]
    for score in cat_agg.scores:
        scenario = scenarios_by_id.get(score.scenario_id)
        title = scenario.title if scenario is not None else score.scenario_id
        lines.append(
            f"**Scenario: {title}** (transcript `{score.transcript_id}`) — "
            f"score {score.score}"
        )
        lines.append(f"\n{score.reasoning}\n")
        for excerpt in score.excerpts:
            quote = excerpt.quote.replace("\n", "\n> ")
            lines.append(f"> **{excerpt.speaker}:** {quote}\n")
    return lines
