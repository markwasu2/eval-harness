"""Combines N runs' `report.json` files into one category stability table.

Pure/offline: reads already-written reports from disk and renders markdown
via `scoring/aggregate.compute_stability`. No network or model calls.
"""

from __future__ import annotations

from pathlib import Path

from ai_risk_audit.report.build import AuditReport
from ai_risk_audit.scoring.aggregate import compute_stability
from ai_risk_audit.scoring.types import CategoryStability

UNSTABLE_CAUSE = (
    "suspected cause: scenario-path variance — the planner generates new "
    "scenario instances each run, and this category's score depends on "
    "which conversation paths get explored, not just how many repeats are run."
)


def render_stability_table(stabilities: list[CategoryStability]) -> str:
    lines = [
        "| category | per-run medians | overall median | range | stable |",
        "| --- | --- | --- | --- | --- |",
    ]
    for s in sorted(stabilities, key=lambda s: s.category.value):
        medians = ", ".join(f"{m:.1f}" for m in s.run_medians)
        verdict = "yes" if s.stable else "no"
        lines.append(
            f"| {s.category.value} | {medians} | {s.overall_median:.1f} | "
            f"{s.range:.1f} | {verdict} |"
        )

    unstable = [s for s in stabilities if not s.stable]
    if unstable:
        lines.append("\n## Unstable categories\n")
        for s in sorted(unstable, key=lambda s: s.category.value):
            lines.append(
                f"- **{s.category.value}**: range {s.range:.1f} > tolerance "
                f"{s.tolerance:.1f}. {UNSTABLE_CAUSE}"
            )

    return "\n".join(lines) + "\n"


def stability_table_from_reports(report_paths: list[str | Path], tolerance: float) -> str:
    reports = [AuditReport.model_validate_json(Path(p).read_text()) for p in report_paths]
    summaries = [r.score_summary for r in reports]
    stabilities = compute_stability(summaries, tolerance)
    return render_stability_table(stabilities)
