"""Renders an `AuditPlan`'s scenarios as Petri seed markdown files.

Petri's `seeds_dataset()` reads a directory of `.md` files: YAML front
matter becomes `Sample.metadata`, the body becomes `Sample.input` (the
auditor's seed instructions), and the filename stem becomes `Sample.id`.
"""

from __future__ import annotations

from pathlib import Path

from ai_risk_audit.planner.types import AuditPlan


def render_seed_files(plan: AuditPlan, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for scenario in plan.scenarios:
        front_matter = (
            "---\n"
            f"category: {scenario.category.value}\n"
            f"title: {scenario.title}\n"
            f"pass_criteria: {scenario.pass_criteria!r}\n"
            f"fail_criteria: {scenario.fail_criteria!r}\n"
            "---\n"
        )
        path = out_dir / f"{scenario.id}.md"
        path.write_text(front_matter + scenario.seed_instructions)
    return out_dir
