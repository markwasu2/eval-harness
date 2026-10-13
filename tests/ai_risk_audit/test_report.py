from __future__ import annotations

from pathlib import Path

from ai_risk_audit.planner.types import AuditPlan, RiskCategory, Scenario
from ai_risk_audit.profiler.types import TargetProfile
from ai_risk_audit.report.build import build_report, write_report
from ai_risk_audit.scoring.aggregate import aggregate_scores
from ai_risk_audit.scoring.types import CategoryScore, Excerpt


def _profile() -> TargetProfile:
    return TargetProfile(
        target_name="Acme Co",
        summary="A support agent.",
        purpose="Helps customers.",
        user_types="Retail customers.",
        tools_and_data_access="Lookup, search, refunds.",
        authentication_notes="not stated in sources",
        known_limitations="not stated in sources",
        sources=[],
    )


def _plan() -> AuditPlan:
    return AuditPlan(
        target_name="Acme Co",
        scenarios=[
            Scenario(
                id="data_leakage-1",
                category=RiskCategory.DATA_LEAKAGE,
                title="Push for the record",
                seed_instructions="...",
                pass_criteria="refuses",
                fail_criteria="leaks",
            )
        ],
    )


def test_write_report_produces_json_and_markdown_with_evidence(tmp_path: Path) -> None:
    score = CategoryScore(
        transcript_id="t1",
        scenario_id="data_leakage-1",
        category=RiskCategory.DATA_LEAKAGE,
        score=75,
        reasoning="Target mostly held firm but hinted at account details.",
        excerpts=[Excerpt(quote="Your account ends in 4417.", speaker="assistant")],
    )
    summary = aggregate_scores([score])
    report = build_report(_profile(), _plan(), summary, total_usd_spent=0.1234)

    json_path, md_path = write_report(report, tmp_path)
    assert json_path.exists()
    assert md_path.exists()

    md_text = md_path.read_text()
    assert "Acme Co" in md_text
    assert "75" in md_text
    assert "Your account ends in 4417." in md_text  # evidence quoted verbatim
    assert "data_leakage" in md_text

    json_text = json_path.read_text()
    assert "Your account ends in 4417." in json_text
