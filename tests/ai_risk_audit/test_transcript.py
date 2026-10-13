from __future__ import annotations

from pathlib import Path

from ai_risk_audit.execution.transcript import (
    TranscriptMessage,
    TranscriptRecord,
    batch_path,
    load_records,
    new_batch_id,
    new_transcript_id,
    write_record,
)
from ai_risk_audit.planner.types import RiskCategory


def _record(scenario_id: str = "data_leakage-1") -> TranscriptRecord:
    return TranscriptRecord(
        transcript_id=new_transcript_id(),
        batch_id="batch-1",
        scenario_id=scenario_id,
        category=RiskCategory.DATA_LEAKAGE,
        epoch=1,
        auditor_model="openrouter/mock",
        target_model="openrouter/mock",
        judge_model="openrouter/mock",
        messages=[TranscriptMessage(role="user", content="hi")],
        usd_cost=0.01,
        prompt_tokens=10,
        completion_tokens=5,
        started_at=None,
        completed_at=None,
    )


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    path = batch_path(tmp_path, new_batch_id())
    r1 = _record("data_leakage-1")
    r2 = _record("ip_infringement-1")

    write_record(path, r1)
    write_record(path, r2)

    loaded = load_records(path)
    assert [r.scenario_id for r in loaded] == ["data_leakage-1", "ip_infringement-1"]
    assert loaded[0].usd_cost == 0.01


def test_write_record_appends_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "out" / "batch.jsonl"
    write_record(path, _record("a"))
    write_record(path, _record("b"))
    assert len(load_records(path)) == 2
