"""Immutable transcript records and their append-only JSONL storage.

Every scenario run against the target is stored in full: scenario id,
category, which model played each role, the complete message trace, cost,
tokens, and timestamps. Mirrors `eval_harness.runs`'s pattern: reruns never
overwrite a previous batch.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ai_risk_audit.planner.types import RiskCategory


class TranscriptMessage(BaseModel):
    """One message in a target conversation, as observed by the auditor."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str
    tool_calls: list[dict[str, Any]] = []
    tool_call_id: str | None = None


class TranscriptRecord(BaseModel):
    """One immutable record of a single scenario run."""

    model_config = ConfigDict(extra="forbid")

    transcript_id: str
    batch_id: str
    scenario_id: str
    category: RiskCategory
    epoch: int
    auditor_model: str
    target_model: str
    judge_model: str
    messages: list[TranscriptMessage]
    usd_cost: float
    prompt_tokens: int
    completion_tokens: int
    started_at: str | None
    completed_at: str | None


def new_batch_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def new_transcript_id() -> str:
    return uuid.uuid4().hex


def batch_path(transcripts_dir: Path, batch_id: str) -> Path:
    return transcripts_dir / f"{batch_id}.jsonl"


def write_record(path: Path, record: TranscriptRecord) -> None:
    """Append one record. Never truncates or rewrites existing lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json() + "\n")


def load_records(path: str | Path) -> list[TranscriptRecord]:
    path = Path(path)
    records: list[TranscriptRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(TranscriptRecord.model_validate_json(line))
    return records
