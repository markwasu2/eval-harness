"""Immutable run records and their append-only JSONL storage.

Every model response is stored in full: task id, model id, provider, full
model config, prompt transcript, raw response, parsed answer, tool trace,
token usage, latency, timestamp, and seed. Reruns never overwrite a previous
batch: each invocation of `eval-harness run` gets a fresh batch id and a
fresh file, so run history is fully reproducible and auditable.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from eval_harness.models import ModelConfig
from eval_harness.providers.base import Message, TokenUsage
from eval_harness.tools import ToolResult


class ToolCallLog(BaseModel):
    """One tool invocation and its result, as observed during a run."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    name: str
    arguments: dict[str, Any]
    result: ToolResult
    duration_ms: float


class ParsedAnswer(BaseModel):
    """A model's final answer: a probability distribution over the answer set."""

    model_config = ConfigDict(extra="forbid")

    probabilities: dict[str, float]


class RunRecord(BaseModel):
    """One immutable record of a single (task, model) evaluation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    batch_id: str
    task_id: str
    model_id: str
    provider: str
    config: ModelConfig
    prompt: list[Message]
    raw_response: str
    parsed_answer: ParsedAnswer | None
    parse_ok: bool
    retried: bool
    tool_trace: list[ToolCallLog]
    usage: TokenUsage
    latency_ms: float
    timestamp: datetime
    seed: int | None


def new_batch_id() -> str:
    """A fresh, sortable, unique id for one `run` invocation."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def new_run_id() -> str:
    return uuid.uuid4().hex


def batch_path(runs_dir: Path, batch_id: str) -> Path:
    return runs_dir / f"{batch_id}.jsonl"


def write_record(path: Path, record: RunRecord) -> None:
    """Append one record. Never truncates or rewrites existing lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(record.model_dump_json() + "\n")


def load_records(path: str | Path) -> list[RunRecord]:
    path = Path(path)
    records: list[RunRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(RunRecord.model_validate_json(line))
    return records
