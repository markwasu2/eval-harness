"""Task specification schema and loader.

Every task is a single scored item: one input packet, one closed answer set,
one ground truth, one scoring rule, one mechanical baseline, and an optional
oracle/ceiling. Task files are YAML or JSON and are validated eagerly with
pydantic so a malformed task fails loudly at load time rather than silently
producing garbage scores later in the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Track = str
Family = Literal["classification", "ordered_band"]


class TaskLoadError(Exception):
    """Raised when a task file fails to parse or validate."""


class InputPacket(BaseModel):
    """The material shown to the model for this task."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["inline", "file"]
    text: str | None = None
    files: list[str] | None = None

    @model_validator(mode="after")
    def _check_payload(self) -> InputPacket:
        if self.kind == "inline" and not self.text:
            raise ValueError("input.kind == 'inline' requires non-empty 'text'")
        if self.kind == "file" and not self.files:
            raise ValueError("input.kind == 'file' requires a non-empty 'files' list")
        return self

    def render(self, base_dir: Path) -> str:
        """Render the packet to plain text for inclusion in a prompt."""
        if self.kind == "inline":
            assert self.text is not None
            return self.text
        assert self.files is not None
        parts: list[str] = []
        for rel in self.files:
            path = base_dir / rel
            if not path.is_file():
                raise TaskLoadError(f"input file not found: {path}")
            parts.append(f"--- file: {rel} ---\n{path.read_text()}")
        return "\n\n".join(parts)


class GroundTruth(BaseModel):
    """The true outcome for a task. Either a single label, explicit per-class
    probabilities (e.g. a soft label), or both (in which case probabilities
    is the distribution used for scoring and label is used for reporting)."""

    model_config = ConfigDict(extra="forbid")

    label: str | None = None
    probabilities: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check_one_present(self) -> GroundTruth:
        if self.label is None and self.probabilities is None:
            raise ValueError("ground_truth requires 'label' and/or 'probabilities'")
        if self.probabilities is not None:
            total = sum(self.probabilities.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"ground_truth.probabilities must sum to 1, got {total}")
            if any(p < 0 for p in self.probabilities.values()):
                raise ValueError("ground_truth.probabilities must be non-negative")
        return self

    def distribution(self, answer_set: list[str]) -> dict[str, float]:
        """The true distribution over the answer set, one-hot if only a label was given."""
        if self.probabilities is not None:
            return dict(self.probabilities)
        assert self.label is not None
        return {c: (1.0 if c == self.label else 0.0) for c in answer_set}

    def true_label(self, answer_set: list[str]) -> str:
        if self.label is not None:
            return self.label
        dist = self.distribution(answer_set)
        return max(answer_set, key=lambda c: dist[c])


class ScoringRule(BaseModel):
    """How this task is scored. classes must equal answer_set; for ordered_band
    the order defines the band ordering used by the ranked probability score."""

    model_config = ConfigDict(extra="forbid")

    metric: Family


class MechanicalBaseline(BaseModel):
    """A cheap, non-LLM reference forecast every model is compared against."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["uniform", "majority_class", "prior_probabilities"]
    label: str | None = None
    probabilities: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check_strategy_fields(self) -> MechanicalBaseline:
        if self.strategy == "majority_class" and self.label is None:
            raise ValueError("baseline.strategy == 'majority_class' requires 'label'")
        if self.strategy == "prior_probabilities" and self.probabilities is None:
            raise ValueError(
                "baseline.strategy == 'prior_probabilities' requires 'probabilities'"
            )
        return self

    def distribution(self, answer_set: list[str]) -> dict[str, float]:
        if self.strategy == "uniform":
            p = 1.0 / len(answer_set)
            return {c: p for c in answer_set}
        if self.strategy == "majority_class":
            assert self.label is not None
            eps = 1e-6
            rest = eps / max(len(answer_set) - 1, 1)
            return {c: (1.0 - eps if c == self.label else rest) for c in answer_set}
        assert self.probabilities is not None
        return dict(self.probabilities)


class OracleSpec(BaseModel):
    """An optional ceiling forecast representing the best achievable score."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ground_truth", "fixed"]
    probabilities: dict[str, float] | None = None

    @model_validator(mode="after")
    def _check_fixed_has_probabilities(self) -> OracleSpec:
        if self.kind == "fixed" and self.probabilities is None:
            raise ValueError("oracle.kind == 'fixed' requires 'probabilities'")
        return self

    def distribution(self, ground_truth: GroundTruth, answer_set: list[str]) -> dict[str, float]:
        if self.kind == "ground_truth":
            return ground_truth.distribution(answer_set)
        assert self.probabilities is not None
        return dict(self.probabilities)


class TaskSpec(BaseModel):
    """A single scored benchmark item."""

    model_config = ConfigDict(extra="forbid")

    id: str
    track: Track
    family: Family
    title: str
    description: str
    input: InputPacket
    answer_set: list[str] = Field(min_length=2)
    ground_truth: GroundTruth
    scoring: ScoringRule
    baseline: MechanicalBaseline
    oracle: OracleSpec | None = None

    # populated by the loader, not part of the file on disk
    source_path: Path = Field(default=Path("."), exclude=True)

    @model_validator(mode="after")
    def _check_consistency(self) -> TaskSpec:
        if len(set(self.answer_set)) != len(self.answer_set):
            raise ValueError("answer_set must not contain duplicate classes")
        if self.scoring.metric != self.family:
            raise ValueError(
                f"scoring.metric ({self.scoring.metric}) must match family ({self.family})"
            )
        answer_set = set(self.answer_set)
        if self.ground_truth.label is not None and self.ground_truth.label not in answer_set:
            raise ValueError(
                f"ground_truth.label {self.ground_truth.label!r} not in answer_set"
            )
        if self.ground_truth.probabilities is not None and set(
            self.ground_truth.probabilities
        ) != answer_set:
            raise ValueError("ground_truth.probabilities keys must equal answer_set")
        if self.baseline.strategy == "majority_class":
            assert self.baseline.label is not None
            if self.baseline.label not in answer_set:
                raise ValueError(f"baseline.label {self.baseline.label!r} not in answer_set")
        if self.baseline.strategy == "prior_probabilities":
            assert self.baseline.probabilities is not None
            if set(self.baseline.probabilities) != answer_set:
                raise ValueError("baseline.probabilities keys must equal answer_set")
        if self.oracle is not None and self.oracle.kind == "fixed":
            assert self.oracle.probabilities is not None
            if set(self.oracle.probabilities) != answer_set:
                raise ValueError("oracle.probabilities keys must equal answer_set")
        return self

    def rendered_input(self) -> str:
        return self.input.render(self.source_path.parent)


def load_task(path: str | Path) -> TaskSpec:
    """Load and validate a single task file. Raises TaskLoadError with a
    clear, file-scoped message on any parse or validation failure."""
    path = Path(path)
    if not path.is_file():
        raise TaskLoadError(f"task file not found: {path}")
    try:
        raw_text = path.read_text()
        is_json = path.suffix.lower() == ".json"
        data = json.loads(raw_text) if is_json else yaml.safe_load(raw_text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise TaskLoadError(f"{path}: could not parse file: {exc}") from exc
    if not isinstance(data, dict):
        raise TaskLoadError(f"{path}: task file must contain a mapping at the top level")
    try:
        task = TaskSpec.model_validate(data)
    except Exception as exc:  # pydantic.ValidationError and friends
        raise TaskLoadError(f"{path}: invalid task spec: {exc}") from exc
    task.source_path = path
    return task


def load_tasks(paths: list[str | Path]) -> list[TaskSpec]:
    """Load a directory (recursively glob *.yaml/*.yml/*.json) or an explicit
    list of task files. Fails loudly: the first invalid task aborts the load."""
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.yaml")))
            files.extend(sorted(p.rglob("*.yml")))
            files.extend(sorted(p.rglob("*.json")))
        elif p.is_file():
            files.append(p)
        else:
            raise TaskLoadError(f"task path not found: {p}")
    if not files:
        raise TaskLoadError(f"no task files found under {paths}")
    tasks = [load_task(f) for f in files]
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise TaskLoadError(f"duplicate task ids found: {dupes}")
    return tasks
