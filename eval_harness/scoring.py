"""Scoring: pointwise metrics, mechanical baseline / oracle comparison, the
skill score, and cluster bootstrap confidence intervals.

Every task in this harness is itself a single scored item (one input, one
closed answer set, one ground truth). "Cluster by item" therefore means
resampling whole tasks with replacement when bootstrapping a model's mean
metric across the task set. Because every model in a run is scored against
the exact identical set of tasks, these per-model bootstrap intervals are
directly, pairwise comparable across models by construction.

A response that failed to parse (`parse_ok=False`) is scored as a confident
wrong answer: all probability mass is placed on whichever answer-set class
is *least* consistent with ground truth. This keeps every metric
well-defined (no NaNs, no special-casing) while guaranteeing a parse failure
can never score better than an honest wrong answer.
"""

from __future__ import annotations

import math
import random
import statistics

from pydantic import BaseModel, ConfigDict

from eval_harness.runs import RunRecord
from eval_harness.tasks import TaskSpec

_LOGLOSS_EPS = 1e-6


def brier_score(pred: dict[str, float], true: dict[str, float], answer_set: list[str]) -> float:
    return sum((pred[c] - true[c]) ** 2 for c in answer_set)


def log_loss_score(pred: dict[str, float], true: dict[str, float], answer_set: list[str]) -> float:
    total = 0.0
    for c in answer_set:
        t = true[c]
        if t <= 0:
            continue
        p = min(max(pred[c], _LOGLOSS_EPS), 1 - _LOGLOSS_EPS)
        total += -t * math.log(p)
    return total


def ranked_probability_score(
    pred: dict[str, float], true: dict[str, float], ordered_classes: list[str]
) -> float:
    """RPS over ordered bands, normalized to [0, 1] by dividing by (K - 1)."""
    k = len(ordered_classes)
    cum_pred = cum_true = 0.0
    total = 0.0
    for cls in ordered_classes[:-1]:
        cum_pred += pred[cls]
        cum_true += true[cls]
        total += (cum_pred - cum_true) ** 2
    return total / (k - 1)


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    """Macro-averaged F1 over (true_label, predicted_label) pairs. Classes
    that never occur as a true label are excluded from the average."""
    classes = sorted({t for t, _ in pairs})
    scores = []
    for c in classes:
        tp = sum(1 for t, p in pairs if t == c and p == c)
        fp = sum(1 for t, p in pairs if t != c and p == c)
        fn = sum(1 for t, p in pairs if t == c and p != c)
        if tp + fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        scores.append(f1)
    return sum(scores) / len(scores) if scores else 0.0


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile cluster bootstrap CI for the mean of `values`, where each
    entry is one item (cluster)."""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_resamples)]
    hi = means[min(int((1 - alpha / 2) * n_resamples), n_resamples - 1)]
    return (lo, hi)


def parse_failure_distribution(task: TaskSpec) -> dict[str, float]:
    true = task.ground_truth.distribution(task.answer_set)
    worst = min(task.answer_set, key=lambda c: true[c])
    return {c: (1.0 if c == worst else 0.0) for c in task.answer_set}


def model_distribution(task: TaskSpec, record: RunRecord) -> dict[str, float]:
    if record.parse_ok and record.parsed_answer is not None:
        return record.parsed_answer.probabilities
    return parse_failure_distribution(task)


def task_loss(task: TaskSpec, dist: dict[str, float], true: dict[str, float]) -> float:
    """The task's primary loss metric: RPS for ordered bands, Brier
    otherwise. This is the metric used for the baseline/oracle/skill triple."""
    if task.family == "ordered_band":
        return ranked_probability_score(dist, true, task.answer_set)
    return brier_score(dist, true, task.answer_set)


class TaskScore(BaseModel):
    """The full scoring breakdown for one (task, model) pair."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    model_id: str
    track: str
    family: str
    predicted_label: str
    true_label: str
    predicted_probabilities: dict[str, float]
    correct: bool
    brier: float
    log_loss: float
    rps: float | None
    model_loss: float
    baseline_loss: float
    oracle_loss: float | None
    skill: float | None
    regret: float | None
    parse_ok: bool
    retried: bool


class ModelSummary(BaseModel):
    """Aggregate metrics for one model across every task it was scored on."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    n_tasks: int
    accuracy: float
    macro_f1: float
    mean_brier: float
    mean_log_loss: float
    mean_rps: float | None
    mean_skill: float | None
    skill_ci_low: float | None
    skill_ci_high: float | None
    mean_regret: float | None
    parse_failure_rate: float


class ScoreReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_id: str
    task_scores: list[TaskScore]
    model_summaries: list[ModelSummary]


def score_task_record(task: TaskSpec, record: RunRecord) -> TaskScore:
    true_dist = task.ground_truth.distribution(task.answer_set)
    true_label = task.ground_truth.true_label(task.answer_set)
    dist = model_distribution(task, record)
    predicted_label = max(task.answer_set, key=lambda c: dist[c])

    model_loss = task_loss(task, dist, true_dist)
    baseline_dist = task.baseline.distribution(task.answer_set)
    baseline_loss = task_loss(task, baseline_dist, true_dist)
    oracle_loss: float | None = None
    if task.oracle is not None:
        oracle_dist = task.oracle.distribution(task.ground_truth, task.answer_set)
        oracle_loss = task_loss(task, oracle_dist, true_dist)

    skill: float | None = None
    regret: float | None = None
    if oracle_loss is not None:
        regret = model_loss - oracle_loss
        denom = baseline_loss - oracle_loss
        if denom != 0:
            skill = (baseline_loss - model_loss) / denom

    return TaskScore(
        task_id=task.id,
        model_id=record.model_id,
        track=task.track,
        family=task.family,
        predicted_label=predicted_label,
        true_label=true_label,
        predicted_probabilities=dist,
        correct=predicted_label == true_label,
        brier=brier_score(dist, true_dist, task.answer_set),
        log_loss=log_loss_score(dist, true_dist, task.answer_set),
        rps=ranked_probability_score(dist, true_dist, task.answer_set)
        if task.family == "ordered_band"
        else None,
        model_loss=model_loss,
        baseline_loss=baseline_loss,
        oracle_loss=oracle_loss,
        skill=skill,
        regret=regret,
        parse_ok=record.parse_ok,
        retried=record.retried,
    )


def summarize_model(
    model_id: str, rows: list[TaskScore], n_resamples: int, seed: int
) -> ModelSummary:
    n = len(rows)
    skills = [r.skill for r in rows if r.skill is not None]
    regrets = [r.regret for r in rows if r.regret is not None]
    skill_lo, skill_hi = (None, None)
    if skills:
        skill_lo, skill_hi = bootstrap_ci(skills, n_resamples=n_resamples, seed=seed)
    rpss = [r.rps for r in rows if r.rps is not None]
    return ModelSummary(
        model_id=model_id,
        n_tasks=n,
        accuracy=sum(r.correct for r in rows) / n,
        macro_f1=macro_f1([(r.true_label, r.predicted_label) for r in rows]),
        mean_brier=statistics.fmean(r.brier for r in rows),
        mean_log_loss=statistics.fmean(r.log_loss for r in rows),
        mean_rps=statistics.fmean(rpss) if rpss else None,
        mean_skill=statistics.fmean(skills) if skills else None,
        skill_ci_low=skill_lo,
        skill_ci_high=skill_hi,
        mean_regret=statistics.fmean(regrets) if regrets else None,
        parse_failure_rate=sum(not r.parse_ok for r in rows) / n,
    )


def score_batch(
    tasks: list[TaskSpec],
    records: list[RunRecord],
    n_resamples: int = 2000,
    seed: int = 0,
) -> ScoreReport:
    """Score every run record against its task and aggregate per model."""
    tasks_by_id = {t.id: t for t in tasks}
    task_scores: list[TaskScore] = []
    for record in records:
        task = tasks_by_id.get(record.task_id)
        if task is None:
            raise ValueError(
                f"run record for task {record.task_id!r} has no matching task spec loaded"
            )
        task_scores.append(score_task_record(task, record))

    model_ids = sorted({r.model_id for r in records})
    summaries = [
        summarize_model(
            mid, [ts for ts in task_scores if ts.model_id == mid], n_resamples, seed
        )
        for mid in model_ids
    ]
    batch_id = records[0].batch_id if records else ""
    return ScoreReport(batch_id=batch_id, task_scores=task_scores, model_summaries=summaries)
