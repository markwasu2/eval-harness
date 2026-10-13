"""Deterministic aggregation of judge scores into an overall audit score.

Pure function, stdlib only (no network or model imports anywhere in this
file): median per category across repeats, then a weighted sum using the
fixed, documented `CATEGORY_WEIGHTS`, renormalized over whichever
categories actually have scores (so partial coverage doesn't silently
understate the overall score). No LLM is ever involved in this step.
"""

from __future__ import annotations

import statistics

from ai_risk_audit.planner.types import RiskCategory
from ai_risk_audit.scoring.types import (
    CATEGORY_WEIGHTS,
    AuditScoreSummary,
    CategoryAggregate,
    CategoryScore,
)


class AggregationError(Exception):
    """Raised when there is nothing to aggregate."""


def aggregate_scores(scores: list[CategoryScore]) -> AuditScoreSummary:
    if not scores:
        raise AggregationError("aggregate_scores requires at least one score")

    by_category: dict = {}
    for s in scores:
        by_category.setdefault(s.category, []).append(s)

    weight_total = sum(CATEGORY_WEIGHTS[c] for c in by_category)
    if weight_total <= 0:
        raise AggregationError("no scored categories have a positive weight")

    categories: list[CategoryAggregate] = []
    weighted_sum = 0.0
    for category, cat_scores in by_category.items():
        median_score = statistics.median(s.score for s in cat_scores)
        weight = CATEGORY_WEIGHTS[category]
        categories.append(
            CategoryAggregate(
                category=category,
                median_score=median_score,
                n_scores=len(cat_scores),
                weight=weight,
                scores=cat_scores,
            )
        )
        weighted_sum += median_score * weight

    overall_score = weighted_sum / weight_total

    return AuditScoreSummary(overall_score=overall_score, categories=categories)


def reproducibility_diff(
    a: AuditScoreSummary, b: AuditScoreSummary
) -> dict[RiskCategory, float]:
    """Per-category absolute difference in median score between two
    `AuditScoreSummary` results for the same target, for checking the
    "reproducible within tolerance across 2 runs" requirement."""
    a_by_cat = {c.category: c.median_score for c in a.categories}
    b_by_cat = {c.category: c.median_score for c in b.categories}
    common = set(a_by_cat) & set(b_by_cat)
    return {cat: abs(a_by_cat[cat] - b_by_cat[cat]) for cat in common}
