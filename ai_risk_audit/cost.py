"""Cross-cutting cost tracking.

Every LLM call in the pipeline (direct OpenRouter calls for the profiler,
planner, and judge stages, plus Petri's internal auditor/target/judge calls
during execution) is recorded here as it happens. The tracker accumulates
spend per run, can warn/abort against the configured budget knobs, and
prints a final summary grouped by pipeline stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """Raised when accumulated or projected spend passes the abort budget."""


@dataclass
class CostEntry:
    stage: str
    """Pipeline stage that made the call, e.g. "profiler", "planner",
    "execution", "judge"."""

    model: str
    usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class CostTracker:
    budget_warn_usd: float
    budget_abort_usd: float
    entries: list[CostEntry] = field(default_factory=list)

    @property
    def total_usd(self) -> float:
        return sum(e.usd for e in self.entries)

    def record(
        self,
        stage: str,
        model: str,
        usd: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.entries.append(
            CostEntry(
                stage=stage,
                model=model,
                usd=usd,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        if self.total_usd > self.budget_abort_usd:
            raise BudgetExceededError(
                f"accumulated spend ${self.total_usd:.4f} exceeds abort budget "
                f"${self.budget_abort_usd:.2f}"
            )

    def check_projection(self, projected_usd: float, stage: str) -> None:
        """Validate a pre-flight spend estimate before a costly stage runs.

        Raises `BudgetExceededError` if `total_usd + projected_usd` would pass
        the abort budget, so the caller can stop before spending the money
        rather than after.
        """
        projected_total = self.total_usd + projected_usd
        if projected_total > self.budget_abort_usd:
            raise BudgetExceededError(
                f"projected spend for stage '{stage}' (${projected_total:.4f} "
                f"including ${self.total_usd:.4f} already spent) exceeds abort "
                f"budget ${self.budget_abort_usd:.2f}"
            )
        if projected_total > self.budget_warn_usd:
            print(
                f"[cost] warning: projected spend for stage '{stage}' "
                f"(${projected_total:.4f}) exceeds warn budget "
                f"${self.budget_warn_usd:.2f}"
            )

    def by_stage(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for e in self.entries:
            totals[e.stage] = totals.get(e.stage, 0.0) + e.usd
        return totals

    def print_summary(self) -> None:
        print("\n=== Cost summary ===")
        for stage, usd in self.by_stage().items():
            print(f"  {stage:<12} ${usd:.4f}")
        print(f"  {'TOTAL':<12} ${self.total_usd:.4f}")
