# Audit Report: openai-cs-agents-demo Airline Agent

## 1. Summary

**Target audited:** `openai-cs-agents-demo Airline Agent` — the
[openai-cs-agents-demo](https://github.com/openai/openai-cs-agents-demo) airline
customer-service multi-agent demo (triage, flight info, booking/cancellation,
seat & special services, FAQ, refunds & compensation, plus relevance/jailbreak
input guardrails), running locally with its own instructions/tools/handoffs/
guardrails unchanged.

**Backend substitution (disclosed):** this evaluation environment has no
`OPENAI_API_KEY`, so all 8 of the demo's agents (6 main agents, normally
`gpt-5.2`, plus 2 input-guardrail agents, normally `gpt-4.1-mini`) were
swapped to run on Claude via litellm (`claude-haiku-4-5-20251001`) instead of
the demo's own hardcoded OpenAI models. This is a backend substitution, not a
change to the target's own behavior — the demo's instructions, tools,
handoffs, and guardrails are otherwise completely unmodified. Verbatim
disclosure text is carried in every run's `report.json` under
`environment_note` and in `configs/openai_cs.yaml`.

**Known reliability gap of the substitution:** the 2 input-guardrail agents'
structured JSON output occasionally fails malformed-JSON validation when
driven via litellm/Claude (not seen with the demo's own `gpt-4.1-mini`, which
uses OpenAI's native JSON-schema response format). The target adapter
(`ai_risk_audit/execution/openai_cs_target_agent.py`) retries a bounded number
of times before surfacing this as a real error — an infrastructure-layer
reliability shim, not a change to the guardrails' own instructions/schema.

**Harness version:** two commits cover the runs in this report —
`42c8511` ("Add openai-cs-agents-demo airline target with disclosed Claude
backend substitution") introduced the target at `tool_call_abuse` repeats=8,
and `3543d55` ("Raise openai_cs tool_call_abuse repeats to 12; disclose
persistent instability") raised that override to 12. Runs 1–3 ran under
`42c8511`'s config; runs 4–6 ran under `3543d55`'s config.

**When:** six full audit runs, all on 2026-10-14 (local run-artifact
timestamps): run1 16:36, run2 16:47, run3 16:58 (repeats=8 config), then
run4 17:38, run5 18:12, run6 18:22 (repeats=12 config).

**Total spend across all 6 runs:** $15.04 ($2.10577785 + $2.50233685 +
$2.2998491 + $2.7090252 + $2.5304119 + $2.8925318), per each run's
`total_usd_spent` field in `audits/openai-cs-run{1..6}/report.json`.

## 2. Method

**Categories tested (5):** `data_leakage`, `harmful_output`,
`ip_infringement`, `prompt_injection`, `tool_call_abuse`.

**Scenarios and repeats:** `scenarios_per_category: 2` (2 scenario
instances generated per category by the planner each run), `repeats: 4`
globally, with a `repeats_overrides` entry for `tool_call_abuse` (8 in runs
1–3, 12 in runs 4–6). This means each run scores 8 transcripts for
`data_leakage`/`harmful_output`/`ip_infringement`/`prompt_injection`
(2 scenarios × 4 repeats) and 16 transcripts for `tool_call_abuse` in runs
1–3, or 24 in runs 4–6 (2 scenarios × 8 or 12 repeats). Confirmed directly
from each run's `report.json` → `score_summary.categories[].n_scores` and
`plan.scenarios[].repeats`.

**Scoring:** each transcript is scored by a judge model
(`ai_risk_audit/scoring/judge.py`) against the scenario's `pass_criteria`/
`fail_criteria`. The judge must back every score with a verbatim quote
excerpted from the transcript; a score that does not pass this evidence
check is rejected and the call is retried, raising `JudgeError` on repeated
failure rather than accepting an uncited score. Per-category medians and an
overall weighted score are then aggregated (`score_summary.categories[].
median_score`, `score_summary.overall_score`).

**Retry policy:** two independent safeguards are never weakened, only
retried:
- `ModelBehaviorError` from the litellm/Claude structured-output reliability
  gap (guardrail agents) — retried a bounded number of times at the
  infrastructure layer; if retries exhaust, the whole run is re-run from
  scratch rather than patching around the error.
- `JudgeError` from the judge's verbatim-excerpt evidence check — the whole
  run is re-run rather than loosening the evidence requirement.

**Budget gates:** `budget_warn_usd: 5.0`, `budget_abort_usd: 10.0` per run
(`configs/openai_cs.yaml`), enforced by `CostTracker` before stage 3
(execution) starts.

## 3. Results — repeats=12 runs (run4, run5, run6)

Per-run medians, computed and verified via `ai-risk-audit stability` against
`audits/openai-cs-run4/report.json`, `run5/report.json`, `run6/report.json`
(output: `audits/openai-cs-stability.md`), tolerance 10.0:

| category | run4 | run5 | run6 | overall median | range | stable (tol. 10.0) |
| --- | --- | --- | --- | --- | --- | --- |
| data_leakage | 82.0 | 82.0 | 82.0 | 82.0 | 0.0 | yes |
| harmful_output | 72.0 | 86.5 | 72.0 | 72.0 | 14.5 | **no** |
| ip_infringement | 83.5 | 83.5 | 86.5 | 83.5 | 3.0 | yes |
| prompt_injection | 91.0 | 88.0 | 94.5 | 91.0 | 6.5 | yes |
| tool_call_abuse | 48.5 | 43.5 | 30.0 | 43.5 | 18.5 | **no** |

## 4. Variance finding: repeats did not fix instability

`tool_call_abuse` and `harmful_output` both exceed the 10.0 stability
tolerance at repeats=12. The repeats=8 → repeats=12 change (`3543d55`) was
an explicit test of whether more samples per run would stabilize these
categories. It did not — and for `tool_call_abuse` it made things worse.

**repeats=8 vs repeats=12, directly from artifacts:**

Regenerated from `audits/openai-cs-run1/report.json`, `run2/report.json`,
`run3/report.json` via `ai-risk-audit stability` (output:
`audits/openai-cs-stability-repeats8.md`):

| category | run1 | run2 | run3 | overall median | range | stable (tol. 10.0) |
| --- | --- | --- | --- | --- | --- | --- |
| data_leakage | 77.0 | 77.0 | 82.0 | 77.0 | 5.0 | yes |
| harmful_output | 72.0 | 67.0 | 77.0 | 72.0 | 10.0 | yes |
| ip_infringement | 82.0 | 82.0 | 88.0 | 82.0 | 6.0 | yes |
| prompt_injection | 87.5 | 88.0 | 86.5 | 87.5 | 1.5 | yes |
| tool_call_abuse | 50.0 | 55.0 | 43.5 | 50.0 | 11.5 | **no** |

**Comparison:**

| category | range @ repeats=8 | range @ repeats=12 | direction |
| --- | --- | --- | --- |
| tool_call_abuse | 11.5 | 18.5 | worse |
| harmful_output | 10.0 (stable, at the boundary) | 14.5 | newly destabilized |

Raising repeats from 8 to 12 did not reduce variance in either category —
`tool_call_abuse`'s range grew from 11.5 to 18.5, and `harmful_output`,
which was stable (right at the 10.0 tolerance boundary) at repeats=8,
became unstable at repeats=12. More samples per run bought no additional
stability.

**Explanation:** this is attributed to scenario-path variance, not an
epoch-count problem. The planner (`ai_risk_audit/planner/plan.py`) generates
fresh scenario instances for `data_leakage`/`harmful_output`/
`ip_infringement`/`prompt_injection`/`tool_call_abuse` on every run — the
specific conversational paths a scenario explores differ run to run. For
`tool_call_abuse` and `harmful_output`, the agent's resulting score depends
heavily on which specific conversation paths get generated and explored in
a given run, not on how many repeats are run within a run. Adding more
repeats samples more of the same run's scenario instances more densely, but
does nothing to average out the run-to-run variance introduced by the
planner generating different scenario instances each time. This is
disclosed plainly rather than hidden: `harmful_output` was stable at
repeats=8 and is not stable at repeats=12.

## 5. Implication (for a non-technical / insurance reader)

This agent's behavior is not a single fixed number — it varies meaningfully
from one evaluation run to the next in two of the five risk categories
tested (`tool_call_abuse` and `harmful_output`), even when nothing about the
agent, its prompts, or its tools changed between runs. In `tool_call_abuse`,
the scored risk ranged from a median of 30.0 to a median of 48.5 across
three otherwise-identical runs — a materially different risk picture
depending purely on which run you happened to look at. `harmful_output`
behaved consistently in one set of runs and then showed the same kind of
swing in another.

Running an AI agent through a single evaluation pass and recording "it
scored X" is therefore not a reliable way to characterize that agent's risk
in these categories — a single run can land anywhere in a wide range purely
by chance of which test scenarios got generated and explored, not because
the agent itself changed. Underwriting or certifying an agent's safety
based on one evaluation run risks materially understating (or
overstating) its actual risk in volatile categories like these. Repeated,
multi-run auditing — and reporting the full range observed, not just one
run's result — is necessary to get a trustworthy picture of risk in
categories that show this kind of run-to-run variance.
