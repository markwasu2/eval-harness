# Audit Report: Two-Target Risk Audit (openai-cs-agents-demo Airline Agent + MiniDesk)

## 1. Summary

**Targets audited (2):**

1. `openai-cs-agents-demo Airline Agent` — the
   [openai-cs-agents-demo](https://github.com/openai/openai-cs-agents-demo)
   airline customer-service multi-agent demo (triage, flight info,
   booking/cancellation, seat & special services, FAQ, refunds &
   compensation, plus relevance/jailbreak input guardrails), running
   locally with its own instructions/tools/handoffs/guardrails unchanged.
2. `MiniDesk Support Agent` — a minimal helpdesk support agent (one
   `Agent`, two tools: FAQ search and account-status lookup) scaffolded
   directly in this repo, not a clone of a third-party demo.

**Backend substitution (disclosed, both targets):** this evaluation
environment has no `OPENAI_API_KEY`. The Airline Agent's all 8 of its own
agents (6 main agents, normally `gpt-5.2`, plus 2 input-guardrail agents,
normally `gpt-4.1-mini`) were swapped to run on Claude via litellm
(`claude-haiku-4-5-20251001`) instead of the demo's own hardcoded OpenAI
models — a backend substitution, not a change to the target's own
behavior; the demo's instructions, tools, handoffs, and guardrails are
otherwise completely unmodified. MiniDesk is built on the same publicly
pip-installable OpenAI Agents SDK (`openai-agents`) the Airline Agent
uses, and is likewise run on Claude via litellm (`claude-haiku-4-5-20251001`);
unlike the Airline Agent, there is no third party's own hardcoded model
being swapped out for MiniDesk, since MiniDesk's instructions, tools, and
agent structure are this repo's own, designed around this backend from
the start. Both disclosures are carried verbatim in every run's
`report.json` under `environment_note`, and in each target's own config
file (`configs/openai_cs.yaml`, `configs/minidesk.yaml`).

**Known reliability gap of the Airline Agent's substitution:** the 2
input-guardrail agents' structured JSON output occasionally fails
malformed-JSON validation when driven via litellm/Claude (not seen with
the demo's own `gpt-4.1-mini`, which uses OpenAI's native JSON-schema
response format). The target adapter
(`ai_risk_audit/execution/openai_cs_target_agent.py`) retries a bounded
number of times before surfacing this as a real error — an
infrastructure-layer reliability shim, not a change to the guardrails'
own instructions/schema. MiniDesk has no guardrail agents and has not
exhibited this gap.

**Harness version:** `42c8511` introduced the Airline Agent target at
`tool_call_abuse` repeats=8; `3543d55` raised that override to 12.
Airline Agent runs 1–3 ran under `42c8511`'s config, runs 4–6 under
`3543d55`'s config. `e930207` introduced MiniDesk (smoke-tested only, 1
sample/category); MiniDesk's `configs/minidesk.yaml` was then promoted to
the same repeats=12 full-audit protocol as the Airline Agent's runs 4–6
(2 scenarios/category, repeats=4 globally, `tool_call_abuse` override=12)
for the 3 full runs reported here.

**When:** all runs on 2026-10-14 (local run-artifact timestamps). Airline
Agent: run1 16:36, run2 16:47, run3 16:58 (repeats=8), run4 17:38, run5
18:12, run6 18:22 (repeats=12). MiniDesk: run1 19:59, run2 20:10, run3
20:27 (all repeats=12; run3 required one full retry after a `JudgeError`
— see Method's retry policy).

**Total spend:** Airline Agent, 6 runs: $15.04 ($2.10577785 +
$2.50233685 + $2.2998491 + $2.7090252 + $2.5304119 + $2.8925318).
MiniDesk, 3 successful runs: $10.14136365 ($3.504642 + $3.4696485 +
$3.16707315). Per each run's `total_usd_spent` field in
`audits/openai-cs-run{1..6}/report.json` and
`audits/minidesk-run{1..3}/report.json`. **Combined total: $25.18.**
(MiniDesk run3's first attempt, which hit the `JudgeError` described
under Method, incurred additional real spend before being discarded; that
attempt's directory was deleted per the retry policy and so left no
`report.json` artifact — it is excluded from the dollar figures above
since it cannot be traced to a file in this repo, consistent with this
report's no-untraceable-numbers rule.)

## 2. Method

Identical for both targets.

**Categories tested (5):** `data_leakage`, `harmful_output`,
`ip_infringement`, `prompt_injection`, `tool_call_abuse`.

**Scenarios and repeats:** `scenarios_per_category: 2` (2 scenario
instances generated per category by the planner each run), `repeats: 4`
globally, with a `repeats_overrides` entry for `tool_call_abuse` (8 for
Airline Agent runs 1–3, 12 for Airline Agent runs 4–6 and all 3 MiniDesk
runs). This means each run scores 8 transcripts for
`data_leakage`/`harmful_output`/`ip_infringement`/`prompt_injection`
(2 scenarios × 4 repeats) and 16 or 24 transcripts for `tool_call_abuse`
(2 scenarios × 8 or 12 repeats). Confirmed directly from each run's
`report.json` → `score_summary.categories[].n_scores` and
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
  gap (Airline Agent guardrail agents) — retried a bounded number of times
  at the infrastructure layer; if retries exhaust, the whole run is re-run
  from scratch rather than patching around the error.
- `JudgeError` from the judge's verbatim-excerpt evidence check — the
  partial run directory is deleted and the whole run is re-run from
  scratch rather than loosening the evidence requirement. This fired once
  during this audit, on MiniDesk's third run (excerpt not found verbatim
  in the transcript); the partial `audits/minidesk-run3/` directory was
  deleted and the run fully retried. The retry completed cleanly and its
  `report.json` is the run3 reported in Section 4 below.

**Budget gates:** `budget_warn_usd: 5.0`, `budget_abort_usd: 10.0` per run
(`configs/openai_cs.yaml`, `configs/minidesk.yaml`), enforced by
`CostTracker` before stage 3 (execution) starts.

## 3. Target 1: openai-cs-agents-demo Airline Agent

### 3.1 Results — repeats=12 runs (run4, run5, run6)

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

### 3.2 Variance finding: repeats did not fix instability

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
stability. This is attributed to scenario-path variance, not an
epoch-count problem (see Section 5 for the full explanation, shared with
Target 2's finding below).

## 4. Target 2: MiniDesk

### 4.1 Results — repeats=12 runs (run1, run2, run3)

Per-run medians, computed and verified via `ai-risk-audit stability` against
`audits/minidesk-run1/report.json`, `run2/report.json`, `run3/report.json`
(output: `audits/minidesk-stability.md`), tolerance 10.0:

| category | run1 | run2 | run3 | overall median | range | stable (tol. 10.0) |
| --- | --- | --- | --- | --- | --- | --- |
| data_leakage | 95.0 | 92.0 | 95.0 | 95.0 | 3.0 | yes |
| harmful_output | 96.0 | 93.5 | 96.0 | 96.0 | 2.5 | yes |
| ip_infringement | 92.0 | 88.0 | 92.0 | 92.0 | 4.0 | yes |
| prompt_injection | 97.0 | 97.0 | 96.5 | 97.0 | 0.5 | yes |
| tool_call_abuse | 72.0 | 47.0 | 72.0 | 72.0 | 25.0 | **no** |

### 4.2 Variance finding

Only `tool_call_abuse` exceeds the 10.0 stability tolerance for MiniDesk,
and it does so by a wider margin than the Airline Agent's own
`tool_call_abuse` instability (range 25.0 vs 18.5) — a median of 47.0 in
run2 against 72.0 in both run1 and run3, across three runs of the
identical config and source material. No repeats=8 run of MiniDesk was
conducted (MiniDesk went directly to the repeats=12 protocol per this
task's instructions), so no repeats=8-vs-12 ablation exists for MiniDesk
— unlike Target 1, this is not something this report can show for
MiniDesk, and no such comparison is claimed.

Per the pre-authorized stop condition for this audit: `tool_call_abuse`
exceeding tolerance on MiniDesk is disclosed here as category-inherent
variance, consistent with Target 1's treatment — repeats were not raised
further and no additional tuning was attempted.

## 5. Cross-target comparison

Both targets were built independently (one is a third-party demo with a
substituted backend; the other is scaffolded from scratch in this repo on
the same backend pattern) and tested with the identical protocol
(same 5 categories, same scenarios_per_category/repeats/`tool_call_abuse`
override, same judge, same tolerance). This lets the per-category
stability verdicts be compared directly:

| category | Airline Agent (repeats=12) | MiniDesk (repeats=12) | agreement |
| --- | --- | --- | --- |
| data_leakage | stable (range 0.0) | stable (range 3.0) | **stable on both** |
| harmful_output | **unstable** (range 14.5) | stable (range 2.5) | **diverges** |
| ip_infringement | stable (range 3.0) | stable (range 4.0) | **stable on both** |
| prompt_injection | stable (range 6.5) | stable (range 0.5) | **stable on both** |
| tool_call_abuse | **unstable** (range 18.5) | **unstable** (range 25.0) | **unstable on both** |

**`tool_call_abuse` instability reproduces on a second, independently
scaffolded agent.** Both targets — one a third-party multi-agent demo
with substituted backend, the other a from-scratch single-agent tool-user
on the same backend pattern — show `tool_call_abuse` exceeding the
tolerance by a wide margin (18.5 and 25.0 respectively), with no
repeats=8-vs-12 ablation available to suggest more samples would help for
MiniDesk, and direct evidence that it did not help for the Airline Agent.
This is consistent with the scenario-path-variance explanation being a
property of the category itself (how varied the generated
tool-call-abuse scenarios' conversational paths can be, and how much the
agent's score swings depending on which paths a given run's planner
happens to generate) rather than a property of either specific target
agent.

**`harmful_output` instability does NOT reproduce.** It is unstable for
the Airline Agent (range 14.5) but stable for MiniDesk (range 2.5) —
the data does not support a single blanket statement that `harmful_output`
is "inherently unstable" across agents in general. It may be that
`harmful_output` instability is driven by something specific to the
Airline Agent (e.g. its multi-agent handoff structure, or its guardrail
agents) rather than being a universal property of the category, or it
may simply not have shown up in these particular 3 MiniDesk runs. This
report does not have evidence to decide between those explanations and
does not claim one.

`data_leakage`, `ip_infringement`, and `prompt_injection` are stable on
both targets at this tolerance.

## 6. Implication (for a non-technical / insurance reader)

Neither agent's behavior is a single fixed number — both vary
meaningfully from one evaluation run to the next in at least one risk
category, even when nothing about the agent, its prompts, or its tools
changed between runs:

- **`tool_call_abuse` is volatile on both agents tested so far** — a
  completely different third-party demo and a from-scratch minimal agent.
  For the Airline Agent, the scored risk ranged from a median of 30.0 to
  48.5 across three otherwise-identical runs; for MiniDesk, from 47.0 to
  72.0. Seeing the same pattern on two independently built agents is
  stronger evidence that this is a property of the risk category itself
  (how much a single evaluation run's results can swing by chance,
  depending on which specific test conversations get generated) rather
  than a flaw unique to one product.
- **`harmful_output` is volatile on one agent but not the other.** This
  is disclosed plainly rather than smoothed over: it would be wrong to
  conclude either that "`harmful_output` is always unstable" or that
  "`harmful_output` is always fine" from this data. The honest reading is
  that this category's stability may depend on the specific agent being
  tested, and a reader should not assume one agent's `harmful_output`
  track record predicts another's.
- The other three categories (`data_leakage`, `ip_infringement`,
  `prompt_injection`) were consistent across repeated runs on both
  agents.

Running an AI agent through a single evaluation pass and recording "it
scored X" is therefore not a reliable way to characterize that agent's
risk in volatile categories — a single run can land anywhere in a wide
range purely by chance of which test scenarios got generated and
explored, not because the agent itself changed. Underwriting or
certifying an agent's safety based on one evaluation run risks materially
understating (or overstating) its actual risk, and the risk is
specifically concentrated in categories like `tool_call_abuse` that this
report shows swinging widely on more than one agent. Repeated, multi-run
auditing — and reporting the full range observed per category per agent,
rather than a single pass/fail number — is necessary to get a
trustworthy picture of risk, and that picture cannot simply be assumed to
transfer from one agent to another without testing each one.
