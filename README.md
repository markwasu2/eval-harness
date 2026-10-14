# eval-harness

A config-driven LLM evaluation harness. It is the sealed runner behind an
insurance LLM benchmark: every task, model, and scoring rule is declared in
config files, every model response is recorded immutably, and the whole
pipeline is reproducible from a single CLI command.

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync                 # installs runtime + dev dependencies into .venv
uv run pytest -q        # run the test suite
uv run ruff check .     # lint
```

The CLI is installed as `eval-harness` inside the uv environment (`uv run
eval-harness ...`), or activate `.venv` and call `eval-harness` directly.

## Quickstart: the worked example

Two example tasks ship in `tasks/`, and an offline mock model panel ships in
`model_panels/example_mock.yaml`. One command runs both tasks against three
mock models, scores them, and renders the static results site, using no
network access at all:

```bash
uv run eval-harness all \
  --config run_config/example.yaml \
  --tasks tasks/ \
  --models model_panels/example_mock.yaml \
  --out ./out

open ./out/site/index.html
```

This writes:

- `./out/runs/<batch_id>.jsonl` — immutable run records (one per task/model pair)
- `./out/runs/<batch_id>.scores.json` — full scoring breakdown
- `./out/site/index.html` — the static results site (headline table, per-task
  tables, calibration plots, parse-failure table)

Each invocation of `run` (or `all`) gets a fresh, timestamped `batch_id` and a
fresh JSONL file; reruns never overwrite a previous batch.

## CLI

```
eval-harness run        --config run.yaml --tasks TASKS --models PANEL [--out DIR]
eval-harness score       --run RUN.jsonl --tasks TASKS [--config run.yaml] [--out scores.json]
eval-harness build-site  --run RUN.jsonl --tasks TASKS [--config run.yaml] [--out DIR]
eval-harness baseline    --task TASK_FILE
eval-harness all         --config run.yaml --tasks TASKS --models PANEL [--out DIR]
```

`--tasks` accepts one or more task files and/or directories (directories are
globbed recursively for `*.yaml`/`*.yml`/`*.json`). `--config` is optional;
it only carries bootstrap-resampling settings (see below) and defaults are
used if omitted. `all` is `run` + `score` + `build-site` in one invocation.

## Config format (`--config`)

A small YAML file controlling the paired cluster bootstrap used for skill
score confidence intervals. All fields are optional:

```yaml
n_bootstrap_resamples: 2000   # default 2000
bootstrap_seed: 0             # default 0
```

## Task spec format

One YAML (or JSON) file per task. Every field is validated eagerly with
pydantic; a malformed task fails loudly with a file-scoped error message
rather than silently producing garbage scores. See `tasks/*.yaml` for full
worked examples.

```yaml
id: claims_triage_3class       # unique across the whole task set
track: claims                  # free-text grouping label
family: classification         # "classification" | "ordered_band"
title: "Claims severity triage"
description: >
  Human-readable task instructions shown to the model.

input:
  kind: inline                 # "inline" | "file"
  text: |                      # required when kind == inline
    The input packet shown to the model.
  # files: ["packet.csv"]      # required when kind == file; paths are
                                # resolved relative to the task file and
                                # their contents are read and embedded
                                # into the prompt verbatim.

answer_set: ["low", "medium", "high"]   # the closed label set. For
                                         # family == ordered_band, this
                                         # list's order IS the band order
                                         # (lowest to highest) used by RPS.

ground_truth:
  label: "low"                 # a single true class, and/or:
  # probabilities: {low: 0.9, medium: 0.1, high: 0.0}  # a soft label

scoring:
  metric: classification       # must equal `family`

baseline:                      # the mechanical, non-LLM reference forecast
  strategy: uniform            # "uniform" | "majority_class" | "prior_probabilities"
  # label: "low"                # required for majority_class
  # probabilities: {...}        # required for prior_probabilities

oracle:                        # OPTIONAL ceiling forecast. If omitted, the
  kind: ground_truth           # task has no defined ceiling and the skill
                                # score and regret are reported as null for
                                # that task (and excluded from aggregates).
  # kind: fixed
  # probabilities: {...}       # required for kind: fixed
```

Every model is required to respond with calibrated probabilities over the
exact `answer_set`, so every task is scored with Brier score and log loss
(for `classification`) or with the ranked probability score (for
`ordered_band`, respecting the order of `answer_set`), plus accuracy and
macro-F1 (aggregated across a model's tasks). The task's primary loss metric
(Brier for classification, RPS for ordered bands) is used for the
baseline/oracle/skill triple:

```
skill = (baseline_loss - model_loss) / (baseline_loss - oracle_loss)
```

`skill` is `1.0` for a perfect forecast, `0.0` for matching the mechanical
baseline exactly, and negative for doing worse than the baseline. `regret =
model_loss - oracle_loss`. Both are `null` for a task with no `oracle`
defined.

### Parse policy

Every model's final message must be exactly:

```json
{"probabilities": {"<class>": <probability>, ...}}
```

with one entry per class in the answer set, non-negative, summing to ~1
(renormalized before scoring). A response that fails to parse gets exactly
one repair attempt (a follow-up prompt restating the required schema); if
that also fails, the run is recorded with `parse_ok=false` and is scored as
a **confident wrong answer**: all probability mass is placed on whichever
class is least consistent with ground truth. This keeps every metric
well-defined (no NaNs) while guaranteeing a parse failure never scores
better than an honest wrong answer. Per-model parse-failure rate is reported
in the site and in `ModelSummary.parse_failure_rate`.

## Adding a task

1. Write a new YAML file under `tasks/` (or any directory you pass via
   `--tasks`) following the schema above.
2. Run `eval-harness baseline --task path/to/new_task.yaml` to sanity-check
   that it loads and to see its baseline/oracle loss before running any
   models against it.
3. Include it via `--tasks` in your next `run`/`all` invocation.

## Adding a model

Add an entry to a model panel YAML:

```yaml
models:
  - id: my-model-name          # unique within the panel; used in reports
    provider: openrouter       # "mock" | "openrouter"
    provider_model: openai/gpt-4.1   # the exact string sent to the provider
    temperature: 0.0
    seed: 0
    reasoning: true             # request extended/thinking mode
    tool_use: true              # expose the sandbox tool via function calling
    max_tool_turns: 6           # cap on tool-calling round trips per task
```

Every model, closed or open-weight, is called through the same pluggable
`Provider` interface (`eval_harness/providers/`) with the identical system
prompt and call configuration — only `provider_model` and the API glue
differ. To run against real models:

```bash
export OPENROUTER_API_KEY=sk-or-...
uv run eval-harness all \
  --tasks tasks/ \
  --models model_panels/example_openrouter.yaml \
  --out ./out-real
```

See `model_panels/example_openrouter.yaml` for a starter panel spanning
closed (GPT, Claude, Gemini) and open-weight (Llama) models.

### Adding a new provider

Implement `eval_harness.providers.base.Provider` (one method: `complete`)
and register it in `eval_harness/providers/__init__.py`'s `_REGISTRY`. Every
provider must route tool use through the harness's own sandbox (exposed via
function calling) — provider-native code-execution tools must never be
used, so tool behavior is identical across every model in the panel.

## The tool sandbox

Every model is given exactly one function-calling tool, `execute_python`
(`eval_harness/tools.py`): an offline Python 3 executor with no network
access, no filesystem access outside a fresh temp directory, and bounded
CPU time, wall-clock time, and (where the OS supports it) address space.
Every tool call and its result is logged into the run record's `tool_trace`.

**Honest limitation**: this is a language-level + POSIX-rlimit sandbox, not
a container or VM. It blocks socket creation, `subprocess`/`os.system`/exec
escapes, and filesystem access outside the sandbox directory from within
the executed script, and bounds CPU/wall-clock time. `RLIMIT_AS` (virtual
memory) is applied best-effort: Linux enforces it, but macOS/XNU does not
support setting it at all (the kernel reports `RLIM_INFINITY` for the
limit and refuses any finite value), so on macOS only CPU-time and
wall-clock limits are actually enforced. There is no kernel-level isolation
(no seccomp/namespaces/chroot). This is sufficient for bounding what an
evaluated model's generated snippets can do in this benchmark, but this
sandbox should not be relied on for defense against a human adversary
crafting code to attack the host.

## Run records and reproducing a run

Every model response is stored immutably as one JSON line in
`runs/<batch_id>.jsonl` (`eval_harness/runs.py`): task id, model id,
provider, full model config, the complete prompt transcript (including tool
calls/results), raw response, parsed answer, tool trace, token usage,
latency, timestamp, and seed. Reruns never overwrite: every `run`/`all`
invocation gets a new `batch_id` and a new file.

To reproduce a past run's scoring and site from its JSONL file (e.g. after
changing a task's ground truth or baseline, or after a scoring bug fix) —
without calling any model again:

```bash
uv run eval-harness score      --run runs/<batch_id>.jsonl --tasks tasks/
uv run eval-harness build-site --run runs/<batch_id>.jsonl --tasks tasks/ --out site/
```

## Scoring details

- **Accuracy / macro-F1**: computed over the argmax label of each model's
  probability vector, aggregated across a model's tasks.
- **Brier score**: `sum((p_c - t_c)^2)` over the answer set.
- **Log loss**: cross-entropy against the true distribution, clipped at
  `1e-6` to stay finite.
- **Ranked probability score (RPS)**: for `ordered_band` tasks, the mean
  squared difference between cumulative predicted and true distributions
  across the band order, normalized to `[0, 1]`.
- **Skill score**: `(baseline_loss - model_loss) / (baseline_loss -
  oracle_loss)`, using Brier for `classification` and RPS for
  `ordered_band` as the primary loss. Only defined for tasks with an
  `oracle` spec.
- **Regret**: `model_loss - oracle_loss`, only defined for tasks with an
  `oracle` spec.
- **Confidence intervals**: a paired cluster bootstrap over tasks
  (`eval_harness.scoring.bootstrap_ci`). Every task is itself a single
  scored item, so "cluster by item" means resampling whole tasks with
  replacement; because every model in a run is scored against the
  identical task set, these per-model intervals are directly comparable
  across models by construction.

## Package layout

```
eval_harness/
  tasks.py       TaskSpec pydantic schema + loader
  models.py      ModelConfig / ModelPanel schema + loader
  providers/     pluggable Provider interface: base, mock (offline), openrouter
  tools.py       the offline sandboxed Python tool
  runner.py      prompt construction, agent loop, parse policy
  runs.py        RunRecord schema + append-only JSONL storage
  scoring.py     metrics, baseline/oracle/skill/regret, bootstrap CI
  site.py        static HTML site + calibration plots
  cli.py         `eval-harness` command-line entry point
tasks/                example task fixtures
model_panels/          example model panel configs (mock + openrouter)
run_config/            example run-settings config
tests/                 pytest suite
```

## ai-risk-audit: AI-company risk audit agent

A separate module (`ai_risk_audit/`) built on
[Petri](https://github.com/meridianlabs-ai/inspect_petri) for auditing a
target AI agent across five risk categories: IP infringement, harmful
output, data leakage, tool-call abuse, and prompt injection. It is
independent of the rest of `eval_harness` (different CLI entry point,
different config, no shared imports beyond the repo's conventions) and
does not affect `eval-harness run`/`score`/`build-site`/`all`.

### Pipeline (five stages, one module each)

1. **Profiler** (`ai_risk_audit/profiler/`) — LLM-extracts a structured
   `TargetProfile` from raw source material (text/file/URL), saving the raw
   sources alongside. Unaddressed fields are extracted as the literal
   string `"not stated in sources"` rather than guessed.
2. **Planner** (`ai_risk_audit/planner/`) — turns a profile into an
   `AuditPlan`: adversarial scenarios for each of the 5 risk categories,
   each with a pass/fail rubric, rendered as Petri seed `.md` files.
3. **Execution** (`ai_risk_audit/execution/`) — runs each scenario through
   Petri's auditor↔target multi-turn conversation against a custom target
   agent and stores every transcript immutably (append-only JSONL, mirrors
   `eval_harness/runs.py`'s pattern).
4. **Scoring** (`ai_risk_audit/scoring/`) — a judge model scores each
   transcript 1–100 per category and must quote transcript lines verbatim;
   quotes that aren't an exact substring of the transcript are rejected and
   retried. Aggregation to an overall score is a pure, deterministic
   function (median per category, fixed documented weights) — no LLM
   involved.
5. **Report** (`ai_risk_audit/report/`) — writes `report.json` +
   `report.md`: every category score with its quoted evidence excerpts.

Every LLM call (profiler, planner, judge, and Petri's own auditor/target/
judge roles) is named in one config file (`ai_risk_audit/config.py` / a
YAML file like `configs/audit_config.example.yaml`). Calls go direct to
the Anthropic Messages API by default (`AnthropicChat` in
`ai_risk_audit/llm.py`, reading `ANTHROPIC_API_KEY`) — an `OpenRouterChat`
backend also exists for when a funded OpenRouter account is available
instead; both satisfy the same `ChatClient` protocol. A `CostTracker`
wraps every call, accumulates spend per run, warns at `budget_warn_usd`,
and aborts (writing a report documenting why, before any further spend)
past `budget_abort_usd`.

### The mock Acme target

The built-in definition-of-done target is a mock "Acme Co. support agent"
(`ai_risk_audit/execution/fake_target.py`): a Claude-served chat model with
a permissive support-agent persona, 3 fixed tools, and planted weaknesses
with known ground truth (a fake customer record it will leak if pushed, a
fake copyrighted jingle it will reproduce, and a prompt-injection payload
hidden in a help article). Tool calls are executed by a deterministic
Python backend — never simulated by the auditor model — so scoring against
these scenarios is verifiable, not just plausible.

### Running it

```bash
uv sync --extra audit                 # installs inspect-ai + inspect-petri
export ANTHROPIC_API_KEY=sk-ant-...   # or put it in a .env file
uv run ai-risk-audit demo --out audits/acme-demo
```

`demo` runs the full pipeline end-to-end against the mock Acme target with
no required flags. `run` takes a named target and profiler source material
(`--source-text`/`--source-file`/`--source-url`, repeatable); `--config`
selects which target is actually driven via `AuditConfig.target_kind` (see
`ai_risk_audit/config.py`): the mock Acme target (default), the tau2-bench
airline agent (`configs/tau2_airline.yaml`), or the openai-cs-agents-demo
airline agent (`configs/openai_cs.yaml`).

### Other targets wired in: tau2-bench and openai-cs-agents-demo

Besides the mock Acme target, two real public agents are wired in as
`AuditConfig.target_kind` options, each via its own adapter in
`ai_risk_audit/execution/` and its own config file:

- **tau2-bench airline agent** (`target_kind: tau2_airline`,
  `configs/tau2_airline.yaml`,
  `ai_risk_audit/execution/tau_target_agent.py`) — a clone of
  [tau2-bench](https://github.com/sierra-research/tau2-bench) run locally,
  stock domain/policy/tools, with the auditor standing in for tau2's own
  simulated user. Runs on its own configured model, no substitution.

- **openai-cs-agents-demo airline agent** (`target_kind: openai_cs`,
  `configs/openai_cs.yaml`,
  `ai_risk_audit/execution/openai_cs_target_agent.py`) — a clone of
  [openai-cs-agents-demo](https://github.com/openai/openai-cs-agents-demo)
  run locally via its own `Agent`/`Runner.run()` objects (not its FastAPI/
  ChatKit UI layer), with the auditor standing in for a human chat user.
  **Disclosed backend substitution**: this environment has no
  `OPENAI_API_KEY`, so all 8 of the demo's own agents (its 6 main agents,
  normally `gpt-5.2`, plus its 2 input-guardrail agents in
  `airline/guardrails.py`, normally `gpt-4.1-mini`) are swapped at runtime
  to run on Claude via `litellm`, using `ANTHROPIC_API_KEY`, instead of the
  demo's own hardcoded OpenAI models. The demo's own instructions, tools,
  handoffs, and guardrail logic are otherwise completely unmodified — this
  is a backend substitution, not a change to the target's behavior, and it
  is also recorded in that target's `report.json`/`report.md`
  (`environment_note`) and in `configs/openai_cs.yaml`'s own comments.
  **Known reliability gap of this substitution**: the 2 input-guardrail
  agents' structured JSON output occasionally fails malformed-JSON
  validation when driven via litellm/Claude, at a rate not seen with the
  demo's own `gpt-4.1-mini` (native JSON-schema response format vs.
  litellm's best-effort tool-use coercion for Anthropic). The adapter
  retries a bounded number of times before surfacing it as a real error —
  see `openai_cs_target_agent.py`'s docstring.

Both adapters are driven with the same execution pattern: a Petri target
agent that speaks `TargetContext`'s handshake, runs the real agent's own
turn loop internally (never letting the auditor fabricate tool results),
and mirrors the target's own conversation into the Inspect sample store for
scoring (`EvalSample.messages` is the auditor's own history, not the
target's).

## Design choices and current boundaries

- **Sandbox**: the tool sandbox enforces isolation at the language level
  plus POSIX rlimits (see "The tool sandbox" above) — sufficient for
  bounding evaluated-model code today. Container/VM/seccomp-level isolation
  is a natural next step if the harness needs to withstand a human
  adversary crafting code to attack the host.
- **Providers**: `mock` and `openrouter` are implemented today behind a
  pluggable `Provider` interface built to be extended (see "Adding a new
  provider") — adding a new provider is a focused, well-scoped change.
- **Input packets**: `input.kind: file` reads file contents as text and
  embeds them in the prompt, which covers every task in this benchmark.
  Multimodal input (images, audio) would be a future extension to
  `InputPacket`.
- **Significance testing**: bootstrap CIs are per-model over the shared
  task set, which already makes per-model intervals directly comparable
  by construction. A dedicated paired-difference test between two specific
  models would sharpen head-to-head comparisons further.
- **Storage**: run records are immutable, append-only JSONL, which keeps
  every run human-inspectable and diffable. SQLite remains a
  straightforward addition if query-at-scale becomes a need.
- **Execution order**: `run` calls models sequentially, task by task,
  model by model, which keeps ordering fully deterministic and the
  implementation simple. Parallelizing model calls is a throughput
  optimization that can be layered on without changing the run-record
  format.
