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

## Explicitly out of scope

- **True sandbox isolation** (containers/VMs/seccomp/namespaces). The tool
  sandbox is language-level + POSIX rlimits only; see "The tool sandbox"
  above.
- **Providers other than OpenRouter.** The `Provider` interface is
  designed to be extended (see "Adding a new provider"), but only
  `mock` and `openrouter` are implemented.
- **Multimodal / non-text input packets.** `input.kind: file` reads file
  contents as text and embeds them in the prompt; images, audio, or other
  binary inputs are not supported.
- **A paired, model-vs-model significance test.** The bootstrap CIs are
  per-model (over the shared task set); a dedicated paired-difference test
  between two specific models is not implemented.
- **SQLite storage.** Run records are JSONL only; the spec allowed either.
- **Concurrency / parallel model calls.** `run` calls models sequentially,
  task by task, model by model. This keeps ordering deterministic and the
  implementation simple; it is not a throughput optimization.
