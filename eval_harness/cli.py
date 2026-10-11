"""Command-line interface.

    eval-harness run --config run.yaml --tasks tasks/ --models models.yaml
    eval-harness score --run runs/<batch>.jsonl --tasks tasks/
    eval-harness build-site --run runs/<batch>.jsonl --tasks tasks/ --out site/
    eval-harness baseline --task tasks/classification_3class.yaml
    eval-harness all --config run.yaml --tasks tasks/ --models models.yaml --out .
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import yaml
from pydantic import BaseModel, ConfigDict

from eval_harness.models import load_panel
from eval_harness.providers import get_provider
from eval_harness.runner import run_task
from eval_harness.runs import RunRecord, batch_path, load_records, new_batch_id, write_record
from eval_harness.scoring import ScoreReport, score_batch
from eval_harness.site import build_site as render_site
from eval_harness.tasks import TaskSpec, load_task, load_tasks


class RunConfig(BaseModel):
    """Settings for `eval-harness run` / `score` not tied to a specific
    model or task: bootstrap resampling parameters for score reporting."""

    model_config = ConfigDict(extra="forbid")

    n_bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0


def _load_run_config(path: str | None) -> RunConfig:
    if path is None:
        return RunConfig()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return RunConfig.model_validate(data)


def _score(cfg: RunConfig, tasks: list[TaskSpec], records: list[RunRecord]) -> ScoreReport:
    return score_batch(
        tasks, records, n_resamples=cfg.n_bootstrap_resamples, seed=cfg.bootstrap_seed
    )


def _print_headline(report: ScoreReport) -> None:
    click.echo(f"batch: {report.batch_id}")
    click.echo(f"{'model':<20}{'n':>4}{'acc':>8}{'macroF1':>9}{'brier':>8}{'skill':>9}{'parsefail':>11}")
    for s in sorted(report.model_summaries, key=lambda s: s.model_id):
        skill = f"{s.mean_skill:.3f}" if s.mean_skill is not None else "—"
        click.echo(
            f"{s.model_id:<20}{s.n_tasks:>4}{s.accuracy:>8.3f}{s.macro_f1:>9.3f}"
            f"{s.mean_brier:>8.3f}{skill:>9}{s.parse_failure_rate:>11.3f}"
        )


@click.group()
def main() -> None:
    """eval-harness: a config-driven LLM evaluation harness."""


@main.command("run")
@click.option("--config", "config_path", default=None, help="run settings YAML (optional)")
@click.option("--tasks", "task_paths", required=True, multiple=True, help="task file(s) or dir(s)")
@click.option("--models", "models_path", required=True, help="model panel YAML/JSON")
@click.option("--out", "out_dir", default="runs", help="directory to write the run batch into")
def run_command(
    config_path: str | None, task_paths: tuple[str, ...], models_path: str, out_dir: str
) -> None:
    """Run every model in the panel against every task and record results."""
    _load_run_config(config_path)  # validated eagerly even though unused here
    tasks = load_tasks(list(task_paths))
    panel = load_panel(models_path)

    batch_id = new_batch_id()
    path = batch_path(Path(out_dir), batch_id)
    providers = {m.provider for m in panel.models}
    provider_instances = {name: get_provider(name) for name in providers}

    n_written = 0
    for task in tasks:
        for model_config in panel.models:
            provider = provider_instances[model_config.provider]
            record = run_task(task, model_config, provider, batch_id)
            write_record(path, record)
            n_written += 1
            click.echo(
                f"[{model_config.id}] {task.id}: parse_ok={record.parse_ok} "
                f"retried={record.retried} latency_ms={record.latency_ms:.0f}"
            )

    click.echo(f"wrote {n_written} run records to {path}")


@main.command("score")
@click.option("--run", "run_path", required=True, help="run batch JSONL file")
@click.option("--tasks", "task_paths", required=True, multiple=True, help="task file(s) or dir(s)")
@click.option("--config", "config_path", default=None, help="run settings YAML (optional)")
@click.option(
    "--out", "out_path", default=None, help="where to write scores.json (default: alongside run)"
)
def score_command(
    run_path: str, task_paths: tuple[str, ...], config_path: str | None, out_path: str | None
) -> None:
    """Score a run batch against its tasks and print/write a summary."""
    cfg = _load_run_config(config_path)
    tasks = load_tasks(list(task_paths))
    records = load_records(run_path)
    report = _score(cfg, tasks, records)

    out = Path(out_path) if out_path else Path(run_path).with_suffix(".scores.json")
    out.write_text(report.model_dump_json(indent=2))
    _print_headline(report)
    click.echo(f"wrote scores to {out}")


@main.command("build-site")
@click.option("--run", "run_path", required=True, help="run batch JSONL file")
@click.option("--tasks", "task_paths", required=True, multiple=True, help="task file(s) or dir(s)")
@click.option("--config", "config_path", default=None, help="run settings YAML (optional)")
@click.option("--out", "out_dir", default="site", help="directory to render the static site into")
def build_site_command(
    run_path: str, task_paths: tuple[str, ...], config_path: str | None, out_dir: str
) -> None:
    """Render the static results site for a run batch."""
    cfg = _load_run_config(config_path)
    tasks = load_tasks(list(task_paths))
    records = load_records(run_path)
    report = _score(cfg, tasks, records)
    render_site(report, tasks, out_dir)
    click.echo(f"rendered site to {Path(out_dir) / 'index.html'}")


@main.command("baseline")
@click.option("--task", "task_path", required=True, help="a single task file")
def baseline_command(task_path: str) -> None:
    """Print a task's mechanical baseline and oracle/ceiling score."""
    task: TaskSpec = load_task(task_path)
    true_dist = task.ground_truth.distribution(task.answer_set)
    from eval_harness.scoring import task_loss

    baseline_dist = task.baseline.distribution(task.answer_set)
    baseline_loss = task_loss(task, baseline_dist, true_dist)
    click.echo(f"task: {task.id} ({task.family})")
    click.echo(f"baseline strategy: {task.baseline.strategy}")
    click.echo(f"baseline distribution: {json.dumps(baseline_dist)}")
    metric_name = "RPS" if task.family == "ordered_band" else "Brier"
    click.echo(f"baseline loss ({metric_name}): {baseline_loss:.4f}")
    if task.oracle is not None:
        oracle_dist = task.oracle.distribution(task.ground_truth, task.answer_set)
        oracle_loss = task_loss(task, oracle_dist, true_dist)
        click.echo(f"oracle distribution: {json.dumps(oracle_dist)}")
        click.echo(f"oracle loss: {oracle_loss:.4f}")
    else:
        click.echo("oracle: not defined for this task")


@main.command("all")
@click.option("--config", "config_path", default=None, help="run settings YAML (optional)")
@click.option("--tasks", "task_paths", required=True, multiple=True, help="task file(s) or dir(s)")
@click.option("--models", "models_path", required=True, help="model panel YAML/JSON")
@click.option("--out", "out_dir", default=".", help="base directory for runs/ and site/")
@click.pass_context
def all_command(
    ctx: click.Context,
    config_path: str | None,
    task_paths: tuple[str, ...],
    models_path: str,
    out_dir: str,
) -> None:
    """Run every model against every task, score the batch, and render the site."""
    base = Path(out_dir)
    runs_dir = base / "runs"
    site_dir = base / "site"

    tasks = load_tasks(list(task_paths))
    panel = load_panel(models_path)
    cfg = _load_run_config(config_path)

    batch_id = new_batch_id()
    path = batch_path(runs_dir, batch_id)
    provider_instances = {name: get_provider(name) for name in {m.provider for m in panel.models}}

    for task in tasks:
        for model_config in panel.models:
            provider = provider_instances[model_config.provider]
            record = run_task(task, model_config, provider, batch_id)
            write_record(path, record)

    records = load_records(path)
    report = _score(cfg, tasks, records)
    scores_path = path.with_suffix(".scores.json")
    scores_path.write_text(report.model_dump_json(indent=2))
    render_site(report, tasks, site_dir)

    _print_headline(report)
    click.echo(f"run batch: {path}")
    click.echo(f"scores: {scores_path}")
    click.echo(f"site: {site_dir / 'index.html'}")


if __name__ == "__main__":
    sys.exit(main())
