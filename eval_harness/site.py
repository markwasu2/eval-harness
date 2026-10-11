"""Static HTML results site: headline table (model x overall skill), one
table per task, calibration plots, and a parse-failure table. No server, no
JS, plain HTML tables and embedded PNGs generated with matplotlib (Agg
backend, no display needed).
"""

from __future__ import annotations

import html
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval_harness.scoring import ScoreReport, TaskScore
from eval_harness.tasks import TaskSpec

_STYLE = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }
table { border-collapse: collapse; margin-bottom: 2rem; }
th, td { border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: right; font-size: 0.9rem; }
th { background: #f0f0f0; }
td.label, th.label { text-align: left; }
h1, h2 { margin-top: 2rem; }
.muted { color: #777; font-size: 0.85rem; }
img.calibration { max-width: 320px; display: block; margin-bottom: 1rem; }
"""


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", s)


def _fmt(x: float | None, nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _page(title: str, body: str) -> str:
    return (
        "<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><h1>{html.escape(title)}</h1>{body}</body></html>\n"
    )


def _calibration_plot(model_id: str, rows: list[TaskScore], path: Path) -> None:
    bin_edges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    xs: list[float] = []
    ys: list[float] = []
    counts: list[int] = []
    for lo, hi in bin_edges:
        selected = [r for r in rows if lo <= max(r.predicted_probabilities.values()) < hi]
        if not selected:
            continue
        xs.append(statistics.fmean(max(r.predicted_probabilities.values()) for r in selected))
        ys.append(statistics.fmean(1.0 if r.correct else 0.0 for r in selected))
        counts.append(len(selected))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    if xs:
        ax.plot(xs, ys, marker="o", color="tab:blue", label=model_id)
        for x, y, n in zip(xs, ys, counts, strict=True):
            ax.annotate(str(n), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(f"calibration — {model_id}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _headline_table(report: ScoreReport) -> str:
    rows_html = []
    for s in sorted(report.model_summaries, key=lambda s: s.model_id):
        skill_cell = _fmt(s.mean_skill)
        if s.skill_ci_low is not None and s.skill_ci_high is not None:
            ci = f"[{_fmt(s.skill_ci_low)}, {_fmt(s.skill_ci_high)}]"
            skill_cell += f" <span class='muted'>{ci}</span>"
        rows_html.append(
            "<tr>"
            f"<td class='label'>{html.escape(s.model_id)}</td>"
            f"<td>{s.n_tasks}</td>"
            f"<td>{_fmt(s.accuracy)}</td>"
            f"<td>{_fmt(s.macro_f1)}</td>"
            f"<td>{_fmt(s.mean_brier)}</td>"
            f"<td>{_fmt(s.mean_log_loss)}</td>"
            f"<td>{_fmt(s.mean_rps)}</td>"
            f"<td>{skill_cell}</td>"
            f"<td>{_fmt(s.mean_regret)}</td>"
            f"<td>{_fmt(s.parse_failure_rate)}</td>"
            "</tr>"
        )
    header = (
        "<tr><th class='label'>model</th><th>n tasks</th><th>accuracy</th><th>macro-F1</th>"
        "<th>mean Brier</th><th>mean log loss</th><th>mean RPS</th>"
        "<th>mean skill [95% CI]</th><th>mean regret</th><th>parse-failure rate</th></tr>"
    )
    return f"<table>{header}{''.join(rows_html)}</table>"


def _parse_failure_table(report: ScoreReport) -> str:
    rows_html = []
    for s in sorted(report.model_summaries, key=lambda s: s.model_id):
        rows_html.append(
            f"<tr><td class='label'>{html.escape(s.model_id)}</td>"
            f"<td>{_fmt(s.parse_failure_rate)}</td></tr>"
        )
    return "<table><tr><th class='label'>model</th><th>parse-failure rate</th></tr>" + "".join(
        rows_html
    ) + "</table>"


def _task_table(task: TaskSpec, rows: list[TaskScore]) -> str:
    rows_html = []
    for r in sorted(rows, key=lambda r: r.model_id):
        probs = ", ".join(f"{c}={r.predicted_probabilities[c]:.3f}" for c in task.answer_set)
        rows_html.append(
            "<tr>"
            f"<td class='label'>{html.escape(r.model_id)}</td>"
            f"<td class='label'>{html.escape(probs)}</td>"
            f"<td class='label'>{html.escape(r.predicted_label)}</td>"
            f"<td>{'yes' if r.correct else 'no'}</td>"
            f"<td>{_fmt(r.brier)}</td>"
            f"<td>{_fmt(r.log_loss)}</td>"
            f"<td>{_fmt(r.rps)}</td>"
            f"<td>{_fmt(r.skill)}</td>"
            f"<td>{_fmt(r.regret)}</td>"
            f"<td>{'yes' if r.parse_ok else 'no'}</td>"
            f"<td>{'yes' if r.retried else 'no'}</td>"
            "</tr>"
        )
    header = (
        "<tr><th class='label'>model</th><th class='label'>predicted probabilities</th>"
        "<th class='label'>predicted label</th><th>correct</th><th>Brier</th>"
        "<th>log loss</th><th>RPS</th><th>skill</th><th>regret</th><th>parsed ok</th>"
        "<th>retried</th></tr>"
    )
    return (
        f"<p>true label: <b>{html.escape(task.ground_truth.true_label(task.answer_set))}</b>"
        f" &mdash; family: {task.family} &mdash; track: {html.escape(task.track)}</p>"
        f"<p>{html.escape(task.description)}</p>"
        f"<table>{header}{''.join(rows_html)}</table>"
    )


def build_site(report: ScoreReport, tasks: list[TaskSpec], output_dir: str | Path) -> None:
    """Render the full static results site into `output_dir`."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    calibration_imgs: dict[str, str] = {}
    for summary in report.model_summaries:
        rows = [ts for ts in report.task_scores if ts.model_id == summary.model_id]
        rel_path = f"plots/{_slug(summary.model_id)}_calibration.png"
        _calibration_plot(summary.model_id, rows, output_dir / rel_path)
        calibration_imgs[summary.model_id] = rel_path

    task_links = "".join(
        f"<li><a href='task_{_slug(t.id)}.html'>{html.escape(t.id)}</a> "
        f"&mdash; {html.escape(t.title)}</li>"
        for t in sorted(tasks, key=lambda t: t.id)
    )
    calibration_html = "".join(
        f"<div><p class='muted'>{html.escape(mid)}</p>"
        f"<img class='calibration' src='{src}'></div>"
        for mid, src in sorted(calibration_imgs.items())
    )
    index_body = (
        f"<p class='muted'>batch: {html.escape(report.batch_id)}</p>"
        "<h2>headline: model &times; overall skill</h2>"
        f"{_headline_table(report)}"
        "<h2>parse-failure rate</h2>"
        f"{_parse_failure_table(report)}"
        "<h2>calibration</h2>"
        f"{calibration_html}"
        "<h2>tasks</h2>"
        f"<ul>{task_links}</ul>"
    )
    (output_dir / "index.html").write_text(_page("eval-harness results", index_body))

    for task in tasks:
        rows = [ts for ts in report.task_scores if ts.task_id == task.id]
        body = (
            "<p><a href='index.html'>&larr; back to headline</a></p>"
            f"{_task_table(task, rows)}"
        )
        (output_dir / f"task_{_slug(task.id)}.html").write_text(_page(task.title, body))
