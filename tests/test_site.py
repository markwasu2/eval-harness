from __future__ import annotations

from pathlib import Path

from eval_harness.models import ModelConfig
from eval_harness.providers.mock import MockProvider
from eval_harness.runner import run_task
from eval_harness.scoring import score_batch
from eval_harness.site import build_site
from eval_harness.tasks import load_tasks


def test_build_site_renders_expected_files(tmp_path: Path, example_tasks_dir: Path) -> None:
    tasks = load_tasks([example_tasks_dir])
    config = ModelConfig(id="echo-baseline", provider="mock", provider_model="echo", seed=0)
    provider = MockProvider()
    records = [run_task(t, config, provider, "batch-1") for t in tasks]
    report = score_batch(tasks, records, n_resamples=50, seed=0)

    out_dir = tmp_path / "site"
    build_site(report, tasks, out_dir)

    assert (out_dir / "index.html").is_file()
    index_html = (out_dir / "index.html").read_text()
    assert "headline" in index_html
    assert "echo-baseline" in index_html

    for task in tasks:
        task_page = out_dir / f"task_{task.id}.html"
        assert task_page.is_file()
        assert task.title in task_page.read_text()

    plot = out_dir / "plots" / "echo-baseline_calibration.png"
    assert plot.is_file()
    assert plot.stat().st_size > 0
