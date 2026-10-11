from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from eval_harness.cli import main


def test_all_command_end_to_end(
    tmp_path: Path, example_tasks_dir: Path, example_mock_panel: Path
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "all",
            "--tasks",
            str(example_tasks_dir),
            "--models",
            str(example_mock_panel),
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "site" / "index.html").is_file()
    runs = list((tmp_path / "runs").glob("*.jsonl"))
    assert len(runs) == 1
    scores = list((tmp_path / "runs").glob("*.scores.json"))
    assert len(scores) == 1


def test_run_then_score_then_build_site_separately(
    tmp_path: Path, example_tasks_dir: Path, example_mock_panel: Path
) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    result = runner.invoke(
        main,
        [
            "run",
            "--tasks",
            str(example_tasks_dir),
            "--models",
            str(example_mock_panel),
            "--out",
            str(runs_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    run_file = next(runs_dir.glob("*.jsonl"))

    result = runner.invoke(
        main, ["score", "--run", str(run_file), "--tasks", str(example_tasks_dir)]
    )
    assert result.exit_code == 0, result.output
    assert run_file.with_suffix(".scores.json").is_file()

    site_dir = tmp_path / "site"
    result = runner.invoke(
        main,
        [
            "build-site",
            "--run",
            str(run_file),
            "--tasks",
            str(example_tasks_dir),
            "--out",
            str(site_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (site_dir / "index.html").is_file()


def test_rerun_creates_new_batch_not_overwrite(
    tmp_path: Path, example_tasks_dir: Path, example_mock_panel: Path
) -> None:
    runner = CliRunner()
    runs_dir = tmp_path / "runs"
    args = [
        "run",
        "--tasks",
        str(example_tasks_dir),
        "--models",
        str(example_mock_panel),
        "--out",
        str(runs_dir),
    ]
    assert runner.invoke(main, args).exit_code == 0
    assert runner.invoke(main, args).exit_code == 0
    assert len(list(runs_dir.glob("*.jsonl"))) == 2


def test_baseline_command(example_tasks_dir: Path) -> None:
    runner = CliRunner()
    task_file = example_tasks_dir / "classification_3class.yaml"
    result = runner.invoke(main, ["baseline", "--task", str(task_file)])
    assert result.exit_code == 0, result.output
    assert "baseline loss" in result.output
    assert "oracle loss" in result.output
