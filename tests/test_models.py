from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eval_harness.models import ModelConfigError, load_panel


def test_load_example_panel(example_mock_panel: Path) -> None:
    panel = load_panel(example_mock_panel)
    ids = {m.id for m in panel.models}
    assert ids == {"echo-baseline", "echo-flaky", "echo-always-fails"}


def test_duplicate_model_ids_rejected(tmp_path: Path) -> None:
    data = {
        "models": [
            {"id": "x", "provider": "mock", "provider_model": "echo"},
            {"id": "x", "provider": "mock", "provider_model": "echo-flaky"},
        ]
    }
    path = tmp_path / "panel.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ModelConfigError):
        load_panel(path)


def test_unknown_provider_rejected(tmp_path: Path) -> None:
    data = {"models": [{"id": "x", "provider": "bogus", "provider_model": "echo"}]}
    path = tmp_path / "panel.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ModelConfigError):
        load_panel(path)


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelConfigError):
        load_panel(tmp_path / "does_not_exist.yaml")
