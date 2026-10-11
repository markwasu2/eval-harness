from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_TASKS_DIR = REPO_ROOT / "tasks"
EXAMPLE_MOCK_PANEL = REPO_ROOT / "model_panels" / "example_mock.yaml"


@pytest.fixture
def example_tasks_dir() -> Path:
    return EXAMPLE_TASKS_DIR


@pytest.fixture
def example_mock_panel() -> Path:
    return EXAMPLE_MOCK_PANEL
