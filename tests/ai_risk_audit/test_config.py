from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ai_risk_audit.config import (
    DEFAULT_CHEAP_MODEL,
    DEFAULT_JUDGE_MODEL,
    AuditConfig,
    AuditConfigError,
    load_config,
)


def test_defaults() -> None:
    config = load_config(None)
    assert config.profiler_model.provider_model == DEFAULT_CHEAP_MODEL
    assert config.planner_model.provider_model == DEFAULT_CHEAP_MODEL
    assert config.auditor_model.provider_model == DEFAULT_CHEAP_MODEL
    assert config.target_model.provider_model == DEFAULT_CHEAP_MODEL
    assert config.judge_model.provider_model == DEFAULT_JUDGE_MODEL
    assert config.scenarios_per_category == 2
    assert config.repeats == 2
    assert config.max_turns == 6
    assert config.budget_warn_usd == 3.0
    assert config.budget_abort_usd == 5.0


def test_load_from_yaml(tmp_path: Path) -> None:
    data = {
        "judge_model": {"provider_model": "some/strong-model"},
        "repeats": 3,
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data))

    config = load_config(path)
    assert config.judge_model.provider_model == "some/strong-model"
    assert config.repeats == 3
    # untouched fields keep their defaults
    assert config.profiler_model.provider_model == DEFAULT_CHEAP_MODEL


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(AuditConfigError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_extra_field_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"not_a_real_field": 1}))
    with pytest.raises(AuditConfigError):
        load_config(path)


def test_extra_field_rejected_at_model_level() -> None:
    with pytest.raises(ValidationError):
        AuditConfig.model_validate({"profiler_model": {"provider_model": "x", "bogus": 1}})
