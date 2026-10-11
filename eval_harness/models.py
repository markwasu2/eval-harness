"""Model panel configuration.

Every model in the panel, closed or open-weight, is called through the same
pluggable Provider interface with one standardized call configuration: the
same system prompt, temperature/seed where supported, extended reasoning on,
and tool use enabled. Providers differ only in how they translate that single
configuration into a concrete API call.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["mock", "openrouter"]


class ModelConfigError(Exception):
    """Raised when a model panel file fails to parse or validate."""


class ModelConfig(BaseModel):
    """One model entry in the panel. `provider_model` is the identifier sent
    to the provider's API (e.g. an OpenRouter route like 'openai/gpt-4o')."""

    model_config = ConfigDict(extra="forbid")

    id: str
    provider: ProviderName
    provider_model: str
    temperature: float = 0.0
    seed: int | None = 0
    max_tokens: int = 2048
    reasoning: bool = True
    tool_use: bool = True
    max_tool_turns: int = 4


class ModelPanel(BaseModel):
    """The full set of models to evaluate."""

    model_config = ConfigDict(extra="forbid")

    models: list[ModelConfig] = Field(min_length=1)


def load_panel(path: str | Path) -> ModelPanel:
    path = Path(path)
    if not path.is_file():
        raise ModelConfigError(f"model panel file not found: {path}")
    try:
        raw_text = path.read_text()
        data = json.loads(raw_text) if path.suffix.lower() == ".json" else yaml.safe_load(raw_text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ModelConfigError(f"{path}: could not parse file: {exc}") from exc
    try:
        panel = ModelPanel.model_validate(data)
    except Exception as exc:
        raise ModelConfigError(f"{path}: invalid model panel: {exc}") from exc
    ids = [m.id for m in panel.models]
    if len(set(ids)) != len(ids):
        raise ModelConfigError(f"{path}: duplicate model ids in panel")
    return panel
