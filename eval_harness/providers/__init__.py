"""Pluggable model providers. Every model in the panel is called through the
same Provider interface regardless of whether it is a closed or open-weight
model; OpenRouter is the first real provider, with a deterministic mock
provider for offline testing."""

from __future__ import annotations

from eval_harness.models import ModelConfig
from eval_harness.providers.base import (
    CompletionResult,
    Message,
    Provider,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)
from eval_harness.providers.mock import MockProvider
from eval_harness.providers.openrouter import OpenRouterProvider

_REGISTRY: dict[str, type[Provider]] = {
    "mock": MockProvider,
    "openrouter": OpenRouterProvider,
}


def get_provider(name: str) -> Provider:
    """Instantiate the provider registered under `name`."""
    try:
        cls = _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown provider '{name}', available: {sorted(_REGISTRY)}"
        ) from exc
    return cls()


__all__ = [
    "CompletionResult",
    "Message",
    "ModelConfig",
    "Provider",
    "TokenUsage",
    "ToolCallRequest",
    "ToolDefinition",
    "get_provider",
]
