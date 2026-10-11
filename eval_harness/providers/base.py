"""Provider interface shared by every backend (mock, OpenRouter, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from eval_harness.models import ModelConfig

Role = Literal["system", "user", "assistant", "tool"]

# Shared markers embedded in prompt text by eval_harness.runner, and read back
# out by eval_harness.providers.mock so the mock provider can behave
# consistently without any provider-specific coupling to the real prompt text.
ANSWER_SET_MARKER = "ANSWER_SET:"
PARSE_REPAIR_MARKER = "PARSE_REPAIR_REQUEST"


class Message(BaseModel):
    """One turn in the conversation sent to/received from a provider."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCallRequest] | None = None


class ToolDefinition(BaseModel):
    """A function-calling tool definition exposed to the model."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCallRequest(BaseModel):
    """A single tool invocation requested by the model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any]


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResult(BaseModel):
    """The normalized result of one provider call."""

    model_config = ConfigDict(extra="forbid")

    content: str | None
    tool_calls: list[ToolCallRequest] = []
    usage: TokenUsage
    raw: dict[str, Any] = {}


class Provider(ABC):
    """A pluggable backend that turns a standardized call configuration into
    a single model completion. Implementations must not use provider-native
    code execution tools; the only tool ever exposed is the offline sandbox
    passed in via `tools`."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        config: ModelConfig,
    ) -> CompletionResult: ...
