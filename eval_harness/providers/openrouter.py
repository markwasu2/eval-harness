"""Real provider: every model in the panel (closed or open-weight) is routed
through the OpenRouter chat-completions API using the OpenAI-compatible
function-calling schema. Requires OPENROUTER_API_KEY in the environment.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from eval_harness.models import ModelConfig
from eval_harness.providers.base import (
    CompletionResult,
    Message,
    Provider,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)

_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    """Raised for API-key, HTTP, or response-shape failures."""


def _to_openai_message(msg: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.role == "tool":
        out["tool_call_id"] = msg.tool_call_id
        out["name"] = msg.name
    if msg.role == "assistant" and msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in msg.tool_calls
        ]
    return out


def _to_openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class OpenRouterProvider(Provider):
    def __init__(self, timeout_s: float = 120.0) -> None:
        self._timeout_s = timeout_s

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        config: ModelConfig,
    ) -> CompletionResult:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise OpenRouterError(
                "OPENROUTER_API_KEY is not set in the environment; "
                "export it before running against the openrouter provider"
            )

        body: dict[str, Any] = {
            "model": config.provider_model,
            "messages": [_to_openai_message(m) for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if config.seed is not None:
            body["seed"] = config.seed
        if config.reasoning:
            body["reasoning"] = {"enabled": True}
        if config.tool_use and tools:
            body["tools"] = [_to_openai_tool(t) for t in tools]
            body["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "eval-harness",
        }

        try:
            response = httpx.post(_API_URL, json=body, headers=headers, timeout=self._timeout_s)
        except httpx.HTTPError as exc:
            raise OpenRouterError(f"request to OpenRouter failed: {exc}") from exc

        if response.status_code != 200:
            raise OpenRouterError(
                f"OpenRouter returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            choice = data["choices"][0]["message"]
        except (KeyError, IndexError) as exc:
            raise OpenRouterError(f"unexpected OpenRouter response shape: {data}") from exc

        tool_calls = [
            ToolCallRequest(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=json.loads(tc["function"]["arguments"] or "{}"),
            )
            for tc in choice.get("tool_calls") or []
        ]

        usage_raw = data.get("usage") or {}
        usage = TokenUsage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )

        return CompletionResult(
            content=choice.get("content"),
            tool_calls=tool_calls,
            usage=usage,
            raw=data,
        )
