from __future__ import annotations

import json

import httpx
import pytest

from eval_harness.models import ModelConfig
from eval_harness.providers.base import Message, ToolDefinition
from eval_harness.providers.openrouter import OpenRouterError, OpenRouterProvider


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    provider = OpenRouterProvider()
    config = ModelConfig(id="m", provider="openrouter", provider_model="openai/gpt-4.1")
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        provider.complete([Message(role="user", content="hi")], [], config)


def test_successful_completion_is_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    captured_body = {}

    def fake_post(url, json, headers, timeout):  # noqa: A002
        captured_body.update(json)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"probabilities": {"a": 1.0}}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    monkeypatch.setattr("eval_harness.providers.openrouter.httpx.post", fake_post)

    provider = OpenRouterProvider()
    config = ModelConfig(
        id="m", provider="openrouter", provider_model="openai/gpt-4.1", temperature=0.0, seed=7
    )
    result = provider.complete(
        [Message(role="system", content="sys"), Message(role="user", content="hi")], [], config
    )

    assert result.content == '{"probabilities": {"a": 1.0}}'
    assert result.usage.total_tokens == 15
    assert captured_body["model"] == "openai/gpt-4.1"
    assert captured_body["seed"] == 7
    assert captured_body["reasoning"] == {"enabled": True}


def test_tool_calls_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_post(url, json, headers, timeout):  # noqa: A002
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "function": {
                                        "name": "execute_python",
                                        "arguments": json_dumps({"code": "print(1)"}),
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def json_dumps(obj):
        return json.dumps(obj)

    monkeypatch.setattr("eval_harness.providers.openrouter.httpx.post", fake_post)

    provider = OpenRouterProvider()
    config = ModelConfig(id="m", provider="openrouter", provider_model="openai/gpt-4.1")
    tools = [ToolDefinition(name="execute_python", description="d", parameters={"type": "object"})]
    result = provider.complete([Message(role="user", content="hi")], tools, config)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "execute_python"
    assert result.tool_calls[0].arguments == {"code": "print(1)"}


def test_non_200_response_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def fake_post(url, json, headers, timeout):  # noqa: A002
        return httpx.Response(500, text="server error")

    monkeypatch.setattr("eval_harness.providers.openrouter.httpx.post", fake_post)

    provider = OpenRouterProvider()
    config = ModelConfig(id="m", provider="openrouter", provider_model="openai/gpt-4.1")
    with pytest.raises(OpenRouterError, match="HTTP 500"):
        provider.complete([Message(role="user", content="hi")], [], config)
