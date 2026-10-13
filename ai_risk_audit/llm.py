"""Decoupled chat-completion client for the profiler/planner/judge stages.

Kept separate from `eval_harness.providers`: those are tuned for the main
harness's sandboxed tool-calling use case, and reusing them here would pull
in unrelated machinery for no benefit. Two real backends are provided:
`AnthropicChat` (direct Anthropic Messages API, used by default — no funded
OpenRouter account was available) and `OpenRouterChat` (kept for when one
is). Petri's own auditor/target/judge calls during stage 3 report cost
through a different path (`EvalSample.role_usage`); both paths feed the
same `CostTracker`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import httpx

_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


class LLMError(RuntimeError):
    """Raised for API-key, HTTP, or response-shape failures."""


@dataclass
class ChatResult:
    content: str
    usd: float
    prompt_tokens: int
    completion_tokens: int


class ChatClient(Protocol):
    """Satisfied by both `OpenRouterChat` (real) and `MockChat` (offline
    tests), so profiler/planner/judge code can be exercised without network
    access or spend."""

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ChatResult: ...


class OpenRouterChat:
    """Real OpenRouter chat-completions client."""

    def __init__(self, timeout_s: float = 120.0) -> None:
        self._timeout_s = timeout_s

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ChatResult:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMError(
                "OPENROUTER_API_KEY is not set in the environment; "
                "export it before making real audit calls"
            )

        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "ai-risk-audit",
        }

        try:
            response = httpx.post(
                _OPENROUTER_API_URL, json=body, headers=headers, timeout=self._timeout_s
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"request to OpenRouter failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"OpenRouter returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected OpenRouter response shape: {data}") from exc

        usage = data.get("usage") or {}
        return ChatResult(
            content=content or "",
            usd=float(usage.get("cost") or 0.0),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
        )


# $/million-token rates used to price Anthropic calls ourselves: the
# Messages API (unlike OpenRouter) never returns cost on the response, and
# Inspect AI's `ModelUsage.total_cost` comes back `None`/0 for these model
# ids too (no pricing table entry) — confirmed by the stage-3 smoke test,
# where execution-stage cost silently read as $0 despite real token usage.
# Public so `execution/run.py` can price Petri's auditor/target/judge usage
# with the same table. Keyed by model-id prefix.
ANTHROPIC_RATES_PER_MTOK = {
    "claude-opus": (15.00, 75.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-haiku": (1.00, 5.00),
}
_CACHE_WRITE_MULTIPLIER = 1.25
"""Anthropic charges a premium to write the (default 5-minute ephemeral)
prompt cache, and a discount to read from it; see Anthropic's pricing docs."""
_CACHE_READ_MULTIPLIER = 0.10


def anthropic_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    for prefix, (in_rate, out_rate) in ANTHROPIC_RATES_PER_MTOK.items():
        if model.startswith(prefix):
            cost = (
                input_tokens * in_rate
                + output_tokens * out_rate
                + cache_write_tokens * in_rate * _CACHE_WRITE_MULTIPLIER
                + cache_read_tokens * in_rate * _CACHE_READ_MULTIPLIER
            )
            return cost / 1_000_000
    raise LLMError(f"no known cost rate for Anthropic model {model!r}")


def _strip_markdown_fence(content: str) -> str:
    """Anthropic has no native JSON-mode flag (unlike OpenRouter's
    `response_format`), so a model sometimes wraps its JSON reply in a
    ```json ... ``` fence despite being told not to. Strip it if present."""
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


class AnthropicChat:
    """Real direct-to-Anthropic Messages API client (default backend)."""

    def __init__(self, timeout_s: float = 120.0) -> None:
        self._timeout_s = timeout_s

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ChatResult:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set in the environment; "
                "export it (see .env) before making real audit calls"
            )

        user_content = user
        if json_mode:
            user_content += (
                "\n\nRespond with ONLY a single valid JSON object. No prose, "
                "no markdown code fences, no commentary before or after."
            )

        body: dict = {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        try:
            response = httpx.post(
                _ANTHROPIC_API_URL, json=body, headers=headers, timeout=self._timeout_s
            )
        except httpx.HTTPError as exc:
            raise LLMError(f"request to Anthropic failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMError(
                f"Anthropic returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        try:
            content = "".join(
                block.get("text", "") for block in data["content"] if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected Anthropic response shape: {data}") from exc

        if json_mode:
            content = _strip_markdown_fence(content)

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)
        return ChatResult(
            content=content,
            usd=anthropic_cost_usd(model, prompt_tokens, completion_tokens),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


class MockChat:
    """Offline scripted client for tests.

    Returns canned responses in call order, at zero cost. Raises `LLMError`
    if more calls are made than responses were scripted, so tests fail loudly
    on an unexpected extra call rather than silently reusing stale output.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.calls: list[dict[str, str]] = []

    def complete(
        self,
        system: str,
        user: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> ChatResult:
        idx = len(self.calls)
        self.calls.append({"system": system, "user": user, "model": model})
        if idx >= len(self._responses):
            raise LLMError(f"MockChat exhausted: call {idx} has no scripted response")
        return ChatResult(
            content=self._responses[idx],
            usd=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )
