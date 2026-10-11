"""Deterministic offline provider used for tests and the worked example.

The mock provider never looks at the task's ground truth. It derives the
answer set from the `ANSWER_SET:` marker line that the runner's prompt
builder embeds in the user message, then produces a deterministic pseudo
random probability distribution seeded from the model id, the answer set,
and the configured seed. This makes example runs fully reproducible without
any network access, while still exercising the full pipeline: tool calls,
parsing, and (for the "echo-flaky" and "echo-fail" variants) the retry and
permanent-parse-failure paths.
"""

from __future__ import annotations

import hashlib
import json
import random
import re

from eval_harness.models import ModelConfig
from eval_harness.providers.base import (
    ANSWER_SET_MARKER,
    PARSE_REPAIR_MARKER,
    CompletionResult,
    Message,
    Provider,
    TokenUsage,
    ToolCallRequest,
    ToolDefinition,
)

_ANSWER_SET_RE = re.compile(re.escape(ANSWER_SET_MARKER) + r"\s*(\[.*?\])")


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _extract_answer_set(messages: list[Message]) -> list[str]:
    for msg in messages:
        match = _ANSWER_SET_RE.search(msg.content)
        if match:
            return json.loads(match.group(1))
    raise ValueError("mock provider: no ANSWER_SET marker found in conversation")

def _has_role(messages: list[Message], role: str) -> bool:
    return any(m.role == role for m in messages)

def _has_repair_request(messages: list[Message]) -> bool:
    return any(PARSE_REPAIR_MARKER in m.content for m in messages)

def _deterministic_probabilities(
    model_id: str, answer_set: list[str], seed: int | None
) -> dict[str, float]:
    digest = hashlib.sha256(f"{model_id}|{seed}|{'|'.join(answer_set)}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    raw = [rng.random() + 0.1 for _ in answer_set]
    total = sum(raw)
    probs = [round(v / total, 4) for v in raw]
    # fix rounding drift on the last entry so values sum to exactly 1.0
    probs[-1] = round(1.0 - sum(probs[:-1]), 4)
    return dict(zip(answer_set, probs, strict=True))


class MockProvider(Provider):
    """Supports three `provider_model` variants, selected by name:

    - "echo": calls the sandbox tool once, then returns a valid parseable
      JSON probability distribution.
    - "echo-flaky": returns unparseable text on the first attempt, then a
      valid distribution once the runner sends a repair prompt.
    - "echo-fail": always returns unparseable text, exercising the
      permanent parse-failure path and the parse-failure-rate report.
    """

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        config: ModelConfig,
    ) -> CompletionResult:
        variant = config.provider_model
        prompt_tokens = sum(_approx_tokens(m.content) for m in messages)

        if (
            variant == "echo"
            and config.tool_use
            and tools
            and not _has_role(messages, "tool")
            and not _has_role(messages, "assistant")
        ):
            call = ToolCallRequest(
                id="call_1", name=tools[0].name, arguments={"code": "print(sum(range(10)))"}
            )
            content = "Let me double-check the arithmetic with the sandbox tool."
            return CompletionResult(
                content=content,
                tool_calls=[call],
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=_approx_tokens(content),
                    total_tokens=prompt_tokens + _approx_tokens(content),
                ),
                raw={"mock_variant": variant, "turn": "tool_call"},
            )

        answer_set = _extract_answer_set(messages)

        if variant == "echo-fail" or (
            variant == "echo-flaky" and not _has_repair_request(messages)
        ):
            content = "I think the answer is probably fine, no JSON here."
            return CompletionResult(
                content=content,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=_approx_tokens(content),
                    total_tokens=prompt_tokens + _approx_tokens(content),
                ),
                raw={"mock_variant": variant, "turn": "malformed"},
            )

        probs = _deterministic_probabilities(config.id, answer_set, config.seed)
        content = json.dumps({"probabilities": probs})
        return CompletionResult(
            content=content,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=_approx_tokens(content),
                total_tokens=prompt_tokens + _approx_tokens(content),
            ),
            raw={"mock_variant": variant, "turn": "final"},
        )
