"""Prompt construction, the tool-calling agent loop, and the parse policy.

Every model is called with the exact same system prompt and the exact same
instructions for how to format its final answer, regardless of provider.
A response that fails to parse into the closed answer set gets exactly one
repair attempt; if that also fails, the run is recorded with
`parse_ok=False` and is scored as a confident wrong answer (see
`eval_harness.scoring`).
"""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime

from eval_harness.models import ModelConfig
from eval_harness.providers.base import (
    ANSWER_SET_MARKER,
    PARSE_REPAIR_MARKER,
    Message,
    Provider,
)
from eval_harness.runs import ParsedAnswer, RunRecord, ToolCallLog, new_run_id
from eval_harness.tasks import TaskSpec
from eval_harness.tools import TOOL_DEFINITION, run_tool_call

SYSTEM_PROMPT = (
    "You are an expert insurance risk analyst completing a forced-choice "
    "classification task for a benchmark. For every task you are given a "
    "closed set of answer classes. You must respond with your best "
    "calibrated probability estimate for each class.\n\n"
    "You have access to an offline Python code execution tool with no "
    "network access, for any calculations you don't trust yourself to do "
    "exactly without help. Use it whenever it would improve your accuracy.\n\n"
    "Think step by step, using the tool if useful. Your FINAL message must "
    "contain ONLY a single JSON object of the form "
    '{"probabilities": {"<class>": <probability>, ...}} '
    "with exactly one entry per class in the answer set, non-negative "
    "values that sum to 1, and no other text, no markdown fences, and no "
    "explanation outside the JSON object."
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_FINAL_INSTRUCTION = (
    "Respond with ONLY the JSON object described above. Do not include any "
    "other text."
)


class ParseError(Exception):
    """Raised when a model's final content cannot be parsed into the
    required `{"probabilities": {...}}` schema over the task's answer set."""


def build_messages(task: TaskSpec) -> list[Message]:
    """The initial (system, user) turn pair sent to every model for a task.
    Identical in structure across every model and provider."""
    answer_set_json = json.dumps(task.answer_set)
    lines = [
        f"Task: {task.title}",
        "",
        task.description,
        "",
        task.rendered_input(),
        "",
        f"Answer set ({len(task.answer_set)} classes): {answer_set_json}",
    ]
    if task.family == "ordered_band":
        lines.append(
            "These classes are ordered bands, listed from lowest to highest: "
            f"{answer_set_json}"
        )
    lines += [
        "",
        f"{ANSWER_SET_MARKER} {answer_set_json}",
        "",
        _FINAL_INSTRUCTION,
    ]
    user = Message(role="user", content="\n".join(lines))
    return [Message(role="system", content=SYSTEM_PROMPT), user]


def _strip_fences(content: str) -> str:
    match = _JSON_FENCE_RE.search(content)
    return match.group(1) if match else content


def parse_answer(content: str, answer_set: list[str]) -> ParsedAnswer:
    """Parse model output into a validated, normalized probability
    distribution over exactly `answer_set`. Raises ParseError on any
    deviation: wrong shape, wrong class keys, non-numeric or negative
    values, or a sum too far from 1 to be a rounding artifact."""
    text = _strip_fences(content).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or "probabilities" not in data:
        raise ParseError("JSON object missing required 'probabilities' key")
    probs = data["probabilities"]
    if not isinstance(probs, dict):
        raise ParseError("'probabilities' must be a JSON object")
    if set(probs) != set(answer_set):
        raise ParseError(f"probabilities keys {sorted(probs)} != answer set {sorted(answer_set)}")
    try:
        values = {k: float(v) for k, v in probs.items()}
    except (TypeError, ValueError) as exc:
        raise ParseError(f"probabilities values must be numeric: {exc}") from exc
    if any(v < 0 for v in values.values()):
        raise ParseError("probabilities must be non-negative")
    total = sum(values.values())
    if not (0.9 <= total <= 1.1):
        raise ParseError(f"probabilities must sum to ~1, got {total}")
    normalized = {k: v / total for k, v in values.items()}
    return ParsedAnswer(probabilities=normalized)


def _repair_message(answer_set: list[str]) -> Message:
    content = (
        "Your previous response could not be parsed. It must be ONLY a JSON "
        'object of the form {"probabilities": {"<class>": <probability>, ...}} '
        f"with exactly these classes, probabilities non-negative and summing to 1: "
        f"{json.dumps(answer_set)}. Respond now with ONLY the corrected JSON object.\n\n"
        f"{PARSE_REPAIR_MARKER}"
    )
    return Message(role="user", content=content)


def run_task(task: TaskSpec, config: ModelConfig, provider: Provider, batch_id: str) -> RunRecord:
    """Run one (task, model) pair end to end: build the prompt, run the
    tool-calling agent loop, apply the parse policy (one repair attempt),
    and return a fully populated, immutable RunRecord."""
    start = time.monotonic()
    messages = build_messages(task)
    tools = [TOOL_DEFINITION] if config.tool_use else []
    tool_trace: list[ToolCallLog] = []

    prompt_tokens = completion_tokens = 0
    content: str | None = None
    result = None
    for _ in range(config.max_tool_turns):
        result = provider.complete(messages, tools, config)
        prompt_tokens += result.usage.prompt_tokens
        completion_tokens += result.usage.completion_tokens
        if result.tool_calls:
            assistant_msg = Message(
                role="assistant", content=result.content or "", tool_calls=result.tool_calls
            )
            messages.append(assistant_msg)
            for call in result.tool_calls:
                call_start = time.monotonic()
                tool_result = run_tool_call(call.name, call.arguments)
                duration_ms = (time.monotonic() - call_start) * 1000
                tool_trace.append(
                    ToolCallLog(
                        timestamp=datetime.now(UTC),
                        name=call.name,
                        arguments=call.arguments,
                        result=tool_result,
                        duration_ms=duration_ms,
                    )
                )
                messages.append(
                    Message(
                        role="tool",
                        content=tool_result.model_dump_json(),
                        tool_call_id=call.id,
                        name=call.name,
                    )
                )
            continue
        content = result.content or ""
        break
    else:
        content = result.content or "" if result is not None else ""

    retried = False
    try:
        parsed = parse_answer(content, task.answer_set)
        parse_ok = True
    except ParseError:
        retried = True
        messages.append(Message(role="assistant", content=content))
        repair = _repair_message(task.answer_set)
        messages.append(repair)
        repair_result = provider.complete(messages, [], config)
        prompt_tokens += repair_result.usage.prompt_tokens
        completion_tokens += repair_result.usage.completion_tokens
        content = repair_result.content or ""
        try:
            parsed = parse_answer(content, task.answer_set)
            parse_ok = True
        except ParseError:
            parsed = None
            parse_ok = False

    latency_ms = (time.monotonic() - start) * 1000
    return RunRecord(
        run_id=new_run_id(),
        batch_id=batch_id,
        task_id=task.id,
        model_id=config.id,
        provider=config.provider,
        config=config,
        prompt=messages,
        raw_response=content,
        parsed_answer=parsed,
        parse_ok=parse_ok,
        retried=retried,
        tool_trace=tool_trace,
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        latency_ms=latency_ms,
        timestamp=datetime.now(UTC),
        seed=config.seed,
    )
