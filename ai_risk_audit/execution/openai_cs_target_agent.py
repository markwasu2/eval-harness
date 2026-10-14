"""Petri target agent that drives the openai-cs-agents-demo airline agent.

openai-cs-agents-demo (https://github.com/openai/openai-cs-agents-demo) is a
multi-agent airline customer-service demo built on the OpenAI Agents SDK plus
ChatKit for its UI/streaming layer. This module drives its agents the same
way `server.py`'s `AirlineServer.respond()` does (import the plain `Agent`
objects from `airline.agents`, call `Runner.run()`, thread `input_items` /
`AirlineAgentContext` state / `current_agent` across turns) with the auditor
standing in for a human user, leaving the demo's own files completely
untouched: we never edit anything under `openai-cs-agents-demo/`, only import
from it after inserting its `python-backend` directory onto `sys.path` (it
has no `pyproject.toml`/`setup.py`, so it can't be `uv`/`pip` installed).

**Backend substitution, explicitly disclosed (per this task's authorization,
not a silent patch):** this environment has no `OPENAI_API_KEY`, so the demo
can't run on its own hardcoded OpenAI models (`gpt-5.2` for the 6 main
agents, `gpt-4.1-mini` for its 2 input-guardrail agents in
`airline/guardrails.py`). `make_openai_cs_target_agent` swaps every one of
those 8 `Agent.model` attributes to `agents.extensions.models.litellm_model.
LitellmModel("anthropic/<agent_model>")`, routed through the `ANTHROPIC_API_KEY`
already configured for this pipeline (see `.env`), so the target actually
audited here runs on a substituted Claude backend. Everything else about the
target -- instructions, tools, handoffs, guardrails, agent structure -- is
the demo's own code, unmodified. `pipeline.py` passes this fact into the
report's `environment_note`, and `configs/openai_cs.yaml` documents it too,
per this task's disclosure requirement.

**A genuine litellm<1.83 bug, worked around here, not papered over in the
target's own behavior:** `openai-agents`'s `LitellmModel` always passes
`reasoning_effort=reasoning_effort` to `litellm.acompletion()`, even when
`None`. Combined with any `tools` being present in the request, this trips a
bug in `litellm==1.82.6` (the version pinned by this project's `tau2` extra,
hence the ceiling -- litellm>=1.83 fixes it but conflicts with tau2's own
pin): `BaseConfig.is_thinking_enabled()` crashes on `None.get(...)`, and even
once that's guarded, an explicit `"thinking": null` leaks into the raw
Anthropic request body and gets rejected. `_patch_litellm_thinking_bug`
monkeypatches both call sites narrowly (verified against a live tool-calling
+ handoff + guardrail run -- see this module's test file and the task's own
verification notes) -- this is a third-party library compatibility shim, not
a change to the target's instructions/tools/handoffs.

**A known reliability gap of the substitution, retried not papered over:**
during live audit runs, the 2 input-guardrail agents' structured JSON output
(`RelevanceOutput`/`JailbreakOutput`) occasionally fails
`agents.exceptions.ModelBehaviorError`'s "Invalid JSON when parsing model
output" validation when driven via litellm/Claude -- a rate not observed
with the demo's own `gpt-4.1-mini` (which uses OpenAI's native JSON-schema
response format; litellm's Anthropic structured-output path is closer to
best-effort tool-use coercion). `DEFAULT_MAX_MODEL_BEHAVIOR_RETRIES` retries
the whole `Runner.run` call (same input, same agent, same guardrails) a
bounded number of times before giving up and re-raising -- this absorbs an
occasional malformed-output glitch without changing the guardrails' own
instructions/schema, and a persistent failure still surfaces as a real
error rather than being silently swallowed.

Design, mirroring `execution/tau_target_agent.py`:
1. `make_openai_cs_target_agent` does the one-time setup (sys.path insertion,
   litellm monkeypatch, 8-agent model swap) and returns a Petri target agent
   closure.
2. That closure speaks Petri's low-level `TargetContext` handshake (matching
   `execution/tau_target_agent.py` / `execution/target_agent.py`'s
   precedent): `wait_for_resume()` -> `user_message()` -> loop (turn ->
   `send_output` -> next `user_message()`), catching `ExitSignal`. Each turn
   appends the auditor's message to `input_items` and calls `Runner.run()`,
   exactly as `server.py`'s `respond()` does for a real UI turn, tracking
   `current_agent` across handoffs via `result.last_agent`.
3. `Runner.run()`'s own `to_input_list()` already returns the full
   conversation so far in a flat, uniform shape (plain `{"role", "content"}`
   dicts for user/assistant text, `{"type": "function_call", ...}` for every
   tool call INCLUDING handoffs, `{"type": "function_call_output", ...}` for
   results) -- confirmed by live inspection. `_input_items_to_dicts` converts
   that directly into the `{"role", "content", "tool_calls", "tool_call_id"}`
   shape `execution/transcript.py`'s `TranscriptMessage` expects, mirrored
   into the sample store on every turn under
   `OPENAI_CS_TARGET_MESSAGES_STORE_KEY` (same reasoning as the tau2 adapter:
   `EvalSample.messages` is the AUDITOR's history, not this target's).
4. The demo's own LLM calls (main agents AND their input guardrails) bypass
   Inspect's role-based usage tracking, so this module prices them itself
   from `RunResult.context_wrapper.usage` (confirmed via a live test to
   aggregate usage across both the main agent's turn and any guardrail calls
   made during it) via `llm.anthropic_cost_usd`, and records that into the
   shared `CostTracker` -- same pattern as `tau_target_agent.py` part 3.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.util import store
from inspect_petri.target import TOOL_RESULT, ExitSignal, TargetContext

from ai_risk_audit.cost import CostTracker
from ai_risk_audit.llm import anthropic_cost_usd

OPENAI_CS_TARGET_MESSAGES_STORE_KEY = "ai_risk_audit:openai_cs_target_messages"
"""Store key this agent writes its own (openai-cs-agents-demo-side)
conversation under, and that `execution/run.py`'s transcript-building code
reads back from `EvalSample.store` after the run."""

DEFAULT_MAX_INTERNAL_TOOL_TURNS = 10
"""Passed through to `Runner.run(..., max_turns=...)`: caps the demo's own
agent<->tool<->handoff loop per auditor turn, same safety-valve role as
`tau_target_agent.py`'s `DEFAULT_MAX_INTERNAL_TOOL_TURNS`."""

DEFAULT_MAX_MODEL_BEHAVIOR_RETRIES = 5
"""Live-verified reliability gap of the litellm/Claude substitution (see
module docstring): the two input-guardrail agents' structured JSON output
occasionally fails `agents.exceptions.ModelBehaviorError`'s
"Invalid JSON when parsing model output" validation when driven via
litellm/Claude, at a low rate not observed with the demo's own
gpt-4.1-mini. This retries the whole `Runner.run` call (same input) up to
this many extra times before giving up -- a bounded reliability retry at
the infrastructure layer, not a change to the guardrails' own logic/schema."""

_DEMO_PYTHON_BACKEND_DIR = (
    Path(__file__).resolve().parents[3] / "openai-cs-agents-demo" / "python-backend"
)
"""Sibling clone of https://github.com/openai/openai-cs-agents-demo, cloned
read-only into the workspace next to `eval-harness/`. Never written to."""

_setup_done = False


class OpenaiCsAdapterError(Exception):
    """Raised when the demo's agents can't be imported/driven as expected."""


def _ensure_demo_importable() -> None:
    if not _DEMO_PYTHON_BACKEND_DIR.is_dir():
        raise OpenaiCsAdapterError(
            f"expected openai-cs-agents-demo clone at {_DEMO_PYTHON_BACKEND_DIR}, "
            "but it doesn't exist -- clone https://github.com/openai/openai-cs-agents-demo "
            "as a sibling of eval-harness/ first"
        )
    path_str = str(_DEMO_PYTHON_BACKEND_DIR)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _patch_litellm_thinking_bug() -> None:
    """Two-part monkeypatch for a genuine litellm<1.83 bug that otherwise
    crashes every tool-calling Anthropic call made through the newer
    openai-agents SDK's `LitellmModel` (see module docstring). Idempotent:
    safe to call more than once."""
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig
    from litellm.llms.base_llm.chat.transformation import BaseConfig

    def _patched_is_thinking_enabled(self: Any, non_default_params: dict) -> bool:
        return (
            (non_default_params.get("thinking") or {}).get("type") == "enabled"
            or non_default_params.get("reasoning_effort") is not None
        )

    BaseConfig.is_thinking_enabled = _patched_is_thinking_enabled

    if not getattr(AnthropicConfig.map_openai_params, "_ai_risk_audit_patched", False):
        _orig_map_openai_params = AnthropicConfig.map_openai_params

        def _patched_map_openai_params(
            self: Any,
            non_default_params: dict,
            optional_params: dict,
            model: str,
            drop_params: bool,
        ) -> dict:
            result = _orig_map_openai_params(
                self, non_default_params, optional_params, model, drop_params
            )
            if result.get("thinking") is None and "thinking" in result:
                del result["thinking"]
            return result

        _patched_map_openai_params._ai_risk_audit_patched = True  # type: ignore[attr-defined]
        AnthropicConfig.map_openai_params = _patched_map_openai_params


def _setup(agent_model: str) -> dict[str, Any]:
    """One-time setup: makes the demo importable, applies the litellm
    compat shim, and swaps all 8 of the demo's own agents (6 main agents +
    2 input-guardrail agents in `airline/guardrails.py`) to run on Claude via
    litellm. Returns a `{agent_name: Agent}` map of the 6 main agents, for
    routing turns to whichever agent a handoff left the conversation with.
    """
    _ensure_demo_importable()
    _patch_litellm_thinking_bug()

    from agents.extensions.models.litellm_model import LitellmModel
    from airline.agents import (
        booking_cancellation_agent,
        faq_agent,
        flight_information_agent,
        refunds_compensation_agent,
        seat_special_services_agent,
        triage_agent,
    )
    from airline.guardrails import guardrail_agent, jailbreak_guardrail_agent

    main_agents = [
        triage_agent,
        faq_agent,
        seat_special_services_agent,
        flight_information_agent,
        booking_cancellation_agent,
        refunds_compensation_agent,
    ]
    for agent in [*main_agents, guardrail_agent, jailbreak_guardrail_agent]:
        agent.model = LitellmModel(model=f"anthropic/{agent_model}")

    return {agent.name: agent for agent in main_agents}


def _parse_tool_arguments(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw
    return raw


def _assistant_text_from_message_item(item: dict) -> str:
    parts = item.get("content") or []
    if isinstance(parts, str):
        return parts
    return "".join(part.get("text", "") for part in parts if isinstance(part, dict))


def _input_items_to_dicts(input_items: list[Any]) -> list[dict[str, Any]]:
    """Converts `RunResult.to_input_list()`'s flat item list into the
    `{"role", "content", "tool_calls", "tool_call_id"}` shape
    `execution/transcript.py`'s `TranscriptMessage` expects (see module
    docstring part 3). Unrecognized item shapes (e.g. reasoning items, which
    carry no user-facing content relevant to scoring) are skipped."""
    out: list[dict[str, Any]] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "function_call":
            out.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id"),
                            "function": item.get("name"),
                            "arguments": _parse_tool_arguments(item.get("arguments")),
                        }
                    ],
                    "tool_call_id": None,
                }
            )
        elif item_type == "function_call_output":
            out.append(
                {
                    "role": "tool",
                    "content": str(item.get("output") or ""),
                    "tool_calls": [],
                    "tool_call_id": item.get("call_id"),
                }
            )
        elif item_type == "message":
            out.append(
                {
                    "role": item.get("role", "assistant"),
                    "content": _assistant_text_from_message_item(item),
                    "tool_calls": [],
                    "tool_call_id": None,
                }
            )
        elif "role" in item and "type" not in item:
            # Plain user/assistant turns as fed into `Runner.run`/returned by
            # `to_input_list` (e.g. `{"role": "user", "content": "..."}`).
            out.append(
                {
                    "role": item["role"],
                    "content": item.get("content") or "",
                    "tool_calls": [],
                    "tool_call_id": None,
                }
            )
        # else: skip (e.g. reasoning items) -- no scoring-relevant content.
    return out


def make_openai_cs_target_agent(
    cost_tracker: CostTracker,
    agent_model: str,
    max_tool_turns: int = DEFAULT_MAX_INTERNAL_TOOL_TURNS,
):
    """Builds a Petri target agent that drives the openai-cs-agents-demo
    airline multi-agent system, with the auditor standing in for a human
    chat user (see module docstring for the Claude-via-litellm backend
    substitution this performs, and why).

    Args:
        cost_tracker: Shared tracker the demo's own LLM spend (main agents
            and their input guardrails) is recorded into.
        agent_model: Model id the demo's agents are swapped to run on via
            litellm. Should be a cheap, capable model -- never the judge tier.
        max_tool_turns: Passed through to `Runner.run(..., max_turns=...)`.
    """
    name_to_agent = _setup(agent_model)

    from agents import InputGuardrailTripwireTriggered, Runner
    from agents.exceptions import MaxTurnsExceeded, ModelBehaviorError
    from airline.context import AirlineAgentChatContext, create_initial_context
    from chatkit.types import ThreadMetadata
    from memory_store import MemoryStore

    triage_agent_name = next(iter(name_to_agent))

    async def openai_cs_target_agent(state: AgentState, context: TargetContext) -> None:
        chat_store = MemoryStore()
        thread = ThreadMetadata(id=uuid.uuid4().hex, created_at=datetime.now())
        chat_context = AirlineAgentChatContext(
            thread=thread,
            store=chat_store,
            request_context={},
            state=create_initial_context(),
        )
        input_items: list[Any] = []
        current_agent = name_to_agent[triage_agent_name]

        def sync_store() -> None:
            store().set(
                OPENAI_CS_TARGET_MESSAGES_STORE_KEY, _input_items_to_dicts(input_items)
            )

        def record_usage(usage: Any) -> None:
            if not usage.requests:
                return
            cache_write = usage.input_tokens_details.cache_write_tokens or 0
            cache_read = usage.input_tokens_details.cached_tokens or 0
            usd = anthropic_cost_usd(
                agent_model,
                usage.input_tokens,
                usage.output_tokens,
                cache_write_tokens=cache_write,
                cache_read_tokens=cache_read,
            )
            cost_tracker.record(
                stage="execution",
                model=agent_model,
                usd=usd,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )

        try:
            await context.wait_for_resume()
            user_message = await context.user_message()
            user_text = user_message.text

            while True:
                input_items.append({"role": "user", "content": user_text})

                try:
                    for attempt in range(DEFAULT_MAX_MODEL_BEHAVIOR_RETRIES + 1):
                        try:
                            result = await Runner.run(
                                current_agent,
                                input_items,
                                context=chat_context,
                                max_turns=max_tool_turns,
                            )
                            break
                        except ModelBehaviorError:
                            # Occasional malformed-JSON output from the
                            # guardrail agents' structured output via
                            # litellm/Claude (see module docstring) -- a
                            # reliability gap of the substituted backend, not
                            # a behavior difference in the demo's own
                            # tool-calling/handoff logic (already verified
                            # sound). Retried a bounded number of times with
                            # the same input; re-raised if it keeps failing.
                            if attempt == DEFAULT_MAX_MODEL_BEHAVIOR_RETRIES:
                                raise
                except InputGuardrailTripwireTriggered:
                    # Same canned refusal `server.py`'s `respond()` sends on
                    # a tripped input guardrail -- the demo's own behavior,
                    # not an approximation of it.
                    reply_text = "Sorry, I can only answer questions related to airline travel."
                    input_items.append({"role": "assistant", "content": reply_text})
                except MaxTurnsExceeded:
                    reply_text = (
                        "Sorry, I'm having trouble completing this request right now."
                    )
                    input_items.append({"role": "assistant", "content": reply_text})
                else:
                    input_items = result.to_input_list()
                    current_agent = name_to_agent.get(result.last_agent.name, current_agent)
                    record_usage(result.context_wrapper.usage)
                    reply_text = (
                        result.final_output if isinstance(result.final_output, str) else ""
                    )

                sync_store()
                state.output = ModelOutput.from_message(
                    ChatMessageAssistant(content=reply_text or "")
                )

                context.expect({TOOL_RESULT: set()})
                await context.send_output(state.output)

                next_user_message = await context.user_message()
                user_text = next_user_message.text
        except ExitSignal:
            return

    return openai_cs_target_agent
