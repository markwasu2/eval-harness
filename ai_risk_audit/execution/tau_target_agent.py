"""Petri target agent that drives the tau2-bench airline agent.

tau2-bench (https://github.com/sierra-research/tau2-bench) is built around a
simulated LLM user (`tau2.user.user_simulator.UserSimulator`) that drives a
stock `tau2.agent.llm_agent.LLMAgent` through its `Orchestrator`. This module
replaces that simulated user with our Petri auditor while leaving the
airline environment, tools, and policy completely untouched: we never import
`Orchestrator` or `UserSimulator`, and never construct the airline
environment with anything other than `get_environment()`'s stock defaults
(no custom `db`, no monkeypatched policy).

Design, in three parts:
1. `_run_agent_turn` hand-rolls the same agent<->environment tool-execution
   cycle `tau2.orchestrator.orchestrator.Orchestrator.step()` uses for
   AGENT/ENV transitions (confirmed by reading that file): call
   `agent.generate_next_message`, and while the result is a tool call,
   execute every call via `environment.get_response` and feed the result(s)
   back to the agent, until it produces a genuine non-tool-call reply. It's
   plain, synchronous, and takes `agent`/`env` as parameters, so tests can
   stub both and exercise it with zero network calls.
2. `make_tau_airline_target_agent` wraps that loop in Petri's low-level
   `TargetContext` handshake (matching `inspect_petri.target._agent.target_agent`,
   confirmed against the reference implementation, and this pipeline's own
   `execution/target_agent.py` precedent): `wait_for_resume()` ->
   `user_message()` -> loop (turn -> `send_output` -> next `user_message()`),
   catching `ExitSignal` so the auditor ending the conversation doesn't crash
   the run. Only the airline agent's final, user-facing reply is ever sent
   to the auditor as a `ModelOutput`; the internal tool-call turns are never
   staged as a pending `TOOL_RESULT` for the auditor to fill (same reasoning
   as `execution/target_agent.py`: the auditor must not be able to fabricate
   tool results tau2's own environment is supposed to produce). The full
   tool-call trace is still mirrored into the sample store (see below), so
   `tool_call_abuse` scoring has real signal.
3. Because the airline agent's own LLM calls go through tau2-bench's own
   `tau2.utils.llm_utils.generate()` (a direct `litellm.completion()` call)
   rather than through `inspect_ai.model.get_model(role=...)`, Inspect's
   automatic `sample.role_usage` tracking never sees them. Each
   `AssistantMessage` tau2 returns already carries `.cost`/`.usage` (computed
   by litellm's own `completion_cost()`), so `make_tau_airline_target_agent`
   takes the shared `CostTracker` as a parameter and records that cost
   itself on every turn, giving the report one all-in spend number across
   the pipeline's own calls and the airline agent's.

Like `execution/target_agent.py`, this agent must never rely on
`EvalSample.messages` to recover its own conversation afterward (that list
reflects the AUDITOR's message history, not this target's): instead it
mirrors its own tau2 message list into the Inspect sample `store()` on every
turn under `TAU_TARGET_MESSAGES_STORE_KEY`, in the same
`{"role", "content", "tool_calls", "tool_call_id"}` dict shape
`execution/transcript.py`'s `TranscriptMessage` expects, so the existing
`execution/run.py` transcript/scoring path needs no changes to consume it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from inspect_ai.agent import AgentState
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.util import store
from inspect_petri.target import TOOL_RESULT, ExitSignal, TargetContext
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau2.domains.airline.environment import get_environment
from tau2.environment.environment import Environment

from ai_risk_audit.cost import CostTracker

TAU_TARGET_MESSAGES_STORE_KEY = "ai_risk_audit:tau_target_messages"
"""Store key this agent writes its own (tau2-side) conversation under, and
that `execution/run.py`'s transcript-building code reads back from
`EvalSample.store` after the run."""

DEFAULT_MAX_INTERNAL_TOOL_TURNS = 10
"""Safety valve: caps the airline agent's own tool-call loop per auditor
turn. If a real stock airline task needs more than this many chained tool
calls to answer one user turn, that's worth surfacing as an error rather
than silently truncating or looping forever."""


class TauAdapterError(Exception):
    """Raised when the tau2 airline agent can't be driven to a user-facing
    reply within the internal tool-call turn budget."""


@dataclass
class TauTurnResult:
    """Result of driving the airline agent through one auditor turn."""

    reply: AssistantMessage
    agent_state: LLMAgentState
    emitted: list[Message] = field(default_factory=list)
    """Every `AssistantMessage`/`ToolMessage` produced this turn, in order
    (never a `MultiToolMessage` — those get expanded to their constituent
    `ToolMessage`s for mirroring), including `reply` itself."""
    usd_cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _wrap_tool_results(tool_results: list[ToolMessage]) -> ToolMessage | MultiToolMessage:
    """Mirrors `Orchestrator._wrap_tool_results`: a single `ToolMessage` if
    there's exactly one, else a `MultiToolMessage`, for feeding back into
    `agent.generate_next_message`."""
    if len(tool_results) > 1:
        return MultiToolMessage(role="tool", tool_messages=tool_results)
    return tool_results[0]


def run_agent_turn(
    agent: LLMAgent,
    env: Environment,
    user_text: str,
    agent_state: LLMAgentState,
    max_tool_turns: int = DEFAULT_MAX_INTERNAL_TOOL_TURNS,
) -> TauTurnResult:
    """Drive `agent` from one auditor user-turn to its next user-facing
    reply, executing any tool calls against `env` along the way.

    Synchronous and side-effect-free beyond `agent`/`env` mutation, so it can
    be exercised directly in tests with stub `agent`/`env` objects — no
    network calls, no asyncio.
    """
    emitted: list[Message] = []
    usd_cost = 0.0
    prompt_tokens = 0
    completion_tokens = 0
    message: UserMessage | ToolMessage | MultiToolMessage = UserMessage(
        role="user", content=user_text
    )

    for _ in range(max_tool_turns):
        assistant_msg, agent_state = agent.generate_next_message(message, agent_state)
        emitted.append(assistant_msg)
        if assistant_msg.cost is not None:
            usd_cost += assistant_msg.cost
        if assistant_msg.usage is not None:
            prompt_tokens += assistant_msg.usage.get("prompt_tokens", 0)
            completion_tokens += assistant_msg.usage.get("completion_tokens", 0)

        if not assistant_msg.is_tool_call():
            return TauTurnResult(
                reply=assistant_msg,
                agent_state=agent_state,
                emitted=emitted,
                usd_cost=usd_cost,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        tool_results = [env.get_response(tc) for tc in assistant_msg.tool_calls]
        emitted.extend(tool_results)
        env.sync_tools()
        message = _wrap_tool_results(tool_results)

    raise TauAdapterError(
        f"airline agent exceeded {max_tool_turns} internal tool-call turns "
        "without producing a user-facing reply"
    )


def _tau_message_to_dict(m: Message) -> dict[str, Any]:
    """Converts one tau2 `AssistantMessage`/`ToolMessage` into the
    `{"role", "content", "tool_calls", "tool_call_id"}` shape
    `execution/transcript.py`'s `TranscriptMessage` expects — the same shape
    `execution/target_agent.py`'s `_message_to_dict` produces for inspect
    `ChatMessage`s, so downstream scoring code is unchanged.

    Tool call names map to `"function"` (not tau2's own `.name`) because
    `scoring/judge.py`'s `render_transcript` reads `tc.get("function")`.
    """
    if isinstance(m, AssistantMessage):
        tool_calls = [
            {"id": tc.id, "function": tc.name, "arguments": tc.arguments}
            for tc in (m.tool_calls or [])
        ]
        return {
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": tool_calls,
            "tool_call_id": None,
        }
    if isinstance(m, ToolMessage):
        return {
            "role": "tool",
            "content": m.content or "",
            "tool_calls": [],
            "tool_call_id": m.id,
        }
    raise TauAdapterError(f"cannot mirror unexpected tau2 message type: {type(m)}")


def make_tau_airline_target_agent(
    cost_tracker: CostTracker,
    agent_model: str,
    max_tool_turns: int = DEFAULT_MAX_INTERNAL_TOOL_TURNS,
):
    """Builds a Petri target agent that drives the stock tau2-bench airline
    `LLMAgent` against the stock airline `Environment`, with the auditor
    standing in for tau2's simulated user.

    Args:
        cost_tracker: Shared tracker the airline agent's own LLM spend is
            recorded into (see module docstring part 3).
        agent_model: Model id for the airline agent's own LLM calls. Should
            be a cheap, capable model — never the judge tier.
        max_tool_turns: Passed through to `run_agent_turn`.
    """

    async def tau_airline_target_agent(state: AgentState, context: TargetContext) -> None:
        env = get_environment()
        agent = LLMAgent(
            tools=env.get_tools(),
            domain_policy=env.get_policy(),
            llm=agent_model,
        )
        agent_state = agent.get_init_state()
        tau_messages: list[Message] = []

        def sync_store() -> None:
            store().set(
                TAU_TARGET_MESSAGES_STORE_KEY,
                [_tau_message_to_dict(m) for m in tau_messages],
            )

        try:
            await context.wait_for_resume()
            user_message = await context.user_message()
            user_text = user_message.text

            while True:
                result = await asyncio.to_thread(
                    run_agent_turn, agent, env, user_text, agent_state, max_tool_turns
                )
                agent_state = result.agent_state
                tau_messages.extend(result.emitted)
                sync_store()

                if result.usd_cost:
                    cost_tracker.record(
                        stage="execution",
                        model=agent_model,
                        usd=result.usd_cost,
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                    )

                state.output = ModelOutput.from_message(
                    ChatMessageAssistant(content=result.reply.content or "")
                )

                context.expect({TOOL_RESULT: set()})
                await context.send_output(state.output)

                next_user_message = await context.user_message()
                user_text = next_user_message.text
        except ExitSignal:
            return

    return tau_airline_target_agent
