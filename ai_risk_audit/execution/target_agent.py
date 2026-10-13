"""Custom Petri target agent for the mock Acme Co. support agent.

Petri's default target agent has the AUDITOR model fabricate every tool
call result, even for pre-registered tools — there's no built-in
deterministic execution backend. That's wrong for this pipeline: the
planted weaknesses (`execution/fake_target.py`) need a ground truth the
auditor cannot tamper with, so scoring stays verifiable.

This agent speaks Petri's low-level `TargetContext` handshake directly
(matching `inspect_petri.target._agent.target_agent`'s protocol) but, after
receiving the auditor's system/user messages, runs its OWN internal
tool-execution loop against `fake_target.execute_tool` instead of ever
staging a `TOOL_RESULT` slot for the auditor to fill. The auditor still
observes the full tool-call trace (it's appended to `state.messages`, which
becomes part of the stored transcript) — it just cannot influence what the
tools return.

Two things this agent must do to behave like a well-formed Petri target
(confirmed against `inspect_petri.target._agent.target_agent`, the
reference implementation):
- Catch `ExitSignal`, raised through `wait_for_resume()` once the auditor
  ends the conversation. Letting it propagate crashes the whole eval run.
- Never rely on `EvalSample.messages` to recover this target's own
  conversation afterward: that list reflects the AUDITOR's (the sample's
  top-level agent) message history, not this sub-agent's. Instead, this
  agent mirrors its own `state.messages` into the Inspect sample `store()`
  on every turn, and `execution/run.py` reads it back from
  `EvalSample.store` after the run.
"""

from __future__ import annotations

from typing import Any

from inspect_ai.agent import AgentState
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    Model,
    get_model,
)
from inspect_ai.tool import ToolInfo, ToolParams
from inspect_ai.util import store
from inspect_petri.target import TOOL_RESULT, ExitSignal, TargetContext

from ai_risk_audit.execution import fake_target

TARGET_MESSAGES_STORE_KEY = "ai_risk_audit:target_messages"
"""Store key this agent writes its own conversation under, and that
`execution/run.py` reads back from `EvalSample.store` after the run."""

_MAX_INTERNAL_TOOL_TURNS = 4
"""Safety valve: caps the target's own tool-call loop per turn so a
confused brain model can't spin forever before replying to the auditor."""

_TOOL_INFOS = [
    ToolInfo(
        name=t["name"],
        description=t["description"],
        parameters=ToolParams.model_validate(t["parameters"]),
    )
    for t in fake_target.TOOLS
]


async def acme_target_agent(state: AgentState, context: TargetContext) -> None:
    model = get_model(role="target")

    try:
        await context.wait_for_resume()
        user_message = await context.user_message()
        state.messages[:] = [
            ChatMessageSystem(content=fake_target.SYSTEM_PROMPT),
            user_message,
        ]
        _sync_store(state)

        while True:
            await _generate_final_reply(state, model)
            _sync_store(state)

            # Never pend on a tool result: our own loop above already
            # resolved every tool call this target made, so the auditor
            # has nothing to supply but the next user message.
            context.expect({TOOL_RESULT: set()})
            await context.send_output(state.output)

            next_user_message = await context.user_message()
            state.messages.append(next_user_message)
            _sync_store(state)
    except ExitSignal:
        return


def _message_to_dict(m: ChatMessage) -> dict[str, Any]:
    tool_calls = [
        {"id": tc.id, "function": tc.function, "arguments": tc.arguments}
        for tc in (getattr(m, "tool_calls", None) or [])
    ]
    return {
        "role": m.role,
        "content": m.text,
        "tool_calls": tool_calls,
        "tool_call_id": getattr(m, "tool_call_id", None),
    }


def _sync_store(state: AgentState) -> None:
    """Mirrors this target's own conversation into the sample store so it
    survives after the run, independent of the auditor's own message
    history (see module docstring)."""
    store().set(TARGET_MESSAGES_STORE_KEY, [_message_to_dict(m) for m in state.messages])


async def _generate_final_reply(state: AgentState, model: Model) -> None:
    """Advance `state` to a reply with no pending tool calls.

    Runs the target's internal brain-model generate/execute loop against
    `fake_target.execute_tool`, appending each `ChatMessageAssistant` and
    resulting `ChatMessageTool` to `state.messages` so the full trace is
    visible to the auditor and preserved in the stored transcript.
    """
    for _ in range(_MAX_INTERNAL_TOOL_TURNS):
        state.output = await model.generate(input=state.messages, tools=_TOOL_INFOS)
        state.messages.append(state.output.message)

        tool_calls = state.output.message.tool_calls
        if not tool_calls:
            return

        for call in tool_calls:
            result = fake_target.execute_tool(call.function, call.arguments)
            state.messages.append(
                ChatMessageTool(
                    content=result.content,
                    tool_call_id=call.id,
                    function=call.function,
                )
            )

    # Safety valve: no tools offered, so this reply cannot have tool calls.
    state.output = await model.generate(input=state.messages, tools=[])
    state.messages.append(state.output.message)
