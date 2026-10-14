"""Offline tests for the tau2-bench airline adapter.

Exercises `run_agent_turn` and `_tau_message_to_dict` directly against stub
`agent`/`env` objects, per the module's own design (see
`ai_risk_audit/execution/tau_target_agent.py` docstring part 1) — no real
`LLMAgent`, no `Environment`, no network/LLM calls anywhere in this file.
"""

from __future__ import annotations

import pytest
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from ai_risk_audit.execution.tau_target_agent import (
    TauAdapterError,
    _tau_message_to_dict,
    run_agent_turn,
)


class FakeAgentState:
    """Stands in for `LLMAgentState`; `run_agent_turn` only ever threads it
    through unchanged, so a bare sentinel object is enough."""


class FakeAgent:
    """Stub agent whose `generate_next_message` returns canned replies in
    order, mirroring `LLMAgent.generate_next_message`'s (message, state) ->
    (AssistantMessage, state) signature."""

    def __init__(self, replies: list[AssistantMessage]) -> None:
        self._replies = list(replies)
        self.received_messages: list = []

    def generate_next_message(self, message, state):
        self.received_messages.append(message)
        return self._replies.pop(0), state


class FakeEnv:
    """Stub environment keyed by tool name, mirroring
    `Environment.get_response`/`sync_tools`."""

    def __init__(self, responses: dict[str, ToolMessage]) -> None:
        self._responses = responses
        self.sync_calls = 0

    def get_response(self, tool_call: ToolCall) -> ToolMessage:
        return self._responses[tool_call.name]

    def sync_tools(self) -> None:
        self.sync_calls += 1


def _text_reply(content: str, cost: float = 0.01) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=content,
        cost=cost,
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


def _tool_call_reply(tool_calls: list[ToolCall], cost: float = 0.02) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=tool_calls,
        cost=cost,
        usage={"prompt_tokens": 20, "completion_tokens": 8},
    )


def test_run_agent_turn_no_tool_calls_returns_immediately() -> None:
    reply = _text_reply("how can I help?")
    agent = FakeAgent([reply])
    env = FakeEnv({})

    result = run_agent_turn(agent, env, "hi", FakeAgentState())

    assert result.reply is reply
    assert result.emitted == [reply]
    assert result.usd_cost == pytest.approx(0.01)
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert isinstance(agent.received_messages[0], UserMessage)
    assert agent.received_messages[0].content == "hi"


def test_run_agent_turn_single_tool_call_feeds_result_back() -> None:
    call = ToolCall(id="c1", name="search_flights", arguments={"origin": "SFO"})
    tool_call_msg = _tool_call_reply([call])
    tool_result = ToolMessage(id="c1", role="tool", content="[]")
    final_reply = _text_reply("no flights found")

    agent = FakeAgent([tool_call_msg, final_reply])
    env = FakeEnv({"search_flights": tool_result})

    result = run_agent_turn(agent, env, "find me a flight", FakeAgentState())

    assert result.reply is final_reply
    assert result.emitted == [tool_call_msg, tool_result, final_reply]
    # Costs from both assistant turns accumulate.
    assert result.usd_cost == pytest.approx(0.02 + 0.01)
    assert env.sync_calls == 1
    # Second generate_next_message call was fed the raw ToolMessage, not
    # wrapped in a MultiToolMessage, since there was only one tool call.
    assert agent.received_messages[1] is tool_result


def test_run_agent_turn_multiple_tool_calls_wrapped_in_multi_tool_message() -> None:
    calls = [
        ToolCall(id="c1", name="a", arguments={}),
        ToolCall(id="c2", name="b", arguments={}),
    ]
    tool_call_msg = _tool_call_reply(calls)
    result_a = ToolMessage(id="c1", role="tool", content="ok-a")
    result_b = ToolMessage(id="c2", role="tool", content="ok-b")
    final_reply = _text_reply("done")

    agent = FakeAgent([tool_call_msg, final_reply])
    env = FakeEnv({"a": result_a, "b": result_b})

    run_agent_turn(agent, env, "do two things", FakeAgentState())

    fed_back = agent.received_messages[1]
    assert isinstance(fed_back, MultiToolMessage)
    assert fed_back.tool_messages == [result_a, result_b]


def test_run_agent_turn_exceeds_max_tool_turns_raises() -> None:
    call = ToolCall(id="c1", name="loop", arguments={})
    replies = [_tool_call_reply([call]) for _ in range(3)]
    agent = FakeAgent(replies)
    env = FakeEnv({"loop": ToolMessage(id="c1", role="tool", content="again")})

    with pytest.raises(TauAdapterError):
        run_agent_turn(agent, env, "go", FakeAgentState(), max_tool_turns=3)


def test_tau_message_to_dict_assistant_maps_tool_call_name_to_function_key() -> None:
    call = ToolCall(id="c1", name="search_flights", arguments={"origin": "SFO"})
    msg = AssistantMessage(role="assistant", content="looking it up", tool_calls=[call])

    d = _tau_message_to_dict(msg)

    assert d == {
        "role": "assistant",
        "content": "looking it up",
        "tool_calls": [{"id": "c1", "function": "search_flights", "arguments": {"origin": "SFO"}}],
        "tool_call_id": None,
    }


def test_tau_message_to_dict_tool_message() -> None:
    msg = ToolMessage(id="c1", role="tool", content="[]")

    d = _tau_message_to_dict(msg)

    assert d == {"role": "tool", "content": "[]", "tool_calls": [], "tool_call_id": "c1"}
