"""Offline tests for the openai-cs-agents-demo adapter's pure conversion
functions.

Exercises `_input_items_to_dicts`, `_parse_tool_arguments`, and
`_assistant_text_from_message_item` directly against dict fixtures shaped
exactly like `RunResult.to_input_list()`'s own item formats (confirmed via a
live test against the real SDK -- see module docstring part 3) -- no real
`Agent`, no `Runner.run`, no network/LLM calls anywhere in this file.
"""

from __future__ import annotations

from ai_risk_audit.execution.openai_cs_target_agent import (
    _assistant_text_from_message_item,
    _input_items_to_dicts,
    _parse_tool_arguments,
)


def test_parse_tool_arguments_decodes_json_string() -> None:
    assert _parse_tool_arguments('{"confirmation_number": "ABC123"}') == {
        "confirmation_number": "ABC123"
    }


def test_parse_tool_arguments_falls_back_to_raw_on_bad_json() -> None:
    assert _parse_tool_arguments("not json") == "not json"


def test_parse_tool_arguments_passes_through_non_string() -> None:
    assert _parse_tool_arguments({"already": "a dict"}) == {"already": "a dict"}


def test_assistant_text_from_message_item_joins_text_parts() -> None:
    item = {
        "content": [
            {"annotations": [], "text": "Sure, ", "type": "output_text", "logprobs": []},
            {"annotations": [], "text": "I can help.", "type": "output_text", "logprobs": []},
        ]
    }
    assert _assistant_text_from_message_item(item) == "Sure, I can help."


def test_input_items_to_dicts_plain_user_turn() -> None:
    items = [{"role": "user", "content": "I need to change my seat"}]

    assert _input_items_to_dicts(items) == [
        {
            "role": "user",
            "content": "I need to change my seat",
            "tool_calls": [],
            "tool_call_id": None,
        }
    ]


def test_input_items_to_dicts_function_call_and_output() -> None:
    items = [
        {
            "arguments": '{"confirmation_number": "ABC123"}',
            "call_id": "call_1",
            "name": "transfer_to_seat_and_special_services_agent",
            "type": "function_call",
            "id": "__fake_id__",
            "provider_data": {},
        },
        {
            "call_id": "call_1",
            "output": "Transferred.",
            "type": "function_call_output",
        },
    ]

    assert _input_items_to_dicts(items) == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": "transfer_to_seat_and_special_services_agent",
                    "arguments": {"confirmation_number": "ABC123"},
                }
            ],
            "tool_call_id": None,
        },
        {
            "role": "tool",
            "content": "Transferred.",
            "tool_calls": [],
            "tool_call_id": "call_1",
        },
    ]


def test_input_items_to_dicts_assistant_message_item() -> None:
    items = [
        {
            "id": "msg_1",
            "content": [
                {
                    "annotations": [],
                    "text": "Happy to help with that.",
                    "type": "output_text",
                    "logprobs": [],
                }
            ],
            "role": "assistant",
            "status": "completed",
            "type": "message",
            "provider_data": {},
        }
    ]

    assert _input_items_to_dicts(items) == [
        {
            "role": "assistant",
            "content": "Happy to help with that.",
            "tool_calls": [],
            "tool_call_id": None,
        }
    ]


def test_input_items_to_dicts_skips_unrecognized_item_types() -> None:
    items = [
        {"type": "reasoning", "id": "r1", "summary": []},
        {"role": "user", "content": "hi"},
    ]

    result = _input_items_to_dicts(items)

    assert len(result) == 1
    assert result[0]["role"] == "user"
