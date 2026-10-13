from __future__ import annotations

import pytest

from ai_risk_audit.llm import LLMError, MockChat


def test_mock_chat_returns_scripted_responses_in_order() -> None:
    client = MockChat(["first", "second"])
    r1 = client.complete(system="s", user="u", model="m")
    r2 = client.complete(system="s", user="u", model="m")
    assert r1.content == "first"
    assert r2.content == "second"
    assert r1.usd == 0.0
    assert len(client.calls) == 2


def test_mock_chat_raises_when_exhausted() -> None:
    client = MockChat(["only"])
    client.complete(system="s", user="u", model="m")
    with pytest.raises(LLMError):
        client.complete(system="s", user="u", model="m")
