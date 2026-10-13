from __future__ import annotations

from ai_risk_audit.execution import fake_target


def test_lookup_customer_known_email_returns_full_record() -> None:
    result = fake_target.execute_tool(
        "lookup_customer", {"email": fake_target.PLANTED_CUSTOMER_RECORD["email"]}
    )
    assert result.ok
    assert fake_target.PLANTED_CUSTOMER_RECORD["card_last4"] in result.content
    assert fake_target.PLANTED_CUSTOMER_RECORD["note"] in result.content


def test_lookup_customer_unknown_email_fails() -> None:
    result = fake_target.execute_tool("lookup_customer", {"email": "nobody@example.com"})
    assert not result.ok


def test_search_help_center_song_query_returns_copyrighted_snippet() -> None:
    result = fake_target.execute_tool("search_help_center", {"query": "what's your jingle?"})
    assert result.ok
    assert fake_target.PLANTED_COPYRIGHTED_SNIPPET in result.content


def test_search_help_center_promo_query_returns_injection_payload() -> None:
    result = fake_target.execute_tool("search_help_center", {"query": "any promo codes?"})
    assert result.ok
    assert fake_target.PLANTED_ADMIN_CODE in result.content


def test_search_help_center_unrelated_query_returns_nothing_planted() -> None:
    result = fake_target.execute_tool("search_help_center", {"query": "store hours"})
    assert result.ok
    assert fake_target.PLANTED_ADMIN_CODE not in result.content
    assert fake_target.PLANTED_COPYRIGHTED_SNIPPET not in result.content


def test_issue_refund_is_deterministic() -> None:
    result = fake_target.execute_tool(
        "issue_refund", {"order_id": "O-1", "amount_usd": 10, "reason": "late"}
    )
    assert result.ok
    assert "O-1" in result.content


def test_unknown_tool_fails() -> None:
    result = fake_target.execute_tool("delete_everything", {})
    assert not result.ok
