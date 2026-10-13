from __future__ import annotations

import json

import pytest

from ai_risk_audit.config import RoleModelConfig
from ai_risk_audit.cost import CostTracker
from ai_risk_audit.llm import MockChat
from ai_risk_audit.profiler.extract import ProfilerError, profile_target
from ai_risk_audit.profiler.sources import source_from_text

_VALID_RESPONSE = json.dumps(
    {
        "summary": "A support agent.",
        "purpose": "Helps customers.",
        "user_types": "Retail customers.",
        "tools_and_data_access": "Customer lookup, help center search, refunds.",
        "authentication_notes": "not stated in sources",
        "known_limitations": "not stated in sources",
    }
)


def test_profile_target_success() -> None:
    sources = [source_from_text("docs", "Acme Assist helps customers 24/7.")]
    client = MockChat([_VALID_RESPONSE])
    tracker = CostTracker(budget_warn_usd=5.0, budget_abort_usd=10.0)

    profile = profile_target(
        "Acme Co", sources, RoleModelConfig(provider_model="mock/model"), client, tracker
    )

    assert profile.target_name == "Acme Co"
    assert profile.summary == "A support agent."
    assert profile.authentication_notes == "not stated in sources"
    assert profile.sources == sources
    assert tracker.total_usd == 0.0
    assert tracker.by_stage() == {"profiler": 0.0}


def test_profile_target_requires_sources() -> None:
    client = MockChat([_VALID_RESPONSE])
    with pytest.raises(ProfilerError):
        profile_target(
            "Acme Co", [], RoleModelConfig(provider_model="mock/model"), client
        )


def test_profile_target_rejects_invalid_json() -> None:
    sources = [source_from_text("docs", "some text")]
    client = MockChat(["not json"])
    with pytest.raises(ProfilerError):
        profile_target(
            "Acme Co", sources, RoleModelConfig(provider_model="mock/model"), client
        )


def test_profile_target_rejects_missing_fields() -> None:
    sources = [source_from_text("docs", "some text")]
    client = MockChat([json.dumps({"summary": "only summary"})])
    with pytest.raises(ProfilerError):
        profile_target(
            "Acme Co", sources, RoleModelConfig(provider_model="mock/model"), client
        )
