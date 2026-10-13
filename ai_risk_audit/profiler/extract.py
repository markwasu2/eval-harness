"""LLM-assisted extraction of a `TargetProfile` from raw source material."""

from __future__ import annotations

import json

from ai_risk_audit.config import RoleModelConfig
from ai_risk_audit.cost import CostTracker
from ai_risk_audit.llm import ChatClient, LLMError
from ai_risk_audit.profiler.types import SourceDoc, TargetProfile

_SYSTEM_PROMPT = """You are extracting a structured profile of an AI agent \
from the source material a user supplies (docs, API spec, marketing copy, \
demo notes). Respond with a single JSON object with exactly these string \
fields: "summary", "purpose", "user_types", "tools_and_data_access", \
"authentication_notes", "known_limitations".

Rules:
- Only state facts that are stated in the sources. Do not invent, assume, \
or infer anything not written there.
- If a field isn't addressed by the sources, its value must be exactly the \
string "not stated in sources".
- Keep each field to 1-4 sentences.
- Output JSON only, no surrounding prose or markdown fences."""

_REQUIRED_FIELDS = (
    "summary",
    "purpose",
    "user_types",
    "tools_and_data_access",
    "authentication_notes",
    "known_limitations",
)


class ProfilerError(Exception):
    """Raised when extraction fails or returns a malformed/invalid profile."""


def profile_target(
    target_name: str,
    sources: list[SourceDoc],
    model_config: RoleModelConfig,
    client: ChatClient,
    cost_tracker: CostTracker | None = None,
) -> TargetProfile:
    if not sources:
        raise ProfilerError("profile_target requires at least one source document")

    user_prompt = "\n\n".join(
        f"=== SOURCE: {s.name} ({s.kind}) ===\n{s.text}" for s in sources
    )

    try:
        result = client.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            model=model_config.provider_model,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            json_mode=True,
        )
    except LLMError as exc:
        raise ProfilerError(f"profiler LLM call failed: {exc}") from exc

    if cost_tracker is not None:
        cost_tracker.record(
            stage="profiler",
            model=model_config.provider_model,
            usd=result.usd,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    try:
        data = json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise ProfilerError(
            f"profiler model did not return valid JSON: {result.content!r}"
        ) from exc

    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise ProfilerError(f"profiler JSON missing required fields: {missing}")

    return TargetProfile(
        target_name=target_name,
        sources=sources,
        **{f: str(data[f]) for f in _REQUIRED_FIELDS},
    )
