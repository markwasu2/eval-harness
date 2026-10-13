"""Data types for the target-profiler stage.

A `TargetProfile` is the structured output stage 1 hands to the planner. It
always carries its `sources` (the raw text it was extracted from) so every
downstream claim about the target can be traced back to something a human
can re-read.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class SourceDoc(BaseModel):
    """One piece of raw source material fed to the profiler."""

    model_config = ConfigDict(extra="forbid")

    name: str
    """Short label, e.g. a filename or URL, used in reports."""

    kind: str
    """Where this text came from: "text", "file", or "url"."""

    text: str


class TargetProfile(BaseModel):
    """Structured profile of what a target AI agent does.

    Every field other than `target_name` and `sources` is LLM-extracted from
    `sources` and should read "not stated in sources" (verbatim) rather than
    a guess when the sources don't say.
    """

    model_config = ConfigDict(extra="forbid")

    target_name: str
    summary: str
    purpose: str
    user_types: str
    tools_and_data_access: str
    authentication_notes: str
    known_limitations: str
    sources: list[SourceDoc]
