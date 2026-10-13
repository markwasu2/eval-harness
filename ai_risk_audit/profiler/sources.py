"""Loading raw source material for the profiler.

These helpers never call a model — they only gather text. `source_from_url`
is the one function here that touches the network (a plain GET to fetch
public docs), and does not make a model call.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from ai_risk_audit.profiler.types import SourceDoc


def source_from_text(name: str, text: str) -> SourceDoc:
    return SourceDoc(name=name, kind="text", text=text)


def source_from_file(path: str | Path) -> SourceDoc:
    path = Path(path)
    return SourceDoc(name=path.name, kind="file", text=path.read_text())


def source_from_url(url: str, timeout_s: float = 30.0) -> SourceDoc:
    response = httpx.get(url, timeout=timeout_s, follow_redirects=True)
    response.raise_for_status()
    return SourceDoc(name=url, kind="url", text=response.text)


def save_sources(sources: list[SourceDoc], out_dir: str | Path) -> Path:
    """Persist raw sources alongside the extracted profile, unmodified."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "sources.json"
    path.write_text(json.dumps([s.model_dump() for s in sources], indent=2))
    return path
