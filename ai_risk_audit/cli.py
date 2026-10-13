"""Command-line interface.

    ai-risk-audit demo --out audits/acme-demo
    ai-risk-audit run --name "Acme Co" --source-text "..." --out audits/acme-demo
    ai-risk-audit run --config audit_config.yaml --name "Acme Co" \\
        --source-file docs.txt --out audits/acme-demo

Both commands currently audit the built-in mock Acme Co. support-agent
target (`execution/fake_target.py`) — that is the pipeline's definition-of-
done target. Auditing a different target requires swapping the `target=`
Agent passed to Petri in `execution/run.py`; see that module's docstring.
"""

from __future__ import annotations

import sys

import click

from ai_risk_audit.config import AuditConfigError, load_config
from ai_risk_audit.execution.fake_target import ACME_PUBLIC_DOCS
from ai_risk_audit.llm import AnthropicChat
from ai_risk_audit.pipeline import PipelineAbortedError, run_audit
from ai_risk_audit.profiler.sources import (
    source_from_file,
    source_from_text,
    source_from_url,
)


@click.group()
def main() -> None:
    """ai-risk-audit: an AI-company risk audit agent built on Petri."""


@main.command("demo")
@click.option("--config", "config_path", default=None, help="audit config YAML (optional)")
@click.option(
    "--out", "out_dir", default="audits/acme-demo", help="directory to write the audit into"
)
def demo_command(config_path: str | None, out_dir: str) -> None:
    """Run the full audit against the built-in mock Acme Co. target.

    Single-command definition-of-done path: no target URL, docs, or creds
    required — uses the built-in public marketing copy as the only source.
    """
    try:
        config = load_config(config_path)
    except AuditConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    sources = [source_from_text("acme_public_docs", ACME_PUBLIC_DOCS)]
    client = AnthropicChat()

    try:
        result_dir = run_audit("Acme Co", sources, config, client, out_dir)
    except PipelineAbortedError as exc:
        click.echo(f"audit aborted: {exc}", err=True)
        sys.exit(1)

    click.echo(f"wrote audit report to {result_dir}")


@main.command("run")
@click.option("--config", "config_path", default=None, help="audit config YAML (optional)")
@click.option("--name", "target_name", required=True, help="name of the target being audited")
@click.option("--source-text", "source_texts", multiple=True, help="raw source text (repeatable)")
@click.option(
    "--source-file", "source_files", multiple=True, help="path to a source file (repeatable)"
)
@click.option(
    "--source-url", "source_urls", multiple=True, help="URL to fetch as a source (repeatable)"
)
@click.option("--out", "out_dir", required=True, help="directory to write the audit into")
def run_command(
    config_path: str | None,
    target_name: str,
    source_texts: tuple[str, ...],
    source_files: tuple[str, ...],
    source_urls: tuple[str, ...],
    out_dir: str,
) -> None:
    """Run the full audit pipeline for a named target against the built-in
    mock Acme target, using the given profiler source material."""
    try:
        config = load_config(config_path)
    except AuditConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    sources = (
        [source_from_text(f"text-{i}", t) for i, t in enumerate(source_texts)]
        + [source_from_file(f) for f in source_files]
        + [source_from_url(u) for u in source_urls]
    )
    if not sources:
        raise click.ClickException(
            "at least one of --source-text, --source-file, --source-url is required"
        )

    client = AnthropicChat()

    try:
        result_dir = run_audit(target_name, sources, config, client, out_dir)
    except PipelineAbortedError as exc:
        click.echo(f"audit aborted: {exc}", err=True)
        sys.exit(1)

    click.echo(f"wrote audit report to {result_dir}")


if __name__ == "__main__":
    sys.exit(main())
