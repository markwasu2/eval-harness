"""The single offline tool exposed to every model via function calling.

`execute_python` runs a short Python snippet with no network access, no
filesystem access outside a fresh temp directory, and CPU/wall-clock/memory
limits. Every model in the panel, including closed ones, is given this same
sandbox through function calling; provider-native code-execution tools are
never used, so tool behavior is identical across providers.

Isolation model and its limits (documented honestly rather than overclaimed):
this is a language-level + POSIX-rlimit sandbox, not a container or VM. It
blocks socket creation, restricts `open()` to the sandbox directory, and
blocks `subprocess`/`os.system`/exec-family escapes from within the script,
and bounds CPU time, wall-clock time, and address space. It does not provide
kernel-level isolation (no seccomp/namespaces/chroot) and should not be
relied on to run untrusted code that is adversarial to the host OS itself;
for this benchmark's purpose (bounding what an evaluated model's generated
snippets can do) it is sufficient.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from eval_harness.providers.base import ToolDefinition

TOOL_NAME = "execute_python"

TOOL_DEFINITION = ToolDefinition(
    name=TOOL_NAME,
    description=(
        "Execute a short Python 3 snippet in an offline sandbox. No network access, "
        "no filesystem access outside the sandbox, and bounded CPU time, wall-clock "
        "time, and memory. Use this for calculations you don't trust yourself to do "
        "exactly in your head. Anything printed to stdout is returned to you."
    ),
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python source code to execute.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
)

_MAX_OUTPUT_CHARS = 20_000

_WRAPPER_PREAMBLE = textwrap.dedent(
    """
    import os as _os
    import socket as _socket
    import subprocess as _subprocess
    import builtins as _builtins

    class _SandboxViolation(OSError):
        pass

    def _blocked(*_args, **_kwargs):
        raise _SandboxViolation("blocked by the eval-harness sandbox")

    _socket.socket = _blocked
    _socket.create_connection = _blocked
    _socket.getaddrinfo = _blocked
    _socket.socketpair = _blocked

    _subprocess.Popen.__init__ = _blocked
    _subprocess.run = _blocked
    _subprocess.call = _blocked
    _subprocess.check_call = _blocked
    _subprocess.check_output = _blocked

    for _name in ("system", "popen", "fork", "forkpty", "execv", "execve", "execvp",
                  "execvpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
        if hasattr(_os, _name):
            setattr(_os, _name, _blocked)

    _real_open = _builtins.open
    _sandbox_dir = _os.path.realpath(_os.getcwd())

    def _restricted_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, _os.PathLike)):
            _resolved = _os.path.realpath(file)
            if not (_resolved == _sandbox_dir or _resolved.startswith(_sandbox_dir + _os.sep)):
                raise PermissionError(f"access outside the sandbox directory: {file!r}")
        return _real_open(file, *args, **kwargs)

    _builtins.open = _restricted_open
    del _name
    """
).strip()


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def execute_python(
    code: str,
    timeout_s: float = 5.0,
    memory_mb: int = 1024,
    cpu_s: int = 5,
) -> ToolResult:
    """Run `code` in the sandbox and capture stdout/stderr."""
    script = _WRAPPER_PREAMBLE + "\n\n# --- user code below ---\n" + code

    with tempfile.TemporaryDirectory(prefix="eval-harness-sandbox-") as sandbox_dir:
        script_path = Path(sandbox_dir) / "snippet.py"
        script_path.write_text(script)

        def _apply_limits() -> None:
            import resource

            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            # RLIMIT_AS (virtual address space) is not enforceable on macOS/XNU:
            # setrlimit raises even though getrlimit reports RLIM_INFINITY. Apply
            # it best-effort so memory is bounded on Linux, and fall back silently
            # to CPU-time + wall-clock enforcement only where the kernel refuses it.
            mem_bytes = memory_mb * 1024 * 1024
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        env = {"PATH": "/usr/bin:/bin", "HOME": sandbox_dir, "LANG": "C.UTF-8"}

        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=sandbox_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                preexec_fn=_apply_limits,
                start_new_session=True,
            )
            return ToolResult(
                stdout=proc.stdout[:_MAX_OUTPUT_CHARS],
                stderr=proc.stderr[:_MAX_OUTPUT_CHARS],
                exit_code=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                stdout=(exc.stdout or "")[:_MAX_OUTPUT_CHARS]
                if isinstance(exc.stdout, str)
                else "",
                stderr=(exc.stderr or "")[:_MAX_OUTPUT_CHARS]
                if isinstance(exc.stderr, str)
                else "",
                exit_code=-1,
                timed_out=True,
            )


def run_tool_call(name: str, arguments: dict[str, Any]) -> ToolResult:
    """Dispatch a model-requested tool call by name, never raising: a
    malformed call from the model becomes an error ToolResult instead of
    crashing the run."""
    if name != TOOL_NAME:
        return ToolResult(
            stdout="", stderr=f"unknown tool: {name!r}", exit_code=-1, timed_out=False
        )
    code = arguments.get("code")
    if not isinstance(code, str):
        return ToolResult(
            stdout="",
            stderr="tool call missing required string argument 'code'",
            exit_code=-1,
            timed_out=False,
        )
    return execute_python(code)
