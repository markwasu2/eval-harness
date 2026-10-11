from __future__ import annotations

from eval_harness.tools import TOOL_DEFINITION, execute_python, run_tool_call


def test_basic_execution() -> None:
    result = execute_python("print(1 + 1)")
    assert result.exit_code == 0
    assert result.stdout == "2\n"
    assert not result.timed_out


def test_network_access_is_blocked() -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    socket.socket()\n"
        "    print('NOT BLOCKED')\n"
        "except OSError as e:\n"
        "    print('blocked')\n"
    )
    result = execute_python(code)
    assert result.stdout.strip() == "blocked"


def test_subprocess_spawning_is_blocked() -> None:
    code = (
        "import subprocess\n"
        "try:\n"
        "    subprocess.run(['echo', 'hi'])\n"
        "    print('NOT BLOCKED')\n"
        "except OSError:\n"
        "    print('blocked')\n"
    )
    result = execute_python(code)
    assert result.stdout.strip() == "blocked"


def test_filesystem_access_outside_sandbox_is_blocked() -> None:
    result = execute_python("open('/etc/passwd').read()")
    assert result.exit_code != 0
    assert "PermissionError" in result.stderr


def test_filesystem_access_inside_sandbox_is_allowed() -> None:
    code = "open('scratch.txt', 'w').write('hi')\nprint(open('scratch.txt').read())"
    result = execute_python(code)
    assert result.exit_code == 0
    assert result.stdout.strip() == "hi"


def test_wall_clock_timeout_is_enforced() -> None:
    result = execute_python("import time; time.sleep(10)", timeout_s=0.5)
    assert result.timed_out
    assert result.exit_code == -1


def test_cpu_time_limit_is_enforced() -> None:
    code = "x = 0\nwhile True:\n    x += 1\n"
    result = execute_python(code, timeout_s=10.0, cpu_s=1)
    # either CPU limit kills it (SIGXCPU, negative exit code) or the wall
    # clock catches it first; both are acceptable bounded outcomes.
    assert result.exit_code != 0 or result.timed_out


def test_run_tool_call_dispatches_execute_python() -> None:
    result = run_tool_call(TOOL_DEFINITION.name, {"code": "print('ok')"})
    assert result.stdout.strip() == "ok"


def test_run_tool_call_rejects_unknown_tool() -> None:
    result = run_tool_call("not_a_real_tool", {})
    assert result.exit_code == -1
    assert "unknown tool" in result.stderr


def test_run_tool_call_rejects_missing_code_argument() -> None:
    result = run_tool_call(TOOL_DEFINITION.name, {})
    assert result.exit_code == -1
    assert "code" in result.stderr
