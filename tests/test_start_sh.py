"""Contract tests for start.sh (spec B1.6).

Static checks read start.sh as text. Runtime checks copy start.sh into a
sandbox whose .venv/bin/python is an argv logging stub, and whose PATH puts
stub lsof/curl/npm/node/ffmpeg/streamlit first, so the real script logic
runs without touching real ports, real processes or the real venv.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"

REMOVED_MSG = "Streamlit UI was removed"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="bash script")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def code_lines() -> list[str]:
    """start.sh lines with full line comments and blanks removed."""
    out = []
    for line in START.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(line)
    return out


def function_body(name: str) -> str:
    text = START.read_text()
    m = re.search(rf"^{re.escape(name)}\(\)\s*\{{\n(.*?)^\}}", text, re.S | re.M)
    assert m, f"function {name}() not found in start.sh"
    return m.group(1)


PY_STUB = r"""#!/bin/sh
# Stand in for .venv/bin/python: log argv, never run anything real.
printf '%s\n' "$*" >> "$STUB_LOG"
case " $* " in
  *" --version"*) echo "Python 3.12.0"; exit 0 ;;
esac
case " $* " in
  *" -c "*) echo "3 12"; exit 0 ;;
esac
case " $* " in
  *uvicorn*)
    echo $$ > "$STUB_PID_FILE"
    if [ "${STUB_UVICORN_SLEEP:-0}" != "0" ]; then
      exec sleep "$STUB_UVICORN_SLEEP"
    fi
    ;;
esac
exit 0
"""

GENERIC_STUB = r"""#!/bin/sh
printf '%s %s\n' "$(basename "$0")" "$*" >> "$STUB_LOG"
exit 0
"""

STUBBED_TOOLS = ("lsof", "curl", "npm", "node", "ffmpeg", "streamlit", "open", "xdg-open", "wslview")


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def sandbox(tmp_path: Path):
    """A copy of start.sh in a throwaway ROOT with stub tools around it."""
    root = tmp_path / "root"
    root.mkdir()
    shutil.copy2(START, root / "start.sh")
    (root / "start.sh").chmod(0o755)
    (root / "scripts").mkdir()
    (root / "frontend").mkdir()

    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _write_exec(venv_bin / "python", PY_STUB)

    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for tool in STUBBED_TOOLS:
        _write_exec(stubs / tool, GENERIC_STUB)

    log = tmp_path / "argv.log"
    pid_file = tmp_path / "uvicorn.pid"
    bash_dir = str(Path(shutil.which("bash") or "/bin/bash").parent)
    env = {
        "PATH": os.pathsep.join([str(stubs), bash_dir, "/usr/bin", "/bin"]),
        "HOME": str(tmp_path),
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        # Legacy resolver reads this; the new script must resolve $ROOT/.venv on its own.
        "VIRTUAL_ENV": str(root / ".venv"),
        "STUB_LOG": str(log),
        "STUB_PID_FILE": str(pid_file),
        "STUB_UVICORN_SLEEP": "0",
    }
    return {"root": root, "script": root / "start.sh", "log": log, "pid_file": pid_file, "env": env}


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_script(sb: dict, *args: str, timeout: float = 60, **env_over: str) -> tuple[int, str]:
    env = dict(sb["env"], **env_over)
    proc = subprocess.Popen(
        [str(sb["script"]), *args],
        cwd=str(sb["root"]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        out, _ = proc.communicate()
        pytest.fail(f"start.sh {' '.join(args)} did not exit within {timeout}s\n{out}")
    return proc.returncode, out


def uvicorn_argv(sb: dict) -> list[str]:
    log = sb["log"]
    assert log.exists(), "the python stub was never invoked"
    lines = [ln for ln in log.read_text().splitlines() if "uvicorn" in ln]
    assert len(lines) == 1, f"expected exactly one uvicorn launch, log was:\n{log.read_text()}"
    return lines[0].split()


def pid_alive(pid: int) -> bool:
    res = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = res.stdout.strip()
    return res.returncode == 0 and bool(state) and not state.startswith("Z")


# ---------------------------------------------------------------------------
# Static: syntax, interpreter, removed lines
# ---------------------------------------------------------------------------


def test_bash_syntax_ok():
    res = subprocess.run(["bash", "-n", str(START)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_py_resolves_to_project_venv():
    pat = re.compile(r'^\s*PY="?\$\{?ROOT\}?/\.venv/bin/python"?\s*(#.*)?$')
    hits = [ln for ln in code_lines() if pat.match(ln)]
    assert hits, "expected PY=$ROOT/.venv/bin/python in start.sh"


def test_no_yt_dlp_pip_install():
    pat = re.compile(r"pip\s+install\b.*yt-dlp")
    hits = [ln for ln in code_lines() if pat.search(ln)]
    assert not hits, f"pip install yt-dlp line still present: {hits}"


# ---------------------------------------------------------------------------
# Static: host, dev flag, kill_port, trap, readiness
# ---------------------------------------------------------------------------


def test_default_host_is_loopback():
    lines = code_lines()
    default = re.compile(r'^\s*[A-Za-z_]*HOST[A-Za-z_]*=["\']?127\.0\.0\.1["\']?\s*(#.*)?$')
    assert any(default.match(ln) for ln in lines), "no HOST variable defaulting to 127.0.0.1"
    wildcard = [ln for ln in lines if "0.0.0.0" in ln]
    assert not wildcard, f"wildcard bind still present: {wildcard}"
    flag = re.compile(r"^\s*(\S+\|)?--host(=\*)?[)|]")
    assert any(flag.match(ln) for ln in lines), "--host is not parsed as a flag"
    uvicorn_lines = [ln for ln in lines if "uvicorn" in ln and "--host" in ln]
    assert uvicorn_lines, "no uvicorn launch line passes --host"
    for ln in uvicorn_lines:
        assert re.search(r'--host\s+"?\$', ln), f"uvicorn --host is a literal, not the variable: {ln.strip()}"


def test_dev_flag_is_parsed():
    flag = re.compile(r"^\s*(\S+\|)?--dev[)|]")
    assert any(flag.match(ln) for ln in code_lines()), "--dev is not parsed as a flag"


def test_kill_port_listen_only_and_term_before_kill():
    body = function_body("kill_port")
    assert re.search(r'lsof\s+-ti\s+tcp:"?\$\{?port\}?"?\s+-sTCP:LISTEN', body), (
        "kill_port must select LISTEN sockets: lsof -ti tcp:$port -sTCP:LISTEN"
    )
    kill_calls = [
        m for m in re.finditer(r"\bkill\b(\s+-[A-Za-z0-9]+|\s+-s\s+\w+)?", body)
    ]
    assert kill_calls, "kill_port never calls kill"
    first_flag = (kill_calls[0].group(1) or "").strip()
    assert first_flag in ("", "-TERM", "-15", "-s TERM"), (
        f"first kill in kill_port must be TERM, got 'kill {first_flag}'"
    )
    hard = [m for m in kill_calls if re.fullmatch(r"-(9|KILL)|-s\s+KILL", (m.group(1) or "").strip())]
    assert hard, "kill_port has no KILL escalation after TERM"
    assert hard[0].start() > kill_calls[0].start()
    assert "sleep" in body[kill_calls[0].end(): hard[0].start()], "no wait between TERM and KILL"


def test_trap_kills_api_and_frontend_pids():
    traps = [ln for ln in code_lines() if re.match(r"^\s*trap\s", ln)]
    assert traps, "no trap line in start.sh"
    good = [
        ln for ln in traps
        if "$API_PID" in ln
        and ("$FE_PID" in ln or "$FRONTEND_PID" in ln)
        and all(sig in ln for sig in ("EXIT", "INT", "TERM"))
    ]
    assert good, f"trap does not kill API and Vite PIDs on EXIT INT TERM: {traps}"


def test_readiness_waits_on_health_live():
    lines = code_lines()
    assert any("/health/live" in ln for ln in lines), "start.sh never polls /health/live"
    fixed_sleeps = [ln for ln in lines if re.match(r"^\s*sleep\s+2\s*(#.*)?$", ln)]
    assert not fixed_sleeps, f"fixed sleep 2 still used for readiness: {fixed_sleeps}"


# ---------------------------------------------------------------------------
# Runtime: sandboxed executions of the real script
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["ui", "both"])
def test_removed_ui_modes_exit_2(sandbox, mode):
    rc, out = run_script(sandbox, mode)
    assert rc == 2, f"./start.sh {mode} exited {rc}, expected 2\n{out}"
    assert REMOVED_MSG in out, f"missing '{REMOVED_MSG}' in output:\n{out}"
    launched = sandbox["log"].read_text() if sandbox["log"].exists() else ""
    assert "uvicorn" not in launched and "streamlit" not in launched, (
        f"{mode} mode launched servers before refusing:\n{launched}"
    )


def test_api_mode_binds_loopback_without_reload(sandbox):
    rc, out = run_script(sandbox, "api", "--skip-setup", "--no-open")
    argv = uvicorn_argv(sandbox)
    assert "--host" in argv, f"uvicorn launched without --host: {argv}\n{out}"
    assert argv[argv.index("--host") + 1] == "127.0.0.1", f"uvicorn host is not loopback: {argv}"
    assert "--reload" not in argv, f"--reload passed without --dev: {argv}"


def test_dev_flag_enables_reload(sandbox):
    rc, out = run_script(sandbox, "api", "--dev", "--skip-setup", "--no-open")
    assert "Unknown argument" not in out, f"--dev rejected by the arg parser\n{out}"
    argv = uvicorn_argv(sandbox)
    assert "--reload" in argv, f"--dev did not enable --reload: {argv}"
    assert argv[argv.index("--host") + 1] == "127.0.0.1", f"--dev changed the bind host: {argv}"


def test_sigint_leaves_no_api_child(sandbox):
    env = dict(sandbox["env"], STUB_UVICORN_SLEEP="60")
    proc = subprocess.Popen(
        [str(sandbox["script"]), "api", "--skip-setup", "--no-open"],
        cwd=str(sandbox["root"]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    child_pid = None
    try:
        deadline = time.time() + 20
        while time.time() < deadline and not sandbox["pid_file"].exists():
            if proc.poll() is not None:
                out, _ = proc.communicate()
                pytest.fail(f"start.sh api exited early with {proc.returncode}\n{out}")
            time.sleep(0.1)
        assert sandbox["pid_file"].exists(), "uvicorn stub never started"
        child_pid = int(sandbox["pid_file"].read_text().strip())
        assert pid_alive(child_pid)
        time.sleep(1.5)  # let the script reach its wait
        os.kill(proc.pid, signal.SIGINT)
        try:
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            child_alive = pid_alive(child_pid)
            _kill_group(proc)
            out, _ = proc.communicate()
            pytest.fail(
                f"start.sh did not exit within 15s of SIGINT; API child pid {child_pid} "
                f"alive={child_alive}\n{out}"
            )
        deadline = time.time() + 8
        while time.time() < deadline and pid_alive(child_pid):
            time.sleep(0.2)
        assert not pid_alive(child_pid), (
            f"API child pid {child_pid} still running after SIGINT to start.sh\n{out}"
        )
    finally:
        _kill_group(proc)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
