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
  *" -c "*) echo "${STUB_PY_VERSION:-3 12}"; exit 0 ;;
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


def run_script(
    sb: dict, *args: str, timeout: float = 60, cwd: Path | None = None, **env_over: str
) -> tuple[int, str]:
    env = dict(sb["env"], **env_over)
    proc = subprocess.Popen(
        [str(sb["script"]), *args],
        cwd=str(cwd or sb["root"]),
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


def test_banner_credits_author_and_repo():
    body = function_body("banner")
    credit = [ln for ln in body.splitlines() if "github.com/Chunduri-Aditya/ai-remixmate" in ln]
    assert credit, f"banner() does not name the repo path:\n{body}"
    assert "Aditya Chunduri" in credit[0], f"credit line lacks the author: {credit[0].strip()}"
    # The credit is wider than the box, so it sits under it, not inside a border.
    assert "\u2551" not in credit[0], f"credit line breaks the box: {credit[0].strip()}"


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


# ---------------------------------------------------------------------------
# Ports, listener ownership, Python 3.12 gate, setup stamp (fix 3)
# ---------------------------------------------------------------------------

# lsof stand in. For `-ti tcp:PORT -sTCP:LISTEN` it prints $STUB_LSOF_PID when
# PORT is $STUB_LSOF_PORT and that pid is alive and not a zombie (the same rule
# as pid_alive). For `-a -p PID -d cwd -Fn` it prints the -Fn record with
# $STUB_LSOF_CWD, or nothing when that variable is empty.
LSOF_STUB = r"""#!/bin/sh
printf 'lsof %s\n' "$*" >> "$STUB_LOG"
mode=listen; port=""; pid=""; prev=""
for a in "$@"; do
  case "$a" in tcp:*) port="${a#tcp:}" ;; esac
  [ "$prev" = "-p" ] && pid="$a"
  [ "$a" = "cwd" ] && mode=cwd
  prev="$a"
done
if [ "$mode" = cwd ]; then
  if [ -n "$STUB_LSOF_CWD" ] && [ "$pid" = "$STUB_LSOF_PID" ]; then
    printf 'p%s\nfcwd\nn%s\n' "$pid" "$STUB_LSOF_CWD"
  fi
  exit 0
fi
if [ -n "$STUB_LSOF_PID" ] && [ "$port" = "$STUB_LSOF_PORT" ]; then
  st=$(ps -o stat= -p "$STUB_LSOF_PID" 2>/dev/null | tr -d ' ')
  case "$st" in
    ""|Z*) ;;
    *) echo "$STUB_LSOF_PID" ;;
  esac
fi
exit 0
"""

# A base interpreter on PATH (python3.12, python3, python). VERSION is what a
# `-c` probe prints; `-m venv DIR` writes the venv python stub into DIR.
BASE_PY_STUB = r"""#!/bin/sh
printf '%s %s\n' "$(basename "$0")" "$*" >> "$STUB_LOG"
case " $* " in
  *" --version"*) echo "Python {dotted}"; exit 0 ;;
esac
case " $* " in
  *" -c "*) echo "{version}"; exit 0 ;;
esac
if [ "$1" = "-m" ] && [ "$2" = "venv" ]; then
  mkdir -p "$3/bin"
  cp "{template}" "$3/bin/python"
  chmod +x "$3/bin/python"
fi
exit 0
"""

BROKEN_SHIM = "#!/bin/sh\nexit 127\n"

APP_ARGV = [sys.executable, "-c", "import time; time.sleep(60)", "scripts.api.main:app"]
FOREIGN_ARGV = ["sleep", "60"]


def _stubs_dir(sb: dict) -> Path:
    return sb["root"].parent / "stubs"


def _install_lsof_stub(sb: dict) -> None:
    _write_exec(_stubs_dir(sb) / "lsof", LSOF_STUB)


def _install_base_pythons(sb: dict, versions: dict[str, str | None]) -> None:
    """Drop .venv and put base interpreter stubs on PATH; None means a broken shim."""
    shutil.rmtree(sb["root"] / ".venv")
    template = _stubs_dir(sb) / "venv_python_template"
    template.write_text(PY_STUB)
    for name, version in versions.items():
        if version is None:
            _write_exec(_stubs_dir(sb) / name, BROKEN_SHIM)
            continue
        body = (
            BASE_PY_STUB.replace("{dotted}", version.replace(" ", ".") + ".0")
            .replace("{version}", version)
            .replace("{template}", str(template))
        )
        _write_exec(_stubs_dir(sb) / name, body)


def _log_text(sb: dict) -> str:
    return sb["log"].read_text() if sb["log"].exists() else ""


def _uvicorn_lines(sb: dict) -> list[str]:
    return [ln for ln in _log_text(sb).splitlines() if "uvicorn" in ln]


def _pip_lines(sb: dict) -> list[str]:
    return [ln for ln in _log_text(sb).splitlines() if "pip install" in ln]


def _reap(child: subprocess.Popen) -> None:
    try:
        os.kill(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait(timeout=10)


def test_api_port_follows_env(sandbox):
    rc, out = run_script(sandbox, "api", "--skip-setup", "--no-open", REMIXMATE_API_PORT="8765")
    argv = uvicorn_argv(sandbox)
    assert argv[argv.index("--port") + 1] == "8765", f"--port does not follow REMIXMATE_API_PORT: {argv}\n{out}"
    sandbox["log"].unlink()
    rc, out = run_script(sandbox, "api", "--skip-setup", "--no-open")
    argv = uvicorn_argv(sandbox)
    assert argv[argv.index("--port") + 1] == "8000", f"default API port is not 8000: {argv}\n{out}"


@pytest.mark.parametrize(
    "env",
    [
        {"REMIXMATE_API_PORT": "abc"},
        {"REMIXMATE_API_PORT": "0"},
        {"REMIXMATE_UI_PORT": "70000"},
        {"REMIXMATE_API_PORT": "8765", "REMIXMATE_UI_PORT": "8765"},
    ],
    ids=["api-abc", "api-zero", "ui-70000", "equal-ports"],
)
def test_invalid_port_env_is_rejected(sandbox, env):
    rc, out = run_script(sandbox, "api", "--skip-setup", "--no-open", **env)
    assert rc != 0, f"start.sh accepted {env}\n{out}"
    assert not _uvicorn_lines(sandbox), f"uvicorn launched with {env}:\n{_log_text(sandbox)}"


@pytest.mark.parametrize(
    "argv,cwd_kind",
    [
        (FOREIGN_ARGV, "repo"),
        (APP_ARGV, "other"),
        (APP_ARGV, "sibling"),
        (APP_ARGV, "empty"),
    ],
    ids=["foreign-command-repo-cwd", "app-command-other-cwd", "app-command-sibling-prefix-cwd", "app-command-empty-cwd"],
)
def test_foreign_listener_is_refused_and_survives(sandbox, argv, cwd_kind):
    _install_lsof_stub(sandbox)
    root_real = os.path.realpath(sandbox["root"])
    cwd = {
        "repo": root_real,
        "other": os.path.realpath(sandbox["root"].parent),
        "sibling": root_real + "-other",
        "empty": "",
    }[cwd_kind]
    child = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        rc, out = run_script(
            sandbox, "api", "--skip-setup", "--no-open",
            STUB_LSOF_PID=str(child.pid), STUB_LSOF_PORT="8000", STUB_LSOF_CWD=cwd,
        )
        assert rc != 0, f"start.sh started over a foreign listener\n{out}"
        assert str(child.pid) in out, f"output does not name pid {child.pid}\n{out}"
        assert "REMIXMATE_API_PORT" in out, f"output does not name REMIXMATE_API_PORT\n{out}"
        assert pid_alive(child.pid), f"foreign listener {child.pid} was signalled\n{out}"
        assert not _uvicorn_lines(sandbox), f"uvicorn launched over a foreign listener:\n{_log_text(sandbox)}"
    finally:
        _reap(child)


def test_own_listener_is_stopped(sandbox):
    _install_lsof_stub(sandbox)
    child = subprocess.Popen(APP_ARGV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        rc, out = run_script(
            sandbox, "api", "--skip-setup", "--no-open",
            STUB_LSOF_PID=str(child.pid), STUB_LSOF_PORT="8000",
            STUB_LSOF_CWD=os.path.realpath(sandbox["root"]),
        )
        # No rc check: when the stub API exits at once, the EXIT trap's kill on
        # dead pids fails under set -e and start.sh exits 1 (pre existing).
        assert any(
            f"-p {child.pid}" in ln and "cwd" in ln for ln in _log_text(sandbox).splitlines()
        ), f"ownership (cwd) was never checked before the signal:\n{_log_text(sandbox)}"
        assert not pid_alive(child.pid), f"own listener {child.pid} still alive\n{out}"
        assert len(_uvicorn_lines(sandbox)) == 1, f"expected one uvicorn launch:\n{_log_text(sandbox)}"
    finally:
        _reap(child)


def test_stop_leaves_foreign_listener(sandbox):
    _install_lsof_stub(sandbox)
    child = subprocess.Popen(FOREIGN_ARGV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        rc, out = run_script(
            sandbox, "stop",
            STUB_LSOF_PID=str(child.pid), STUB_LSOF_PORT="8000",
            STUB_LSOF_CWD=os.path.realpath(sandbox["root"]),
        )
        assert rc == 0, f"stop exited {rc}\n{out}"
        assert pid_alive(child.pid), f"stop signalled a foreign listener {child.pid}\n{out}"
        assert str(child.pid) in out, f"stop does not name who holds the port\n{out}"
    finally:
        _reap(child)


def test_existing_venv_not_312_fails(sandbox):
    rc, out = run_script(sandbox, "api", "--skip-setup", "--no-open", STUB_PY_VERSION="3 13")
    assert rc != 0, f"start.sh ran on a 3.13 .venv\n{out}"
    assert "3.12" in out, f"message does not name Python 3.12\n{out}"
    assert not _uvicorn_lines(sandbox), f"uvicorn launched on a 3.13 .venv:\n{_log_text(sandbox)}"


def test_no_python_312_fails(sandbox):
    _install_base_pythons(sandbox, {"python3.12": "3 13", "python3": "3 13", "python": "3 13"})
    rc, out = run_script(sandbox, "api", "--skip-setup", "--no-open")
    assert rc != 0, f"start.sh accepted a non 3.12 interpreter\n{out}"
    assert "3.12" in out, f"message does not name Python 3.12\n{out}"
    assert not (sandbox["root"] / ".venv").exists(), f".venv created without Python 3.12\n{out}"
    assert not _uvicorn_lines(sandbox)


def test_python3_reporting_312_is_accepted(sandbox):
    # python3.12 is a shim that does not run; python3 reports 3.12.
    _install_base_pythons(sandbox, {"python3.12": None, "python3": "3 12", "python": "3 13"})
    rc, out = run_script(sandbox, "api", "--skip-setup", "--no-open")
    assert "python3 -m venv" in _log_text(sandbox), f".venv not built with python3:\n{_log_text(sandbox)}\n{out}"
    assert len(_uvicorn_lines(sandbox)) == 1, f"run did not reach the uvicorn launch:\n{_log_text(sandbox)}\n{out}"


def test_setup_installs_requirements_dev(sandbox):
    rc, out = run_script(sandbox, "setup")
    assert rc == 0, f"setup exited {rc}\n{out}"
    assert any("pip install -r requirements-dev.txt" in ln for ln in _pip_lines(sandbox)), (
        f"setup does not install requirements-dev.txt:\n{_log_text(sandbox)}"
    )


def test_dev_extra_covers_requirements_dev():
    import tomllib

    def name(req: str) -> str:
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
        assert m, f"unparseable requirement: {req!r}"
        return re.sub(r"[-_.]+", "-", m.group(1)).lower()

    pinned = set()
    for line in (ROOT / "requirements-dev.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-r"):
            continue
        pinned.add(name(line))
    with (ROOT / "pyproject.toml").open("rb") as fh:
        dev = {name(r) for r in tomllib.load(fh)["project"]["optional-dependencies"]["dev"]}
    assert pinned, "no packages parsed from requirements-dev.txt"
    missing = sorted(pinned - dev)
    assert not missing, f"requirements-dev.txt pins packages the dev extra lacks: {missing}"


def test_run_installs_once_then_skips(sandbox):
    req = sandbox["root"] / "requirements.txt"
    req.write_text("numpy\n")
    rc, out = run_script(sandbox, "api", "--no-open")
    assert len(_uvicorn_lines(sandbox)) == 1, out
    assert _pip_lines(sandbox), f"first run did not install:\n{_log_text(sandbox)}"

    (sandbox["root"] / "frontend" / "node_modules").mkdir()
    sandbox["log"].unlink()
    rc, out = run_script(sandbox, "api", "--no-open")
    assert not _pip_lines(sandbox), f"second run installed again:\n{_log_text(sandbox)}"
    assert "Dependencies up to date" in out, out
    assert len(_uvicorn_lines(sandbox)) == 1

    req.write_text("numpy\nscipy\n")
    sandbox["log"].unlink()
    rc, out = run_script(sandbox, "api", "--no-open")
    assert len(_uvicorn_lines(sandbox)) == 1, out
    assert _pip_lines(sandbox), f"run after a requirements change did not install:\n{_log_text(sandbox)}"


def test_vite_config_reads_port_env():
    text = (ROOT / "frontend" / "vite.config.ts").read_text()
    assert "REMIXMATE_API_PORT" in text and "REMIXMATE_UI_PORT" in text, "vite.config.ts ignores the port env vars"
    assert "localhost:8000" not in text, "vite.config.ts still hardcodes the API port"


def test_port_env_vars_load_cleanly(monkeypatch):
    from scripts.core.config import load_config

    monkeypatch.setenv("REMIXMATE_API_PORT", "8765")
    monkeypatch.setenv("REMIXMATE_UI_PORT", "5199")
    cfg = load_config()
    assert cfg.api.port == 8765


# ---------------------------------------------------------------------------
# Scan mode and the UI port handed to the API
# ---------------------------------------------------------------------------

SCAN_PY_STUB = r"""#!/bin/sh
# Stand in for .venv/bin/python in scan mode: log cwd and each arg, exit 7.
{
  printf 'cwd=%s\n' "$(pwd -P)"
  for a in "$@"; do printf 'arg=%s\n' "$a"; done
} >> "$STUB_LOG"
exit 7
"""


def test_scan_mode_execs_scan_with_absolute_folder(sandbox, tmp_path):
    _write_exec(sandbox["root"] / ".venv" / "bin" / "python", SCAN_PY_STUB)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "my music").mkdir(parents=True)

    rc, out = run_script(sandbox, "scan", "my music", "--stems", "--analyze", cwd=elsewhere)

    assert rc == 7, f"scan did not pass the python exit code through, got {rc}\n{out}"
    lines = _log_text(sandbox).splitlines()
    assert lines == [
        f"cwd={sandbox['root'].resolve()}",
        "arg=-m",
        "arg=scripts.scan",
        f"arg={(elsewhere / 'my music').resolve()}",
        "arg=--stems",
        "arg=--analyze",
    ], f"unexpected scan invocation:\n{_log_text(sandbox)}\n{out}"
    assert "AI RemixMate" not in out, f"scan printed the start banner\n{out}"


def test_scan_mode_never_touches_ports(sandbox, tmp_path):
    _write_exec(sandbox["root"] / ".venv" / "bin" / "python", SCAN_PY_STUB)
    (tmp_path / "music").mkdir()

    rc, out = run_script(sandbox, "scan", str(tmp_path / "music"))

    assert rc == 7, f"scan exited {rc}\n{out}"
    lsof_calls = [ln for ln in _log_text(sandbox).splitlines() if ln.startswith("lsof")]
    assert lsof_calls == [], f"scan mode ran lsof: {lsof_calls}"


def test_scan_mode_without_venv_asks_for_setup(sandbox, tmp_path):
    shutil.rmtree(sandbox["root"] / ".venv")
    (tmp_path / "music").mkdir()

    rc, out = run_script(sandbox, "scan", str(tmp_path / "music"))

    assert rc == 1, f"scan without .venv exited {rc}\n{out}"
    assert "Run ./start.sh setup first" in out, out
    assert not (sandbox["root"] / ".venv").exists(), f"scan created a .venv\n{out}"


def test_usage_names_scan_mode():
    usage = [ln for ln in code_lines() if "Usage:" in ln]
    assert len(usage) == 2, f"expected two Usage strings, got {usage}"
    assert all("scan" in ln for ln in usage), f"a Usage string omits scan: {usage}"


def test_api_gets_ui_port_outside_the_config_prefix(sandbox):
    stub = PY_STUB.replace(
        "    echo $$ > \"$STUB_PID_FILE\"\n",
        "    echo $$ > \"$STUB_PID_FILE\"\n"
        "    printf 'env RMX_UI_PORT=%s REMIXMATE_UI_PORT=%s\\n' "
        "\"${RMX_UI_PORT-unset}\" \"${REMIXMATE_UI_PORT-unset}\" >> \"$STUB_LOG\"\n",
    )
    assert stub != PY_STUB, "stub patch point moved"
    _write_exec(sandbox["root"] / ".venv" / "bin" / "python", stub)

    rc, out = run_script(sandbox, "api", "--skip-setup", "--no-open", REMIXMATE_UI_PORT="5199")

    env_lines = [ln for ln in _log_text(sandbox).splitlines() if ln.startswith("env ")]
    assert env_lines == ["env RMX_UI_PORT=5199 REMIXMATE_UI_PORT=unset"], (
        f"API subshell env: {env_lines}\n{_log_text(sandbox)}\n{out}"
    )
