"""tests/test_deploy_scripts.py — static checks on app/deploy (spec B8.2).

Each test greps one fact the deploy runbook depends on. Documentation heredocs
(``cat <<EOF ... EOF``) are stripped before grepping so an instruction printed
for the operator never counts as a command the script runs. ``bash -n`` runs on
both scripts; shellcheck runs only when it is already on PATH.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "app" / "deploy"
BACKUP = DEPLOY / "backup.sh"
BOOTSTRAP = DEPLOY / "bootstrap.sh"
UNIT = DEPLOY / "remixmate.service"
CADDYFILE = DEPLOY / "Caddyfile"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_cat_heredocs(text: str) -> str:
    """Drop the bodies of ``cat <<EOF``/``cat <<'EOF'`` blocks (operator prose).

    Heredocs fed to ``bash``/``sudo`` are kept: those lines execute.
    """
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    opener = re.compile(r"^\s*cat\s+<<-?\s*['\"]?(\w+)['\"]?")
    while i < len(lines):
        m = opener.match(lines[i])
        if m:
            tag = m.group(1)
            i += 1
            while i < len(lines) and lines[i].strip() != tag:
                i += 1
            i += 1  # skip terminator
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _code_lines(text: str) -> list[str]:
    """Non-comment, non-blank lines with documentation heredocs removed."""
    stripped = _strip_cat_heredocs(text)
    return [
        ln for ln in stripped.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def _unit_values(text: str, key: str) -> list[str]:
    return [
        ln.split("=", 1)[1].strip()
        for ln in text.splitlines()
        if ln.strip().startswith(f"{key}=")
    ]


class SyntaxTests(unittest.TestCase):
    def test_bash_n_backup(self):
        proc = subprocess.run(["bash", "-n", str(BACKUP)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_bash_n_bootstrap(self):
        proc = subprocess.run(["bash", "-n", str(BOOTSTRAP)], capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)

    @unittest.skipUnless(shutil.which("shellcheck"), "shellcheck not installed (never install it here)")
    def test_shellcheck_both_scripts(self):
        proc = subprocess.run(
            ["shellcheck", "-S", "warning", str(BACKUP), str(BOOTSTRAP)],
            capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


class BackupScriptTests(unittest.TestCase):
    def setUp(self):
        self.text = _read(BACKUP)
        self.code = "\n".join(_code_lines(self.text))

    def test_executable_bit_on_disk(self):
        mode = BACKUP.stat().st_mode
        self.assertTrue(mode & stat.S_IXUSR, f"backup.sh mode {oct(mode)} lacks u+x; cron cannot run it")

    def test_git_index_mode_100755(self):
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-s", "app/deploy/backup.sh"],
            capture_output=True, text=True, check=True,
        )
        self.assertTrue(proc.stdout.strip(), "backup.sh is not tracked by git")
        self.assertTrue(
            proc.stdout.startswith("100755 "),
            f"git index mode is not 100755: {proc.stdout.strip()!r}",
        )

    def test_strips_psycopg_from_url_before_pg_dump(self):
        # pg_dump rejects postgresql+psycopg://; the SQLAlchemy driver suffix must go.
        self.assertRegex(
            self.code,
            r"\+psycopg",
            "backup.sh never mentions +psycopg, so pg_dump receives the SQLAlchemy URL",
        )
        strip_idx = self.code.find("+psycopg")
        dump_idx = self.code.find("pg_dump")
        self.assertGreater(dump_idx, -1, "pg_dump call missing")
        self.assertLess(strip_idx, dump_idx, "+psycopg is handled after pg_dump, not before")

    def test_prunes_dumps_older_than_14_days(self):
        self.assertRegex(
            self.code,
            r"find\b[^\n]*-mtime\s+\+(14|13|\$\{?[A-Za-z_]+\}?)",
            "no find -mtime +N prune of old dumps",
        )
        self.assertRegex(self.code, r"\b14\b", "retention of 14 days not stated")
        self.assertRegex(
            self.code, r"-delete|\brm\b", "find prune has no -delete or rm action"
        )

    def test_tars_data_dir(self):
        self.assertRegex(
            self.code,
            r"\btar\b[^\n]*data",
            "no tar of data/ (audit log and model JSON)",
        )

    def test_outputs_tar_is_optional_via_backup_outputs(self):
        self.assertIn("BACKUP_OUTPUTS", self.code, "BACKUP_OUTPUTS opt in switch missing")
        self.assertRegex(
            self.code,
            r"BACKUP_OUTPUTS[^\n]*1|\"?\$\{?BACKUP_OUTPUTS[^\n]*",
            "BACKUP_OUTPUTS is not tested against 1",
        )
        self.assertRegex(
            self.code,
            r"\btar\b[^\n]*outputs|outputs[^\n]*\btar\b",
            "outputs/ is never handed to tar",
        )

    def test_env_file_parsed_line_by_line_not_sourced(self):
        self.assertNotRegex(
            self.code,
            r"^\s*(source|\.)\s+/etc/remixmate\.env",
            "env file is still sourced; source executes arbitrary shell",
        )
        self.assertRegex(
            self.code,
            r"while\s+(IFS=\S*\s+)?read\b",
            "no while read loop over the env file",
        )


class BootstrapScriptTests(unittest.TestCase):
    def setUp(self):
        self.text = _read(BOOTSTRAP)
        self.code = "\n".join(_code_lines(self.text))

    def test_root_check(self):
        self.assertRegex(
            self.code,
            r"EUID|\bid\s+-u\b|\bwhoami\b",
            "no root check; apt/useradd/systemctl fail half way as a normal user",
        )

    def test_useradd_M_with_explicit_home(self):
        lines = [ln for ln in self.code.splitlines() if re.search(r"\buseradd\b", ln)]
        self.assertTrue(lines, "no useradd call")
        for ln in lines:
            self.assertRegex(ln, r"\s-M(\s|$)", f"useradd creates a home (-m or no -M): {ln.strip()!r}")
            self.assertRegex(ln, r"\s-d\s+/\S+", f"useradd has no explicit -d directory: {ln.strip()!r}")

    def test_refuses_until_repo_cloned(self):
        self.assertRegex(self.code, r"/opt/remixmate/\.git", "no check for the cloned repo")
        idx = self.code.find("/opt/remixmate/.git")
        after = self.code[idx:]
        self.assertRegex(after, r"exit\s+1", "missing repo does not exit 1")

    def test_python312_explicit(self):
        self.assertRegex(
            self.code,
            r"python3\.12\s+-m\s+venv",
            "venv is created with a bare python3, not python3.12",
        )

    def test_chown_only_data_and_outputs(self):
        chowns = [ln for ln in self.code.splitlines() if re.search(r"\bchown\b", ln)]
        self.assertTrue(chowns, "no chown at all; data/ and outputs/ stay root owned")
        joined = "\n".join(chowns)
        self.assertRegex(joined, r"/opt/remixmate/data\b", "chown does not cover data/")
        self.assertRegex(joined, r"/opt/remixmate/outputs\b", "chown does not cover outputs/")
        for ln in chowns:
            self.assertNotRegex(
                ln,
                r"chown\b[^\n]*\s/opt/remixmate/?(\s|$|\")",
                f"chown targets the whole repo: {ln.strip()!r}",
            )

    def test_installs_caddyfile_and_reloads_caddy(self):
        self.assertRegex(
            self.code,
            r"\b(cp|install)\b[^\n]*Caddyfile[^\n]*/etc/caddy/",
            "Caddyfile is not copied to /etc/caddy/ (only mentioned in printed prose)",
        )
        self.assertRegex(
            self.code, r"systemctl\s+reload\s+caddy", "caddy is not reloaded"
        )

    def test_restarts_remixmate(self):
        self.assertRegex(
            self.code, r"systemctl\s+restart\s+remixmate", "remixmate is not restarted"
        )

    def test_prints_health_ready(self):
        self.assertRegex(
            self.code,
            r"curl\b[^\n]*/health/ready",
            "bootstrap never fetches /health/ready",
        )


class SystemdUnitTests(unittest.TestCase):
    def setUp(self):
        self.text = _read(UNIT)

    def test_exec_start_has_graceful_shutdown_timeout(self):
        exec_start = _unit_values(self.text, "ExecStart")
        self.assertEqual(len(exec_start), 1, exec_start)
        self.assertRegex(
            exec_start[0],
            r"--timeout-graceful-shutdown\s+30\b",
            "uvicorn has no --timeout-graceful-shutdown 30",
        )

    def test_after_network_online(self):
        after = " ".join(_unit_values(self.text, "After"))
        self.assertIn("network-online.target", after, f"After= is {after!r}")

    def test_requires_mounts_for_catalog(self):
        values = _unit_values(self.text, "RequiresMountsFor")
        self.assertTrue(
            any("/mnt/catalog" in v for v in values),
            f"RequiresMountsFor=/mnt/catalog missing: {values}",
        )

    def test_protect_system_strict(self):
        self.assertEqual(_unit_values(self.text, "ProtectSystem"), ["strict"])

    def test_read_write_paths_data_and_outputs(self):
        paths = " ".join(_unit_values(self.text, "ReadWritePaths")).split()
        self.assertIn("/opt/remixmate/data", paths)
        self.assertIn("/opt/remixmate/outputs", paths)


class CaddyfileTests(unittest.TestCase):
    def setUp(self):
        self.text = _read(CADDYFILE)

    def test_request_body_max_size_120mb(self):
        self.assertRegex(
            self.text,
            r"request_body\s*\{[^}]*max_size\s+120MB",
            "no request_body { max_size 120MB } block",
        )

    def test_events_flush_interval_minus_one(self):
        self.assertRegex(self.text, r"/events/\*", "no /events/* matcher")
        self.assertRegex(self.text, r"flush_interval\s+-1\b", "no flush_interval -1 for SSE")


if __name__ == "__main__":
    unittest.main()
