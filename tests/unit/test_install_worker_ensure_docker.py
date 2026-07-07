"""Phase H turnkey installer — ensure_docker() + --turnkey (install-worker.sh).

The worker installer gained ``ensure_docker`` (STEP 0, before ``preflight``) — a
mirror of the swarm installer's copy — plus a ``--turnkey`` alias that sets
``APPLY`` + ``RESTART_DOCKERD`` for a blank compute node. It still hard-fails
without ``--manager-addr`` / a join token.

Behavioral tests: run the real installer with a stub ``docker`` (and, for the
apply/turnkey path, a stub ``systemctl`` and a temp ``DAEMON_JSON_DST``) on PATH,
asserting the planned argv / ``die`` text — never the source text.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKER_INSTALLER = ROOT / "scripts" / "install-worker.sh"


def _make_exec(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _bin_dir(tmp_path: Path, version: str, with_systemctl: bool = False) -> Path:
    """docker stub (chosen VERSION, INACTIVE swarm, all else exit 0 so a `swarm
    join` under --apply succeeds); optional no-op systemctl stub for the apply
    path so `systemctl restart docker` never touches the real host."""
    bin_dir = tmp_path / "bin"
    _make_exec(
        bin_dir / "docker",
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        f"  version) echo '{version}' ;;\n"
        "  info) echo 'inactive' ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
    )
    if with_systemctl:
        _make_exec(bin_dir / "systemctl", "#!/usr/bin/env bash\nexit 0\n")
    return bin_dir


def _run(
    tmp_path: Path,
    *args: str,
    version: str = "29.2.1",
    with_systemctl: bool = False,
    set_join_token: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = _bin_dir(tmp_path, version, with_systemctl)
    env = dict(os.environ)
    env.pop("JOIN_TOKEN", None)
    if set_join_token:
        env["JOIN_TOKEN"] = "join-xyz"
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(WORKER_INSTALLER), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_worker_is_bash_n_clean() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(WORKER_INSTALLER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr!r}"


def test_ensure_docker_planned_before_preflight(tmp_path: Path) -> None:
    result = _run(tmp_path, "--manager-addr", "1.2.3.4:2377", "--workload", "cpu")
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    step0 = "STEP 0 ensure_docker"
    step1 = "STEP 1/3 preflight"
    assert step0 in result.stdout
    assert step1 in result.stdout
    assert result.stdout.index(step0) < result.stdout.index(step1)


def test_recent_docker_skips_install(tmp_path: Path) -> None:
    result = _run(tmp_path, "--manager-addr", "1.2.3.4:2377", "--workload", "cpu")
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "skipping install (idempotent)" in result.stdout
    assert "get.docker.com" not in result.stdout


def test_old_docker_plans_get_docker(tmp_path: Path) -> None:
    # Worker preflight only WARNS on an old engine (it does not die), so the run
    # still completes; ensure_docker plans the get.docker.com install (dry-run).
    result = _run(
        tmp_path,
        "--manager-addr",
        "1.2.3.4:2377",
        "--workload",
        "cpu",
        version="24.0.0",
    )
    assert "https://get.docker.com" in result.stdout
    assert "curl -fsSL -o" in result.stdout


def test_skip_docker_install_hard_fails_when_old(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "--manager-addr",
        "1.2.3.4:2377",
        "--workload",
        "cpu",
        version="24.0.0",
        extra_env={"SKIP_DOCKER_INSTALL": "true"},
    )
    assert result.returncode != 0, f"stdout={result.stdout!r}"
    assert "SKIP_DOCKER_INSTALL=true" in result.stderr


def test_turnkey_sets_apply_and_restart_dockerd(tmp_path: Path) -> None:
    # --turnkey => APPLY + RESTART_DOCKERD. Sandboxed: docker + systemctl stubs and
    # a temp DAEMON_JSON_DST so nothing touches the real host.
    result = _run(
        tmp_path,
        "--manager-addr",
        "1.2.3.4:2377",
        "--workload",
        "cpu",
        "--turnkey",
        with_systemctl=True,
        extra_env={"DAEMON_JSON_DST": str(tmp_path / "daemon.json")},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    combined = result.stdout + result.stderr
    assert "RUNNING IN --apply MODE" in combined
    # RESTART_DOCKERD=true takes the install branch (absent without the flag).
    assert "install -m 0644 " in result.stdout
    assert str(tmp_path / "daemon.json") in result.stdout


def test_turnkey_still_requires_manager_addr(tmp_path: Path) -> None:
    # --turnkey (APPLY) must still hard-fail without --manager-addr; ensure_docker
    # skips (recent engine) so nothing mutates before the preflight die.
    result = _run(
        tmp_path,
        "--workload",
        "cpu",
        "--turnkey",
        with_systemctl=True,
    )
    assert result.returncode != 0, f"stdout={result.stdout!r}"
    assert "--manager-addr" in result.stderr


def test_still_requires_join_token(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "--manager-addr",
        "1.2.3.4:2377",
        "--workload",
        "cpu",
        set_join_token=False,
    )
    assert result.returncode != 0, f"stdout={result.stdout!r}"
    assert "join token required" in result.stderr
