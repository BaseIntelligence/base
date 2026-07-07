"""Phase H turnkey installer — ensure_docker() Docker auto-install (install-swarm.sh).

``ensure_docker`` runs as STEP 0 (BEFORE ``preflight``) and, on a blank/old host,
installs Docker Engine >= ``MIN_DOCKER_MAJOR`` via ``get.docker.com`` (default) or
``apt``, all behind ``plan`` (dry-run-safe) and idempotent (skips when a recent
engine is already present). ``SKIP_DOCKER_INSTALL=true`` opts out with a
hard-fail; ``DOCKER_INSTALL_METHOD`` selects the install method.

These are behavioral dry-run tests: they run the real installer with a stub
``docker`` on PATH reporting a chosen server version and assert the planned argv /
``die`` text, never the source text.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWARM_INSTALLER = ROOT / "deploy" / "swarm" / "install-swarm.sh"

# Full env that satisfies every hard requirement so the happy-path dry-run runs
# to completion (values are throwaway — dry-run never uses them).
REQUIRED_SECRET_ENV: dict[str, str] = {
    "BASE_ADMIN_TOKEN": "x",
    "MASTER_DATABASE_URL": "postgresql+asyncpg://base@base-master-postgres:5432/base",
    "MASTER_PG_PASSWORD": "x",
    "AGENT_CHALLENGE_CHALLENGE_TOKEN": "x",
    "AGENT_CHALLENGE_DOCKER_BROKER_TOKEN": "x",
    "AGENT_CHALLENGE_SUBMISSION_ENV_KEY": "x",
    "AGENT_CHALLENGE_DATABASE_URL": "postgresql+asyncpg://challenge@h:5432/challenge",
    "AGENT_CHALLENGE_PG_PASSWORD": "x",
    "PRISM_CHALLENGE_TOKEN": "x",
    "PRISM_DOCKER_BROKER_TOKEN": "x",
    "PRISM_DATABASE_URL": "postgresql+asyncpg://challenge@h:5432/challenge",
    "PRISM_PG_PASSWORD": "x",
    "GATEWAY_TOKEN": "x",
    "CENTRAL_GATEWAY_TOKEN": "x",
    "YUNWU_API_KEY": "x",
}


def _docker_stub(bin_dir: Path, version: str) -> None:
    """`docker` stub reporting a chosen server VERSION and an INACTIVE swarm.

    Every non version/info subcommand exits non-zero so all `inspect` probes MISS
    (resources are planned) and no mutating subcommand is silently satisfied.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        f"  version) echo '{version}' ;;\n"
        "  info) echo 'inactive' ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    tmp_path: Path,
    *extra_args: str,
    version: str = "29.1.3",
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _docker_stub(bin_dir, version)
    env = dict(os.environ)
    env.update(REQUIRED_SECRET_ENV)
    if extra_env:
        env.update(extra_env)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["BROKER_WORKSPACE_DIR"] = str(tmp_path / "broker-ws")
    env["MASTER_CONFIG_PATH"] = str(tmp_path / "master.yaml")
    return subprocess.run(
        [
            "bash",
            str(SWARM_INSTALLER),
            "--backup-dir",
            str(tmp_path / "missing"),
            "--greenfield",
            "--static-challenges",
            "--skip-ghcr-login",
            *extra_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_installer_is_bash_n_clean() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(SWARM_INSTALLER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr!r}"


def test_recent_docker_skips_install_idempotent(tmp_path: Path) -> None:
    # docker >= MIN_DOCKER_MAJOR present => ensure_docker is a no-op skip and plans
    # NO get.docker.com install; the rest of the dry-run completes cleanly.
    result = _run(tmp_path, version="29.1.3")
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "skipping install (idempotent)" in result.stdout
    assert "get.docker.com" not in result.stdout
    assert "apt-get install -y docker.io" not in result.stdout


def test_old_docker_plans_get_docker_convenience_script(tmp_path: Path) -> None:
    # docker present but < MIN_DOCKER_MAJOR => plan fetches + runs get.docker.com
    # (dry-run only). preflight then hard-fails on the old version (dry-run does
    # not actually upgrade), so returncode is non-zero — the plan lines still print.
    result = _run(tmp_path, version="24.0.0")
    assert "curl -fsSL -o" in result.stdout
    assert "https://get.docker.com" in result.stdout
    # The fetched script is then executed via `sh <tmpfile>`.
    assert "get-docker" in result.stdout


def test_skip_docker_install_hard_fails_when_docker_old(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        version="24.0.0",
        extra_env={"SKIP_DOCKER_INSTALL": "true"},
    )
    assert result.returncode != 0, f"stdout={result.stdout!r}"
    assert "SKIP_DOCKER_INSTALL=true" in result.stderr
    # No install was planned when opted out.
    assert "get.docker.com" not in result.stdout


def test_docker_install_method_apt_plans_apt_package(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        version="24.0.0",
        extra_env={"DOCKER_INSTALL_METHOD": "apt"},
    )
    assert "apt-get install -y docker.io" in result.stdout
    # The apt method must NOT also fetch the convenience script.
    assert "get.docker.com" not in result.stdout


def test_ensure_docker_planned_before_preflight(tmp_path: Path) -> None:
    # ensure_docker (STEP 0) must run before preflight (STEP 1) so a blank host
    # gets an engine before the version/swarm/creds checks read docker.
    result = _run(tmp_path, version="29.1.3")
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    step0 = "STEP 0 ensure_docker"
    step1 = "STEP 1/12 preflight"
    assert step0 in result.stdout
    assert step1 in result.stdout
    assert result.stdout.index(step0) < result.stdout.index(step1)
