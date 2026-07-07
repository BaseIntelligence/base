"""Turnkey installer — ensure_uv() uv auto-install (install-swarm.sh).

``ensure_uv`` runs as STEP 0b (AFTER ``ensure_docker``, BEFORE ``preflight``). When
the supervisor is being installed (``--install-supervisor``) it installs the ``uv``
runtime the ``base-supervisor.service`` unit launches through
(``ExecStart=/usr/local/bin/uv run --project current ...``) via the astral.sh
installer, all behind ``plan`` (dry-run-safe) and idempotent (skips when ``uv`` is
already present at ``${UV_INSTALL_DIR:-/usr/local/bin}/uv``). ``SKIP_UV_INSTALL=true``
opts out with a hard-fail; when the supervisor is NOT being installed it is a no-op.

These are behavioral dry-run tests: they run the real installer with a stub
``docker`` on PATH reporting a recent server version + inactive swarm and assert the
planned argv / ``die`` text / step ordering, never the source text. uv presence is
controlled by pointing ``UV_INSTALL_DIR`` at a tmp dir (the idempotency check reads
``${UV_INSTALL_DIR:-/usr/local/bin}/uv``).
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWARM_INSTALLER = ROOT / "deploy" / "swarm" / "install-swarm.sh"

# Full env that satisfies every hard requirement so the happy-path dry-run runs to
# completion (values are throwaway — dry-run never uses them). GHCR creds are omitted
# because these runs pass --skip-ghcr-login.
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


def _docker_stub(bin_dir: Path, version: str = "29.1.3") -> None:
    """`docker` stub reporting a recent server VERSION and an INACTIVE swarm.

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


def _make_uv(uv_dir: Path) -> None:
    """Create an executable file named ``uv`` in ``uv_dir`` (models uv present)."""
    uv_dir.mkdir(parents=True, exist_ok=True)
    uv = uv_dir / "uv"
    uv.write_text("#!/usr/bin/env bash\necho 'uv 0.0.0'\n", encoding="utf-8")
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    tmp_path: Path,
    *extra_args: str,
    uv_install_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _docker_stub(bin_dir)
    env = dict(os.environ)
    env.update(REQUIRED_SECRET_ENV)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["BROKER_WORKSPACE_DIR"] = str(tmp_path / "broker-ws")
    env["MASTER_CONFIG_PATH"] = str(tmp_path / "master.yaml")
    if uv_install_dir is not None:
        env["UV_INSTALL_DIR"] = str(uv_install_dir)
    if extra_env:
        env.update(extra_env)
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


def test_supervisor_install_uv_absent_plans_astral_install(tmp_path: Path) -> None:
    # --install-supervisor => ensure_uv engages; uv ABSENT (empty UV_INSTALL_DIR) =>
    # plan fetches + runs the astral.sh installer with UV_INSTALL_DIR pinned + no PATH
    # edits (dry-run only; plan lines print but nothing executes).
    uv_dir = tmp_path / "uvbin"
    uv_dir.mkdir()
    result = _run(tmp_path, "--install-supervisor", uv_install_dir=uv_dir)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    assert "STEP 0b ensure_uv" in out
    assert "curl -fsSL -o /tmp/uv-install." in out
    assert "https://astral.sh/uv/install.sh" in out
    assert f"env UV_INSTALL_DIR={uv_dir} INSTALLER_NO_MODIFY_PATH=1 sh" in out


def test_uv_present_skips_install_idempotent(tmp_path: Path) -> None:
    # uv already present at ${UV_INSTALL_DIR}/uv => ensure_uv is an idempotent skip
    # and plans NO astral.sh install; the rest of the dry-run completes cleanly.
    uv_dir = tmp_path / "uvbin"
    _make_uv(uv_dir)
    result = _run(tmp_path, "--install-supervisor", uv_install_dir=uv_dir)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    assert f"uv present at {uv_dir}/uv" in out
    assert "skipping install (idempotent)" in out
    assert "astral.sh" not in out


def test_skip_uv_install_hard_fails_when_uv_absent(tmp_path: Path) -> None:
    # Opted out of uv auto-install but supervisor being installed + uv absent =>
    # hard-fail with a message that names SKIP_UV_INSTALL; no astral install planned.
    uv_dir = tmp_path / "uvbin"
    uv_dir.mkdir()
    result = _run(
        tmp_path,
        "--install-supervisor",
        uv_install_dir=uv_dir,
        extra_env={"SKIP_UV_INSTALL": "true"},
    )
    assert result.returncode != 0, f"stdout={result.stdout!r}"
    assert "SKIP_UV_INSTALL" in result.stderr
    assert "astral.sh" not in result.stdout


def test_no_supervisor_install_uv_is_noop(tmp_path: Path) -> None:
    # No --install-supervisor (INSTALL_SUPERVISOR stays false) => ensure_uv is a
    # no-op skip and plans NO astral.sh install, even with uv absent.
    uv_dir = tmp_path / "uvbin"
    uv_dir.mkdir()
    result = _run(tmp_path, uv_install_dir=uv_dir)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    assert "supervisor not being installed" in out
    assert "astral.sh" not in out


def test_ensure_uv_planned_after_docker_before_preflight(tmp_path: Path) -> None:
    # ensure_uv (STEP 0b) must run AFTER ensure_docker (STEP 0) and BEFORE preflight
    # (STEP 1), so a blank host gets the supervisor runtime staged in the right order.
    uv_dir = tmp_path / "uvbin"
    _make_uv(uv_dir)
    result = _run(tmp_path, "--install-supervisor", uv_install_dir=uv_dir)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    step0 = "STEP 0 ensure_docker"
    step0b = "STEP 0b ensure_uv"
    step1 = "STEP 1/12 preflight"
    assert step0 in out and step0b in out and step1 in out
    assert out.index(step0) < out.index(step0b) < out.index(step1)
