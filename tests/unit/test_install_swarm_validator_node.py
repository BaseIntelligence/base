"""m6 G-B1/G-B3 guard tests for ``install-swarm.sh``: turnkey supervisor staging
and the validator-NODE install path.

Encodes:

* VAL-CODE-AUTO-004 — ``--install-supervisor`` STAGES the unit's release checkout
  (``releases/<version>`` + ``uv sync``) and atomically points ``current`` at it
  BEFORE ``systemctl enable --now``, so the unit ExecStart
  (``uv run --project .../current base master supervisor``) actually boots.
* VAL-CODE-AUTO-005 — ``--validator-node`` renders an AUTO-UPDATABLE
  ``base validator agent`` swarm service PLUS a node-local base-supervisor
  configured with the validator-agent updater target, so a validator's ``base``
  code auto-updates.

BEHAVIORAL tests: run the real installer in DEFAULT dry-run (mutates nothing) with
a stub ``docker`` whose every ``inspect`` misses, so resources are PLANNED, then
assert the rendered plan ordering/content (not the source text).
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWARM_INSTALLER = ROOT / "deploy" / "swarm" / "install-swarm.sh"

MANIFEST_URL = "https://raw.example/base/release/supervisor-manifest.json"

# Master path needs every hard-required secret so a full dry-run reaches
# deploy_supervisor (STEP 12) instead of dying earlier on a missing secret.
MASTER_SECRET_ENV = {
    "GHCR_USER": "ci-user",
    "GHCR_TOKEN": "ci-token",
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

# The validator-NODE flow brings up only the agent + node-local supervisor, so it
# hard-requires the GHCR login, the validator's own broker token, AND the master
# coordination/gateway URL (VALIDATOR_MASTER_URL is required — no self-referential
# default; see test_validator_node_requires_master_url).
VALIDATOR_SECRET_ENV = {
    "GHCR_USER": "ci-user",
    "GHCR_TOKEN": "ci-token",
    "VALIDATOR_BROKER_TOKEN": "btok",
    "VALIDATOR_MASTER_URL": "http://master-host:19080",
}


def _docker_stub(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        "  version) echo '29.1.3' ;;\n"
        "  info) echo 'inactive' ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    tmp_path: Path,
    *extra_args: str,
    env_overrides: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _docker_stub(bin_dir)
    env = dict(os.environ)
    env.update(env_overrides)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["BROKER_WORKSPACE_DIR"] = str(tmp_path / "broker-ws")
    env["MASTER_CONFIG_PATH"] = str(tmp_path / "master.yaml")
    argv = [
        "bash",
        str(SWARM_INSTALLER),
        "--backup-dir",
        str(tmp_path / "missing"),
        *extra_args,
    ]
    return subprocess.run(
        argv, env=env, capture_output=True, text=True, timeout=120, check=False
    )


def _plan_lines(out: str) -> list[str]:
    """Only the executable plan lines (``  + ...``)."""
    return [ln for ln in out.splitlines() if ln.lstrip().startswith("+ ")]


def _index_of(lines: list[str], needle: str) -> int:
    for i, ln in enumerate(lines):
        if needle in ln:
            return i
    raise AssertionError(f"no plan line containing {needle!r}; lines={lines!r}")


# ---------------------------------------------------------------------------
# VAL-CODE-AUTO-004: --install-supervisor stages current BEFORE enable
# ---------------------------------------------------------------------------


def test_install_supervisor_stages_current_before_enable(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "--greenfield",
        "--static-challenges",
        "--install-supervisor",
        env_overrides=MASTER_SECRET_ENV,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = _plan_lines(result.stdout)

    # The release is exported from the COMMITTED HEAD tree (git archive -> tar), NOT
    # cp -a of the working tree (no .git, no uncommitted/untracked files), + a
    # per-release venv provisioned (uv sync), then current is atomically flipped
    # (mv -T ... /current) — all BEFORE the unit is enabled, so ExecStart
    # --project .../current can boot.
    archive_idx = _index_of(lines, "archive --format=tar")
    uv_sync_idx = _index_of(lines, "uv sync --project")
    symlink_idx = _index_of(lines, "ln -sfn releases/")
    swap_idx = _index_of(lines, "mv -T")
    enable_idx = _index_of(lines, "systemctl enable --now base-supervisor.service")

    assert not any("cp -a" in ln for ln in lines), (
        "must export the committed HEAD tree (git archive), not cp -a the working tree"
    )
    assert archive_idx < enable_idx, (
        "committed-tree snapshot must be staged before enable"
    )
    assert uv_sync_idx < enable_idx, "uv sync must run before enable"
    assert symlink_idx < swap_idx < enable_idx, (
        "current must be atomically swapped to releases/<version> before enable"
    )
    # The swap targets the unit's `current` symlink under the release root.
    swap_line = lines[swap_idx]
    assert "/var/lib/base/supervisor/current" in swap_line
    assert "releases/" in lines[symlink_idx]


def test_install_supervisor_staging_uses_atomic_relative_symlink(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        "--greenfield",
        "--static-challenges",
        "--install-supervisor",
        env_overrides=MASTER_SECRET_ENV,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = _plan_lines(result.stdout)

    # A relative symlink target (releases/<version>) + mv -T mirrors
    # self_update.atomic_symlink_swap, so current always points at a COMPLETE dir.
    symlink_line = lines[_index_of(lines, "ln -sfn releases/")]
    assert "current.staging" in symlink_line
    swap_line = lines[_index_of(lines, "mv -T")]
    assert "current.staging" in swap_line and swap_line.rstrip().endswith("current")


def test_default_without_install_supervisor_documents_staging_step(
    tmp_path: Path,
) -> None:
    # Without the flag the staging+enable are INSTRUCTIONS only (no `  + ` plan
    # lines), but the staging-before-enable cutover step is documented.
    result = _run(
        tmp_path,
        "--greenfield",
        "--static-challenges",
        env_overrides=MASTER_SECRET_ENV,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    assert "MANDATORY CUTOVER STEP" in out
    assert "uv sync --project" in out  # documented in the instruction block
    assert "+ systemctl enable --now base-supervisor.service" not in out


# ---------------------------------------------------------------------------
# VAL-CODE-AUTO-005: --validator-node renders an auto-updatable agent + supervisor
# ---------------------------------------------------------------------------


def test_validator_node_renders_autoupdatable_agent_service(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        "--validator-node",
        env_overrides=VALIDATOR_SECRET_ENV,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = _plan_lines(result.stdout)

    svc = lines[_index_of(lines, "docker service create")]
    # The agent runs as a swarm service on the node's OWN swarm, running the agent
    # CMD from the auto-rolled validator runtime image.
    assert "--name base-validator-agent" in svc
    assert "base validator agent --config" in svc
    assert "ghcr.io/baseintelligence/base-validator-runtime" in svc
    # Its own broker token + rendered validator.yaml are mounted.
    assert "base_broker_token" in svc
    assert "base_validator_yaml" in svc


def test_validator_node_renders_supervisor_with_validator_agent_target(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        "--validator-node",
        env_overrides=VALIDATOR_SECRET_ENV,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # The node-local supervisor config drives the image-updater at THIS node's
    # agent service (validator_agent_target_enabled) and drops the master targets.
    assert "validator_agent_target_enabled: true" in out
    assert "validator_agent_service: base-validator-agent" in out
    assert (
        "validator_agent_image: ghcr.io/baseintelligence/base-validator-runtime:latest"
        in out
    )
    assert "image_updater_targets: []" in out
    # The watched image announced by deploy_supervisor is the validator runtime,
    # NOT the master proxy/broker.
    assert "watched image (image-updater, 60s): base-validator-agent" in out


def test_validator_node_install_supervisor_stages_current_before_enable(
    tmp_path: Path,
) -> None:
    result = _run(
        tmp_path,
        "--validator-node",
        "--install-supervisor",
        env_overrides=VALIDATOR_SECRET_ENV,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    lines = _plan_lines(result.stdout)

    # The validator NODE's own supervisor unit is staged + enabled (same G-B1
    # staging-before-enable guarantee as the master path).
    svc_idx = _index_of(lines, "--name base-validator-agent")
    swap_idx = _index_of(lines, "mv -T")
    enable_idx = _index_of(lines, "systemctl enable --now base-supervisor.service")
    assert svc_idx < enable_idx, "agent service is created before the supervisor enable"
    assert swap_idx < enable_idx, "current is staged before the supervisor is enabled"


def test_validator_node_self_update_enabled_when_manifest_url_set(
    tmp_path: Path,
) -> None:
    env = dict(VALIDATOR_SECRET_ENV)
    env["SUPERVISOR_SELF_UPDATE_MANIFEST_URL"] = MANIFEST_URL
    result = _run(tmp_path, "--validator-node", env_overrides=env)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    assert "self_update_enabled: true" in out
    assert f"self_update_manifest_url: {MANIFEST_URL}" in out


def test_validator_node_dry_run_is_non_mutating(tmp_path: Path) -> None:
    cfg = tmp_path / "master.yaml"
    result = _run(
        tmp_path,
        "--validator-node",
        env_overrides=VALIDATOR_SECRET_ENV,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    # Dry-run writes NOTHING: the node-local supervisor config file is only shown.
    assert not cfg.exists()
    assert "(dry-run)" in result.stdout
    # Imperative Swarm only (mirrors the test_docker_compose_deploy.py contract).
    assert "docker compose" not in result.stdout.lower()


def test_validator_node_requires_master_url(tmp_path: Path) -> None:
    # VALIDATOR_MASTER_URL has NO default (a validator must never point at its own
    # advertise address — a footgun), so --validator-node fails fast when it is unset,
    # even in dry-run, before anything is planned.
    env = {k: v for k, v in VALIDATOR_SECRET_ENV.items() if k != "VALIDATOR_MASTER_URL"}
    result = _run(tmp_path, "--validator-node", env_overrides=env)
    assert result.returncode != 0, f"expected failure; stdout={result.stdout!r}"
    assert "VALIDATOR_MASTER_URL is required" in result.stderr
    # Fails before planning any mutating command.
    assert not _plan_lines(result.stdout)


def test_validator_node_master_url_is_not_self_referential(tmp_path: Path) -> None:
    # The rendered validator.yaml points registry/master/gateway at the explicit
    # VALIDATOR_MASTER_URL (the MASTER), never the node's own advertise address.
    result = _run(
        tmp_path,
        "--validator-node",
        env_overrides=VALIDATOR_SECRET_ENV,
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    assert "master_url: http://master-host:19080" in out
    assert "gateway_url: http://master-host:19080" in out
    assert "registry_url: http://master-host:19080" in out


def test_installer_is_bash_n_clean() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(SWARM_INSTALLER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr!r}"
