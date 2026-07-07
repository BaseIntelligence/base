"""Phase H turnkey installer — ``--auto-secrets`` provisioning (install-swarm.sh).

``auto_provision_secrets`` (STEP 5b) generates (AUTOGEN), derives (DB URLs), and
mints (central gate token) every secret EXCEPT the single external
``YUNWU_API_KEY``, so a blank box needs only that one key. Values persist to
``SECRETS_ENV_FILE`` (mode 600) under ``--apply`` for coherent re-runs and are
never printed to plan/log output. When ``--auto-secrets`` is absent the step is a
no-op and the operator env flows through ``create_secrets`` unchanged.

Behavioral tests: run the real installer with a stub ``docker`` on PATH,
asserting the planned argv / ``die`` text (dry-run) or the persisted secrets file
(apply). Because DB-URL / Fernet VALUES never reach argv (stdin/file only), the
apply-path test asserts derivation via the persisted file. Secret VALUE bytes on
disk are constructed here from parts — the DSN literals render REDACTED in a file
view but the real bytes are the standard SQLAlchemy async DSNs asserted below.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parents[2]
SWARM_INSTALLER = ROOT / "deploy" / "swarm" / "install-swarm.sh"

# The base-master image ref carries "base-master"; the mint runs inside it.
IMAGE_MASTER_MARKER = "base-master"

# The 15 docker secret names create_secrets provisions (14 required + yunwu).
ALL_SECRET_NAMES = (
    "base_admin_token",
    "base_master_database_url",
    "base_master_pg_password",
    "base_agent_challenge_challenge_token",
    "base_agent_challenge_docker_broker_token",
    "base_agent_challenge_submission_env_encryption_key",
    "base_agent_challenge_database_url",
    "base_agent_challenge_pg_password",
    "base_prism_challenge_token",
    "base_prism_docker_broker_token",
    "base_prism_database_url",
    "base_prism_pg_password",
    "base_gateway_token_secret",
    "base_gateway_yunwu_api_key",
    "base_gateway_token",
)


def _write_stub(bin_dir: Path, body: str) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(body, encoding="utf-8")
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# Dry-run stub: recent engine, INACTIVE swarm, every `inspect` MISSES.
_STUB_DRYRUN = (
    "#!/usr/bin/env bash\n"
    'case "$1" in\n'
    "  version) echo '29.1.3' ;;\n"
    "  info) echo 'inactive' ;;\n"
    "  *) exit 1 ;;\n"
    "esac\n"
)

# Idempotency stub: like the dry-run stub but every `docker secret inspect` HITS
# (exit 0), so _ensure_secret takes its "already exists — skipping" branch and
# plans NO create. Other inspects still MISS so their resources are planned.
_STUB_SECRET_HIT = (
    "#!/usr/bin/env bash\n"
    'sub="$1"; shift || true\n'
    'case "$sub" in\n'
    "  version) echo '29.1.3' ;;\n"
    "  info) echo 'inactive' ;;\n"
    '  secret) case "$1" in inspect) exit 0 ;; *) exit 1 ;; esac ;;\n'
    "  *) exit 1 ;;\n"
    "esac\n"
)

# Apply stub: wallet-gen `docker run … python3` echoes a fake ss58 and the mint
# `docker run … mint-central-gate-token` echoes a fake token (so command
# substitutions succeed without a real image); `docker secret create` FAILS so the
# apply run aborts at create_secrets (STEP 6) — AFTER auto_provision_secrets (STEP
# 5b) has already persisted SECRETS_ENV_FILE, and BEFORE the real cache-download /
# healthcheck steps. Everything else succeeds.
_STUB_APPLY = (
    "#!/usr/bin/env bash\n"
    'sub="$1"; shift || true\n'
    'case "$sub" in\n'
    "  version) echo '29.1.3' ;;\n"
    "  info) echo 'inactive' ;;\n"
    "  run)\n"
    '    case "$*" in\n'
    "      *mint-central-gate-token*) echo 'FAKECENTRALTOKEN' ;;\n"
    "      *python3*) echo '5FAKEhotkeyss58ADDRESS' ;;\n"
    "      *) : ;;\n"
    "    esac ;;\n"
    "  secret) exit 1 ;;\n"
    "  image) exit 1 ;;\n"
    "  *) exit 0 ;;\n"
    "esac\n"
)


def _run(
    tmp_path: Path,
    *extra_args: str,
    stub: str = _STUB_DRYRUN,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _write_stub(bin_dir, stub)
    # Start from a pristine env WITHOUT any inherited secret vars, then add PATH and
    # only what a test opts into — auto-secrets must generate the rest.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            (
                "BASE_",
                "MASTER_",
                "AGENT_CHALLENGE_",
                "PRISM_",
                "GATEWAY_",
                "YUNWU_",
                "CENTRAL_",
                "HF_",
            )
        )
    }
    if extra_env:
        env.update(extra_env)
    env["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
    env["BROKER_WORKSPACE_DIR"] = str(tmp_path / "broker-ws")
    env["MASTER_CONFIG_PATH"] = str(tmp_path / "master.yaml")
    return subprocess.run(
        [
            "bash",
            str(SWARM_INSTALLER),
            "--backup-dir",
            str(tmp_path / "missing"),
            "--greenfield",
            "--skip-ghcr-login",
            *extra_args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


# ---------------------------------------------------------------------------
# Dry-run: --auto-secrets provisions everything but the external YUNWU key
# ---------------------------------------------------------------------------


def test_autosecrets_plans_all_15_secrets_with_only_yunwu(tmp_path: Path) -> None:
    # Every secret env var unset EXCEPT YUNWU_API_KEY; --auto-secrets generates /
    # derives / mints the other 14 so create_secrets plans all 15 with NO
    # "required secret env var … is empty" die.
    result = _run(
        tmp_path,
        "--auto-secrets",
        "--static-challenges",
        extra_env={"YUNWU_API_KEY": "external-key"},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "required secret env var" not in result.stderr
    for name in ALL_SECRET_NAMES:
        assert f"docker secret create {name}" in result.stdout, name


def test_yunwu_still_required_in_real_mode(tmp_path: Path) -> None:
    # YUNWU_API_KEY stays EXTERNAL: --auto-secrets + provider_mode=real + unset key
    # hard-fails with the single turnkey message (proves it is never synthesized).
    result = _run(
        tmp_path,
        "--auto-secrets",
        "--static-challenges",
        extra_env={"GATEWAY_PROVIDER_MODE": "real"},
    )
    assert result.returncode != 0, f"stdout={result.stdout!r}"
    assert "only required external secret" in result.stderr


def test_mock_mode_plans_no_yunwu_secret(tmp_path: Path) -> None:
    # provider_mode=mock needs no provider key: --auto-secrets completes with YUNWU
    # unset and plans NO yunwu secret.
    result = _run(
        tmp_path,
        "--auto-secrets",
        "--static-challenges",
        extra_env={"GATEWAY_PROVIDER_MODE": "mock"},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "docker secret create base_gateway_yunwu_api_key" not in result.stdout


def test_mint_plan_is_shown_in_dry_run(tmp_path: Path) -> None:
    # Dry-run prints the planned central-gate mint as an auditable `docker run …
    # ${IMAGE_MASTER}` argv carrying `mint-central-gate-token --source llm_review`.
    result = _run(
        tmp_path,
        "--auto-secrets",
        "--static-challenges",
        extra_env={"YUNWU_API_KEY": "external-key"},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    mint_lines = [
        line
        for line in result.stdout.splitlines()
        if "docker run" in line and "mint-central-gate-token" in line
    ]
    assert mint_lines, f"no planned mint docker-run line; stdout={result.stdout!r}"
    line = mint_lines[0]
    assert IMAGE_MASTER_MARKER in line
    assert "--source llm_review" in line


def test_idempotent_when_secrets_already_exist(tmp_path: Path) -> None:
    # A docker stub whose `secret inspect` HITS => each secret is "already exists —
    # skipping" and NO new `docker secret create` is planned (idempotent re-run).
    result = _run(
        tmp_path,
        "--auto-secrets",
        "--static-challenges",
        stub=_STUB_SECRET_HIT,
        extra_env={"YUNWU_API_KEY": "external-key"},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    assert "already exists — skipping" in result.stdout
    assert "docker secret create" not in result.stdout


# ---------------------------------------------------------------------------
# Apply-path: DB-URL derivation + persistence to SECRETS_ENV_FILE (mode 600)
# ---------------------------------------------------------------------------


def _run_apply(tmp_path: Path) -> Path:
    """Run `--apply --auto-secrets` with only the 3 PG passwords preset and return
    the persisted SECRETS_ENV_FILE.

    provider_mode=mock (no external key) and a preset CENTRAL_GATEWAY_TOKEN skip
    the root-only /run mint path (minting is covered by the dry-run mint-plan
    test). The apply run intentionally aborts at create_secrets (the stub fails
    `secret create`), AFTER STEP 5b has persisted the file and BEFORE the real
    cache-download / healthcheck steps — so the file is always written.
    """
    secrets_file = tmp_path / "secrets.env"
    result = _run(
        tmp_path,
        "--apply",
        "--auto-secrets",
        "--static-challenges",
        stub=_STUB_APPLY,
        extra_env={
            "GATEWAY_PROVIDER_MODE": "mock",
            "CENTRAL_GATEWAY_TOKEN": "preset-central-token",
            "SECRETS_ENV_FILE": str(secrets_file),
            "VALIDATOR_WALLET_PATH": str(tmp_path / "wallets"),
            "MASTER_PG_PASSWORD": "pwMASTER123",
            "AGENT_CHALLENGE_PG_PASSWORD": "pwAC456",
            "PRISM_PG_PASSWORD": "pwPRISM789",
        },
    )
    # The run aborts by design at the first `docker secret create`; the file must
    # already be persisted at that point.
    assert secrets_file.is_file(), (
        f"SECRETS_ENV_FILE not written; rc={result.returncode} stderr={result.stderr!r}"
    )
    return secrets_file


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"export (\w+)='(.*)'$", raw.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def test_apply_derives_dsns_and_persists_mode_600(tmp_path: Path) -> None:
    secrets_file = _run_apply(tmp_path)

    # mode 600 (root-only secret-at-rest).
    assert stat.S_IMODE(secrets_file.stat().st_mode) == 0o600

    values = _parse_env_file(secrets_file)
    # Each derived DSN embeds the matching PG password + the fixed host/user/db
    # literals from _deploy_postgres_service (driver postgresql+asyncpg, port 5432).
    assert values["MASTER_DATABASE_URL"] == (
        "postgresql+asyncpg://base:pwMASTER123@base-master-postgres:5432/base"
    )
    assert values["AGENT_CHALLENGE_DATABASE_URL"] == (
        "postgresql+asyncpg://challenge:pwAC456"
        "@challenge-agent-challenge-postgres:5432/challenge"
    )
    assert values["PRISM_DATABASE_URL"] == (
        "postgresql+asyncpg://challenge:pwPRISM789"
        "@challenge-prism-postgres:5432/challenge"
    )


def test_apply_persists_valid_fernet_submission_key(tmp_path: Path) -> None:
    secrets_file = _run_apply(tmp_path)
    values = _parse_env_file(secrets_file)

    key = values["AGENT_CHALLENGE_SUBMISSION_ENV_KEY"]
    # Fernet key = urlsafe-b64(32 bytes) = 44 chars; the consumer builds Fernet(key).
    assert len(key) == 44
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}=", key), key
    Fernet(key.encode())  # must not raise
