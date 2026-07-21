from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from base.config.loader import load_settings
from base.config.policy import is_production_environment, validate_settings_policy
from base.config.settings import MasterSettings, Settings, ValidatorSettings
from base.security.tokens import generate_token, hash_token, verify_token
from base.template_engine import (
    ChallengeTemplateContext,
    render_challenge_template,
)

SWARM_INSTALLER = (
    Path(__file__).resolve().parents[2] / "deploy" / "swarm" / "install-swarm.sh"
)

# Minimal required env for the installer to reach _render_master_config in the
# DEFAULT dry-run (mirrors the harness in test_install_swarm_master_subsystems.py).
# Values are placeholders; the dry-run mutates nothing and prints the plan only.
_INSTALLER_REQUIRED_ENV = {
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


def test_registry_url_defaults_and_examples_use_chain_endpoint() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = "https://chain.joinbase.ai"

    assert MasterSettings().registry_url == expected
    assert ValidatorSettings().registry_url == expected

    master_example = yaml.safe_load(
        (root / "config" / "master.example.yaml").read_text(encoding="utf-8")
    )
    validator_example = yaml.safe_load(
        (root / "config" / "validator.example.yaml").read_text(encoding="utf-8")
    )
    master_compose = yaml.safe_load(
        (root / "deploy" / "compose" / "config" / "master.compose.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert master_example["master"]["registry_url"] == expected
    assert master_example["validator"]["registry_url"] == expected
    assert validator_example["validator"]["registry_url"] == expected
    assert master_compose["master"]["registry_url"] == expected
    assert master_compose["validator"]["registry_url"] == expected
    assert ValidatorSettings().weights_url is None
    assert ValidatorSettings().resolved_weights_url == expected
    assert (
        ValidatorSettings(registry_url="https://master.example").resolved_weights_url
        == "https://master.example"
    )
    assert (
        ValidatorSettings(
            registry_url="https://registry.example",
            weights_url="https://weights.example",
        ).resolved_weights_url
        == "https://weights.example"
    )
    assert ValidatorSettings().weights_interval_seconds == 360
    assert ValidatorSettings().weights_timeout_seconds == 15.0
    assert ValidatorSettings().weights_retries == 3
    assert ValidatorSettings().weights_freshness_seconds == 720
    assert validator_example["validator"]["weights_url"] is None
    assert master_compose["validator"]["weights_url"] is None
    assert validator_example["validator"]["weights_interval_seconds"] == 360
    assert validator_example["validator"]["weights_timeout_seconds"] == 15.0
    assert validator_example["validator"]["weights_retries"] == 3
    assert validator_example["validator"]["weights_freshness_seconds"] == 720


def test_compose_shipping_defaults_pin_public_chain_joinbase_url() -> None:
    """Compose install/templates keep registry defaults on chain.joinbase.ai.

    Operator ``--master-url`` stays an explicit coordination root and must not be
    invented as a hard-coded public IP default. Public registry/weights Settings
    defaults and shipping docs recommend only ``https://chain.joinbase.ai`` as the
    public Base master API. Preferred-product / cutover messaging for
    ``chain.platform.network`` must not appear on shipping paths.
    """
    root = Path(__file__).resolve().parents[2]
    expected = "https://chain.joinbase.ai"
    retired_product_host = "https://chain.platform.network"
    retired_public_hosts = (
        "86.38.238.235",
        "51.83.112.164",
        "88.216.198.199",
    )

    install_master = (root / "deploy" / "compose" / "install-master.sh").read_text(
        encoding="utf-8"
    )
    install_validator = (
        root / "deploy" / "compose" / "install-validator.sh"
    ).read_text(encoding="utf-8")
    env_validator_example = (
        root / "deploy" / "compose" / ".env.validator.example"
    ).read_text(encoding="utf-8")
    master_compose = (
        root / "deploy" / "compose" / "config" / "master.compose.yaml"
    ).read_text(encoding="utf-8")
    master_example = (root / "config" / "master.example.yaml").read_text(
        encoding="utf-8"
    )
    validator_example = (root / "config" / "validator.example.yaml").read_text(
        encoding="utf-8"
    )
    validator_docs = (root / "docs" / "validator" / "README.md").read_text(
        encoding="utf-8"
    )
    ops_validator_docs = (root / "docs" / "operations" / "validator.md").read_text(
        encoding="utf-8"
    )
    compose_docs = (root / "docs" / "compose.md").read_text(encoding="utf-8")
    deploy_docs = (root / "docs" / "deploy.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    for shipping in (
        install_master,
        master_compose,
        master_example,
        validator_example,
    ):
        assert f"registry_url: {expected}" in shipping
        assert f"registry_url: {retired_product_host}" not in shipping
        assert "preferred product" not in shipping.lower()
        assert "chain.platform.network" not in shipping
        for host in retired_public_hosts:
            assert host not in shipping

    # Operator-copyable validator Compose env template: joinbase-only + explicit
    # --master-url and local loopback smoke. Must never reintroduce preferred-
    # product / cutover wording for chain.platform.network.
    assert expected in env_validator_example
    assert "--master-url" in env_validator_example
    assert "http://127.0.0.1:3180" in env_validator_example
    assert "never run master" in env_validator_example.lower()
    assert "chain.platform.network" not in env_validator_example
    assert "preferred product" not in env_validator_example.lower()
    assert "after cutover" not in env_validator_example.lower()
    for host in retired_public_hosts:
        assert host not in env_validator_example

    # Master install renders both master + validator registry defaults to public chain.
    assert install_master.count(f"registry_url: {expected}") >= 2
    assert "weights_url: null" in install_master

    # Validator install is agent-only, requires explicit --master-url, and never
    # invents a hard-coded public IP or platform.network default. Generated
    # registry/weights follow --master-url when master hosts both.
    assert "--master-url" in install_validator
    assert "VALIDATOR_MASTER_URL" in install_validator
    assert "validator install requires --master-url" in install_validator
    assert (
        "NEVER run master" in install_validator
        or "never run master" in install_validator
    )
    assert "agent-only" in install_validator
    assert "master_url: ${MASTER_URL}" in install_validator
    assert "registry_url: ${MASTER_URL}" in install_validator
    assert "weights_url: ${MASTER_URL}" in install_validator
    assert expected in install_validator
    assert "chain.platform.network" not in install_validator
    assert "preferred product" not in install_validator.lower()
    for host in retired_public_hosts:
        assert host not in install_validator
    # No default IP master for empty --master-url.
    assert (
        re.search(r'MASTER_URL="\$\{VALIDATOR_MASTER_URL:-[^"]+\}"', install_validator)
        is None
    )

    assert expected in validator_docs
    assert f"{expected}/v1/weights/latest" in validator_docs
    assert expected in ops_validator_docs
    assert expected in compose_docs
    assert expected in deploy_docs
    assert expected in readme
    # Shipping docs + operator-copyable env examples are joinbase-only:
    # no preferred-product / cutover framing.
    for body in (
        validator_docs,
        ops_validator_docs,
        compose_docs,
        deploy_docs,
        readme,
        install_validator,
        install_master,
        env_validator_example,
    ):
        assert "chain.platform.network" not in body
        assert "preferred product" not in body.lower()
    assert (
        "never run master" in validator_docs.lower()
        or "never run master" in compose_docs.lower()
    )
    # Local disposable-master smoke may use loopback in YAML or CLI form
    # (weight-only install-validator uses --master-url).
    assert (
        "master_url: http://127.0.0.1:3180" in ops_validator_docs
        or "--master-url http://127.0.0.1:3180" in ops_validator_docs
    )


def test_master_url_role_distinguished_from_registry_aliases() -> None:
    """master_url is coordination API; registry/weights are separate aliases."""
    root = Path(__file__).resolve().parents[2]
    install_validator = (
        root / "deploy" / "compose" / "install-validator.sh"
    ).read_text(encoding="utf-8")
    compose_docs = (root / "docs" / "compose.md").read_text(encoding="utf-8")
    validator_docs = (root / "docs" / "validator" / "README.md").read_text(
        encoding="utf-8"
    )

    # Clear distinction across installer help + docs.
    for body in (install_validator, compose_docs, validator_docs):
        assert "master_url" in body
        assert "registry_url" in body or "registry" in body.lower()
        assert "coordination" in body.lower() or "Base master" in body

    # Settings defaults remain the public joinbase master API only.
    assert MasterSettings().registry_url == "https://chain.joinbase.ai"
    assert ValidatorSettings().registry_url == "https://chain.joinbase.ai"
    assert "platform.network" not in MasterSettings().registry_url


def test_registry_facing_defaults_docs_and_examples_do_not_use_rpc_endpoint() -> None:
    root = Path(__file__).resolve().parents[2]
    registry_facing_files = [
        root / "src" / "base" / "config" / "settings.py",
        root / "config" / "master.example.yaml",
        root / "config" / "validator.example.yaml",
        root / "docs" / "validator" / "README.md",
        root / "deploy" / "compose" / "config" / "master.compose.yaml",
        root / "deploy" / "compose" / "install-master.sh",
        root / "deploy" / "swarm" / "master.yaml",
    ]

    retired_rpc_host = ".".join(["rpc", "platform", "network"])
    retired_rpc_base_url = "https://" + retired_rpc_host
    retired_registry_url_path = retired_rpc_host + "/v1/registry"

    for registry_facing_file in registry_facing_files:
        content = registry_facing_file.read_text(encoding="utf-8")
        assert retired_rpc_base_url not in content
        assert retired_registry_url_path not in content


def test_token_hash_verify() -> None:
    token = generate_token()
    assert verify_token(token, hash_token(token))
    assert not verify_token("wrong", hash_token(token))


def test_load_settings_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("network:\n  netuid: 42\n", encoding="utf-8")
    assert load_settings(config).network.netuid == 42


def test_load_settings_parses_complex_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "BASE_DOCKER__BROKER_ALLOWED_IMAGES",
        '["ghcr.io/baseintelligence/"]',
    )

    assert load_settings().docker.broker_allowed_images == ["ghcr.io/baseintelligence/"]


def test_render_challenge_template(tmp_path: Path) -> None:
    out = tmp_path / "challenge"
    files = render_challenge_template(
        out, ChallengeTemplateContext.from_slug("demo-challenge")
    )
    assert Path("pyproject.toml") in files
    assert Path("Dockerfile") in files
    assert Path("src/demo_challenge/sdk/executors/docker.py") in files
    assert (out / "src" / "demo_challenge" / "app.py").exists()
    assert (out / "src" / "demo_challenge" / "sdk" / "executors" / "docker.py").exists()
    assert "docker-cli" in (out / "Dockerfile").read_text(encoding="utf-8")


def test_production_settings_require_postgres_safe_prefixes_and_tls(
    tmp_path: Path,
) -> None:
    config = tmp_path / "prod.yaml"
    secret = tmp_path / "admin_token"
    secret.write_text("test-admin-token\n", encoding="utf-8")
    secret.chmod(0o600)
    config.write_text(
        "\n".join(
            [
                "environment: production",
                "database:",
                "  url: postgresql+asyncpg://base:secret@postgres:5432/platform",
                "security:",
                f"  admin_token_file: {secret}",
                "docker:",
                "  broker_allowed_images:",
                "    - ghcr.io/baseintelligence/",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.environment == "production"
    assert settings.database.url.startswith("postgresql+asyncpg://")


def test_production_settings_reject_sqlite_broad_prefixes_and_insecure_tls(
    tmp_path: Path,
) -> None:
    config = tmp_path / "bad-prod.yaml"
    config.write_text(
        "\n".join(
            [
                "environment: production",
                "database:",
                "  url: sqlite+aiosqlite:////tmp/base.sqlite3",
                "docker:",
                "  broker_allowed_images:",
                "    - baseintelligence/",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PostgreSQL|external PostgreSQL"):
        load_settings(config)

    config.write_text(
        "\n".join(
            [
                "environment: production",
                "database:",
                "  url: postgresql+asyncpg://user:pass@postgres.platform/platform",
                "docker:",
                "  broker_allowed_images:",
                "    - baseintelligence/",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="too broad"):
        load_settings(config)


def _docker_stub(bin_dir: Path) -> None:
    """`docker` stub: recent engine, INACTIVE swarm, every `inspect` MISSES.

    A missed inspect makes the installer *plan* (print) every create, so the full
    plan (including the rendered master config) is exercised in dry-run.
    """
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


def _run_installer_dry_run(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _docker_stub(bin_dir)
    env = dict(os.environ)
    env.update(_INSTALLER_REQUIRED_ENV)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["BROKER_WORKSPACE_DIR"] = str(tmp_path / "broker-ws")
    env["MASTER_CONFIG_PATH"] = str(tmp_path / "master.yaml")
    env["GATEWAY_PUBLIC_BASE_URL"] = "http://master.example:19080"
    return subprocess.run(
        [
            "bash",
            str(SWARM_INSTALLER),
            "--backup-dir",
            str(tmp_path / "missing"),
            "--greenfield",
            "--static-challenges",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _extract_rendered_master_config(stdout: str) -> str:
    """Reconstruct the dry-run-rendered master config YAML from installer stdout.

    The dry-run prints the staged config indented under a marker line. Rather than
    assume a fixed indent width, detect the indent from the first config line and
    collect the contiguous indented block, stripping that indent back off. This
    survives a change to the installer's presentation (e.g. the ``sed`` prefix
    width) as long as the marker and an indented block remain.
    """
    lines = stdout.splitlines()
    try:
        start = next(
            i
            for i, line in enumerate(lines)
            if "master config that WOULD be written to" in line
        )
    except StopIteration as exc:  # pragma: no cover - guards a presentation change
        raise AssertionError(
            "installer dry-run did not print the rendered master config marker"
        ) from exc

    block = lines[start + 1 :]
    indent = next(
        (len(line) - len(line.lstrip(" ")) for line in block if line.strip()),
        0,
    )
    prefix = " " * indent
    body: list[str] = []
    for line in block:
        if line.strip() == "":
            body.append("")
        elif line.startswith(prefix):
            body.append(line[indent:])
        else:
            break
    return "\n".join(body)


def test_rendered_master_config_is_production_and_passes_policy(
    tmp_path: Path,
) -> None:
    """H3 guard (VAL-HARD-ENV-001): the installer-rendered master config forces
    ``environment: production`` (so the image-pin/TLS/Postgres/broker-allowlist
    policy guards activate) AND that config passes ``validate_settings_policy``.

    Proves the narrowed ``broker_allowed_images`` is accepted by the production
    ``validate_allowed_image_prefixes`` and the external-Postgres DB URL passes.
    """
    result = _run_installer_dry_run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"

    rendered = _extract_rendered_master_config(result.stdout)
    data = yaml.safe_load(rendered)

    # The rendered config forces production (a development default would leave the
    # policy guards inert on the live box).
    assert data["environment"] == "production"
    assert is_production_environment(data["environment"])

    # The narrowed broker allowlist is namespaced+repo (NOT the too-broad whole
    # 'ghcr.io/baseintelligence/' namespace the prod policy rejects).
    allowed = data["docker"]["broker_allowed_images"]
    assert allowed == [
        "ghcr.io/baseintelligence/agent-challenge",
        "ghcr.io/baseintelligence/prism",
    ]
    assert "ghcr.io/baseintelligence/" not in allowed

    # The full rendered config loads into Settings (which runs the policy model
    # validator) AND passes validate_settings_policy explicitly — under production.
    # Build directly from the parsed mapping so ambient BASE_* env cannot perturb
    # the assertion (the model validator fires inside model_validate).
    settings = Settings.model_validate(data)
    assert settings.environment == "production"
    validate_settings_policy(settings)


def test_installer_default_image_refs_are_tag_and_digest_pinned() -> None:
    """Production image-reference policy form: each rendered IMAGE_* default
    carries BOTH a tag and an ``@sha256:`` digest (validate_image_reference).

    Covers the prism GPU evaluator (``IMAGE_PRISM_EVALUATOR``) too: it is pinned
    into the rendered prism challenge service (``PRISM_BASE_EVAL_IMAGE``) and must
    satisfy the same production image-reference policy as the other images.
    """
    from base.config.policy import validate_image_reference

    text = SWARM_INSTALLER.read_text(encoding="utf-8")
    image_defaults: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        for var in (
            "IMAGE_MASTER",
            "IMAGE_AGENT_CHALLENGE",
            "IMAGE_PRISM_EVALUATOR",
            "IMAGE_PRISM",
        ):
            prefix = f'{var}="${{{var}:-'
            if stripped.startswith(prefix):
                image_defaults[var] = stripped[len(prefix) :].rstrip('}"')
                break

    assert set(image_defaults) == {
        "IMAGE_MASTER",
        "IMAGE_AGENT_CHALLENGE",
        "IMAGE_PRISM",
        "IMAGE_PRISM_EVALUATOR",
    }
    for var, ref in image_defaults.items():
        assert ":" in ref.split("@")[0].rsplit("/", 1)[-1], f"{var} missing tag: {ref}"
        assert "@sha256:" in ref, f"{var} missing digest: {ref}"
        # Production policy accepts the rendered ref (tag is latest/semver; sha256).
        validate_image_reference(ref, production=True)
