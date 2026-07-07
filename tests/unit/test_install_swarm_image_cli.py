"""Turnkey installer — ephemeral CLI image vs pinned service image split.

The turnkey validator installer (``deploy/swarm/install-swarm.sh``) runs several
EPHEMERAL one-shot ``base-master`` CLI containers during install: central-gate
token mint (``_auto_mint_central_gate_token``), validator wallet generation
(``ensure_validator_wallet``), and runtime-uid inspect (``_master_runtime_uid``).
Those MUST use the MUTABLE ``:latest`` tag (``IMAGE_MASTER_CLI``) so newly-added
CLI subcommands (e.g. ``base master mint-central-gate-token``) exist — a stale
digest pin drops them ("No such command"). The long-running DEPLOYED SERVICES
(``base-master-proxy`` + ``base-docker-broker``, created by
``_deploy_master_service``) keep the DIGEST-PINNED ``IMAGE_MASTER`` (the
supervisor image-updater rolls them to ``:latest`` afterwards). This
ephemeral-vs-service split is the load-bearing invariant.

Additionally the ephemeral mint + wallet-gen runs bind-mount root-owned mode-600
temp files, so both run ``--user 0:0`` (the image default user is non-root and
cannot read them → "Path '/mint.yaml' is not readable").

These regression guards parse the shell script as text (no docker execution) and
fail loudly if the ephemeral-vs-service split, the digest pin, or the
``--user 0:0`` root flag ever regress.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTALL_SWARM = ROOT / "deploy" / "swarm" / "install-swarm.sh"

# The two literal shell image refs. IMAGE_MASTER_CLI is a superset string of
# IMAGE_MASTER, so ``SERVICE_IMAGE_REF not in body`` is a valid "not the pinned
# ref" check (``${IMAGE_MASTER}`` is NOT a substring of ``${IMAGE_MASTER_CLI}``).
CLI_IMAGE_REF = "${IMAGE_MASTER_CLI}"
SERVICE_IMAGE_REF = "${IMAGE_MASTER}"


def _installer_text() -> str:
    return INSTALL_SWARM.read_text(encoding="utf-8")


def _image_var_default(var: str) -> str:
    """Return the ``:-DEFAULT`` value of a top-level ``VAR="${VAR:-DEFAULT}"`` line."""

    match = re.search(
        r"^" + re.escape(var) + r'="\$\{' + re.escape(var) + r':-([^}]+)\}"',
        _installer_text(),
        re.MULTILINE,
    )
    assert match is not None, f"could not find {var} image default assignment"
    return match.group(1)


def _function_body(name: str) -> str:
    """Return the body of the bash ``name() { ... }`` function.

    The installer defines top-level functions with the closing brace in column 0,
    so the body is everything between the opening ``{`` and the first line-start
    ``}`` (no nested column-0 ``}`` occurs inside these functions).
    """

    match = re.search(
        r"^" + re.escape(name) + r"\(\)\s*\{\n(.*?)\n\}",
        _installer_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"could not locate bash function {name!r} in installer"
    return match.group(1)


# --- (a) IMAGE_MASTER_CLI is the MUTABLE :latest tag (no digest pin) ----------


def test_image_master_cli_defaults_to_mutable_latest_tag() -> None:
    default = _image_var_default("IMAGE_MASTER_CLI")
    assert default == "ghcr.io/baseintelligence/base-master:latest"
    # A digest pin here would drop newly-added CLI subcommands on the ephemeral runs.
    assert "@sha256:" not in default


# --- (b) IMAGE_MASTER (deployed service image) STAYS digest-pinned ------------


def test_image_master_service_ref_stays_digest_pinned() -> None:
    default = _image_var_default("IMAGE_MASTER")
    # Regression guard: the service image ref must NOT be accidentally unpinned.
    assert default.startswith("ghcr.io/baseintelligence/base-master:latest@sha256:")
    assert re.search(r"@sha256:[0-9a-f]{64}$", default), default


# --- (c) mint runs the CLI image as root -------------------------------------


def test_mint_docker_run_uses_cli_image_as_root() -> None:
    body = _function_body("_auto_mint_central_gate_token")
    # docker run … --user 0:0 … ${IMAGE_MASTER_CLI} … mint-central-gate-token
    assert re.search(
        r"docker\s+run\b.*?--user\s+0:0.*?"
        + re.escape(CLI_IMAGE_REF)
        + r".*?mint-central-gate-token",
        body,
        re.DOTALL,
    ), body
    # the pinned service image is NEVER the ephemeral CLI run target
    assert SERVICE_IMAGE_REF not in body


# --- (d) wallet-gen runs the CLI image as root -------------------------------


def test_wallet_gen_docker_run_uses_cli_image_as_root() -> None:
    body = _function_body("ensure_validator_wallet")
    # docker run … --user 0:0 … --entrypoint python3 … ${IMAGE_MASTER_CLI}
    assert re.search(
        r"docker\s+run\b.*?--user\s+0:0.*?--entrypoint\s+python3.*?"
        + re.escape(CLI_IMAGE_REF),
        body,
        re.DOTALL,
    ), body
    # the wallet-gen ephemeral run never targets the pinned service image
    assert SERVICE_IMAGE_REF not in body


# --- (e) runtime-uid inspect targets the CLI image ---------------------------


def test_master_runtime_uid_inspects_cli_image() -> None:
    body = _function_body("_master_runtime_uid")
    assert "docker image inspect" in body
    assert CLI_IMAGE_REF in body
    assert SERVICE_IMAGE_REF not in body


# --- (f) deployed services keep the PINNED image (ephemeral-vs-service split) -


def test_deploy_master_service_uses_pinned_image_not_cli() -> None:
    body = _function_body("_deploy_master_service")
    # long-running services run the DIGEST-PINNED image, never the mutable CLI tag
    assert SERVICE_IMAGE_REF in body
    assert CLI_IMAGE_REF not in body


def test_broker_and_proxy_are_deployed_via_deploy_master_service() -> None:
    # Both long-running master services go through _deploy_master_service, so they
    # inherit the pinned ${IMAGE_MASTER} asserted above (not the CLI tag).
    text = _installer_text()
    assert re.search(r'_deploy_master_service\s+"base-docker-broker"\s+"broker"', text)
    assert re.search(r'_deploy_master_service\s+"base-master-proxy"\s+"proxy"', text)
