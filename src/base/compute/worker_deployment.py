"""Declarative worker-image deploy definitions for both compute providers.

The miner deploys the (future) worker image into a GPU instance they pay for. The
image is referenced the same way everywhere: BY DIGEST, so a rented pod runs the
exact bytes we published. This module builds the two declarative definitions that
carry that pin:

* :func:`build_lium_worker_template` -- a Lium ``CustomTemplateRequest`` payload
  for ``POST /templates`` (architecture.md sec 3.2 / lium-api.md), and
* :func:`build_targon_worker_app` -- a Targon app definition referencing the same
  pinned image with an explicit GPU resource shape.

For M1 the image defaults to the already-published ``prism-evaluator`` image
pinned by digest (a placeholder standing in for M2's ``docker/Dockerfile.worker``
image). Both builders take ``image``/``image_digest`` as inputs so M2 swaps in the
worker image with a one-line change; nothing else about the definitions moves.

The definitions embed NO credentials: environment plumbing is caller-provided and
carries only non-secret configuration (master URL, worker role, ...).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

# M1 placeholder: the published prism-evaluator image, pinned by digest. This is
# the same digest the swarm deploy pins (deploy/swarm/install-swarm.sh). M2 swaps
# these two constants for docker/Dockerfile.worker's published image + digest.
WORKER_IMAGE = "ghcr.io/baseintelligence/prism-evaluator"
WORKER_IMAGE_TAG = "latest"
WORKER_IMAGE_DIGEST = (
    "sha256:713b39f13af69dbaf229e67fb682df8a2b7ac93dd02d9e60867ff021d4edb3c9"
)

WORKER_TEMPLATE_NAME = "prism-worker"
WORKER_APP_NAME = "prism-worker"

# SSH (22) is mandatory so the miner can reach the pod; the worker's local broker
# is bound to loopback inside the instance and needs no external mapping.
WORKER_INTERNAL_PORTS: tuple[int, ...] = (22,)

# Metachar-free keep-alive for the Lium template startup_commands. Lium rejects a
# rent whose template startup_commands contain shell metacharacters ("Malicious
# startup command detected"; live-confirmed, library/lium-api.md), so the value
# MUST NOT contain '&&', ';', or '|'. The container's own entrypoint launches the
# agent; this keep-alive only guarantees the pod stays up (Lium's pod agent
# provides SSH independently of the container command).
WORKER_STARTUP_COMMANDS = "tail -f /dev/null"

# Shell metacharacters Lium's rent guard rejects in a template startup command.
_SHELL_METACHARACTERS: tuple[str, ...] = ("&&", "||", ";", "|", "&", "`", "$(")

# Default Targon GPU resource shape (inventory shape id + human GPU type). Real
# Targon inventory ids carry a size suffix (h100-small, b200-large per
# library/targon-api.md); a bare 'h100' would be rejected by a live deploy.
WORKER_GPU_SHAPE = "h100-small"
WORKER_GPU_TYPE = "H100"

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def is_pinned_digest(value: str) -> bool:
    """Return ``True`` if ``value`` is a well-formed ``sha256:<64-hex>`` digest."""
    return bool(_DIGEST_RE.match(value))


def is_metachar_free(command: str) -> bool:
    """Return ``True`` if ``command`` carries no shell metacharacter Lium rejects.

    Lium's rent guard refuses a template startup command containing shell
    metacharacters (``&&``, ``;``, ``|`` ...); a compliant keep-alive is a single
    plain command such as ``tail -f /dev/null``.
    """
    return not any(token in command for token in _SHELL_METACHARACTERS)


def _require_metachar_free(command: str) -> str:
    if not is_metachar_free(command):
        raise ValueError(
            "startup_commands must be a single metachar-free command (no "
            "'&&', ';', '|', ...): Lium rejects shell metacharacters"
        )
    return command


def _require_digest(image_digest: str) -> str:
    if not is_pinned_digest(image_digest):
        raise ValueError(
            "image_digest must be a pinned digest of the form 'sha256:<64 hex>'"
        )
    return image_digest


def pinned_image_reference(image: str, digest: str, *, tag: str | None = None) -> str:
    """Return a fully qualified, digest-pinned image reference.

    With a tag: ``<image>:<tag>@<digest>``; without: ``<image>@<digest>``. The
    ``@<digest>`` suffix is what makes the reference immutable.
    """
    _require_digest(digest)
    if tag:
        return f"{image}:{tag}@{digest}"
    return f"{image}@{digest}"


def build_lium_worker_template(
    *,
    image: str = WORKER_IMAGE,
    image_digest: str = WORKER_IMAGE_DIGEST,
    image_tag: str | None = WORKER_IMAGE_TAG,
    name: str = WORKER_TEMPLATE_NAME,
    environment: Mapping[str, str] | None = None,
    internal_ports: Sequence[int] = WORKER_INTERNAL_PORTS,
    entrypoint: str = "",
    startup_commands: str = WORKER_STARTUP_COMMANDS,
    is_private: bool = True,
    container_start_immediately: bool = True,
) -> dict[str, Any]:
    """Build the Lium ``CustomTemplateRequest`` payload for the worker image.

    The returned dict is the exact JSON body for ``POST /templates``. ``image``
    is pinned by ``image_digest`` (validated to be ``sha256:<64 hex>``) and the
    internal ports MUST include 22 (SSH). ``startup_commands`` is validated to be
    metachar-free (Lium rejects shell metacharacters at rent time), defaulting to
    a plain keep-alive so the pod stays up while the image entrypoint runs the
    agent.
    """
    _require_digest(image_digest)
    _require_metachar_free(startup_commands)
    ports = [int(port) for port in internal_ports]
    if 22 not in ports:
        raise ValueError("internal_ports must include 22 (SSH)")
    return {
        "name": name,
        "docker_image": image,
        "docker_image_tag": image_tag or "",
        "docker_image_digest": image_digest,
        "environment": dict(environment or {}),
        "entrypoint": entrypoint,
        "startup_commands": startup_commands,
        "internal_ports": ports,
        "is_private": is_private,
        "container_start_immediately": container_start_immediately,
    }


def build_targon_worker_app(
    *,
    image: str = WORKER_IMAGE,
    image_digest: str = WORKER_IMAGE_DIGEST,
    image_tag: str | None = WORKER_IMAGE_TAG,
    name: str = WORKER_APP_NAME,
    gpu_shape: str = WORKER_GPU_SHAPE,
    gpu_type: str = WORKER_GPU_TYPE,
    gpu_count: int = 1,
    environment: Mapping[str, str] | None = None,
    ports: Sequence[int] = WORKER_INTERNAL_PORTS,
    min_replicas: int = 0,
    max_replicas: int = 1,
) -> dict[str, Any]:
    """Build the Targon app definition for the worker image.

    The app references a fully qualified, digest-pinned image and declares its GPU
    resource shape (``resource`` inventory shape name + ``gpu_type``/``gpu_count``).
    Environment is plumbed as Targon ``{"name", "value"}`` pairs and carries no
    secrets.
    """
    _require_digest(image_digest)
    if gpu_count < 1:
        raise ValueError("gpu_count must be >= 1")
    return {
        "name": name,
        "image": pinned_image_reference(image, image_digest, tag=image_tag),
        "image_digest": image_digest,
        "resource": gpu_shape,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "min_replicas": min_replicas,
        "max_replicas": max_replicas,
        "envs": [{"name": k, "value": v} for k, v in dict(environment or {}).items()],
        "ports": [int(port) for port in ports],
    }
