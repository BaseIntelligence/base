"""Shipping AC runtime image must package Phala pre-launch helper.

Residual class (live dual flags, pin pack complete):
  ``image_package_prelaunch_script_missing``

``create_review_session`` / offline review+eval compose generators resolve
``REPO_ROOT / docker/review/phala_pre_launch.sh``.  With WORKDIR ``/app`` and
``PYTHONPATH=/app/src``, ``REPO_ROOT`` inherits as ``/app``, so the shipping
runtime image must contain ``/app/docker/review/phala_pre_launch.sh`` (the
checked-in vendor helper, not a reinvented trust root).
"""

from __future__ import annotations

import re
from pathlib import Path

from agent_challenge.canonical import compose as eval_compose
from agent_challenge.review import compose as review_compose

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"
VENDOR_SCRIPT = REPO_ROOT / "docker" / "review" / "phala_pre_launch.sh"
EXPECTED_IMAGE_PATH = "/app/docker/review/phala_pre_launch.sh"


def _runtime_stage_text(dockerfile: str) -> str:
    """Extract the ``runtime`` stage (before ``terminal-bench-runner``)."""

    m = re.search(
        r"FROM\s+\S+\s+AS\s+runtime\b(.*?)(?=\nFROM\s|\Z)",
        dockerfile,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert m is not None, "Dockerfile missing runtime stage"
    return m.group(1)


def test_vendor_phala_pre_launch_script_is_checked_in():
    assert VENDOR_SCRIPT.is_file()
    text = VENDOR_SCRIPT.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash")
    assert "Phala Cloud Pre-Launch Script" in text


def test_repo_root_paths_align_for_image_layout():
    """Compose generators and the server runtime share the same relative path."""

    assert review_compose.PHALA_PRE_LAUNCH_SCRIPT_PATH == (
        review_compose.REPO_ROOT / "docker" / "review" / "phala_pre_launch.sh"
    )
    assert eval_compose.PHALA_PRE_LAUNCH_SCRIPT_PATH == (
        eval_compose.REPO_ROOT / "docker" / "review" / "phala_pre_launch.sh"
    )
    # Under the shipping container layout REPO_ROOT must resolve to /app, so
    # relative docker/review/phala_pre_launch.sh becomes EXPECTED_IMAGE_PATH.
    rel = Path("docker") / "review" / "phala_pre_launch.sh"
    assert (review_compose.REPO_ROOT / rel) == review_compose.PHALA_PRE_LAUNCH_SCRIPT_PATH
    assert str(rel).replace("\\", "/") == "docker/review/phala_pre_launch.sh"
    assert f"/app/{rel.as_posix()}" == EXPECTED_IMAGE_PATH


def test_runtime_dockerfile_copies_phala_pre_launch_to_expected_path():
    """Fail closed if the published runtime image omits the pre-launch helper."""

    assert DOCKERFILE.is_file()
    body = DOCKERFILE.read_text(encoding="utf-8")
    runtime = _runtime_stage_text(body)

    # Exact destination used on prod: /app/docker/review/phala_pre_launch.sh
    # Accept either absolute dest or relative dest under WORKDIR /app.
    copy_to_expected = (
        re.search(
            r"^\s*COPY\s+(?:--chown=\S+\s+)?docker/review/phala_pre_launch\.sh\s+"
            r"(?:/app/)?docker/review/phala_pre_launch\.sh\s*$",
            runtime,
            flags=re.MULTILINE,
        )
        is not None
    )
    # Also accept a recursive copy of docker/review if destination keeps layout.
    recursive_review = (
        re.search(
            r"^\s*COPY\s+(?:--chown=\S+\s+)?docker/review(?:/\s+|\s+)"
            r"(?:/app/)?docker/review(?:/)?\s*$",
            runtime,
            flags=re.MULTILINE,
        )
        is not None
    )
    assert copy_to_expected or recursive_review, (
        "runtime Dockerfile must COPY docker/review/phala_pre_launch.sh into "
        f"{EXPECTED_IMAGE_PATH} (source of residual_class="
        "image_package_prelaunch_script_missing)"
    )


def test_runtime_dockerfile_chowns_app_after_prelaunch_copy():
    """Challenge user must own the script after packaging."""

    body = DOCKERFILE.read_text(encoding="utf-8")
    runtime = _runtime_stage_text(body)
    copy_idx = runtime.find("phala_pre_launch.sh")
    chown_idx = runtime.find("chown -R challenge:challenge /app")
    assert copy_idx != -1, "pre-launch COPY missing from runtime stage"
    assert chown_idx != -1, "chown of /app missing from runtime stage"
    assert copy_idx < chown_idx, "COPY pre-launch script must precede chown of /app"
