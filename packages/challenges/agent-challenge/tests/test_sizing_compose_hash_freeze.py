"""Freeze compose hashes so sizing changes cannot move measured documents.

Sizing (instance_type / disk_size) must stay OUTSIDE the measured app-compose
documents. These constants were measured from the generators BEFORE any sizing
edit on branch feat/agent-challenge-cvm-sizing.
"""

from __future__ import annotations

from agent_challenge.canonical.compose import app_compose_hash, generate_app_compose
from agent_challenge.review.compose import generate_review_app_compose, review_app_compose_hash

# Measured on clean HEAD 263eeb1b before sizing edits (same fixtures as
# tests/test_canonical_compose.py and default review moniker).
_CANONICAL_IMAGE = "ghcr.io/baseintelligence/agent-challenge-canonical@sha256:" + ("a" * 64)
_REVIEW_IMAGE = "ghcr.io/baseintelligence/agent-challenge-review@sha256:" + ("b" * 64)

FREEZE_EVAL_APP_COMPOSE_HASH = (
    "f8f05273959469a2b8eb3e599863cdb2ddc7c741055d1e6830f101fc2e79d334"
)
FREEZE_REVIEW_APP_COMPOSE_HASH = (
    "9ef4435f4bd3e938f371c93c5ee8076fabf16b75a1ea18f3bdb9c0e24176325f"
)


def test_eval_app_compose_hash_frozen_against_sizing_work():
    compose = generate_app_compose(orchestrator_image=_CANONICAL_IMAGE)
    assert app_compose_hash(compose) == FREEZE_EVAL_APP_COMPOSE_HASH
    # Sizing keys must never appear inside the measured document.
    blob = str(compose)
    assert "disk_size" not in blob
    assert "instance_type" not in blob
    assert "tdx.xlarge" not in blob


def test_review_app_compose_hash_frozen_against_sizing_work():
    compose = generate_review_app_compose(review_image=_REVIEW_IMAGE)
    assert review_app_compose_hash(compose) == FREEZE_REVIEW_APP_COMPOSE_HASH
    blob = str(compose)
    assert "disk_size" not in blob
    assert "instance_type" not in blob
    assert "tdx.xlarge" not in blob
