"""Behavioral tests for the generated Phala app-compose (M2, VAL-ORCH-032/033/034).

Covers, offline:
  * VAL-ORCH-032 orchestrator-only compose; task images pinned by digest via the
    golden manifest and NOT declared as static per-task services.
  * VAL-ORCH-033 the generated compose contains no secrets.
  * VAL-ORCH-034 generation is deterministic; the compose-hash is stable; and the
    deployable bytes equal ``normalize_app_compose`` verbatim (so the offline
    hash matches the value dstack measures on the live CVM).
The live deploy/validator + RTMR3 correlation is VAL-ORCH-031 / the live half of
VAL-ORCH-034 (manual-cvm / phala, exercised at M6).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from agent_challenge.canonical import compose as c
from agent_challenge.canonical import measurement as m

CANONICAL_IMAGE = "ghcr.io/baseintelligence/agent-challenge-canonical@sha256:" + ("a" * 64)
GOLDEN_MANIFEST = Path(__file__).resolve().parents[1] / "golden" / "dataset-digest.json"


def _compose(**kwargs):
    return c.generate_app_compose(orchestrator_image=CANONICAL_IMAGE, **kwargs)


def _docker_compose(compose) -> dict:
    return yaml.safe_load(compose["docker_compose_file"])


# --------------------------------------------------------------------------- #
# VAL-ORCH-032: orchestrator-only; task images pinned by digest, not services
# --------------------------------------------------------------------------- #
def test_generated_compose_declares_only_the_orchestrator_service():
    compose = _compose()
    services = _docker_compose(compose)["services"]
    assert set(services) == {c.ORCHESTRATOR_SERVICE}


def test_generated_compose_has_no_per_task_static_services():
    manifest = c.load_golden_manifest(GOLDEN_MANIFEST)
    task_ids = set(manifest["tasks"])
    services = _docker_compose(_compose())["services"]
    # None of the ~89 Terminal-Bench task ids appear as a compose service.
    assert task_ids.isdisjoint(set(services))
    assert len(task_ids) >= 80  # sanity: the manifest really has the task set


def test_golden_manifest_task_images_are_digest_pinned():
    manifest = c.load_golden_manifest(GOLDEN_MANIFEST)
    pins = c.golden_task_image_digests(manifest)
    assert set(pins) == set(manifest["tasks"])
    for task_id, ref in pins.items():
        assert ref.startswith("sha256:"), (task_id, ref)
        assert len(ref) == len("sha256:") + 64


def test_golden_manifest_rejects_floating_task_digest():
    bad = {"tasks": {"t1": {"harbor_registry_ref": "latest"}}}
    with pytest.raises(c.ComposeGenerationError):
        c.golden_task_image_digests(bad)


def test_orchestrator_image_must_be_digest_pinned():
    with pytest.raises(c.ComposeGenerationError):
        c.generate_app_compose(orchestrator_image="ghcr.io/base/agent-challenge:latest")


def test_orchestrator_is_not_privileged_and_starts_no_inner_dockerd():
    service = _docker_compose(_compose())["services"][c.ORCHESTRATOR_SERVICE]
    assert "privileged" not in service
    # DooD: the guest docker socket is bind-mounted; no inner dockerd command.
    joined = json.dumps(service)
    assert "dockerd" not in joined
    assert any(vol.startswith("/var/run/docker.sock:") for vol in service["volumes"])
    assert any(vol.startswith("/var/run/dstack.sock:") for vol in service["volumes"])


def test_orchestrator_does_not_bind_mount_over_image_golden_or_task_cache():
    """Image-baked assets must stay visible; empty guest binds would hide them."""

    service = _docker_compose(_compose())["services"][c.ORCHESTRATOR_SERVICE]
    volumes = list(service.get("volumes") or [])
    assert all(not any(token in vol for token in ("/golden", "task-cache")) for vol in volumes), (
        volumes
    )
    assert any(vol.startswith("/var/run/docker.sock:") for vol in volumes)
    assert any(vol.startswith("/var/run/dstack.sock:") for vol in volumes)


# --------------------------------------------------------------------------- #
# VAL-ORCH-033: no secrets in the generated compose
# --------------------------------------------------------------------------- #
def test_generated_compose_contains_no_secret_values():
    # Sentinel secret VALUES of every class the scan must never find in the bytes.
    sentinels = [
        "phak_" + "d" * 32,  # Phala API key
        "ghp_" + "e" * 36,  # provider/GitHub-style token
        "sk-" + "f" * 40,  # provider API key
        "super-secret-gateway-token-value",  # gateway token value
        "miner-private-env-value",  # miner-env value
    ]
    blob = c.render_app_compose(_compose())
    for sentinel in sentinels:
        assert sentinel not in blob, sentinel


def test_base_gateway_names_are_not_in_eval_compose_environment():
    """VAL-ACAT-013: Base gateway secrets are absent from measured eval compose."""

    service = _docker_compose(_compose())["services"][c.ORCHESTRATOR_SERVICE]
    env = service["environment"]
    assert "BASE_GATEWAY_TOKEN" not in env
    assert "BASE_LLM_GATEWAY_URL" not in env
    # Required eval run token name may be present for encrypted_env injection
    # as a name only (never NAME=value).
    if "EVAL_RUN_TOKEN" in env:
        assert not any(e.startswith("EVAL_RUN_TOKEN=") for e in env)


def test_no_provider_api_key_or_phala_key_names_leak_as_values():
    service = _docker_compose(_compose())["services"][c.ORCHESTRATOR_SERVICE]
    # Only non-secret static config carries a value; no *_API_KEY value appears.
    for entry in service["environment"]:
        if "=" in entry:
            name, value = entry.split("=", 1)
            assert not name.endswith("_API_KEY")
            assert "phak_" not in value


def test_allowed_envs_are_names_only():
    compose = _compose()
    for name in compose["allowed_envs"]:
        assert "=" not in name  # a NAME, never NAME=value
        assert name == name.strip()


# --------------------------------------------------------------------------- #
# VAL-ORCH-034: deterministic generation + stable, matching compose-hash
# --------------------------------------------------------------------------- #
def test_generation_is_byte_identical_for_same_inputs():
    a = c.render_app_compose(_compose())
    b = c.render_app_compose(_compose())
    assert a == b


def test_deployable_bytes_equal_normalize_app_compose_verbatim():
    # CRITICAL contract (library/measurement-tooling.md): the deployed bytes MUST
    # equal normalize_app_compose output verbatim (no separate re-serialization).
    compose = _compose()
    assert c.render_app_compose(compose) == m.normalize_app_compose(compose)
    assert c.render_app_compose_bytes(compose) == m.normalize_app_compose(compose).encode("utf-8")


def test_compose_hash_matches_measurement_and_sha256_of_bytes():
    compose = _compose()
    expected = hashlib.sha256(c.render_app_compose_bytes(compose)).hexdigest()
    assert c.app_compose_hash(compose) == expected
    assert c.app_compose_hash(compose) == m.compose_hash(compose)
    assert len(c.app_compose_hash(compose)) == 64


def test_material_change_changes_hash_and_revert_restores():
    baseline = c.app_compose_hash(_compose())

    other_image = "ghcr.io/baseintelligence/agent-challenge-canonical@sha256:" + ("b" * 64)
    changed = c.app_compose_hash(c.generate_app_compose(orchestrator_image=other_image))
    assert changed != baseline

    # Reverting to the original inputs reproduces the baseline hash exactly.
    assert c.app_compose_hash(_compose()) == baseline


def test_changing_allowed_envs_changes_hash():
    baseline = c.app_compose_hash(_compose())
    fewer = c.app_compose_hash(_compose(allowed_envs=("EVAL_RUN_TOKEN",)))
    assert fewer != baseline


def test_write_app_compose_writes_deployable_bytes(tmp_path):
    compose = _compose()
    dest = tmp_path / "app-compose.json"
    written = c.write_app_compose(dest, compose)
    on_disk = dest.read_text(encoding="utf-8")
    assert on_disk == written == c.render_app_compose(compose)
    # The on-disk file hashes to the compose-hash (what dstack measures).
    assert hashlib.sha256(dest.read_bytes()).hexdigest() == c.app_compose_hash(compose)


def test_docker_compose_file_is_valid_yaml_round_trip():
    compose = _compose()
    parsed = yaml.safe_load(compose["docker_compose_file"])
    assert "services" in parsed
    assert parsed["services"][c.ORCHESTRATOR_SERVICE]["image"] == CANONICAL_IMAGE


def test_app_compose_is_valid_json_document():
    compose = _compose()
    # The deployable text is parseable JSON that round-trips to the same document.
    assert json.loads(c.render_app_compose(compose)) == compose
