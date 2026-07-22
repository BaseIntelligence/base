"""VAL-ACLOCK-011..016: TB 2.1 integrity contracts + allow_internet policy."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.benchmarks import (
    TERMINAL_BENCH_2_1_DIGEST_SHA256,
    TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS,
)
from agent_challenge.evaluation.own_runner.container_builder import network_arg
from agent_challenge.evaluation.own_runner.taskdefs import (
    DigestMismatch,
    ResourceLimits,
    bare_task_name,
    compute_task_digest,
    load_dataset_digest,
    load_task,
    load_task_from_manifest,
)
from agent_challenge.evaluation.tbench_integrity import (
    ALLOW_INTERNET_POLICY_ID,
    ANSWER_HARDCODING_IS_CHEAT,
    FORBIDDEN_TASK_SOURCE_KEYS,
    REQUIRED_HARNESS_PINS,
    TbenchIntegrityError,
    allow_internet_policy_snapshot,
    assert_fallback_ids_subset_of_frozen,
    assert_no_miner_task_source_fields,
    assert_selected_task_ids_in_frozen,
    assert_taskdefs_loader_is_local_only,
    effective_network_arg,
    frozen_digest_path,
    inventory_allow_internet,
    load_frozen_task_ids,
    selected_task_item_allowed_keys,
)
from agent_challenge.selfdeploy import eval as eval_deploy

_REPO = Path(__file__).resolve().parents[1]
_DIGEST = _REPO / "golden" / "dataset-digest.json"
_LIVE_CACHE = _REPO / "docker" / "canonical" / "live-task-cache"
_HARDCODING_RULES = _REPO / ".rules" / "hardcoding.md"
_ANTI_CHEAT_RULES = _REPO / ".rules" / "anti-cheat.md"
_SECURITY_DOC = _REPO / "docs" / "security.md"
_EVAL_DOC = _REPO / "docs" / "evaluation.md"


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-011 — no network at eval for task defs
# --------------------------------------------------------------------------- #
def test_taskdefs_loader_has_no_network_clients() -> None:
    assert_taskdefs_loader_is_local_only()


def test_taskdefs_loader_contract_language_present() -> None:
    text = (_REPO / "src/agent_challenge/evaluation/own_runner/taskdefs.py").read_text(
        encoding="utf-8"
    )
    assert "No network at eval time" in text or "never network-fetches" in text
    assert "DigestMismatch" in text
    assert "dataset-digest.json" in text


def test_digest_mismatch_fail_closed(tmp_path: Path) -> None:
    root = tmp_path / "toy"
    root.mkdir()
    (root / "task.toml").write_text(
        '[task]\nname="toy"\n[environment]\nallow_internet=false\n',
        encoding="utf-8",
    )
    (root / "instruction.md").write_text("do the thing\n", encoding="utf-8")
    env = root / "environment"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM ubuntu:24.04\n", encoding="utf-8")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\necho ok\n", encoding="utf-8")
    actual = compute_task_digest(root)
    with pytest.raises(DigestMismatch):
        load_task(root, task_id="toy", expected_digest="0" * 64)
    with pytest.raises(DigestMismatch):
        load_task_from_manifest(
            root,
            task_id="toy",
            digest_manifest={"tasks": {"toy": {"content_digest_sha256": "1" * 64}}},
        )
    # Matching digest still loads.
    loaded = load_task(root, task_id="toy", expected_digest=actual)
    assert loaded.content_digest_sha256 == actual


def test_eval_docs_state_no_network_fetch() -> None:
    text = _EVAL_DOC.read_text(encoding="utf-8")
    assert "no network" in text.lower() or "does not network-fetch" in text.lower()
    assert "dataset-digest.json" in text or "task-cache" in text


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-012 — miner cannot supply alternate task URL/git
# --------------------------------------------------------------------------- #
def test_forbidden_task_source_keys_rejected() -> None:
    for key in sorted(FORBIDDEN_TASK_SOURCE_KEYS):
        with pytest.raises(TbenchIntegrityError, match="alternate task"):
            assert_no_miner_task_source_fields({key: "https://evil.example/tasks.git"})


def test_honest_plan_fields_have_no_task_url() -> None:
    assert_no_miner_task_source_fields(
        {
            "task_id": "terminal-bench/bn-fit-modify",
            "image_ref": "registry.example/t@sha256:" + "a" * 64,
            "task_config_sha256": "b" * 64,
        }
    )


def test_selected_tasks_schema_rejects_task_url_extra() -> None:
    """Plan selected_tasks[] is schema-closed; task_url is unknown → reject."""
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    base = {
        "schema_version": 1,
        "eval_run_id": "eval-run-tbench",
        "submission_id": "submission-001",
        "submission_version": 1,
        "authorizing_review_digest": "1" * 64,
        "agent_hash": "a" * 64,
        "package_tree_sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "selected_tasks": [
            {
                "task_id": "task-a",
                "image_ref": "registry.example/task@sha256:" + "d" * 64,
                "task_config_sha256": "2" * 64,
                "task_url": "https://evil.example/task.git",
            }
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "d" * 64,
            "compose_hash": "c" * 64,
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "3" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("3" * 64)).hexdigest(),
            "measurement": {
                "mrtd": "a1" * 48,
                "rtmr0": "a2" * 48,
                "rtmr1": "a3" * 48,
                "rtmr2": "a4" * 48,
                "os_image_hash": "a5" * 32,
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "ratls://kr.internal:8701",
        "result_endpoint": "/evaluation/v1/runs/eval-run-tbench/result",
        "key_release_nonce": "key-nonce-001",
        "score_nonce": "score-nonce-001",
        "run_token_sha256": "5" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }
    with pytest.raises(ew.EvalWireError, match="invalid fields|unknown"):
        ew.validate_eval_plan(base)


def test_prepare_response_has_no_miner_task_body_fields() -> None:
    """eval/prepare deploy helper only accepts signed plan wrapper (validator only)."""
    src = inspect.getsource(eval_deploy.build_eval_deployment_plan)
    assert "task_url" not in src
    assert "plan" in src


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-013 — frozen set + fallback IDs
# --------------------------------------------------------------------------- #
def test_digest_pin_matches_constant() -> None:
    actual = hashlib.sha256(_DIGEST.read_bytes()).hexdigest()
    assert actual == TERMINAL_BENCH_2_1_DIGEST_SHA256
    assert frozen_digest_path() == _DIGEST


def test_fallback_ids_subset_of_frozen() -> None:
    assert_fallback_ids_subset_of_frozen()
    frozen = load_frozen_task_ids()
    for task_id in TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS:
        assert task_id in frozen
        assert bare_task_name(task_id) in frozen


def test_manifest_task_count_and_load() -> None:
    manifest = load_dataset_digest(_DIGEST)
    assert len(manifest["tasks"]) == 89
    assert set(TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS).issubset(
        {f"terminal-bench/{n}" for n in manifest["tasks"]}
    )


def test_unknown_selected_id_fails() -> None:
    with pytest.raises(TbenchIntegrityError, match="not in frozen"):
        assert_selected_task_ids_in_frozen(["terminal-bench/this-is-not-a-real-task"])


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-014 — selected_tasks validator-authored only
# --------------------------------------------------------------------------- #
def test_selected_tasks_allowed_keys_only() -> None:
    assert selected_task_item_allowed_keys() == frozenset(
        {"task_id", "image_ref", "task_config_sha256"}
    )


def test_authorization_build_plan_uses_validator_selection() -> None:
    """_build_plan pulls tasks from load_benchmark_tasks + settings, not miner body."""
    from agent_challenge.evaluation import authorization as auth

    src = inspect.getsource(auth._build_plan)
    assert "select_benchmark_tasks" in src
    assert "load_benchmark_tasks" in src
    assert "task_url" not in src
    assert "selected_tasks" in src


def test_prepare_route_has_no_selected_tasks_request_model() -> None:
    routes_src = (_REPO / "src/agent_challenge/api/routes.py").read_text(encoding="utf-8")
    # prepare_submission_eval takes submission_id + auth only (no body tasks).
    assert "async def prepare_submission_eval" in routes_src
    start = routes_src.index("async def prepare_submission_eval")
    chunk = routes_src[start : start + 500]
    assert "selected_tasks" not in chunk
    assert "task_url" not in chunk


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-015 — hardcode answers cheat; harness pins required
# --------------------------------------------------------------------------- #
def test_hardcoding_rules_flag_answer_cheat() -> None:
    text = _HARDCODING_RULES.read_text(encoding="utf-8")
    assert "branches_on_task_identity" in text
    assert "tailors_to_tests" in text or "static output" in text.lower()
    anti = _ANTI_CHEAT_RULES.read_text(encoding="utf-8")
    assert "branches_on_task_identity" in anti
    assert "hardcode" in anti.lower() or "expected result" in anti.lower()


def test_harness_pins_documented_and_listed() -> None:
    assert len(REQUIRED_HARNESS_PINS) >= 6
    snap = allow_internet_policy_snapshot()
    joined = " ".join(snap["required_harness_pins"]).lower()
    assert "dataset-digest" in joined or "digest" in joined
    assert "hardcoding" in ANSWER_HARDCODING_IS_CHEAT.lower()
    assert "cheat" in ANSWER_HARDCODING_IS_CHEAT.lower()
    sec = _SECURITY_DOC.read_text(encoding="utf-8")
    # Product docs distinguish harness pins vs answer hardcoding.
    assert "hardcod" in sec.lower() or "anti-cheat" in sec.lower() or "digest" in sec.lower()


def test_security_doc_states_hardcode_vs_pin_policy() -> None:
    text = _SECURITY_DOC.read_text(encoding="utf-8")
    assert "allow_internet" in text or "Terminal-Bench" in text
    assert "dataset-digest" in text or "task-cache" in text or "content-address" in text.lower()


# --------------------------------------------------------------------------- #
# VAL-ACLOCK-016 — allow_internet policy documented + gated
# --------------------------------------------------------------------------- #
def test_allow_internet_policy_id_locked() -> None:
    snap = allow_internet_policy_snapshot()
    assert snap["policy_id"] == ALLOW_INTERNET_POLICY_ID
    assert snap["default_scored_behavior"] == "honor_task_toml_allow_internet"
    assert snap["opt_in_restrict_default"] is False
    assert snap["task_def_network_at_eval"] is False
    assert snap["selected_tasks_author"] == "validator_prepare_only"


def test_inventory_live_cache_all_allow_true() -> None:
    if not _LIVE_CACHE.is_dir():
        pytest.skip("live-task-cache not present in checkout")
    inv = inventory_allow_internet(_LIVE_CACHE)
    assert inv["counts"]["scanned"] == 89
    assert inv["counts"]["true"] == 89
    assert inv["counts"]["false"] == 0
    assert inv["policy_id"] == ALLOW_INTERNET_POLICY_ID


def test_default_network_arg_honors_task_authored() -> None:
    assert network_arg(ResourceLimits(allow_internet=True)) is None
    assert network_arg(ResourceLimits(allow_internet=False)) == "none"
    assert network_arg(ResourceLimits(allow_internet=None)) == "none"
    assert effective_network_arg(ResourceLimits(allow_internet=True), scored_run=True) is None


def test_opt_in_restrict_forces_none(monkeypatch: pytest.MonkeyPatch) -> None:
    env = {"CHALLENGE_SCORED_TASK_NETWORK_RESTRICT": "1"}
    assert (
        effective_network_arg(
            ResourceLimits(allow_internet=True),
            scored_run=True,
            environ=env,
        )
        == "none"
    )
    # Unscored / non-restrict path still honors task when env not set.
    assert (
        effective_network_arg(
            ResourceLimits(allow_internet=True),
            scored_run=True,
            environ={},
        )
        is None
    )
    monkeypatch.setenv("CHALLENGE_SCORED_TASK_NETWORK_RESTRICT", "true")
    assert network_arg(ResourceLimits(allow_internet=True)) == "none"
    monkeypatch.delenv("CHALLENGE_SCORED_TASK_NETWORK_RESTRICT", raising=False)
    assert network_arg(ResourceLimits(allow_internet=True)) is None


def test_security_doc_documents_allow_internet_choice() -> None:
    text = _SECURITY_DOC.read_text(encoding="utf-8")
    assert "allow_internet" in text
    assert "retain" in text.lower() or "review-class" in text.lower() or "residual" in text.lower()


def test_policy_json_serializable() -> None:
    payload = json.dumps(allow_internet_policy_snapshot())
    assert ALLOW_INTERNET_POLICY_ID in payload
