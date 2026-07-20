"""Product miner harness entry: ZIP+script sole review path; refuse parity harness.

Ports VAL-ACAT-001 / VAL-ACAT-002 into agent-challenge production.
No Base gateway. agent_parity_harness is never product review.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agent_challenge.review.harness_entry import (
    PARITY_HARNESS_KIND,
    PRODUCT_HARNESS_KIND,
    REFUSE_EMPTY_RULES,
    REFUSE_EMPTY_ZIP,
    REFUSE_MISSING_ENTRY_SCRIPT,
    REFUSE_MISSING_RULES,
    REFUSE_MISSING_ZIP,
    REFUSE_OR_BEFORE_RULES,
    REFUSE_PARITY_HARNESS,
    REFUSE_UNMEASURED_HOST,
    ProductHarnessAdmissionError,
    admit_product_review_entry,
    digest_agent_zip,
    inventory_product_vs_parity,
    is_parity_harness_entry,
    is_product_entry_script,
    load_rules_pack_digests,
    refuse_parity_harness_as_review,
    require_rules_before_openrouter,
    sha256_hex,
)

SAMPLE_ZIP = b"PK\x03\x04fake-agent-zip-v1-for-contract"
ENTRY_ID = "python -m agent_challenge.selfdeploy"
ENTRY_BYTES = b'#!/usr/bin/env python3\n"""selfdeploy entry marker"""\n'

SAMPLE_RULES: dict[str, bytes] = {
    ".rules/acceptance.md": b"# acceptance\nMust use measured review path.\n",
    ".rules/anti-cheat.md": b"# anti-cheat\nNo unmeasured shortcuts.\n",
    ".rules/hardcoding.md": b"# hardcoding\nNo hardcoded secrets.\n",
    ".rules/security.md": b"# security\nFail closed on missing attestation.\n",
}


def test_missing_zip_refuses_product_review() -> None:
    with pytest.raises(ProductHarnessAdmissionError) as exc:
        admit_product_review_entry(
            agent_zip_bytes=None,
            entry_script_identity=ENTRY_ID,
            entry_script_bytes=ENTRY_BYTES,
            rules_files=SAMPLE_RULES,
        )
    assert exc.value.code == REFUSE_MISSING_ZIP


def test_empty_zip_refuses_product_review() -> None:
    with pytest.raises(ProductHarnessAdmissionError) as exc:
        admit_product_review_entry(
            agent_zip_bytes=b"",
            entry_script_identity=ENTRY_ID,
            entry_script_bytes=ENTRY_BYTES,
            rules_files=SAMPLE_RULES,
        )
    assert exc.value.code == REFUSE_EMPTY_ZIP


def test_missing_entry_script_refuses() -> None:
    with pytest.raises(ProductHarnessAdmissionError) as exc:
        admit_product_review_entry(
            agent_zip_bytes=SAMPLE_ZIP,
            entry_script=None,
            entry_script_identity=None,
            entry_script_bytes=None,
            rules_files=SAMPLE_RULES,
        )
    assert exc.value.code == REFUSE_MISSING_ENTRY_SCRIPT


def test_missing_rules_refuses_before_openrouter() -> None:
    with pytest.raises(ProductHarnessAdmissionError) as exc:
        admit_product_review_entry(
            agent_zip_bytes=SAMPLE_ZIP,
            entry_script_identity=ENTRY_ID,
            entry_script_bytes=ENTRY_BYTES,
            rules_dir="/nonexistent/rules/pack/xyz",
        )
    assert exc.value.code == REFUSE_MISSING_RULES


def test_empty_rules_refuses() -> None:
    with pytest.raises(ProductHarnessAdmissionError) as exc:
        admit_product_review_entry(
            agent_zip_bytes=SAMPLE_ZIP,
            entry_script_identity=ENTRY_ID,
            entry_script_bytes=ENTRY_BYTES,
            rules_files={},
        )
    assert exc.value.code == REFUSE_EMPTY_RULES


def test_parity_harness_not_accepted_as_product_review() -> None:
    with pytest.raises(ProductHarnessAdmissionError) as exc:
        admit_product_review_entry(
            agent_zip_bytes=SAMPLE_ZIP,
            entry_script="tools/agent_parity_harness.py",
            rules_files=SAMPLE_RULES,
        )
    assert exc.value.code == REFUSE_PARITY_HARNESS

    with pytest.raises(ProductHarnessAdmissionError) as exc2:
        admit_product_review_entry(
            agent_zip_bytes=SAMPLE_ZIP,
            entry_script_identity=ENTRY_ID,
            entry_script_bytes=ENTRY_BYTES,
            rules_files=SAMPLE_RULES,
            harness_kind=PARITY_HARNESS_KIND,
        )
    assert exc2.value.code == REFUSE_PARITY_HARNESS

    with pytest.raises(ProductHarnessAdmissionError) as exc3:
        refuse_parity_harness_as_review("tools/agent_parity_harness.py")
    assert exc3.value.code == REFUSE_PARITY_HARNESS


def test_unmeasured_host_kind_refused() -> None:
    with pytest.raises(ProductHarnessAdmissionError) as exc:
        admit_product_review_entry(
            agent_zip_bytes=SAMPLE_ZIP,
            entry_script_identity=ENTRY_ID,
            entry_script_bytes=ENTRY_BYTES,
            rules_files=SAMPLE_RULES,
            harness_kind="offline_ast",
        )
    assert exc.value.code == REFUSE_UNMEASURED_HOST


def test_zip_and_entry_script_digests_bound_into_session_identity() -> None:
    identity = admit_product_review_entry(
        agent_zip_bytes=SAMPLE_ZIP,
        entry_script_identity=ENTRY_ID,
        entry_script_bytes=ENTRY_BYTES,
        rules_files=SAMPLE_RULES,
    )
    assert identity.harness_kind == PRODUCT_HARNESS_KIND
    assert identity.zip_sha256 == hashlib.sha256(SAMPLE_ZIP).hexdigest()
    assert identity.entry_script_identity == ENTRY_ID
    assert identity.entry_script_content_sha256 == hashlib.sha256(ENTRY_BYTES).hexdigest()
    assert identity.entry_script_path_sha256 == hashlib.sha256(ENTRY_ID.encode()).hexdigest()
    assert identity.openrouter_allowed is True
    material = identity.as_dict()
    assert material["zip_sha256"] == identity.zip_sha256
    assert material["entry_script_content_sha256"] == identity.entry_script_content_sha256
    assert len(identity.session_identity_sha256()) == 64
    other = admit_product_review_entry(
        agent_zip_bytes=SAMPLE_ZIP + b"-tamper",
        entry_script_identity=ENTRY_ID,
        entry_script_bytes=ENTRY_BYTES,
        rules_files=SAMPLE_RULES,
    )
    assert other.session_identity_sha256() != identity.session_identity_sha256()


def test_rules_version_and_bundle_bound_before_openrouter() -> None:
    identity = admit_product_review_entry(
        agent_zip_bytes=SAMPLE_ZIP,
        entry_script_identity=ENTRY_ID,
        entry_script_bytes=ENTRY_BYTES,
        rules_files=SAMPLE_RULES,
    )
    pack = load_rules_pack_digests(files=SAMPLE_RULES)
    assert identity.rules_version == pack.rules_version
    assert identity.rules_bundle_sha256 == pack.bundle_sha256
    assert set(identity.rules_files) == set(SAMPLE_RULES)
    assert identity.rules_file_digests[".rules/acceptance.md"] == sha256_hex(
        SAMPLE_RULES[".rules/acceptance.md"]
    )
    assert require_rules_before_openrouter(identity) is None
    assert require_rules_before_openrouter(None) == REFUSE_OR_BEFORE_RULES


def test_openrouter_before_rules_refused() -> None:
    with pytest.raises(ProductHarnessAdmissionError) as exc:
        admit_product_review_entry(
            agent_zip_bytes=SAMPLE_ZIP,
            entry_script_identity=ENTRY_ID,
            entry_script_bytes=ENTRY_BYTES,
            rules_files=SAMPLE_RULES,
            openrouter_call_attempted=True,
        )
    assert exc.value.code == REFUSE_OR_BEFORE_RULES


def test_rules_digest_mutates_when_pack_changes() -> None:
    a = load_rules_pack_digests(files=SAMPLE_RULES)
    mutated = dict(SAMPLE_RULES)
    mutated[".rules/acceptance.md"] = b"# acceptance\nCHANGED\n"
    b = load_rules_pack_digests(files=mutated)
    assert a.rules_version != b.rules_version
    assert a.bundle_sha256 != b.bundle_sha256


def test_product_entry_detection_inventory() -> None:
    assert is_product_entry_script("python -m agent_challenge.selfdeploy")
    assert is_product_entry_script("agent_challenge.selfdeploy")
    assert is_product_entry_script("docker/review/review_runtime.py")
    assert not is_product_entry_script("tools/agent_parity_harness.py")
    assert is_parity_harness_entry("tools/agent_parity_harness.py")
    inv = inventory_product_vs_parity()
    assert inv["product_harness"]["kind"] == PRODUCT_HARNESS_KIND
    assert inv["not_product"]["agent_parity_harness"]["refuse_code"] == REFUSE_PARITY_HARNESS
    assert inv["rules_before_openrouter"] is True


def test_load_rules_from_real_agent_challenge_pack() -> None:
    rules_dir = Path("/projects/platform-network/agent-challenge/.rules")
    if not rules_dir.is_dir():
        pytest.skip("agent-challenge .rules not present")
    pack = load_rules_pack_digests(rules_dir)
    assert pack.rules_version
    assert len(pack.rules_version) == 64
    assert pack.files
    assert any("acceptance" in f for f in pack.files)
    identity = admit_product_review_entry(
        agent_zip_bytes=SAMPLE_ZIP,
        entry_script_identity=ENTRY_ID,
        entry_script_bytes=ENTRY_BYTES,
        rules_dir=rules_dir,
    )
    assert identity.rules_version == pack.rules_version
    assert identity.openrouter_allowed is True


def test_digest_agent_zip_helpers() -> None:
    assert digest_agent_zip(SAMPLE_ZIP) == hashlib.sha256(SAMPLE_ZIP).hexdigest()
    with pytest.raises(ValueError) as exc:
        digest_agent_zip(None)  # type: ignore[arg-type]
    assert str(exc.value) == REFUSE_MISSING_ZIP


def test_session_identity_canonical_field_set() -> None:
    identity = admit_product_review_entry(
        agent_zip_bytes=SAMPLE_ZIP,
        entry_script_identity=ENTRY_ID,
        entry_script_bytes=ENTRY_BYTES,
        rules_files=SAMPLE_RULES,
    )
    keys = set(identity.as_dict().keys())
    assert keys == {
        "schema_version",
        "harness_kind",
        "zip_sha256",
        "entry_script_identity",
        "entry_script_path_sha256",
        "entry_script_content_sha256",
        "rules_version",
        "rules_bundle_sha256",
        "rules_files",
        "rules_file_digests",
        "rules_policy_text_sha256",
        "openrouter_allowed",
    }
    raw = json.dumps(identity.as_dict(), sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(raw).hexdigest() == identity.session_identity_sha256()
