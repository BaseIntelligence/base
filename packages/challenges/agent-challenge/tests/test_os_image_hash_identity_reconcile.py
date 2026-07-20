"""Product single OS identity: guest bind + Phala provision (residual bd369a vs 5c6d).

Live residual after tip b2f7ce76 (sub23 / eval-1task-litellm-speed):

* Guest ``quote_measurement_mismatch`` diag=os
  actual_prefix=5c6d8f757e3adb05 expected_prefix=bd369a8c2f9edb2b
* Phala CVM attestation tcb registers are the offline dstack-0.5.9 MRTD/RTMRs
* Product ``os_image_hash_from_registers(MRTD, RTMR1, RTMR2)`` = 5c6d8f75… (matches actual)
* Offline pin / dstack-mr-product injected catalog ``mr_image`` = bd369a8c… (digest.txt)
* Honest repin of assignment.os_image_hash to 5c6d… makes dry IN-LIST, but Phala
  provision still returns catalog bd369a… → product previously failed closed:
  ``Phala provision os_image_hash mismatches signed assignment measurement``

Product Mode B (this module):

A) Dual semantics are documented with the residual vector (bd369a catalog ≠ 5c6d formula).
B) Sealed ``assignment.measurement.os_image_hash`` is the **product formula** identity
   (sha256 of registers) used by guest review + keyrelease + allowlist.
C) Phala provision must accept that seal: when completed measurement is product-formula
   consistent with its own MRTD/RTMR1/RTMR2, provision's separate dstack catalog
   ``os_image_hash`` is **not** overloaded against the product field. Optionally it
   may still be bound to an allowlisted ``dstack_mr_image`` catalog digest.
D) No invent MRTD / allow / KR; no product OpenRouter limiter.
"""

from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_challenge.canonical import measurement as m
from agent_challenge.keyrelease.quote import os_image_hash_from_registers
from agent_challenge.selfdeploy import eval as eval_deploy
from agent_challenge.selfdeploy import review as review_deploy

# Residual vector (honest dstack-0.5.9 / tdx.small; no invent).
# Source: evidence/ac-attestation/e2e-live-v7/eval-1task-litellm-speed/
RESIDUAL_MRTD = (
    "f06dfda6dce1cf904d4e2bab1dc370634cf95cefa2ceb2de2eee127c93826980"
    "90d7a4a13e14c536ec6c9c3c8fa87077"
)
RESIDUAL_RTMR0 = (
    "68102e7b524af310f7b7d426ce75481e36c40f5d513a9009c046e9d37e31551f"
    "0134d954b496a3357fd61d03f07ffe96"
)
RESIDUAL_RTMR1 = (
    "07e6f51aa763abfe75c3ddfbf4f425fe3f0ceff66d807a75e049303dce9addf6"
    "8e7218729bd419638af63a370f65878c"
)
RESIDUAL_RTMR2 = (
    "df67e467e60edc1737bcf8e682d48131bfb427f523226aa7f197a7608e9b3784"
    "783fa759ef5b28191fa12f9ddb36b858"
)
# Offline dstack-mr-product / digest.txt catalog identity (NOT product formula).
RESIDUAL_DSTACK_MR_IMAGE = "bd369a8c2f9edb2b52dad48ac8e0b32dde5f1337c423a506b48d07403a7d8033"
# Product + guest quote identity.
RESIDUAL_PRODUCT_OS = "5c6d8f757e3adb0563efc809710076a631442db3b4de02ad32d33fe1994721e0"


def test_residual_vector_documents_dual_os_semantics() -> None:
    """bd369a catalog mr_image ≠ sha256(registers)=5c6d on identical MRTD/RTMR1/RTMR2."""

    product = os_image_hash_from_registers(RESIDUAL_MRTD, RESIDUAL_RTMR1, RESIDUAL_RTMR2)
    assert product == RESIDUAL_PRODUCT_OS
    assert product.startswith("5c6d8f757e3adb05")
    assert RESIDUAL_DSTACK_MR_IMAGE.startswith("bd369a8c2f9edb2b")
    assert product != RESIDUAL_DSTACK_MR_IMAGE
    # Formula is literally SHA-256 of the three register *bytes* (not hex text).
    raw = (
        bytes.fromhex(RESIDUAL_MRTD) + bytes.fromhex(RESIDUAL_RTMR1) + bytes.fromhex(RESIDUAL_RTMR2)
    )
    assert hashlib.sha256(raw).hexdigest() == RESIDUAL_PRODUCT_OS


def test_compute_image_measurement_seals_product_formula_not_catalog_mr_image(
    monkeypatch, tmp_path
) -> None:
    """Offline dstack-mr may emit catalog digest as mr_image; product pins formula."""

    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")

    class _Proc:
        returncode = 0
        # Same residual registers; mr_image is the *wrong* field for product pin.
        stdout = __import__("json").dumps(
            {
                "mrtd": RESIDUAL_MRTD,
                "rtmr0": RESIDUAL_RTMR0,
                "rtmr1": RESIDUAL_RTMR1,
                "rtmr2": RESIDUAL_RTMR2,
                "mr_image": RESIDUAL_DSTACK_MR_IMAGE,
            }
        )
        stderr = ""

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Proc())
    image = m.compute_image_measurement(metadata, cpu=1, memory="2G", dstack_mr_bin="dstack-mr")
    assert image.os_image_hash == RESIDUAL_PRODUCT_OS
    assert image.os_image_hash != RESIDUAL_DSTACK_MR_IMAGE
    # Catalog digest may be retained separately for provision binding, never overloading.
    assert getattr(image, "dstack_mr_image", None) == RESIDUAL_DSTACK_MR_IMAGE


def test_compute_image_measurement_derives_formula_when_mr_image_absent(
    monkeypatch, tmp_path
) -> None:
    """Rust dstack-mr may omit mr_image; product still seals register formula."""

    metadata = tmp_path / "metadata.json"
    metadata.write_text("{}")

    class _Proc:
        returncode = 0
        stdout = __import__("json").dumps(
            {
                "mrtd": RESIDUAL_MRTD,
                "rtmr0": RESIDUAL_RTMR0,
                "rtmr1": RESIDUAL_RTMR1,
                "rtmr2": RESIDUAL_RTMR2,
            }
        )
        stderr = ""

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Proc())
    image = m.compute_image_measurement(metadata, cpu=1, memory="2G", dstack_mr_bin="dstack-mr")
    assert image.os_image_hash == RESIDUAL_PRODUCT_OS
    assert getattr(image, "dstack_mr_image", None) in {None, ""}


def _review_plan_measurement(
    *, os_image_hash: str, dstack_mr_image: str | None = None
) -> dict[str, str]:
    out: dict[str, str] = {
        "mrtd": RESIDUAL_MRTD,
        "rtmr0": RESIDUAL_RTMR0,
        "rtmr1": RESIDUAL_RTMR1,
        "rtmr2": RESIDUAL_RTMR2,
        "os_image_hash": os_image_hash,
        "key_provider": "phala",
        "vm_shape": "tdx.small",
    }
    if dstack_mr_image is not None:
        out["dstack_mr_image"] = dstack_mr_image
    return out


def _stub_review_plan(measurement: dict[str, str]) -> Any:
    plan = MagicMock()
    plan.compose_hash = "a" * 64
    plan.app_identity = "b" * 40
    plan.kms_public_key_hex = "c" * 64
    plan.measurement = measurement
    return plan


def test_review_provision_accepts_catalog_os_when_seal_is_product_formula() -> None:
    """After honest 5c6d repin, provision may still return bd369a catalog — accept."""

    plan = _stub_review_plan(
        _review_plan_measurement(
            os_image_hash=RESIDUAL_PRODUCT_OS,
            dstack_mr_image=RESIDUAL_DSTACK_MR_IMAGE,
        )
    )
    provision = {
        "compose_hash": plan.compose_hash,
        "app_id": plan.app_identity,
        "app_env_encrypt_pubkey": plan.kms_public_key_hex,
        # Phala catalog field (residual) — NOT product formula.
        "os_image_hash": RESIDUAL_DSTACK_MR_IMAGE,
    }
    # Must not raise the residual provision mismatch error.
    review_deploy.HttpReviewPhalaDeployment._verify_provision_response(plan, provision)


def test_review_provision_accepts_catalog_os_without_explicit_dstack_mr_image() -> None:
    """Product seal alone is enough; catalog OS is observed, not overloaded onto seal."""

    plan = _stub_review_plan(_review_plan_measurement(os_image_hash=RESIDUAL_PRODUCT_OS))
    provision = {
        "compose_hash": plan.compose_hash,
        "app_id": plan.app_identity,
        "app_env_encrypt_pubkey": plan.kms_public_key_hex,
        "os_image_hash": RESIDUAL_DSTACK_MR_IMAGE,
    }
    review_deploy.HttpReviewPhalaDeployment._verify_provision_response(plan, provision)


def test_review_provision_still_binds_optional_allowlisted_dstack_mr_image() -> None:
    """When dstack_mr_image is sealed, provision catalog must match that allowlist.

    This prevents invent loophole of accepting any catalog digest when pin names one.
    """

    plan = _stub_review_plan(
        _review_plan_measurement(
            os_image_hash=RESIDUAL_PRODUCT_OS,
            dstack_mr_image=RESIDUAL_DSTACK_MR_IMAGE,
        )
    )
    provision = {
        "compose_hash": plan.compose_hash,
        "app_id": plan.app_identity,
        "app_env_encrypt_pubkey": plan.kms_public_key_hex,
        "os_image_hash": "ff" * 32,  # wrong catalog
    }
    with pytest.raises(review_deploy.ReviewDeploymentError, match="dstack_mr_image|catalog"):
        review_deploy.HttpReviewPhalaDeployment._verify_provision_response(plan, provision)


def test_review_provision_legacy_catalog_seal_still_requires_equality() -> None:
    """Pre-fix pins that overloading catalog as os_image_hash still equality-check."""

    plan = _stub_review_plan(_review_plan_measurement(os_image_hash=RESIDUAL_DSTACK_MR_IMAGE))
    # Matching catalog seal ↔ provision catalog still OK.
    review_deploy.HttpReviewPhalaDeployment._verify_provision_response(
        plan,
        {
            "compose_hash": plan.compose_hash,
            "app_id": plan.app_identity,
            "app_env_encrypt_pubkey": plan.kms_public_key_hex,
            "os_image_hash": RESIDUAL_DSTACK_MR_IMAGE,
        },
    )
    # Mismatch still fails closed (no invent privilege for legacy seal).
    with pytest.raises(
        review_deploy.ReviewDeploymentError,
        match="Phala provision os_image_hash mismatches signed assignment measurement",
    ):
        review_deploy.HttpReviewPhalaDeployment._verify_provision_response(
            plan,
            {
                "compose_hash": plan.compose_hash,
                "app_id": plan.app_identity,
                "app_env_encrypt_pubkey": plan.kms_public_key_hex,
                "os_image_hash": RESIDUAL_PRODUCT_OS,
            },
        )


def test_eval_provision_accepts_catalog_os_when_seal_is_product_formula() -> None:
    """Eval path shares the same provision OS identity policy (tdx.small residual class)."""

    plan = MagicMock()
    plan.compose_hash = "a" * 64
    plan.app_identity = "b" * 40
    plan.kms_public_key_hex = "c" * 64
    plan.measurement = _review_plan_measurement(os_image_hash=RESIDUAL_PRODUCT_OS)
    # Use the shared helper / eval method — eval inlines the same policy.
    eval_deploy.HttpEvalPhalaDeployment._verify_provision_os_identity(
        plan,
        {
            "compose_hash": plan.compose_hash,
            "app_id": plan.app_identity,
            "app_env_encrypt_pubkey": plan.kms_public_key_hex,
            "os_image_hash": RESIDUAL_DSTACK_MR_IMAGE,
        },
    )


def test_guest_and_pin_share_product_os_identity_on_residual_registers() -> None:
    """Single sealed identity for guest bind: formula from the residual registers."""

    sealed = m.product_os_image_hash(
        mrtd=RESIDUAL_MRTD,
        rtmr1=RESIDUAL_RTMR1,
        rtmr2=RESIDUAL_RTMR2,
    )
    guest_computed = os_image_hash_from_registers(RESIDUAL_MRTD, RESIDUAL_RTMR1, RESIDUAL_RTMR2)
    assert sealed == guest_computed == RESIDUAL_PRODUCT_OS
    assert m.measurement_uses_product_os_identity(
        {
            "mrtd": RESIDUAL_MRTD,
            "rtmr1": RESIDUAL_RTMR1,
            "rtmr2": RESIDUAL_RTMR2,
            "os_image_hash": sealed,
        }
    )
    assert not m.measurement_uses_product_os_identity(
        {
            "mrtd": RESIDUAL_MRTD,
            "rtmr1": RESIDUAL_RTMR1,
            "rtmr2": RESIDUAL_RTMR2,
            "os_image_hash": RESIDUAL_DSTACK_MR_IMAGE,
        }
    )
