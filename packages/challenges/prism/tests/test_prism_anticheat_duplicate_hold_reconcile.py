"""v2 reconciliation of the (vestigial NAS) duplicate-review HOLD band.

The v1-NAS decommission removed the operator hold-resolution endpoints
(``resolve_component_hold`` / ``list_component_review_holds``) but left the
duplicate-review HOLD creation on the live worker path, so a near-duplicate that
landed in the quarantine band STRANDED in HELD with no resolve/expire path
(``expire_stale_held`` intentionally skips component holds).

v2 behavior: there is no operator review surface, so the quarantine band is folded
into a terminal rejection at static review (before any GPU work). Exact-source-hash
duplicate dedup is preserved.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator import source_similarity
from prism_challenge.evaluator.source_similarity import (
    DuplicatePolicyDecision,
    SimilarityCandidate,
    SourceSnapshot,
)

QUARANTINE_REASON = "borderline source or semantic graph similarity requires review"
EXACT_REASON = "exact source hash duplicate of an existing submission"


def _settings(tmp_path: Path, name: str) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / name}",
        shared_token="secret",
        allow_insecure_signatures=True,
        plagiarism_enabled=True,
        fineweb_sample_count=4,
        # No OpenRouter key in the unit env; disable the gate (covered in test_*llm*).
        distributed_contract_policy="off",
    )


def _candidate() -> SimilarityCandidate:
    return SimilarityCandidate(
        submission_id="prior-candidate",
        hotkey="miner-prior",
        code_hash="prior-code-hash",
        score=0.9,
        ast_similarity=0.9,
        token_similarity=0.88,
        file_similarity=0.0,
        snapshot=SourceSnapshot(
            files=(),
            ast_features=frozenset(),
            token_shingles=frozenset(),
            fingerprint="prior-fp",
        ),
    )


def _submit(client: TestClient, nonce: str) -> str:
    payload = {"code": two_script_bundle(), "filename": "project.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _process(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200, response.text


def _gpu_lease_count(db_path: Path, submission_id: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM gpu_leases WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    return int(count)


def test_anticheat_duplicate_quarantine_is_rejected_not_held(tmp_path, monkeypatch) -> None:
    """A near-duplicate routed to the quarantine band must terminate in ``rejected``,
    never stranding in ``held``, and must not reach GPU work."""

    def fake_classify(**_kwargs: object) -> DuplicatePolicyDecision:
        return DuplicatePolicyDecision(
            outcome="quarantine",
            reason=QUARANTINE_REASON,
            candidate=_candidate(),
            report={
                "source_similarity": 0.9,
                "graph_similarity": 0.88,
                "outcome": "quarantine",
            },
        )

    monkeypatch.setattr(source_similarity, "classify_duplicate", fake_classify)

    db_name = "dup-quarantine.sqlite3"
    settings = _settings(tmp_path, db_name)
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, nonce="dup-q-1")
        _process(client)
        status = client.get(f"/v1/submissions/{submission_id}").json()

    assert status["status"] == "rejected", status
    assert status["status"] != "held"
    assert status["error"] == QUARANTINE_REASON
    # Rejected at static review, before any GPU lease was taken.
    assert _gpu_lease_count(tmp_path / db_name, submission_id) == 0


def test_anticheat_exact_duplicate_still_rejected(tmp_path, monkeypatch) -> None:
    """Exact-source-hash duplicate dedup is preserved: it is rejected at static review."""

    def fake_classify(**_kwargs: object) -> DuplicatePolicyDecision:
        return DuplicatePolicyDecision(
            outcome="reject",
            reason=EXACT_REASON,
            candidate=_candidate(),
            report={
                "source_similarity": 1.0,
                "outcome": "reject",
                "exact_source_hash": True,
            },
        )

    monkeypatch.setattr(source_similarity, "classify_duplicate", fake_classify)

    settings = _settings(tmp_path, "dup-exact.sqlite3")
    with TestClient(create_app(settings)) as client:
        submission_id = _submit(client, nonce="dup-x-1")
        _process(client)
        status = client.get(f"/v1/submissions/{submission_id}").json()

    assert status["status"] == "rejected", status
    assert status["error"] == EXACT_REASON
