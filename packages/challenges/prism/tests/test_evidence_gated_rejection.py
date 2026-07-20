from __future__ import annotations

from hashlib import sha256

import anyio

from prism_challenge.db import Database, loads
from prism_challenge.evaluator.schemas import DeterministicEvidence
from prism_challenge.models import SubmissionCreate
from prism_challenge.repository import PrismRepository


async def _repo(tmp_path):
    database = Database(tmp_path / "evidence-gate.sqlite3")
    await database.init()
    return PrismRepository(database, epoch_seconds=3600)


async def _submission(repo: PrismRepository) -> str:
    created = await repo.create_submission(
        "miner", SubmissionCreate(code="def build_model(ctx):\n    return None\n")
    )
    return created.id


def _evidence() -> dict[str, object]:
    return DeterministicEvidence(
        rule_id="prism:no-escape",
        artifact_path="model.py",
        line=2,
        snippet_hash=sha256(b"os.system('curl bad')").hexdigest(),
        explanation="process escape attempt is present in submitted source",
    ).model_dump(mode="json")


def test_evidence_backed_rejects_submission(tmp_path):
    async def run() -> None:
        repo = await _repo(tmp_path)
        submission_id = await _submission(repo)
        evidence = [_evidence()]

        await repo.submit_llm_mermaid(
            submission_id=submission_id,
            mermaid="flowchart LR\n  A[Source] --> B[Escape]",
        )
        await repo.submit_llm_verdict(
            submission_id=submission_id,
            approved=False,
            reason="deterministic escape evidence found",
            violations=["prism:no-escape"],
            confidence=0.96,
            raw={"reason": "deterministic escape evidence found", "verdict": False},
            evidence=evidence,
        )

        status = await repo.get_submission(submission_id)
        async with repo.database.connect() as conn:
            review = list(
                await conn.execute_fetchall(
                    "SELECT final_state, evidence FROM llm_reviews WHERE submission_id=?",
                    (submission_id,),
                )
            )[0]
            events = await conn.execute_fetchall(
                "SELECT state FROM llm_review_events WHERE submission_id=? ORDER BY sequence",
                (submission_id,),
            )

        assert status is not None
        assert status.status.value == "rejected"
        assert status.error == "deterministic escape evidence found"
        assert review["final_state"] == "rejected"
        assert loads(str(review["evidence"])) == evidence
        assert [row["state"] for row in events][-1] == "rejected"

    anyio.run(run)


def test_explicit_held_is_terminal_rejected(tmp_path):
    # After gateway removal there is no HELD submission status: held=True quarantine
    # still records final_state=quarantined on the review row, but the submission is
    # terminally rejected (no held/wait path, no live LLM gate).
    async def run() -> None:
        repo = await _repo(tmp_path)
        submission_id = await _submission(repo)

        await repo.submit_llm_mermaid(
            submission_id=submission_id,
            mermaid="flowchart LR\n  A[Source] --> B[Review]",
        )
        await repo.submit_llm_verdict(
            submission_id=submission_id,
            approved=False,
            reason="LLM suspects hidden behavior but has no deterministic evidence",
            violations=["suspicion"],
            confidence=0.61,
            raw={"reason": "LLM suspects hidden behavior", "verdict": False},
            evidence=[],
            held=True,
        )

        status = await repo.get_submission(submission_id)
        async with repo.database.connect() as conn:
            review = list(
                await conn.execute_fetchall(
                    "SELECT final_state, evidence FROM llm_reviews WHERE submission_id=?",
                    (submission_id,),
                )
            )[0]
            events = await conn.execute_fetchall(
                "SELECT state, reason FROM llm_review_events "
                "WHERE submission_id=? ORDER BY sequence",
                (submission_id,),
            )

        assert status is not None
        assert status.status.value == "rejected"
        assert status.error == "LLM suspects hidden behavior but has no deterministic evidence"
        assert review["final_state"] == "quarantined"
        assert loads(str(review["evidence"])) == []
        assert [row["state"] for row in events][-1] == "quarantined"

    anyio.run(run)
