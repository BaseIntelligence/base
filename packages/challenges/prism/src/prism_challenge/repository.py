from __future__ import annotations

from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, SupportsFloat, SupportsInt, cast
from uuid import uuid4

import aiosqlite

from .db import Database, dumps, loads
from .evaluator.schemas import (
    DeterministicEvidence,
)
from .evaluator.scoring import LeaderboardRow, dedupe_best_per_hotkey, rank_leaderboard
from .models import (
    SubmissionCreate,
    SubmissionResponse,
    SubmissionStatus,
    SubmissionStatusResponse,
)
from .runtime_config import RuntimePolicy, resolve_runtime_policy


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def epoch_id_for(timestamp: datetime, epoch_seconds: int) -> int:
    return int(timestamp.timestamp()) // epoch_seconds


async def ensure_epoch(conn: aiosqlite.Connection, epoch_id: int, epoch_seconds: int) -> None:
    starts = datetime.fromtimestamp(epoch_id * epoch_seconds, UTC)
    ends = datetime.fromtimestamp((epoch_id + 1) * epoch_seconds, UTC)
    await conn.execute(
        "INSERT OR IGNORE INTO epochs(id, starts_at, ends_at, status) VALUES (?, ?, ?, ?)",
        (epoch_id, starts.isoformat(), ends.isoformat(), "open"),
    )


class PrismRepository:
    def __init__(
        self,
        database: Database,
        epoch_seconds: int,
        worker_claim_timeout_seconds: int = 900,
        held_review_timeout_seconds: int = 86400,
    ) -> None:
        self.database = database
        self.epoch_seconds = epoch_seconds
        self.worker_claim_timeout_seconds = worker_claim_timeout_seconds
        self.held_review_timeout_seconds = held_review_timeout_seconds

    async def create_submission(self, hotkey: str, request: SubmissionCreate) -> SubmissionResponse:
        created = datetime.now(UTC)
        epoch_id = epoch_id_for(created, self.epoch_seconds)
        submission_id = str(uuid4())
        code_hash = sha256(request.code.encode()).hexdigest()
        async with self.database.connect() as conn:
            await ensure_epoch(conn, epoch_id, self.epoch_seconds)
            await conn.execute(
                "INSERT OR IGNORE INTO miners(hotkey, first_seen, last_seen) VALUES (?, ?, ?)",
                (hotkey, created.isoformat(), created.isoformat()),
            )
            await conn.execute(
                "UPDATE miners SET last_seen=? WHERE hotkey=?", (created.isoformat(), hotkey)
            )
            await conn.execute(
                "INSERT INTO submissions("
                "id, hotkey, epoch_id, filename, code, code_hash, metadata, status, "
                "created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    hotkey,
                    epoch_id,
                    request.filename,
                    request.code,
                    code_hash,
                    dumps(request.metadata),
                    SubmissionStatus.PENDING.value,
                    created.isoformat(),
                    created.isoformat(),
                ),
            )
            await conn.execute(
                "INSERT INTO eval_jobs("
                "id, submission_id, level, status, metrics, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    submission_id,
                    "l1",
                    "pending",
                    "{}",
                    created.isoformat(),
                    created.isoformat(),
                ),
            )
        return SubmissionResponse(
            id=submission_id,
            hotkey=hotkey,
            epoch_id=epoch_id,
            status=SubmissionStatus.PENDING,
            code_hash=code_hash,
            created_at=created,
        )

    async def get_submission(self, submission_id: str) -> SubmissionStatusResponse | None:
        async with self.database.connect() as conn:
            row = await conn.execute_fetchall(
                "SELECT s.*, sc.q_arch, sc.q_recipe, sc.final_score, "
                "sc.anti_cheat_multiplier, sc.diversity_bonus, sc.penalty FROM submissions s "
                "LEFT JOIN scores sc ON sc.submission_id=s.id WHERE s.id=?",
                (submission_id,),
            )
        if not row:
            return None
        item = list(row)[0]
        return SubmissionStatusResponse(
            id=item["id"],
            hotkey=item["hotkey"],
            epoch_id=item["epoch_id"],
            status=SubmissionStatus(item["status"]),
            code_hash=item["code_hash"],
            created_at=datetime.fromisoformat(item["created_at"]),
            error=item["error"],
            final_score=item["final_score"],
            q_arch=item["q_arch"],
            q_recipe=item["q_recipe"],
            anti_cheat_multiplier=item["anti_cheat_multiplier"],
            diversity_bonus=item["diversity_bonus"],
            penalty=item["penalty"],
        )

    async def previous_codes(self, current_submission_id: str) -> list[str]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT code FROM submissions WHERE id != ? ORDER BY created_at DESC",
                (current_submission_id,),
            )
        return [str(row["code"]) for row in rows]

    async def store_source_snapshot(
        self,
        *,
        submission_id: str,
        hotkey: str,
        code_hash: str,
        payload: dict[str, Any],
    ) -> None:
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO submission_sources("
                "submission_id, hotkey, code_hash, files, ast_features, token_shingles, "
                "fingerprint, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    hotkey,
                    code_hash,
                    dumps(payload["files"]),
                    dumps(payload["ast_features"]),
                    dumps(payload["token_shingles"]),
                    str(payload["fingerprint"]),
                    now_iso(),
                ),
            )

    async def source_snapshots(
        self, *, exclude_submission_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM submission_sources"
        params: tuple[str, ...] = ()
        if exclude_submission_id:
            query += " WHERE submission_id != ?"
            params = (exclude_submission_id,)
        query += " ORDER BY created_at DESC"
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(query, params)
        out = []
        for row in rows:
            item = dict(row)
            item["files"] = loads(item["files"])
            item["ast_features"] = loads(item["ast_features"])
            item["token_shingles"] = loads(item["token_shingles"])
            out.append(item)
        return out

    async def source_similarity_candidates(
        self, *, exclude_submission_id: str | None = None
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT ss.*, cs.architecture_id, cs.architecture_graph, "
            "af.canonical_graph_hash AS architecture_graph_hash "
            "FROM submission_sources ss "
            "LEFT JOIN component_signatures cs ON cs.submission_id=ss.submission_id "
            "LEFT JOIN architecture_families af ON af.id=cs.architecture_id"
        )
        params: tuple[str, ...] = ()
        if exclude_submission_id:
            query += " WHERE ss.submission_id != ?"
            params = (exclude_submission_id,)
        query += " ORDER BY ss.created_at DESC"
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(query, params)
        out = []
        for row in rows:
            item = dict(row)
            item["files"] = loads(item["files"])
            item["ast_features"] = loads(item["ast_features"])
            item["token_shingles"] = loads(item["token_shingles"])
            graph = loads(item.get("architecture_graph")) if item.get("architecture_graph") else {}
            item["architecture_graph"] = graph if isinstance(graph, dict) else {}
            out.append(item)
        return out

    async def store_plagiarism_review(
        self,
        *,
        submission_id: str,
        candidate_submission_id: str | None,
        similarity: float,
        verdict: bool,
        reason: str,
        violations: list[str],
        report: dict[str, Any],
    ) -> None:
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO plagiarism_reviews("
                "submission_id, candidate_submission_id, similarity, verdict, reason, violations, "
                "report, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    candidate_submission_id,
                    similarity,
                    int(verdict),
                    reason,
                    dumps(violations),
                    dumps(report),
                    now_iso(),
                ),
            )

    async def store_llm_review(
        self,
        *,
        submission_id: str,
        approved: bool,
        reason: str,
        violations: list[str],
        confidence: float,
        raw: dict[str, Any],
        mermaid: str | None = None,
        evidence: list[dict[str, Any]] | None = None,
        held: bool = False,
    ) -> None:
        raw_mermaid = raw.get("mermaid") if isinstance(raw.get("mermaid"), dict) else None
        mermaid_text = mermaid or (str(raw_mermaid.get("mermaid")) if raw_mermaid else None)
        raw_verdict = raw.get("verdict") if isinstance(raw.get("verdict"), dict) else None
        evidence_payload = _validate_evidence(evidence or raw.get("evidence") or [])
        if raw_verdict is not None:
            if mermaid_text is None:
                raise ValueError(
                    "llm_review_order_error: submit_mermaid required before submit_verdict"
                )
            await self.submit_llm_mermaid(
                submission_id=submission_id,
                mermaid=mermaid_text,
                payload=cast(dict[str, Any], raw_mermaid or {"mermaid": mermaid_text}),
            )
            # Persist the reviewed-bytes fingerprint alongside the verdict so an allow stays bound
            # to the exact bytes it reviewed; a tampered resubmission cannot reuse it (VAL-LLM-023).
            verdict_raw = dict(raw_verdict)
            reviewed_sha = raw.get("reviewed_code_sha256")
            if reviewed_sha is not None:
                verdict_raw.setdefault("reviewed_code_sha256", reviewed_sha)
            await self.submit_llm_verdict(
                submission_id=submission_id,
                approved=approved,
                reason=reason,
                violations=violations,
                confidence=confidence,
                raw=verdict_raw,
                evidence=evidence_payload or _validate_evidence(raw_verdict.get("evidence") or []),
                mermaid=mermaid_text,
                held=held,
            )
            return
        final_state = "quarantined" if held else "accepted" if approved else "rejected"
        evidence_payload = _validate_evidence(evidence_payload or raw.get("evidence") or [])
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO llm_reviews("
                "submission_id, approved, reason, violations, confidence, raw, mermaid, evidence, "
                "final_state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    int(approved),
                    reason,
                    dumps(violations),
                    confidence,
                    dumps(raw),
                    mermaid_text,
                    dumps(evidence_payload),
                    final_state,
                    now_iso(),
                    now_iso(),
                ),
            )
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state=final_state,
                actor="system",
                tool_name="deterministic_review",
                payload={
                    "approved": approved,
                    "violations": violations,
                    "evidence": evidence_payload,
                },
                reason=reason,
                idempotency_key=f"deterministic:{final_state}",
            )
            if final_state == "quarantined":
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.REJECTED.value, reason, now_iso(), submission_id),
                )

    async def record_llm_review_event(
        self,
        *,
        submission_id: str,
        state: str,
        actor: str,
        tool_name: str,
        payload: dict[str, Any] | None = None,
        reason: str = "",
        idempotency_key: str | None = None,
    ) -> None:
        async with self.database.connect() as conn:
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state=state,
                actor=actor,
                tool_name=tool_name,
                payload=payload or {},
                reason=reason,
                idempotency_key=idempotency_key,
            )

    async def submit_llm_mermaid(
        self,
        *,
        submission_id: str,
        mermaid: str,
        actor: str = "llm",
        tool_name: str = "SubmitMermaid",
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        if not mermaid.strip():
            raise ValueError("llm_review_mermaid_empty")
        event_payload = payload or {"mermaid": mermaid}
        stable_key = idempotency_key or _stable_key("SubmitMermaid", event_payload)
        async with self.database.connect() as conn:
            verdict_rows = await conn.execute_fetchall(
                "SELECT 1 FROM llm_review_events WHERE submission_id=? AND state=? LIMIT 1",
                (submission_id, "verdict_submitted"),
            )
            if verdict_rows:
                raise ValueError("llm_review_order_error: mermaid submitted after verdict")
            rows = await conn.execute_fetchall(
                "SELECT payload FROM llm_review_events WHERE submission_id=? AND tool_name=?",
                (submission_id, "SubmitMermaid"),
            )
            for row in rows:
                existing = loads(str(row["payload"]))
                if isinstance(existing, dict) and existing.get("mermaid") == mermaid:
                    return
            if rows:
                raise ValueError("llm_review_mermaid_already_submitted")
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state="mermaid_submitted",
                actor=actor,
                tool_name=tool_name,
                payload=event_payload,
                reason="LLM submitted readable Mermaid review metadata",
                idempotency_key=stable_key,
            )

    async def submit_llm_verdict(
        self,
        *,
        submission_id: str,
        approved: bool,
        reason: str,
        violations: list[str],
        confidence: float,
        raw: dict[str, Any],
        evidence: list[dict[str, Any]] | None = None,
        mermaid: str | None = None,
        held: bool = False,
        actor: str = "llm",
        tool_name: str = "SubmitVerdict",
        idempotency_key: str | None = None,
    ) -> None:
        evidence_payload = _validate_evidence(evidence or [])
        # Hard gate: honor the caller's verdict. A safety reject is TERMINAL (rejected) and is
        # NOT downgraded to a hold for lacking deterministic evidence; only an explicit held=True
        # (e.g. a fail-closed LLM error / plagiarism band) quarantines.
        final_state = "quarantined" if held else "accepted" if approved else "rejected"
        async with self.database.connect() as conn:
            mermaid_rows = await conn.execute_fetchall(
                "SELECT payload FROM llm_review_events WHERE submission_id=? AND state=? "
                "ORDER BY sequence LIMIT 1",
                (submission_id, "mermaid_submitted"),
            )
            if not mermaid_rows:
                raise ValueError(
                    "llm_review_order_error: submit_mermaid required before submit_verdict"
                )
            mermaid_payload = loads(str(list(mermaid_rows)[0]["payload"]))
            mermaid_text = mermaid or (
                str(mermaid_payload.get("mermaid")) if isinstance(mermaid_payload, dict) else None
            )
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state="verdict_submitted",
                actor=actor,
                tool_name=tool_name,
                payload={**raw, "evidence": evidence_payload},
                reason=reason,
                idempotency_key=idempotency_key or _stable_key("SubmitVerdict", raw),
            )
            now = now_iso()
            await conn.execute(
                "INSERT OR REPLACE INTO llm_reviews("
                "submission_id, approved, reason, violations, confidence, raw, mermaid, evidence, "
                "final_state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    int(approved),
                    reason,
                    dumps(violations),
                    confidence,
                    dumps(raw),
                    mermaid_text,
                    dumps(evidence_payload),
                    final_state,
                    now,
                    now,
                ),
            )
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state=final_state,
                actor="system",
                tool_name="llm_review_state_machine",
                payload={"approved": approved, "held": held, "evidence": evidence_payload},
                reason=reason,
                idempotency_key=f"final:{final_state}:{_stable_key('reason', {'reason': reason})}",
            )
            if final_state == "rejected":
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.REJECTED.value, reason, now, submission_id),
                )
            elif final_state == "quarantined":
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.REJECTED.value, reason, now, submission_id),
                )

    async def quarantine_submission_for_llm_review(
        self, *, submission_id: str, reason: str, payload: dict[str, Any]
    ) -> None:
        now = now_iso()
        async with self.database.connect() as conn:
            await conn.execute(
                "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                (SubmissionStatus.REJECTED.value, reason, now, submission_id),
            )
            await self._record_llm_review_event(
                conn,
                submission_id=submission_id,
                state="quarantined",
                actor="system",
                tool_name="llm_review_quarantine",
                payload=payload,
                reason=reason,
                idempotency_key="submission-status:quarantined",
            )

    async def leaderboard(self, epoch_id: int, limit: int = 50) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT s.hotkey, s.id, s.created_at, sc.final_score FROM scores sc "
                "JOIN submissions s ON s.id=sc.submission_id "
                "WHERE s.epoch_id=? AND s.status=? "
                "ORDER BY sc.final_score DESC, s.created_at ASC, s.id ASC",
                (epoch_id, SubmissionStatus.COMPLETED.value),
            )
        # Dedup to ONE surviving submission per hotkey (the leaderboard-best) BEFORE applying the
        # display LIMIT, so a worse same-hotkey duplicate falling inside the top window never holds
        # a board slot a distinct hotkey would otherwise fill; the board shows a full window of
        # DISTINCT hotkeys. The at-most-once-per-hotkey invariant holds, and score_rows()/weights
        # stay unlimited + correct (they dedupe over the whole epoch).
        survivors = dedupe_best_per_hotkey(
            LeaderboardRow(
                submission_id=str(row["id"]),
                hotkey=str(row["hotkey"]),
                final_score=float(cast(SupportsFloat, row["final_score"])),
                accepted_at=str(row["created_at"]),
            )
            for row in rows
        )
        ranked = rank_leaderboard(survivors)
        return [
            {"hotkey": entry.hotkey, "id": entry.submission_id, "final_score": entry.final_score}
            for entry in ranked[:limit]
        ]

    async def score_rows(self, epoch_id: int) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT s.hotkey, s.id, s.created_at, sc.final_score FROM scores sc "
                "JOIN submissions s ON s.id=sc.submission_id WHERE s.epoch_id=? AND s.status=?",
                (epoch_id, SubmissionStatus.COMPLETED.value),
            )
        # Best-per-miner supersede so weights never sum multiple same-hotkey submissions: keep the
        # single submission the canonical leaderboard total-order ranks first for each hotkey.
        survivors = dedupe_best_per_hotkey(
            LeaderboardRow(
                submission_id=str(row["id"]),
                hotkey=str(row["hotkey"]),
                final_score=float(cast(SupportsFloat, row["final_score"])),
                accepted_at=str(row["created_at"]),
            )
            for row in rows
        )
        return [
            {"hotkey": entry.hotkey, "id": entry.submission_id, "final_score": entry.final_score}
            for entry in survivors
        ]

    async def list_epochs(self, limit: int = 50) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, starts_at, ends_at, status FROM epochs "
                "ORDER BY starts_at DESC, id DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    async def list_eval_job_health(self, limit: int = 50) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, submission_id, level, status, attempts, created_at, updated_at "
                "FROM eval_jobs ORDER BY created_at DESC, id DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    async def submission_history(self, days: int = 90) -> list[dict[str, object]]:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT date(created_at) AS day, COUNT(*) AS count "
                "FROM submissions "
                "WHERE date(created_at) >= date('now', ?) "
                "GROUP BY day ORDER BY day ASC",
                (f"-{days} days",),
            )
        return [dict(row) for row in rows]

    async def gpu_status_summary(
        self,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        async with self.database.connect() as conn:
            status_rows = await conn.execute_fetchall(
                "SELECT status, COUNT(*) AS lease_count, "
                "COALESCE(SUM(gpu_count), 0) AS gpu_total "
                "FROM gpu_leases GROUP BY status",
            )
            tier_rows = await conn.execute_fetchall(
                "SELECT tier, COUNT(*) AS lease_count FROM gpu_leases GROUP BY tier",
            )
        return [dict(row) for row in status_rows], [dict(row) for row in tier_rows]

    async def list_architectures(
        self, epoch_id: int | None = None
    ) -> tuple[int, list[dict[str, object]]]:
        """Return (resolved_epoch_id, families) ranked by best final score descending.

        Mirrors the leaderboard's epoch fallback: an explicit ``epoch_id`` scopes to architectures
        with a completed submission in that epoch; ``None`` resolves to the most-recent non-empty
        epoch (the current epoch when nothing has scored yet). Each family carries its cross-epoch
        aggregates (best score / canonical submission / variant + submission counts).
        """
        async with self.database.connect() as conn:
            resolved_epoch = epoch_id
            if resolved_epoch is None:
                latest_rows = await conn.execute_fetchall(
                    "SELECT MAX(epoch_id) AS latest FROM submissions "
                    "WHERE status=? AND arch_hash IS NOT NULL",
                    (SubmissionStatus.COMPLETED.value,),
                )
                latest = list(latest_rows)[0]["latest"] if latest_rows else None
                resolved_epoch = (
                    int(cast(SupportsInt, latest))
                    if latest is not None
                    else epoch_id_for(datetime.now(UTC), self.epoch_seconds)
                )
            rows = await conn.execute_fetchall(
                "SELECT af.id AS architecture_id, af.family_hash AS arch_hash, "
                "af.display_name AS name, af.owner_hotkey AS owner_hotkey, "
                "af.q_arch_best AS best_final_score, "
                "af.canonical_submission_id AS best_submission_id, af.updated_at AS updated_at, "
                "(SELECT COUNT(*) FROM training_variants tv WHERE tv.architecture_id=af.id) "
                "AS variant_count, "
                "(SELECT COUNT(*) FROM submissions s WHERE s.arch_hash=af.family_hash) "
                "AS submission_count "
                "FROM architecture_families af "
                "WHERE EXISTS (SELECT 1 FROM submissions s2 WHERE s2.arch_hash=af.family_hash "
                "AND s2.epoch_id=? AND s2.status=?) "
                "ORDER BY af.q_arch_best DESC, af.created_at ASC, af.id ASC",
                (resolved_epoch, SubmissionStatus.COMPLETED.value),
            )
        return resolved_epoch, [dict(row) for row in rows]

    async def get_architecture(self, architecture_id: str) -> dict[str, object] | None:
        """Return one architecture family's detail fields, or ``None`` when it does not exist."""
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT af.id AS architecture_id, af.family_hash AS arch_hash, "
                "af.display_name AS name, af.owner_hotkey AS owner_hotkey, "
                "af.q_arch_best AS best_final_score, "
                "af.canonical_submission_id AS best_submission_id, "
                "af.created_at AS first_seen_at, af.updated_at AS updated_at, "
                "(SELECT COUNT(*) FROM training_variants tv WHERE tv.architecture_id=af.id) "
                "AS variant_count, "
                "(SELECT COUNT(*) FROM submissions s WHERE s.arch_hash=af.family_hash) "
                "AS submission_count "
                "FROM architecture_families af WHERE af.id=?",
                (architecture_id,),
            )
        row_list = list(rows)
        return dict(row_list[0]) if row_list else None

    async def list_training_variants(self, architecture_id: str) -> list[dict[str, object]]:
        """Return the family's training variants, best first (current-best then score)."""
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id AS variant_id, training_hash, owner_hotkey, submission_id, "
                "q_recipe AS final_score, metric_mean, metric_std, is_current_best, created_at "
                "FROM training_variants WHERE architecture_id=? "
                "ORDER BY is_current_best DESC, q_recipe DESC, created_at ASC, id ASC",
                (architecture_id,),
            )
        return [dict(row) for row in rows]

    async def best_architecture(self) -> dict[str, object] | None:
        """Return the single global (cross-epoch) best architecture family, or ``None`` if empty.

        The crown is persistent and NOT epoch-scoped: ranking by ``q_arch_best DESC`` then
        earliest-created then id makes the all-time best hold the crown until a strictly higher
        score beats it (the tiebreak keeps the crown stable across epochs).

        VAL-RESLAB-004/005: revoked provisional crowns are excluded from the emission map so a
        dead promote failure cannot keep raw-weight share. ``crown_status`` of ``none`` (legacy)
        remains eligible for compatibility with pre-ladder rows.
        """
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, owner_hotkey, q_arch_best, family_hash, "
                "COALESCE(crown_status, 'none') AS crown_status, "
                "COALESCE(param_ladder_stage, 'explore') AS param_ladder_stage, "
                "COALESCE(package_pin, '') AS package_pin "
                "FROM architecture_families "
                "WHERE COALESCE(crown_status, 'none') IN ('none', 'provisional', 'confirmed') "
                "ORDER BY q_arch_best DESC, created_at ASC, id ASC LIMIT 1",
            )
        row_list = list(rows)
        return dict(row_list[0]) if row_list else None

    async def best_training_variant(self, architecture_id: str) -> dict[str, object] | None:
        """Return the single best training variant for ``architecture_id``, or ``None`` if none.

        Revoked crowns are excluded so dead provisional training winners do not keep training
        share after a failed/lost promote (VAL-RESLAB-005).
        """
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, owner_hotkey, q_recipe, is_current_best, "
                "COALESCE(crown_status, 'none') AS crown_status, "
                "COALESCE(param_ladder_stage, 'explore') AS param_ladder_stage, "
                "COALESCE(package_pin, '') AS package_pin "
                "FROM training_variants "
                "WHERE architecture_id=? "
                "AND COALESCE(crown_status, 'none') IN ('none', 'provisional', 'confirmed') "
                "ORDER BY is_current_best DESC, q_recipe DESC, created_at ASC, id ASC LIMIT 1",
                (architecture_id,),
            )
        row_list = list(rows)
        return dict(row_list[0]) if row_list else None

    async def get_submission_curve(self, submission_id: str) -> dict[str, Any] | None:
        """Return the persisted loss curve + reconciled compute + bpb scalars for a submission.

        Reads the centralised ``submission_curves`` row (arrays / step0 / baseline / compute JSON)
        and folds in the ``prequential_bpb`` / ``bits_per_byte`` / ``tokens_consumed`` scalars from
        the ``scores.metrics`` JSON. Returns ``None`` when no curve row was persisted.
        """
        async with self.database.connect() as conn:
            curve_rows = await conn.execute_fetchall(
                "SELECT online_loss, covered_bytes_cumulative, step0_loss, baseline_nats, compute,"
                " train_series "
                "FROM submission_curves WHERE submission_id=?",
                (submission_id,),
            )
            curve_list = list(curve_rows)
            if not curve_list:
                return None
            score_rows = await conn.execute_fetchall(
                "SELECT metrics FROM scores WHERE submission_id=?",
                (submission_id,),
            )
        curve = curve_list[0]
        compute = loads(curve["compute"])
        metrics = loads(list(score_rows)[0]["metrics"]) if score_rows else {}
        metrics = metrics if isinstance(metrics, dict) else {}
        train_series_raw = curve["train_series"] if "train_series" in curve.keys() else None
        train_series = loads(train_series_raw) if train_series_raw else None
        if not isinstance(train_series, dict):
            train_series = None
        return {
            "submission_id": submission_id,
            "online_loss": loads(curve["online_loss"]),
            "covered_bytes_cumulative": loads(curve["covered_bytes_cumulative"]),
            "step0_loss": curve["step0_loss"],
            "baseline_nats": curve["baseline_nats"],
            "compute": compute if isinstance(compute, dict) else {},
            "train_series": train_series,
            "prequential_bpb": metrics.get("prequential_bpb"),
            "bits_per_byte": metrics.get("bits_per_byte"),
            "tokens_consumed": metrics.get("tokens_consumed"),
        }

    async def get_architecture_report(self, architecture_id: str) -> dict[str, object] | None:
        """Return the cached LLM auto-report row for an architecture, or ``None``."""
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT architecture_id, content, model, source_submission_id, generated_at "
                "FROM architecture_reports WHERE architecture_id=?",
                (architecture_id,),
            )
        row_list = list(rows)
        return dict(row_list[0]) if row_list else None

    async def store_architecture_report(
        self,
        *,
        architecture_id: str,
        content: str,
        model: str,
        source_submission_id: str,
        generated_at: str | None = None,
    ) -> None:
        """Upsert the cached auto-report keyed by ``architecture_id`` (source sub = cache key)."""
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO architecture_reports("
                "architecture_id, content, model, source_submission_id, generated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (architecture_id, content, model, source_submission_id, generated_at or now_iso()),
            )

    async def store_runtime_config(
        self,
        *,
        config_key: str,
        value: dict[str, Any] | list[Any] | str | int | float | bool | None,
        updated_by: str,
        schema_version: int = 1,
        effective_from: str | None = None,
        enabled: bool = True,
    ) -> None:
        updated_at = now_iso()
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT INTO runtime_config("
                "config_key, value_json, schema_version, updated_by, updated_at, "
                "effective_from, enabled) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    config_key,
                    dumps(value),
                    schema_version,
                    updated_by,
                    updated_at,
                    effective_from or updated_at,
                    int(enabled),
                ),
            )

    async def active_runtime_config_rows(self, *, at: str | None = None) -> list[dict[str, Any]]:
        effective_at = at or now_iso()
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT rc.* FROM runtime_config rc "
                "JOIN ("
                "SELECT config_key, "
                "MAX(effective_from || '|' || updated_at || '|' || id) AS marker "
                "FROM runtime_config WHERE enabled=1 AND effective_from <= ? GROUP BY config_key"
                ") active ON active.config_key=rc.config_key "
                "AND active.marker=(rc.effective_from || '|' || rc.updated_at || '|' || rc.id) "
                "ORDER BY rc.config_key",
                (effective_at,),
            )
        return [dict(row) for row in rows]

    async def runtime_config(
        self,
        settings: Any,
        *,
        official: bool = True,
    ) -> RuntimePolicy:
        rows = await self.active_runtime_config_rows()
        return resolve_runtime_policy(settings, rows, allow_sql_fallback=not official)

    async def expire_stale_held(self) -> list[str]:
        # After the LLM gating inversion a reject is TERMINAL ('rejected'), so the remaining HELD
        # source is an explicit held=True quarantine (a fail-closed LLM error / plagiarism band),
        # which has no resolve surface in v2; every stale held row is reaped to the terminal
        # rejected state. Cutoff/compare mirror requeue_orphaned_running.
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=self.held_review_timeout_seconds)
        ).isoformat()
        now = now_iso()
        reason = "review hold expired without resolution"
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id FROM submissions WHERE status=? AND updated_at < ?",
                ("held", cutoff),
            )
            expired = [str(row["id"]) for row in rows]
            if expired:
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? "
                    "WHERE status=? AND updated_at < ?",
                    (
                        SubmissionStatus.REJECTED.value,
                        reason,
                        now,
                        "held",
                        cutoff,
                    ),
                )
        return expired

    async def requeue_orphaned_running(self) -> list[str]:
        # claimed_at and cutoff share one tz-aware ISO formatter, so lexicographic
        # `<` is a valid time test (mirrors expire_stale_assignments' deadline_at).
        cutoff = (
            datetime.now(UTC) - timedelta(seconds=self.worker_claim_timeout_seconds)
        ).isoformat()
        now = now_iso()
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id FROM submissions "
                "WHERE status=? AND claimed_at IS NOT NULL AND claimed_at < ?",
                (SubmissionStatus.RUNNING.value, cutoff),
            )
            requeued = [str(row["id"]) for row in rows]
            if requeued:
                await conn.execute(
                    "UPDATE submissions SET status=?, claimed_at=NULL, updated_at=? "
                    "WHERE status=? AND claimed_at IS NOT NULL AND claimed_at < ?",
                    (
                        SubmissionStatus.PENDING.value,
                        now,
                        SubmissionStatus.RUNNING.value,
                        cutoff,
                    ),
                )
        return requeued

    async def claim_next(self) -> dict[str, object] | None:
        await self.requeue_orphaned_running()
        await self.expire_stale_held()
        claimed_at = now_iso()
        async with self.database.connect() as conn:
            row: aiosqlite.Row | None = None
            while True:
                # Atomic compare-and-swap: the AND status='pending' guard + RETURNING
                # close the read-then-write gap that let two callers double-claim.
                rows = await conn.execute_fetchall(
                    "UPDATE submissions SET status=?, updated_at=?, claimed_at=? "
                    "WHERE id=("
                    "SELECT id FROM submissions WHERE status=? ORDER BY created_at LIMIT 1"
                    ") AND status=? "
                    "RETURNING *",
                    (
                        SubmissionStatus.RUNNING.value,
                        claimed_at,
                        claimed_at,
                        SubmissionStatus.PENDING.value,
                        SubmissionStatus.PENDING.value,
                    ),
                )
                row_list = list(rows)
                if row_list:
                    row = row_list[0]
                    break
                # 0 rows: lost the race for that row -> retry the NEXT pending row;
                # only return None when no pending row remains.
                pending = await conn.execute_fetchall(
                    "SELECT 1 FROM submissions WHERE status=? LIMIT 1",
                    (SubmissionStatus.PENDING.value,),
                )
                if not list(pending):
                    return None
        data = dict(cast(Any, row))
        metadata = loads(str(data.get("metadata", "{}")))
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        return data

    async def claim_submission(self, submission_id: str) -> dict[str, object] | None:
        """Claim a SPECIFIC pending submission for the validator that was assigned it.

        Mirrors :meth:`claim_next` but targets one submission so a coordination-plane pull
        processes exactly its assigned work unit. The ``AND status='pending'`` guard + ``RETURNING``
        make the claim atomic, so two validators can never double-claim the same submission and
        re-processing an already-terminal (or in-flight) unit returns ``None`` (a safe no-op).
        """
        claimed_at = now_iso()
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "UPDATE submissions SET status=?, updated_at=?, claimed_at=? "
                "WHERE id=? AND status=? RETURNING *",
                (
                    SubmissionStatus.RUNNING.value,
                    claimed_at,
                    claimed_at,
                    submission_id,
                    SubmissionStatus.PENDING.value,
                ),
            )
        row_list = list(rows)
        if not row_list:
            return None
        data = dict(cast(Any, row_list[0]))
        metadata = loads(str(data.get("metadata", "{}")))
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        return data

    async def submission_execution_row(self, submission_id: str) -> dict[str, object] | None:
        """Return the full submission row (code + metadata) for a re-execution, ignoring status.

        Unlike :meth:`claim_submission` this takes NO claim and mutates nothing, so an already
        terminal (``completed``/``failed``) submission can be replayed for a validator audit without
        disturbing its finalized record (architecture.md 3.5; VAL-FINAL-005). ``None`` when absent.
        """
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM submissions WHERE id=?", (submission_id,)
            )
        row_list = list(rows)
        if not row_list:
            return None
        data = dict(cast(Any, row_list[0]))
        metadata = loads(str(data.get("metadata", "{}")))
        data["metadata"] = metadata if isinstance(metadata, dict) else {}
        return data

    async def list_pending_submissions(self) -> list[dict[str, object]]:
        """Return submissions awaiting re-execution (one prism work unit each), oldest first."""
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT id, hotkey, created_at FROM submissions WHERE status=? ORDER BY created_at",
                (SubmissionStatus.PENDING.value,),
            )
        return [dict(cast(Any, row)) for row in rows]

    async def count_in_flight_submissions(self) -> int:
        """Return how many submissions are currently in-flight (claimed, not yet terminal).

        A claimed submission sits in ``running`` until it reaches a terminal status, so this is the
        validator's true concurrency draw. ``run_validator_cycle`` uses it to enforce the
        concurrency-1 cap against reality rather than assuming zero, so a cycle started while a
        prism unit is still running pulls nothing more (defense-in-depth; the master assign engine
        already enforces concurrency 1).
        """
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT COUNT(*) AS count FROM submissions WHERE status=?",
                (SubmissionStatus.RUNNING.value,),
            )
        row_list = list(rows)
        return int(cast(Any, row_list[0])["count"]) if row_list else 0

    async def record_published_checkpoint(
        self,
        *,
        submission_id: str,
        attempt: int,
        validator_hotkey: str,
        checkpoint_ref: str,
        arch_hash: str = "",
    ) -> None:
        """Record a published HF checkpoint ref on the submission's assignment row.

        The validator persists a crash-recovery checkpoint and pushes it to the master, which
        publishes it (mock publisher in tests) and stores the returned ``checkpoint_ref`` here so a
        later reassignment of this submission resumes from the last PUBLIC checkpoint rather than
        from scratch (architecture.md sections 3.3, 7; VAL-PRISM-022). Upserts the per-attempt
        ``evaluation_assignments`` row so a re-push for the same attempt overwrites the ref.
        """
        timestamp = now_iso()
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "UPDATE evaluation_assignments SET checkpoint_ref=?, validator_hotkey=?, "
                "updated_at=? WHERE submission_id=? AND attempt=? RETURNING id",
                (checkpoint_ref, validator_hotkey, timestamp, submission_id, attempt),
            )
            if not list(rows):
                await conn.execute(
                    "INSERT INTO evaluation_assignments("
                    "id, submission_id, validator_hotkey, status, attempt, deadline_at, arch_hash,"
                    " metrics, checkpoint_ref, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'running', ?, '', ?, '{}', ?, ?, ?)",
                    (
                        str(uuid4()),
                        submission_id,
                        validator_hotkey,
                        attempt,
                        arch_hash,
                        checkpoint_ref,
                        timestamp,
                        timestamp,
                    ),
                )

    async def latest_checkpoint_ref(self, submission_id: str) -> str | None:
        """Return the most recent published checkpoint ref for ``submission_id`` (resume base).

        Returns the highest-attempt non-null ``checkpoint_ref`` so a reassigned run resumes from the
        last public checkpoint; ``None`` when no checkpoint was ever published (a from-scratch run).
        """
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT checkpoint_ref FROM evaluation_assignments "
                "WHERE submission_id=? AND checkpoint_ref IS NOT NULL "
                "ORDER BY attempt DESC, updated_at DESC LIMIT 1",
                (submission_id,),
            )
        row_list = list(rows)
        if not row_list:
            return None
        value = row_list[0]["checkpoint_ref"]
        return str(value) if value is not None else None

    async def submission_status(self, submission_id: str) -> str | None:
        """Return the current status of ``submission_id`` (``None`` when it does not exist)."""
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status FROM submissions WHERE id=?",
                (submission_id,),
            )
        row_list = list(rows)
        if not row_list:
            return None
        return str(row_list[0]["status"])

    async def get_work_unit_result(self, work_unit_id: str) -> dict[str, object] | None:
        """Return the accepted worker-plane result recorded for ``work_unit_id`` (else ``None``).

        The row is the idempotency/conflict key for base->prism result ingestion: a redelivery of
        the same ``manifest_sha256`` is a no-op, a different one for an already-accepted unit is a
        conflict, and the persisted claimed/effective tier is the audit-sampling record.
        """
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM work_unit_results WHERE work_unit_id=?",
                (work_unit_id,),
            )
        row_list = list(rows)
        if not row_list:
            return None
        return dict(cast(Any, row_list[0]))

    async def record_work_unit_result(
        self,
        *,
        work_unit_id: str,
        submission_id: str,
        manifest_sha256: str,
        claimed_tier: int,
        effective_tier: int,
        tier_downgraded: bool,
        worker_pubkey: str | None,
    ) -> None:
        """Record the accepted worker-plane result for ``work_unit_id`` (first accept wins).

        ``INSERT OR IGNORE`` keeps the FIRST accepted delivery authoritative: a later same-manifest
        redelivery does not rewrite it and a conflicting one is refused upstream, so the accepted
        digest + verified tier are stable for idempotency, conflict detection and audit sampling.
        """
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO work_unit_results("
                "work_unit_id, submission_id, manifest_sha256, claimed_tier, effective_tier,"
                "tier_downgraded, worker_pubkey, accepted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    work_unit_id,
                    submission_id,
                    manifest_sha256,
                    int(claimed_tier),
                    int(effective_tier),
                    1 if tier_downgraded else 0,
                    worker_pubkey,
                    now_iso(),
                ),
            )

    async def create_audit_unit(
        self,
        *,
        submission_id: str,
        origin_work_unit_id: str,
        audited_manifest_sha256: str,
        effective_tier: int,
        replication: int = 2,
        max_attempts: int = 3,
    ) -> str:
        """Create the validator audit unit for a sampled result (idempotent per submission).

        The audit unit id is DISTINCT from the primary unit id (``== submission_id``), so an audit
        never collides with the submission's own evaluation unit (VAL-PRISM-012). ``INSERT OR
        IGNORE`` keeps a single audit unit per sampled submission; the audited submission is NOT
        reverted to pending by this call. Returns the audit unit id.
        """
        from .audit import AUDIT_STATUS_PENDING, audit_unit_id_for

        audit_unit_id = audit_unit_id_for(submission_id)
        now = now_iso()
        async with self.database.connect() as conn:
            epoch_rows = await conn.execute_fetchall(
                "SELECT epoch_id FROM submissions WHERE id=?", (submission_id,)
            )
            epoch_list = list(epoch_rows)
            epoch_id = int(cast(SupportsInt, epoch_list[0]["epoch_id"])) if epoch_list else 0
            await conn.execute(
                "INSERT OR IGNORE INTO audit_units("
                "audit_unit_id, submission_id, origin_work_unit_id, epoch_id,"
                "audited_manifest_sha256, effective_tier, replication, required_capability,"
                "executor_kind, status, attempts, max_attempts, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'gpu', 'validator', ?, 0, ?, ?, ?)",
                (
                    audit_unit_id,
                    submission_id,
                    origin_work_unit_id,
                    epoch_id,
                    audited_manifest_sha256,
                    int(effective_tier),
                    int(replication),
                    AUDIT_STATUS_PENDING,
                    int(max_attempts),
                    now,
                    now,
                ),
            )
        return audit_unit_id

    async def get_audit_unit(self, audit_unit_id: str) -> dict[str, object] | None:
        """Return one audit unit row (``None`` when it does not exist)."""
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM audit_units WHERE audit_unit_id=?", (audit_unit_id,)
            )
        row_list = list(rows)
        return dict(cast(Any, row_list[0])) if row_list else None

    async def list_pending_audit_units(self) -> list[dict[str, object]]:
        """Return pending audit units joined with the audited submission's hotkey (oldest first).

        Only ``pending`` audit units are exposed on the coordination plane; a resolved (or
        exhausted) audit is no longer listed (pending-only listing semantics; VAL-PRISM-012).
        """
        from .audit import AUDIT_STATUS_PENDING

        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT au.*, s.hotkey AS hotkey FROM audit_units au "
                "JOIN submissions s ON s.id=au.submission_id "
                "WHERE au.status=? ORDER BY au.created_at, au.audit_unit_id",
                (AUDIT_STATUS_PENDING,),
            )
        return [dict(cast(Any, row)) for row in rows]

    async def claim_audit_unit(
        self, audit_unit_id: str, *, claimant: str, lease_seconds: float
    ) -> bool:
        """Atomically claim a pending audit unit under a lease; ``True`` when this caller won it.

        A single-consumer guard for the validator audit cycle: in a MULTI-validator deployment each
        validator enumerates the same pending audits, so without a claim they would all redundantly
        replay every one. The claim is an atomic CAS -- only the caller whose ``UPDATE`` matches
        (the unit is still ``pending`` AND unclaimed or its lease has expired) stamps ``claimed_at``
        and wins; concurrent claimers get ``False`` and skip. The claim is orthogonal to the
        lifecycle ``status`` (the unit stays ``pending``), so :func:`resolve_audit_unit` is
        unaffected. A crashed claimant's lease expires (``claimed_at <= now - lease_seconds``) and
        the audit becomes reclaimable, so a claim is never permanently orphaned.
        """
        from .audit import AUDIT_STATUS_PENDING

        now = datetime.now(UTC)
        now_str = now.isoformat()
        cutoff = (now - timedelta(seconds=max(lease_seconds, 0.0))).isoformat()
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "UPDATE audit_units SET claimed_at=?, claimed_by=? "
                "WHERE audit_unit_id=? AND status=? "
                "AND (claimed_at IS NULL OR claimed_at <= ?) RETURNING audit_unit_id",
                (now_str, claimant, audit_unit_id, AUDIT_STATUS_PENDING, cutoff),
            )
        return bool(list(rows))

    async def record_audit_resolution(
        self,
        *,
        audit_unit_id: str,
        status: str,
        attempts: int,
        resolution: str | None,
        resolved_manifest_sha256: str | None,
        error: str | None,
    ) -> None:
        """Persist an audit unit's new lifecycle state after a resolution attempt.

        The per-audit claim is cleared (``claimed_at``/``claimed_by`` -> NULL) on every resolution
        so a unit returned to ``pending`` for a bounded re-audit is immediately reclaimable by any
        validator rather than blocked until the previous claimant's lease expires.
        """
        async with self.database.connect() as conn:
            await conn.execute(
                "UPDATE audit_units SET status=?, attempts=?, resolution=?, "
                "resolved_manifest_sha256=?, error=?, claimed_at=NULL, claimed_by=NULL, "
                "updated_at=? WHERE audit_unit_id=?",
                (
                    status,
                    int(attempts),
                    resolution,
                    resolved_manifest_sha256,
                    error,
                    now_iso(),
                    audit_unit_id,
                ),
            )

    async def record_worker_fault(
        self,
        *,
        audit_unit_id: str,
        submission_id: str,
        worker_pubkey: str | None,
        audited_manifest_sha256: str,
        replay_manifest_sha256: str,
        reason: str,
    ) -> None:
        """Record a fault against the worker whose manifest diverged from the validator replay.

        Written on an audit MISMATCH (architecture.md 4; VAL-FINAL-005): the audited submission's
        finalized result named ``worker_pubkey`` as its producer, and the authoritative replay
        proved that manifest wrong. The fault is observational (it never mutates the submission or
        any worker record); it is the durable record that this worker lied on this audited unit.
        """
        async with self.database.connect() as conn:
            await conn.execute(
                "INSERT INTO worker_faults("
                "audit_unit_id, submission_id, worker_pubkey, audited_manifest_sha256,"
                "replay_manifest_sha256, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    audit_unit_id,
                    submission_id,
                    worker_pubkey,
                    audited_manifest_sha256,
                    replay_manifest_sha256,
                    reason,
                    now_iso(),
                ),
            )

    async def list_worker_faults(
        self, *, submission_id: str | None = None
    ) -> list[dict[str, object]]:
        """Return recorded worker faults (optionally scoped to one submission), oldest first."""
        async with self.database.connect() as conn:
            if submission_id is None:
                rows = await conn.execute_fetchall("SELECT * FROM worker_faults ORDER BY id")
            else:
                rows = await conn.execute_fetchall(
                    "SELECT * FROM worker_faults WHERE submission_id=? ORDER BY id",
                    (submission_id,),
                )
        return [dict(cast(Any, row)) for row in rows]

    async def invalidate_submission_score(self, submission_id: str, *, reason: str) -> bool:
        """Invalidate a finalized submission's score and recompute crown/weights aggregates.

        Deletes the ``scores`` row and moves the submission to ``failed`` (so it drops out of the
        epoch leaderboard, ``score_rows`` and weights; VAL-PRISM-013), then recomputes the affected
        architecture family's ``q_arch_best``/``canonical_submission_id`` and its training variants
        from the REMAINING valid (completed + scored) submissions so the crown falls back to the
        best non-invalidated submission or to the BURN state, and ``is_current_best`` moves off the
        invalidated submission (VAL-PRISM-023). Returns ``True`` when a submission row existed.
        """
        now = now_iso()
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT arch_hash FROM submissions WHERE id=?", (submission_id,)
            )
            row_list = list(rows)
            if not row_list:
                return False
            arch_hash = row_list[0]["arch_hash"]
            await conn.execute("DELETE FROM scores WHERE submission_id=?", (submission_id,))
            await conn.execute(
                "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                (SubmissionStatus.FAILED.value, reason, now, submission_id),
            )
            if arch_hash:
                await self._recompute_family_after_invalidation(
                    conn,
                    family_hash=str(arch_hash),
                    invalidated_submission_id=submission_id,
                    now=now,
                )
        return True

    async def _recompute_family_after_invalidation(
        self,
        conn: aiosqlite.Connection,
        *,
        family_hash: str,
        invalidated_submission_id: str,
        now: str,
    ) -> None:
        """Recompute an architecture family's crown aggregates after an invalidation."""
        fam_rows = await conn.execute_fetchall(
            "SELECT id FROM architecture_families WHERE family_hash=?", (family_hash,)
        )
        fam_list = list(fam_rows)
        if not fam_list:
            return
        architecture_id = str(fam_list[0]["id"])
        best_rows = await conn.execute_fetchall(
            "SELECT s.id AS sid, s.hotkey AS owner, sc.final_score AS fs FROM submissions s "
            "JOIN scores sc ON sc.submission_id=s.id "
            "WHERE s.arch_hash=? AND s.status=? "
            "ORDER BY sc.final_score DESC, s.created_at ASC, s.id ASC LIMIT 1",
            (family_hash, SubmissionStatus.COMPLETED.value),
        )
        best = list(best_rows)
        if best:
            best_score = float(cast(SupportsFloat, best[0]["fs"]))
            survivor_id = str(best[0]["sid"])
            # Advance the weight-bearing owner_hotkey (and owner_submission_id) to the surviving
            # best submission's owner. get_weights rewards owner_hotkey for the architecture share,
            # so a proven-faulty owner must not keep it when a co-owner's valid submission survives.
            await conn.execute(
                "UPDATE architecture_families SET canonical_submission_id=?, owner_hotkey=?, "
                "owner_submission_id=?, q_arch_best=?, updated_at=? WHERE id=?",
                (
                    survivor_id,
                    str(best[0]["owner"]),
                    survivor_id,
                    best_score,
                    now,
                    architecture_id,
                ),
            )
        else:
            # No valid submission remains for the family: drop q_arch_best to 0 so the crown falls
            # to another family or BURNs (get_weights treats a non-positive best as no crown).
            await conn.execute(
                "UPDATE architecture_families SET q_arch_best=0.0, updated_at=? WHERE id=?",
                (now, architecture_id),
            )
        # The invalidated submission was a training variant's representative; drop that variant (its
        # only evidence is gone) and recompute is_current_best across the family's survivors.
        await conn.execute(
            "DELETE FROM training_variants WHERE architecture_id=? AND submission_id=?",
            (architecture_id, invalidated_submission_id),
        )
        await conn.execute(
            "UPDATE training_variants SET is_current_best=0 WHERE architecture_id=?",
            (architecture_id,),
        )
        await conn.execute(
            "UPDATE training_variants SET is_current_best=1 WHERE id=("
            "SELECT id FROM training_variants WHERE architecture_id=? "
            "ORDER BY q_recipe DESC, created_at ASC, id ASC LIMIT 1)",
            (architecture_id,),
        )

    async def container_job_attempt_count(self, submission_id: str, level: str) -> int:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT COUNT(*) AS count FROM eval_jobs WHERE submission_id=? AND level=?",
                (submission_id, level),
            )
        return int(list(rows)[0]["count"])

    async def latest_run_manifest_path(self, submission_id: str, level: str) -> str | None:
        """On-disk path of the most recent eval job's ``prism_run_manifest.v2.json`` for a run.

        Used at successful finalization to hash the exact on-disk manifest bytes for the
        ExecutionProof (architecture.md 3.4). Returns ``None`` when no eval job recorded a manifest.
        """
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT run_manifest_path FROM eval_jobs WHERE submission_id=? AND level=? "
                "AND run_manifest_path IS NOT NULL "
                "ORDER BY attempts DESC, created_at DESC LIMIT 1",
                (submission_id, level),
            )
        row_list = list(rows)
        if not row_list:
            return None
        value = row_list[0]["run_manifest_path"]
        return str(value) if value is not None else None

    async def latest_retryable_container_job(
        self, submission_id: str, level: str
    ) -> dict[str, object] | None:
        async with self.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT * FROM eval_jobs WHERE submission_id=? AND level=? "
                "AND status='infra_failed' AND infra_retryable=1 "
                "AND artifact_output_path IS NOT NULL AND run_manifest_path IS NOT NULL "
                "ORDER BY attempts DESC, created_at DESC LIMIT 1",
                (submission_id, level),
            )
        return dict(list(rows)[0]) if rows else None

    async def _record_llm_review_event(
        self,
        conn: aiosqlite.Connection,
        *,
        submission_id: str,
        state: str,
        actor: str,
        tool_name: str,
        payload: dict[str, Any],
        reason: str,
        idempotency_key: str | None,
    ) -> None:
        stable_key = idempotency_key or _stable_key(tool_name, payload)
        sequence_rows = await conn.execute_fetchall(
            "SELECT COALESCE(MAX(sequence), 0) AS sequence FROM llm_review_events "
            "WHERE submission_id=?",
            (submission_id,),
        )
        sequence = int(list(sequence_rows)[0]["sequence"]) + 1
        await conn.execute(
            "INSERT OR IGNORE INTO llm_review_events("
            "id, submission_id, sequence, state, actor, tool_name, idempotency_key, payload, "
            "reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid4()),
                submission_id,
                sequence,
                state,
                actor,
                tool_name,
                stable_key,
                dumps(payload),
                reason,
                now_iso(),
            ),
        )


def _validate_evidence(items: Any) -> list[dict[str, Any]]:
    if not items:
        return []
    if not isinstance(items, list):
        items = [items]
    return [DeterministicEvidence.model_validate(item).model_dump(mode="json") for item in items]


def _stable_key(tool_name: str, payload: dict[str, Any]) -> str:
    payload_hash = sha256(dumps(payload).encode("utf-8")).hexdigest()
    return f"{tool_name}:{payload_hash}"
