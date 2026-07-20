from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.analyzer import base_skeleton
from agent_challenge.core.config import settings
from agent_challenge.core.models import (
    AgentSubmission,
    AnalysisRun,
    EvaluationJob,
    PythonAstFeature,
    SimilarityMatch,
)

ALGORITHM_VERSION = "python-ast-similarity-v1"
MATCH_KIND = "python_ast_similarity"

_FEATURE_WEIGHTS = {
    "ast_hash": 0.35,
    "name_shingles": 0.25,
    "call_shingles": 0.20,
    "signature": 0.20,
}


@dataclass(frozen=True)
class AstFileFeatureSet:
    file_path: str
    artifact_hash: str | None
    file_hash: str | None
    ast_hash: str | None
    function_count: int
    class_count: int
    import_count: int
    call_shingles: frozenset[str]
    name_shingles: frozenset[str]


@dataclass(frozen=True)
class SubmissionFeatureSet:
    analysis_run_id: int | None
    submission_id: int | None
    artifact_hash: str | None
    files: tuple[AstFileFeatureSet, ...]


@dataclass(frozen=True)
class FilePairScore:
    source_file_path: str
    matched_file_path: str
    source_file_hash: str | None
    matched_file_hash: str | None
    score_percent: float
    ast_hash_match: bool
    feature_scores: dict[str, float]


@dataclass(frozen=True)
class SubmissionSimilarityScore:
    source_submission_id: int | None
    matched_submission_id: int | None
    matched_analysis_run_id: int | None
    matched_artifact_uri: str | None
    score_percent: float
    risk_band: str
    top_file_pairs: tuple[FilePairScore, ...]
    algorithm_version: str = ALGORITHM_VERSION
    corpus_snapshot_at: datetime | None = None

    def evidence_json(self) -> str:
        snapshot_at = self.corpus_snapshot_at or datetime.now(UTC)
        return _stable_json(_evidence(self, snapshot_at))


def feature_rows_to_feature_set(
    rows: Iterable[PythonAstFeature | dict[str, object]],
    *,
    analysis_run_id: int | None = None,
    submission_id: int | None = None,
) -> SubmissionFeatureSet:
    return build_submission_feature_set(
        rows,
        analysis_run_id=analysis_run_id,
        submission_id=submission_id,
    )


def risk_band(score_percent: float) -> str:
    if score_percent >= settings.analyzer_similarity_high_risk_threshold:
        return "high"
    if score_percent >= settings.analyzer_similarity_medium_risk_threshold:
        return "medium"
    return "low"


def build_submission_feature_set(
    rows: Iterable[PythonAstFeature | dict[str, object]],
    *,
    analysis_run_id: int | None = None,
    submission_id: int | None = None,
) -> SubmissionFeatureSet:
    grouped: dict[str, dict[str, object]] = {}
    artifact_hash: str | None = None

    for row in rows:
        file_path = str(_row_value(row, "file_path") or "")
        feature_name = _feature_name(str(_row_value(row, "feature_key") or ""), file_path)
        feature_value = str(_row_value(row, "feature_value") or "")
        metadata = _json_object(str(_row_value(row, "metadata_json") or "{}"))
        if isinstance(metadata.get("artifact_hash"), str):
            artifact_hash = str(metadata["artifact_hash"])
        if not file_path:
            continue

        bucket = grouped.setdefault(file_path, {"file_path": file_path})
        bucket[feature_name] = feature_value
        if isinstance(metadata.get("artifact_hash"), str):
            bucket["artifact_hash"] = str(metadata["artifact_hash"])
        if isinstance(metadata.get("file_hash"), str):
            bucket["file_hash"] = str(metadata["file_hash"])

    files = tuple(
        _file_feature_set(values)
        for _, values in sorted(grouped.items())
        if values.get("parser_status") == "ok" and values.get("ast_hash")
    )
    return SubmissionFeatureSet(
        analysis_run_id=analysis_run_id,
        submission_id=submission_id,
        artifact_hash=artifact_hash,
        files=files,
    )


def score_submission_similarity(
    source: SubmissionFeatureSet,
    matched: SubmissionFeatureSet,
    *,
    matched_artifact_uri: str | None = None,
    corpus_snapshot_at: datetime | None = None,
) -> SubmissionSimilarityScore | None:
    if not source.files or not matched.files:
        return None
    if source.artifact_hash and source.artifact_hash == matched.artifact_hash:
        return None

    # Subtract the shared baseagent skeleton from BOTH sides so identical base
    # files contribute nothing and only each submission's DELTA is scored. With
    # no delta left the pair is not a meaningful match (same shape as the
    # byte-identical early return above).
    fingerprint = base_skeleton.base_skeleton_fingerprint()
    source_files = _subtract_base_skeleton(source.files, fingerprint)
    matched_files = _subtract_base_skeleton(matched.files, fingerprint)
    if not source_files or not matched_files:
        return None

    best_pairs: list[FilePairScore] = []
    for source_file in source_files:
        best = max(
            (_score_file_pair(source_file, matched_file) for matched_file in matched_files),
            key=lambda pair: (pair.score_percent, pair.matched_file_path),
        )
        best_pairs.append(best)

    score_percent = round(
        sum(pair.score_percent for pair in best_pairs) / len(best_pairs),
        2,
    )
    top_pairs = tuple(
        sorted(best_pairs, key=lambda pair: (-pair.score_percent, pair.source_file_path))[
            : settings.analyzer_similarity_top_file_pair_limit
        ]
    )
    return SubmissionSimilarityScore(
        source_submission_id=source.submission_id,
        matched_submission_id=matched.submission_id,
        matched_analysis_run_id=matched.analysis_run_id,
        matched_artifact_uri=matched_artifact_uri,
        score_percent=score_percent,
        risk_band=risk_band(score_percent),
        top_file_pairs=top_pairs,
        corpus_snapshot_at=corpus_snapshot_at,
    )


def score_feature_sets(
    source: SubmissionFeatureSet,
    matched: SubmissionFeatureSet,
    *,
    matched_artifact_uri: str | None = None,
    corpus_snapshot_at: datetime | None = None,
) -> SubmissionSimilarityScore | None:
    return score_submission_similarity(
        source,
        matched,
        matched_artifact_uri=matched_artifact_uri,
        corpus_snapshot_at=corpus_snapshot_at,
    )


def score_against_corpus(
    source: SubmissionFeatureSet,
    corpus: Iterable[tuple[SubmissionFeatureSet, str | None]],
) -> tuple[SubmissionSimilarityScore, ...]:
    scores: list[SubmissionSimilarityScore] = []
    for matched, matched_artifact_uri in corpus:
        score = score_submission_similarity(
            source,
            matched,
            matched_artifact_uri=matched_artifact_uri,
        )
        if score is not None:
            scores.append(score)
    return tuple(
        sorted(
            scores,
            key=lambda item: (-item.score_percent, item.matched_submission_id or 0),
        )
    )


def build_similarity_match_rows(
    *,
    analysis_run_id: int,
    scores: Iterable[SubmissionSimilarityScore],
    corpus_snapshot_at: datetime | None = None,
) -> list[SimilarityMatch]:
    snapshot_at = corpus_snapshot_at or datetime.now(UTC)
    return [
        SimilarityMatch(
            analysis_run_id=analysis_run_id,
            source_submission_id=score.source_submission_id,
            matched_submission_id=score.matched_submission_id,
            matched_artifact_uri=score.matched_artifact_uri,
            match_kind=MATCH_KIND,
            score=score.score_percent,
            evidence_json=_stable_json(_evidence(score, snapshot_at)),
        )
        for score in scores
    ]


async def build_same_challenge_similarity_matches(
    session: AsyncSession,
    *,
    analysis_run_id: int,
    corpus_snapshot_at: datetime | None = None,
) -> list[SimilarityMatch]:
    current_run = await session.get(AnalysisRun, analysis_run_id)
    if current_run is None:
        raise ValueError(f"analysis_run_id {analysis_run_id} was not found")

    current_job = (
        await session.get(EvaluationJob, current_run.job_id) if current_run.job_id else None
    )
    current_rows = await _feature_rows_for_runs(session, [analysis_run_id])
    source = build_submission_feature_set(
        current_rows.get(analysis_run_id, ()),
        analysis_run_id=current_run.id,
        submission_id=current_run.submission_id,
    )

    prior_runs = await _prior_same_challenge_runs(session, current_run, current_job)
    if not prior_runs:
        return []

    rows_by_run_id = await _feature_rows_for_runs(session, [run.id for run, _ in prior_runs])
    latest_by_submission: dict[int, tuple[SubmissionFeatureSet, str | None]] = {}
    for run, submission in prior_runs:
        feature_set = build_submission_feature_set(
            rows_by_run_id.get(run.id, ()),
            analysis_run_id=run.id,
            submission_id=run.submission_id,
        )
        latest_by_submission[run.submission_id] = (feature_set, submission.artifact_uri)

    scores = score_against_corpus(source, latest_by_submission.values())
    return build_similarity_match_rows(
        analysis_run_id=analysis_run_id,
        scores=scores,
        corpus_snapshot_at=corpus_snapshot_at,
    )


async def build_similarity_matches_for_analysis_run(
    session: AsyncSession,
    analysis_run_id: int,
    *,
    corpus_snapshot_at: datetime | None = None,
) -> list[SimilarityMatch]:
    return await build_same_challenge_similarity_matches(
        session,
        analysis_run_id=analysis_run_id,
        corpus_snapshot_at=corpus_snapshot_at,
    )


async def persist_same_challenge_similarity_matches(
    session: AsyncSession,
    *,
    analysis_run_id: int,
    corpus_snapshot_at: datetime | None = None,
) -> list[SimilarityMatch]:
    rows = await build_same_challenge_similarity_matches(
        session,
        analysis_run_id=analysis_run_id,
        corpus_snapshot_at=corpus_snapshot_at,
    )
    session.add_all(rows)
    await session.flush()
    return rows


def _subtract_base_skeleton(
    files: tuple[AstFileFeatureSet, ...],
    fingerprint: base_skeleton.BaseSkeletonFingerprint,
) -> tuple[AstFileFeatureSet, ...]:
    if not fingerprint:
        return files
    return tuple(file for file in files if not _is_base_skeleton_file(file, fingerprint))


def _is_base_skeleton_file(
    file: AstFileFeatureSet,
    fingerprint: base_skeleton.BaseSkeletonFingerprint,
) -> bool:
    if file.ast_hash is not None and file.ast_hash in fingerprint.ast_hashes:
        return True
    return file.file_hash is not None and file.file_hash in fingerprint.file_hashes


def _score_file_pair(source: AstFileFeatureSet, matched: AstFileFeatureSet) -> FilePairScore:
    feature_scores = {
        "ast_hash": 1.0 if source.ast_hash and source.ast_hash == matched.ast_hash else 0.0,
        "name_shingles": _jaccard(source.name_shingles, matched.name_shingles),
        "call_shingles": _jaccard(source.call_shingles, matched.call_shingles),
        "signature": _signature_similarity(source, matched),
    }
    score = (
        100.0
        if feature_scores["ast_hash"] == 1.0
        else round(
            sum(feature_scores[name] * weight for name, weight in _FEATURE_WEIGHTS.items()) * 100,
            2,
        )
    )
    return FilePairScore(
        source_file_path=source.file_path,
        matched_file_path=matched.file_path,
        source_file_hash=source.file_hash,
        matched_file_hash=matched.file_hash,
        score_percent=score,
        ast_hash_match=feature_scores["ast_hash"] == 1.0,
        feature_scores={name: round(value, 4) for name, value in feature_scores.items()},
    )


def _signature_similarity(source: AstFileFeatureSet, matched: AstFileFeatureSet) -> float:
    return (
        sum(
            _count_similarity(source_count, matched_count)
            for source_count, matched_count in (
                (source.function_count, matched.function_count),
                (source.class_count, matched.class_count),
                (source.import_count, matched.import_count),
            )
        )
        / 3
    )


def _count_similarity(source: int, matched: int) -> float:
    maximum = max(source, matched)
    if maximum == 0:
        return 1.0
    return 1.0 - (abs(source - matched) / maximum)


def _jaccard(source: frozenset[str], matched: frozenset[str]) -> float:
    if not source and not matched:
        return 0.0
    if not source or not matched:
        return 0.0
    return len(source & matched) / len(source | matched)


def _file_feature_set(values: dict[str, object]) -> AstFileFeatureSet:
    return AstFileFeatureSet(
        file_path=str(values["file_path"]),
        artifact_hash=_optional_str(values.get("artifact_hash")),
        file_hash=_optional_str(values.get("file_hash")),
        ast_hash=_optional_str(values.get("ast_hash")),
        function_count=_int_value(values.get("function_count")),
        class_count=_int_value(values.get("class_count")),
        import_count=_int_value(values.get("import_count")),
        call_shingles=frozenset(_json_list(values.get("call_shingles"))),
        name_shingles=frozenset(_json_list(values.get("name_shingles"))),
    )


def _evidence(score: SubmissionSimilarityScore, snapshot_at: datetime) -> dict[str, object]:
    return {
        "algorithm_version": ALGORITHM_VERSION,
        "corpus_snapshot_at": snapshot_at.astimezone(UTC).isoformat(),
        "matched_analysis_run_id": score.matched_analysis_run_id,
        "matched_submission_id": score.matched_submission_id,
        "risk_band": score.risk_band,
        "score_percent": score.score_percent,
        "top_file_pairs": [
            {
                "ast_hash_match": pair.ast_hash_match,
                "feature_scores": pair.feature_scores,
                "matched_file_hash": pair.matched_file_hash,
                "matched_file_path": pair.matched_file_path,
                "score_percent": pair.score_percent,
                "source_file_hash": pair.source_file_hash,
                "source_file_path": pair.source_file_path,
            }
            for pair in score.top_file_pairs
        ],
    }


async def _prior_same_challenge_runs(
    session: AsyncSession,
    current_run: AnalysisRun,
    current_job: EvaluationJob | None,
) -> list[tuple[AnalysisRun, AgentSubmission]]:
    statement = (
        select(AnalysisRun, AgentSubmission, EvaluationJob)
        .join(AgentSubmission, AnalysisRun.submission_id == AgentSubmission.id)
        .outerjoin(EvaluationJob, AnalysisRun.job_id == EvaluationJob.id)
        .where(AnalysisRun.id < current_run.id)
        .where(AnalysisRun.submission_id != current_run.submission_id)
        .order_by(AnalysisRun.id)
    )
    if current_job is not None:
        statement = statement.where(
            EvaluationJob.selected_tasks_json == current_job.selected_tasks_json
        )
    result = await session.execute(statement)
    return [(run, submission) for run, submission, _job in result.all()]


async def _feature_rows_for_runs(
    session: AsyncSession,
    analysis_run_ids: list[int],
) -> dict[int, list[PythonAstFeature]]:
    if not analysis_run_ids:
        return {}
    result = await session.execute(
        select(PythonAstFeature)
        .where(PythonAstFeature.analysis_run_id.in_(analysis_run_ids))
        .order_by(PythonAstFeature.analysis_run_id, PythonAstFeature.feature_key)
    )
    rows_by_run_id: dict[int, list[PythonAstFeature]] = {run_id: [] for run_id in analysis_run_ids}
    for row in result.scalars().all():
        rows_by_run_id.setdefault(row.analysis_run_id, []).append(row)
    return rows_by_run_id


def _feature_name(feature_key: str, file_path: str) -> str:
    prefix = f"{file_path}:"
    if file_path and feature_key.startswith(prefix):
        return feature_key[len(prefix) :]
    return feature_key


def _row_value(row: PythonAstFeature | dict[str, object], name: str) -> object:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name)


def _json_object(value: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _json_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(loaded, list):
        return ()
    return tuple(str(item) for item in loaded)


def _optional_str(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _int_value(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
