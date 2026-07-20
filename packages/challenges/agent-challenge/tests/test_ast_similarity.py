from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from agent_challenge.analyzer import base_skeleton
from agent_challenge.analyzer.ast_features import (
    build_python_ast_feature_rows,
    extract_python_ast_features,
)
from agent_challenge.analyzer.similarity import (
    ALGORITHM_VERSION,
    MATCH_KIND,
    build_same_challenge_similarity_matches,
    build_similarity_match_rows,
    build_submission_feature_set,
    score_submission_similarity,
)
from agent_challenge.core.db import Base
from agent_challenge.models import AgentSubmission, AnalysisRun, EvaluationJob
from agent_challenge.submissions.artifacts import store_zip_bytes

ENTRYPOINT_SOURCE = "class Agent:\n    pass\n"


def agent_source(contents: str | bytes) -> str | bytes:
    if isinstance(contents, bytes):
        return contents
    if "class Agent" in contents:
        return contents
    return f"{ENTRYPOINT_SOURCE}\n{contents}"


def test_identical_normalized_python_scores_at_least_99_percent(tmp_path: Path) -> None:
    source = _feature_set(
        {
            "agent.py": (
                "class Agent:\n"
                "    pass\n\n"
                "def solve(value):\n"
                "    result = value + 1\n"
                "    return result\n"
            )
        },
        tmp_path / "source",
        analysis_run_id=1,
        submission_id=1,
    )
    matched = _feature_set(
        {
            "agent.py": (
                "class Agent:\n"
                "    pass\n\n"
                "def solve(x):\n\n"
                "    renamed = x + 1\n"
                "    return renamed\n"
            )
        },
        tmp_path / "matched",
        analysis_run_id=2,
        submission_id=2,
    )

    score = score_submission_similarity(source, matched, matched_artifact_uri="/tmp/matched.zip")

    assert score is not None
    assert score.score_percent >= 99
    assert score.risk_band == "high"
    assert score.top_file_pairs[0].ast_hash_match is True


def test_variable_rename_and_format_change_score_none_after_base_subtraction(
    tmp_path: Path, monkeypatch
) -> None:
    source = _feature_set(
        {
            "agent.py": (
                "import os\nimport sys\n\n"
                "class Agent:\n"
                "    def solve(self, value):\n"
                "        helper = str(value).strip()\n"
                "        return helper.lower()\n"
            )
        },
        tmp_path / "source",
        analysis_run_id=1,
        submission_id=1,
    )
    matched = _feature_set(
        {
            "agent.py": (
                "import sys\nimport os\n\n\n"
                "class Agent:\n"
                "    def solve(self, payload):\n"
                "        normalized = str(payload).strip()\n"
                "        return normalized.lower()\n"
            )
        },
        tmp_path / "matched",
        analysis_run_id=2,
        submission_id=2,
    )

    # The renamed/reformatted fork is rename/format-invariant to the base
    # skeleton: once the base is subtracted there is no delta left, so the pair no
    # longer flags as similar (the false positive this feature removes).
    _register_base_skeleton(monkeypatch, source)

    assert score_submission_similarity(source, matched) is None


def test_unrelated_structures_score_low_similarity(tmp_path: Path) -> None:
    source = _feature_set(
        {"agent.py": "def solve(value):\n    return value + 1\n"},
        tmp_path / "source",
        analysis_run_id=1,
        submission_id=1,
    )
    matched = _feature_set(
        {
            "agent.py": (
                "import json\n\n"
                "class Runner:\n"
                "    def load(self, text):\n"
                "        data = json.loads(text)\n"
                "        return [item['name'] for item in data]\n"
            )
        },
        tmp_path / "matched",
        analysis_run_id=2,
        submission_id=2,
    )

    score = score_submission_similarity(source, matched)

    assert score is not None
    assert score.score_percent < 70
    assert score.risk_band == "low"


def test_base_skeleton_subtracted_from_score(tmp_path: Path, monkeypatch) -> None:
    base_agent = "class Agent:\n    pass\n\ndef base_helper(value):\n    return value + 1\n"
    renamed_base = "class Agent:\n    pass\n\ndef base_helper(other):\n    return other + 1\n"

    # Register agent.py as the shared baseagent skeleton.
    base_only = _feature_set(
        {"agent.py": base_agent}, tmp_path / "base", analysis_run_id=9, submission_id=9
    )
    _register_base_skeleton(monkeypatch, base_only)

    # Forks whose only shared code is the (renamed) base skeleton have no delta.
    fork_a = _feature_set(
        {"agent.py": base_agent}, tmp_path / "a", analysis_run_id=1, submission_id=1
    )
    fork_b = _feature_set(
        {"agent.py": renamed_base}, tmp_path / "b", analysis_run_id=2, submission_id=2
    )
    assert score_submission_similarity(fork_a, fork_b) is None

    # Forks sharing a NON-base delta still score high on that delta alone.
    delta_a = _feature_set(
        {"agent.py": base_agent, "solver.py": "def solve(x):\n    return x * 2\n"},
        tmp_path / "da",
        analysis_run_id=3,
        submission_id=3,
    )
    delta_b = _feature_set(
        {"agent.py": renamed_base, "solver.py": "def solve(z):\n    return z * 2\n"},
        tmp_path / "db",
        analysis_run_id=4,
        submission_id=4,
    )
    score = score_submission_similarity(delta_a, delta_b)

    assert score is not None
    assert score.score_percent >= 90
    assert score.risk_band == "high"
    assert [pair.source_file_path for pair in score.top_file_pairs] == ["solver.py"]


def test_similarity_threshold_knobs_are_live(tmp_path: Path, monkeypatch) -> None:
    source = _feature_set(
        {"agent.py": "class Agent:\n    pass\n", "solver.py": "def solve(x):\n    return x + 1\n"},
        tmp_path / "s",
        analysis_run_id=1,
        submission_id=1,
    )
    matched = _feature_set(
        {"agent.py": "class Agent:\n    pass\n", "solver.py": "def solve(y):\n    return y + 1\n"},
        tmp_path / "m",
        analysis_run_id=2,
        submission_id=2,
    )

    baseline = score_submission_similarity(source, matched)
    assert baseline is not None
    assert baseline.score_percent >= 99
    assert baseline.risk_band == "high"

    # Raising the high cutoff above the score reclassifies the band -> knob is live.
    monkeypatch.setattr(
        "agent_challenge.analyzer.similarity.settings.analyzer_similarity_high_risk_threshold",
        101.0,
    )
    monkeypatch.setattr(
        "agent_challenge.analyzer.similarity.settings.analyzer_similarity_medium_risk_threshold",
        50.0,
    )
    reclassified = score_submission_similarity(source, matched)
    assert reclassified is not None
    assert reclassified.risk_band == "medium"

    # The top-file-pair cap is read from settings too (2 source files -> capped to 1).
    monkeypatch.setattr(
        "agent_challenge.analyzer.similarity.settings.analyzer_similarity_top_file_pair_limit",
        1,
    )
    capped = score_submission_similarity(source, matched)
    assert capped is not None
    assert len(capped.top_file_pairs) == 1


def test_base_skeleton_fail_open_when_manifest_missing(tmp_path: Path) -> None:
    missing = base_skeleton.load_base_skeleton_fingerprint(tmp_path / "nope.json")
    assert bool(missing) is False
    assert missing.ast_hashes == frozenset()
    assert missing.file_hashes == frozenset()

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{ not json", encoding="utf-8")
    assert bool(base_skeleton.load_base_skeleton_fingerprint(invalid)) is False


def test_base_skeleton_ignores_malformed_manifest(tmp_path: Path) -> None:
    not_a_dict = tmp_path / "list.json"
    not_a_dict.write_text("[]", encoding="utf-8")
    assert bool(base_skeleton.load_base_skeleton_fingerprint(not_a_dict)) is False

    partial = tmp_path / "partial.json"
    partial.write_text('{"ast_hashes": "nope", "file_hashes": ["h1", ""]}', encoding="utf-8")
    fingerprint = base_skeleton.load_base_skeleton_fingerprint(partial)
    assert fingerprint.ast_hashes == frozenset()
    assert fingerprint.file_hashes == frozenset({"h1"})


def test_base_skeleton_default_manifest_is_packaged() -> None:
    # The manifest ships next to base_skeleton.py inside the analyzer package, so
    # the default fingerprint is non-empty in every image/install (no config or
    # repo-relative ``golden/`` path required).
    base_skeleton.reset_base_skeleton_cache()
    try:
        fingerprint = base_skeleton.base_skeleton_fingerprint()
        assert bool(fingerprint) is True
        assert fingerprint.ast_hashes
        assert fingerprint.file_hashes
    finally:
        base_skeleton.reset_base_skeleton_cache()


def test_base_skeleton_fingerprint_is_cached() -> None:
    base_skeleton.reset_base_skeleton_cache()
    try:
        first = base_skeleton.base_skeleton_fingerprint()
        second = base_skeleton.base_skeleton_fingerprint()
        assert first is second
    finally:
        base_skeleton.reset_base_skeleton_cache()


def test_same_artifact_hash_is_excluded(tmp_path: Path) -> None:
    source = _feature_set(
        {"agent.py": "def solve():\n    return 1\n"},
        tmp_path,
        analysis_run_id=1,
        submission_id=1,
    )
    matched = build_submission_feature_set(
        build_python_ast_feature_rows(
            analysis_run_id=2,
            report=_extract({"agent.py": "def solve():\n    return 1\n"}, tmp_path),
        ),
        analysis_run_id=2,
        submission_id=2,
    )

    assert source.artifact_hash == matched.artifact_hash
    assert score_submission_similarity(source, matched) is None


def test_similarity_evidence_contains_metadata_not_raw_source(tmp_path: Path) -> None:
    secret_source = (
        "class Agent:\n    pass\n\ndef solve():\n    return 'SECRET_LITERAL_SHOULD_NOT_APPEAR'\n"
    )
    source = _feature_set(
        {"agent.py": secret_source},
        tmp_path / "source",
        analysis_run_id=1,
        submission_id=1,
    )
    matched = _feature_set(
        {"agent.py": secret_source, "README.md": "distinct artifact\n"},
        tmp_path / "matched",
        analysis_run_id=2,
        submission_id=2,
    )
    score = score_submission_similarity(
        source,
        matched,
        matched_artifact_uri="/private/matched.zip",
    )

    assert score is not None
    rows = build_similarity_match_rows(
        analysis_run_id=1,
        scores=[score],
        corpus_snapshot_at=datetime(2026, 5, 24, tzinfo=UTC),
    )

    assert len(rows) == 1
    row = rows[0]
    evidence = json.loads(row.evidence_json)
    assert row.match_kind == MATCH_KIND
    assert row.score >= 99
    assert evidence["algorithm_version"] == ALGORITHM_VERSION
    assert evidence["corpus_snapshot_at"] == "2026-05-24T00:00:00+00:00"
    assert evidence["risk_band"] == "high"
    assert evidence["top_file_pairs"][0]["source_file_path"] == "agent.py"
    assert "SECRET_LITERAL_SHOULD_NOT_APPEAR" not in row.evidence_json
    assert "return" not in row.evidence_json


async def test_same_challenge_matches_include_rejected_and_escalated_submissions(
    tmp_path: Path, monkeypatch
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    snapshot_at = datetime(2026, 5, 24, 12, 0, tzinfo=UTC)

    # agent.py is the shared baseagent stub every submission carries; register it
    # as base so matches are driven purely by each submission's non-base delta
    # (strategy.py), demonstrating base subtraction through the corpus path.
    _register_base_skeleton(
        monkeypatch, _feature_set({}, tmp_path / "base", analysis_run_id=99, submission_id=99)
    )

    async with session_factory() as session:
        invalid_submission_id, _invalid_run_id = await _insert_submission_with_features(
            session,
            tmp_path / "invalid",
            submission_id_seed="invalid",
            status="invalid",
            selected_tasks_json='["task-a"]',
            source={"strategy.py": "def solve(x):\n    renamed = x + 1\n    return renamed\n"},
        )
        suspicious_submission_id, _suspicious_run_id = await _insert_submission_with_features(
            session,
            tmp_path / "suspicious",
            submission_id_seed="suspicious",
            status="suspicious",
            selected_tasks_json='["task-a"]',
            source={
                "strategy.py": (
                    "def solve(payload):\n    normalized = payload + 1\n    return normalized\n"
                )
            },
        )
        await _insert_submission_with_features(
            session,
            tmp_path / "other-task",
            submission_id_seed="other-task",
            status="invalid",
            selected_tasks_json='["task-b"]',
            source={"strategy.py": "def solve(value):\n    return value + 1\n"},
        )
        current_submission_id, current_run_id = await _insert_submission_with_features(
            session,
            tmp_path / "current",
            submission_id_seed="current",
            status="pending",
            selected_tasks_json='["task-a"]',
            source={
                "strategy.py": "def solve(value):\n    result = value + 1\n    return result\n"
            },
        )

        matches = await build_same_challenge_similarity_matches(
            session,
            analysis_run_id=current_run_id,
            corpus_snapshot_at=snapshot_at,
        )

    await engine.dispose()

    matched_ids = {match.matched_submission_id for match in matches}
    assert len(matches) == 2
    assert matched_ids == {invalid_submission_id, suspicious_submission_id}
    assert current_submission_id not in matched_ids
    assert all(match.score >= 90 for match in matches)
    assert all(json.loads(match.evidence_json)["risk_band"] == "high" for match in matches)
    assert all(
        json.loads(match.evidence_json)["algorithm_version"] == ALGORITHM_VERSION
        for match in matches
    )


async def _insert_submission_with_features(
    session,
    tmp_path: Path,
    *,
    submission_id_seed: str,
    status: str,
    selected_tasks_json: str,
    source: dict[str, str | bytes],
) -> tuple[int, int]:
    submission = AgentSubmission(
        miner_hotkey=f"miner-{submission_id_seed}",
        name=f"agent-{submission_id_seed}",
        agent_hash=f"hash-{submission_id_seed}",
        artifact_uri=f"/tmp/{submission_id_seed}.zip",
        status=status,
        raw_status=status,
        effective_status=status,
    )
    session.add(submission)
    await session.flush()

    job = EvaluationJob(
        job_id=f"job-{submission_id_seed}",
        submission_id=submission.id,
        status="completed",
        selected_tasks_json=selected_tasks_json,
    )
    session.add(job)
    await session.flush()

    run = AnalysisRun(
        submission_id=submission.id,
        job_id=job.id,
        analyzer_name="python_ast",
        analyzer_version="test",
        status="completed",
        verdict=status,
    )
    session.add(run)
    await session.flush()

    rows = build_python_ast_feature_rows(analysis_run_id=run.id, report=_extract(source, tmp_path))
    session.add_all(rows)
    await session.flush()
    return submission.id, run.id


def _register_base_skeleton(monkeypatch, *feature_sets) -> None:
    ast_hashes = frozenset(
        file.ast_hash for fs in feature_sets for file in fs.files if file.ast_hash
    )
    file_hashes = frozenset(
        file.file_hash for fs in feature_sets for file in fs.files if file.file_hash
    )
    fingerprint = base_skeleton.BaseSkeletonFingerprint(
        ast_hashes=ast_hashes, file_hashes=file_hashes
    )
    monkeypatch.setattr(
        "agent_challenge.analyzer.base_skeleton.base_skeleton_fingerprint",
        lambda: fingerprint,
    )


def _feature_set(
    entries: dict[str, str | bytes],
    tmp_path: Path,
    *,
    analysis_run_id: int,
    submission_id: int,
):
    report = _extract(entries, tmp_path)
    rows = build_python_ast_feature_rows(analysis_run_id=analysis_run_id, report=report)
    return build_submission_feature_set(
        rows,
        analysis_run_id=analysis_run_id,
        submission_id=submission_id,
    )


def _extract(entries: dict[str, str | bytes], tmp_path: Path):
    metadata = store_zip_bytes(zip_bytes=_zip_bytes(entries), artifact_root=str(tmp_path))
    assert metadata.manifest is not None
    return extract_python_ast_features(
        zip_path=metadata.artifact_path,
        manifest=metadata.manifest,
    )


def _zip_bytes(entries: dict[str, str | bytes]) -> bytes:
    buffer = io.BytesIO()
    archive_entries = {"agent.py": ENTRYPOINT_SOURCE, **entries}
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in archive_entries.items():
            if filename == "agent.py":
                contents = agent_source(contents)
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()
