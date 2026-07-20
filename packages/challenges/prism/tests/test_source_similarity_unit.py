from __future__ import annotations

import base64
import io
import json
import zipfile

import pytest

from prism_challenge.evaluator.source_similarity import (
    DuplicateThresholdMatrix,
    SourceSnapshot,
    build_pair_report,
    classify_duplicate,
    jaccard,
    primary_python_code,
    rank_similar,
    run_pair_sandbox,
    snapshot_from_named_sources,
    snapshot_from_submission,
    write_snapshot_dir,
)

MODEL_CODE = """
import torch

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 4)

    def forward(self, x):
        return self.layer(x)

def build_model(ctx):
    return Net()
"""

OTHER_CODE = """
import numpy

class Different:
    def run(self):
        total = 0
        for value in range(10):
            total = total + value
        return total
"""


def _zip_b64(files: dict[str, str]) -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return base64.b64encode(stream.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------


def test_jaccard_identical_sets_is_one():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_both_empty_is_one():
    assert jaccard(set(), set()) == 1.0


def test_jaccard_partial_overlap():
    # intersection {a} size 1, union {a,b,c} size 3
    assert jaccard({"a", "b"}, {"a", "c"}) == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# snapshot construction + features
# ---------------------------------------------------------------------------


def test_snapshot_from_named_sources_basic_features():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    assert len(snap.files) == 1
    assert snap.files[0].path == "model.py"
    assert snap.files[0].sha256
    # AST features should capture class/function/import nodes.
    assert "class:Net:torch.nn.Module" in snap.ast_features
    assert "function:build_model" in snap.ast_features
    assert "import:torch" in snap.ast_features
    assert any(feat.startswith("call:") for feat in snap.ast_features)
    assert snap.token_shingles
    assert snap.fingerprint


def test_snapshot_syntax_error_records_feature_not_crash():
    snap = snapshot_from_named_sources([("broken.py", "def f(:\n  pass")])
    assert any(feat.startswith("syntax_error:") for feat in snap.ast_features)
    # No ast_dump feature when parsing failed.
    assert not any(feat.startswith("ast_dump_sha256:") for feat in snap.ast_features)


def test_snapshot_non_python_file_only_contributes_path_and_ext():
    snap = snapshot_from_named_sources([("notes.txt", "hello world hello")])
    assert "path:notes.txt" in snap.ast_features
    assert "ext:.txt" in snap.ast_features
    assert not any(feat.startswith("node:") for feat in snap.ast_features)


def test_snapshot_files_are_sorted_by_path():
    snap = snapshot_from_named_sources([("b.py", "x = 1"), ("a.py", "y = 2")])
    assert [f.path for f in snap.files] == ["a.py", "b.py"]


def test_snapshot_max_files_exceeded_raises():
    sources = [(f"f{i}.py", "x = 1") for i in range(5)]
    with pytest.raises(ValueError, match="exceeds 2 files"):
        snapshot_from_named_sources(sources, max_files=2)


def test_snapshot_max_bytes_exceeded_raises():
    big = "x" * 100
    with pytest.raises(ValueError, match="exceeds 10 bytes"):
        snapshot_from_named_sources([("f.py", big)], max_bytes=10)


def test_snapshot_unsafe_path_rejected():
    with pytest.raises(ValueError, match="unsafe source path"):
        snapshot_from_named_sources([("../escape.py", "x = 1")])


# ---------------------------------------------------------------------------
# zip extraction within a snapshot
# ---------------------------------------------------------------------------


def test_snapshot_extracts_zip_sources():
    archive = _zip_b64({"inner.py": "z = 3\n"})
    snap = snapshot_from_named_sources([("bundle.zip", archive)])
    paths = {f.path for f in snap.files}
    assert "bundle/inner.py" in paths


def test_snapshot_zip_bad_base64_raises():
    with pytest.raises(ValueError, match="must be base64 encoded"):
        snapshot_from_named_sources([("bundle.zip", "!!!not base64!!!")])


def test_snapshot_zip_bad_archive_raises():
    not_a_zip = base64.b64encode(b"definitely not a zip").decode("ascii")
    with pytest.raises(ValueError, match="not a valid zip archive"):
        snapshot_from_named_sources([("bundle.zip", not_a_zip)])


def test_snapshot_zip_unsupported_suffix_raises():
    archive = _zip_b64({"evil.exe": "binary"})
    with pytest.raises(ValueError, match="unsupported file"):
        snapshot_from_named_sources([("bundle.zip", archive)])


# ---------------------------------------------------------------------------
# snapshot_from_submission
# ---------------------------------------------------------------------------


def test_snapshot_from_submission_plain_code():
    snap = snapshot_from_submission(MODEL_CODE)
    assert [f.path for f in snap.files] == ["model.py"]


def test_snapshot_from_submission_with_archive_metadata():
    archive = _zip_b64({"extra.py": "q = 9\n"})
    snap = snapshot_from_submission(MODEL_CODE, metadata={"archive_base64": archive})
    paths = {f.path for f in snap.files}
    assert "model.py" in paths
    assert "metadata/extra.py" in paths


# ---------------------------------------------------------------------------
# payload round-trip
# ---------------------------------------------------------------------------


def test_snapshot_payload_round_trip_is_lossless():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    restored = SourceSnapshot.from_payload(snap.to_payload())
    assert restored.fingerprint == snap.fingerprint
    assert restored.ast_features == snap.ast_features
    assert restored.token_shingles == snap.token_shingles
    assert [f.path for f in restored.files] == [f.path for f in snap.files]


def test_snapshot_from_payload_recomputes_missing_sha_and_fingerprint():
    payload = {
        "files": [{"path": "a.py", "content": "x = 1"}],
        "ast_features": ["function:foo"],
        "token_shingles": ["a b c"],
    }
    snap = SourceSnapshot.from_payload(payload)
    assert snap.files[0].sha256  # recomputed from content
    assert snap.fingerprint  # recomputed from features+shingles


def test_snapshot_from_payload_skips_entries_without_path():
    payload = {"files": [{"content": "x"}, {"path": "ok.py", "content": "y = 2"}]}
    snap = SourceSnapshot.from_payload(payload)
    assert [f.path for f in snap.files] == ["ok.py"]


# ---------------------------------------------------------------------------
# python helpers
# ---------------------------------------------------------------------------


def test_python_files_property_filters_non_python():
    snap = snapshot_from_named_sources([("a.py", "x=1"), ("b.txt", "hi")])
    assert [f.path for f in snap.python_files] == ["a.py"]


def test_primary_python_code_prefers_model_py():
    snap = snapshot_from_named_sources([("agent.py", "AGENT = 1"), ("model.py", "MODEL = 2")])
    assert "MODEL = 2" in primary_python_code(snap)


def test_primary_python_code_falls_back_to_agent_py():
    snap = snapshot_from_named_sources([("agent.py", "AGENT = 1")])
    assert "AGENT = 1" in primary_python_code(snap)


def test_primary_python_code_combined_when_no_named_entrypoint():
    snap = snapshot_from_named_sources([("util.py", "UTIL = 7")])
    assert "UTIL = 7" in primary_python_code(snap)


def test_combined_python_truncates_at_max_chars():
    snap = snapshot_from_named_sources([("model.py", "a" * 500)])
    combined = snap.combined_python(max_chars=50)
    assert len(combined) <= 50


# ---------------------------------------------------------------------------
# rank_similar
# ---------------------------------------------------------------------------


def _row_from(code: str, **extra):
    snap = snapshot_from_named_sources([("model.py", code)])
    row = {
        "submission_id": "row-1",
        "files": [f.__dict__ for f in snap.files],
        "ast_features": sorted(snap.ast_features),
        "token_shingles": sorted(snap.token_shingles),
        "fingerprint": snap.fingerprint,
    }
    row.update(extra)
    return row


def test_rank_similar_identical_code_scores_high():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    rows = [_row_from(MODEL_CODE, submission_id="dup")]
    ranked = rank_similar(snap, rows, min_similarity=0.5)
    assert ranked
    assert ranked[0].submission_id == "dup"
    assert ranked[0].score == pytest.approx(1.0)


def test_rank_similar_filters_below_min_similarity():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    rows = [_row_from(OTHER_CODE)]
    ranked = rank_similar(snap, rows, min_similarity=0.95)
    assert ranked == []


def test_rank_similar_respects_top_k():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    rows = [
        _row_from(MODEL_CODE, submission_id="a"),
        _row_from(MODEL_CODE, submission_id="b"),
        _row_from(MODEL_CODE, submission_id="c"),
    ]
    ranked = rank_similar(snap, rows, top_k=2, min_similarity=0.0)
    assert len(ranked) == 2


def test_rank_similar_carries_hotkey_and_code_hash():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    rows = [_row_from(MODEL_CODE, hotkey="hk1", code_hash="abc")]
    ranked = rank_similar(snap, rows, min_similarity=0.0)
    assert ranked[0].hotkey == "hk1"
    assert ranked[0].code_hash == "abc"


def test_rank_similar_graph_similarity_computed():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    graph = {"classes": ["Net"], "functions": ["build_model"]}
    rows = [_row_from(MODEL_CODE, architecture_graph=graph)]
    ranked = rank_similar(snap, rows, min_similarity=0.0, architecture_graph=graph)
    assert ranked[0].graph_similarity == pytest.approx(1.0)
    assert ranked[0].architecture_graph_hash


# ---------------------------------------------------------------------------
# classify_duplicate
# ---------------------------------------------------------------------------


def test_classify_duplicate_allow_when_no_rows():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    decision = classify_duplicate(
        submission_id="s1",
        code_hash="hash-1",
        snapshot=snap,
        architecture_graph={},
        rows=[],
    )
    assert decision.outcome == "allow"
    assert decision.candidate is None
    assert decision.report["schema_version"] == "duplicate_report.v1"
    assert not decision.rejected
    assert not decision.held


def test_classify_duplicate_rejects_exact_source_hash():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    rows = [_row_from(MODEL_CODE, code_hash="same-hash", submission_id="prev")]
    decision = classify_duplicate(
        submission_id="s2",
        code_hash="same-hash",
        snapshot=snap,
        architecture_graph={},
        rows=rows,
    )
    assert decision.outcome == "reject"
    assert decision.rejected
    assert decision.report["exact_source_hash"] is True
    assert decision.report["evidence"]


def test_classify_duplicate_attach_on_identical_graph():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    graph = {"classes": ["Net"], "functions": ["build_model"]}
    rows = [_row_from(OTHER_CODE, code_hash="other-hash", architecture_graph=graph)]
    decision = classify_duplicate(
        submission_id="s3",
        code_hash="my-hash",
        snapshot=snap,
        architecture_graph=graph,
        rows=rows,
    )
    assert decision.outcome == "attach"


def test_classify_duplicate_quarantine_on_borderline_similarity():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    rows = [_row_from(MODEL_CODE, code_hash="different", submission_id="prev")]
    decision = classify_duplicate(
        submission_id="s4",
        code_hash="mine",
        snapshot=snap,
        architecture_graph={},
        rows=rows,
    )
    assert decision.outcome == "quarantine"
    assert decision.held


def test_classify_duplicate_allow_when_below_thresholds():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    # Low source similarity AND distinct (disjoint) architecture graphs => allow.
    rows = [
        _row_from(
            OTHER_CODE,
            code_hash="different",
            submission_id="prev",
            architecture_graph={"classes": ["Bar"]},
        )
    ]
    decision = classify_duplicate(
        submission_id="s5",
        code_hash="mine",
        snapshot=snap,
        architecture_graph={"classes": ["Foo"]},
        rows=rows,
    )
    assert decision.outcome == "allow"


def test_classify_duplicate_accepts_threshold_mapping():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    # Near-identical source, but distinct graphs so only the source threshold matters.
    rows = [
        _row_from(
            MODEL_CODE,
            code_hash="different",
            submission_id="prev",
            architecture_graph={"classes": ["Bar"]},
        )
    ]
    decision = classify_duplicate(
        submission_id="s6",
        code_hash="mine",
        snapshot=snap,
        architecture_graph={"classes": ["Foo"]},
        rows=rows,
        # Quarantine threshold above 1.0 so even identical source stays below it.
        thresholds={"quarantine_source_similarity": 1.01},
    )
    assert decision.outcome == "allow"


# ---------------------------------------------------------------------------
# DuplicateThresholdMatrix
# ---------------------------------------------------------------------------


def test_threshold_matrix_defaults():
    matrix = DuplicateThresholdMatrix()
    assert matrix.exact_source_similarity == 0.98
    assert matrix.quarantine_source_similarity == 0.85


def test_threshold_matrix_from_mapping_and_payload_round_trip():
    matrix = DuplicateThresholdMatrix.from_mapping({"static_reject_similarity": 0.5})
    assert matrix.static_reject_similarity == 0.5
    assert matrix.to_payload()["static_reject_similarity"] == 0.5


def test_threshold_matrix_from_none_uses_defaults():
    matrix = DuplicateThresholdMatrix.from_mapping(None)
    assert matrix.same_architecture_similarity == 0.82


# ---------------------------------------------------------------------------
# build_pair_report / run_pair_sandbox
# ---------------------------------------------------------------------------


def test_build_pair_report_identical_snapshots():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    report = build_pair_report(snap, snap)
    assert report["ast_similarity"] == pytest.approx(1.0)
    assert report["file_similarity"] == pytest.approx(1.0)
    assert report["shared_paths"] == ["model.py"]
    assert report["exact_file_matches"]
    assert report["current_file_count"] == 1


def test_build_pair_report_disjoint_snapshots():
    left = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    right = snapshot_from_named_sources([("other.py", OTHER_CODE)])
    report = build_pair_report(left, right)
    assert report["file_similarity"] == pytest.approx(0.0)
    assert report["shared_paths"] == []
    assert report["exact_file_matches"] == []


def test_run_pair_sandbox_local_static_default():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])
    report = run_pair_sandbox(snap, snap)
    assert report["sandbox"] == "local-static"


def test_run_pair_sandbox_with_runner_invokes_comparison():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])

    def fake_runner(left, right, script):
        # The real runner would execute ``script`` inside docker; we just return JSON.
        assert left.exists() and right.exists() and script.exists()
        return json.dumps({"token_similarity": 1.0, "file_similarity": 1.0})

    report = run_pair_sandbox(snap, snap, runner=fake_runner)
    assert report["sandbox"] == "docker-alpine"
    assert report["token_similarity"] == 1.0


def test_run_pair_sandbox_runner_non_object_json_raises():
    snap = snapshot_from_named_sources([("model.py", MODEL_CODE)])

    def bad_runner(left, right, script):
        return json.dumps([1, 2, 3])

    with pytest.raises(ValueError, match="non-object JSON"):
        run_pair_sandbox(snap, snap, runner=bad_runner)


# ---------------------------------------------------------------------------
# write_snapshot_dir
# ---------------------------------------------------------------------------


def test_write_snapshot_dir_materializes_files(tmp_path):
    snap = snapshot_from_named_sources([("pkg/model.py", MODEL_CODE), ("README.md", "# hi")])
    write_snapshot_dir(snap, tmp_path)
    assert (tmp_path / "pkg" / "model.py").read_text() == MODEL_CODE
    assert (tmp_path / "README.md").read_text() == "# hi"
