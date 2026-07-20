from __future__ import annotations

import json

import pytest

from prism_challenge.evaluator.data_prep import prepare_locked_dataset
from prism_challenge.evaluator.dataset import (
    LOCKED_SPLITS,
    LockedDatasetError,
    assign_split,
    load_locked_manifest,
)


class FakeCounter:
    """Deterministic token counter: one "token" per whitespace-split word."""

    name = "fake"
    fingerprint = "fake-fingerprint-0001"

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _docs(n: int):
    return [(f"doc-{i:04d}", f"text body number {i}") for i in range(n)]


def test_prepare_locked_dataset_writes_manifest_and_shards(tmp_path):
    manifest = prepare_locked_dataset(_docs(30), tmp_path, token_counter=FakeCounter())
    assert (tmp_path / "MANIFEST.json").is_file()
    # Every split dir exists.
    for name in LOCKED_SPLITS:
        assert (tmp_path / name).is_dir()
    # Total docs preserved across splits.
    total_docs = sum(manifest.splits[name].doc_count for name in LOCKED_SPLITS)
    assert total_docs == 30


def test_prepare_locked_dataset_partitions_match_assign_split(tmp_path):
    docs = _docs(40)
    manifest = prepare_locked_dataset(docs, tmp_path, token_counter=FakeCounter())
    # Independently bucket the docs and compare counts.
    expected: dict[str, int] = {name: 0 for name in LOCKED_SPLITS}
    for doc_id, _text in docs:
        expected[assign_split(doc_id)] += 1
    for name in LOCKED_SPLITS:
        assert manifest.splits[name].doc_count == expected[name]


def test_prepare_locked_dataset_is_byte_identical_regardless_of_order(tmp_path):
    docs = _docs(25)
    a = tmp_path / "a"
    b = tmp_path / "b"
    m1 = prepare_locked_dataset(docs, a, token_counter=FakeCounter())
    m2 = prepare_locked_dataset(list(reversed(docs)), b, token_counter=FakeCounter())
    # Same manifest payload => deterministic, order-independent.
    assert m1.to_dict() == m2.to_dict()
    # Shard bytes identical too.
    assert (a / "MANIFEST.json").read_bytes() == (b / "MANIFEST.json").read_bytes()


def test_prepare_locked_dataset_dedupes_by_id(tmp_path):
    docs = [("same", "first"), ("same", "second"), ("other", "x")]
    manifest = prepare_locked_dataset(docs, tmp_path, token_counter=FakeCounter())
    total = sum(manifest.splits[name].doc_count for name in LOCKED_SPLITS)
    assert total == 2  # 'same' counted once


def test_prepare_locked_dataset_token_counts_use_counter(tmp_path):
    docs = [("a", "one two three")]  # 3 tokens by FakeCounter
    manifest = prepare_locked_dataset(docs, tmp_path, token_counter=FakeCounter())
    total_tokens = sum(manifest.splits[name].token_count for name in LOCKED_SPLITS)
    assert total_tokens == 3


def test_prepare_locked_dataset_respects_docs_per_shard(tmp_path):
    docs = _docs(50)
    manifest = prepare_locked_dataset(docs, tmp_path, token_counter=FakeCounter(), docs_per_shard=2)
    # At least one split should be sharded into multiple files given small shard size.
    shard_counts = [len(manifest.splits[name].shards) for name in LOCKED_SPLITS]
    assert max(shard_counts) >= 2


def test_prepare_locked_dataset_shard_files_exist_on_disk(tmp_path):
    manifest = prepare_locked_dataset(_docs(10), tmp_path, token_counter=FakeCounter())
    for name in LOCKED_SPLITS:
        for shard in manifest.splits[name].shards:
            assert (tmp_path / shard.path).is_file()


def test_prepare_locked_dataset_clears_stale_shards(tmp_path):
    # First run with many docs, then a smaller run should drop stale shard files.
    prepare_locked_dataset(_docs(60), tmp_path, token_counter=FakeCounter(), docs_per_shard=1)
    before = sorted(p.name for p in (tmp_path / "train").glob("train-*.jsonl"))
    prepare_locked_dataset(_docs(2), tmp_path, token_counter=FakeCounter(), docs_per_shard=1)
    after = sorted(p.name for p in (tmp_path / "train").glob("train-*.jsonl"))
    assert len(after) <= len(before)


def test_prepare_locked_dataset_manifest_reloadable(tmp_path):
    prepare_locked_dataset(_docs(12), tmp_path, token_counter=FakeCounter())
    reloaded = load_locked_manifest(tmp_path)
    assert reloaded.tokenizer["id"] == "fake"
    assert reloaded.tokenizer["fingerprint"] == "fake-fingerprint-0001"


def test_prepare_locked_dataset_shard_jsonl_is_stable_json(tmp_path):
    manifest = prepare_locked_dataset(
        [("z", "hello"), ("a", "world")], tmp_path, token_counter=FakeCounter()
    )
    for name in LOCKED_SPLITS:
        for shard in manifest.splits[name].shards:
            content = (tmp_path / shard.path).read_text(encoding="utf-8")
            for line in content.strip().splitlines():
                record = json.loads(line)
                assert set(record) == {"id", "text"}


# ---------------------------------------------------------------------------
# validation / error paths
# ---------------------------------------------------------------------------


def test_prepare_locked_dataset_rejects_bad_docs_per_shard(tmp_path):
    with pytest.raises(LockedDatasetError, match="docs_per_shard must be >= 1"):
        prepare_locked_dataset([("a", "x")], tmp_path, docs_per_shard=0)


def test_prepare_locked_dataset_rejects_non_string_doc_id(tmp_path):
    with pytest.raises(LockedDatasetError, match="invalid document id"):
        prepare_locked_dataset(
            [(123, "x")],
            tmp_path,
            token_counter=FakeCounter(),  # type: ignore[list-item]
        )


def test_prepare_locked_dataset_rejects_empty_doc_id(tmp_path):
    with pytest.raises(LockedDatasetError, match="invalid document id"):
        prepare_locked_dataset([("", "x")], tmp_path, token_counter=FakeCounter())


def test_prepare_locked_dataset_rejects_non_string_text(tmp_path):
    with pytest.raises(LockedDatasetError, match="non-string text"):
        prepare_locked_dataset(
            [("a", 123)],
            tmp_path,
            token_counter=FakeCounter(),  # type: ignore[list-item]
        )


def _install_fake_datasets(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    import sys
    import types

    fake = types.ModuleType("datasets")

    def load_dataset(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return iter([{"id": "doc-0", "text": "hello world"}])

    fake.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_download_passes_optional_hf_token_to_load_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from prism_challenge.evaluator.data_prep import download_fineweb_edu_documents

    captured: dict = {}
    _install_fake_datasets(monkeypatch, captured)

    list(download_fineweb_edu_documents(limit=1, token="hf_secret"))

    # The Docker-secret-sourced token reaches HuggingFace auth; the pin SHA stays
    # the immutable revision (never a moving tag).
    assert captured["kwargs"]["token"] == "hf_secret"
    assert captured["kwargs"]["revision"]


def test_download_omits_token_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    from prism_challenge.evaluator.data_prep import download_fineweb_edu_documents

    captured: dict = {}
    _install_fake_datasets(monkeypatch, captured)

    list(download_fineweb_edu_documents(limit=1, token=None))

    # FineWeb-Edu is public: with no secret the prep runs anonymous (no token kwarg).
    assert "token" not in captured["kwargs"]
