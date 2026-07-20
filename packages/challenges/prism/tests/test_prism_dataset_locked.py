from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pytest

from prism_challenge.evaluator.data_prep import (
    download_fineweb_edu_documents,
    prepare_locked_dataset,
)
from prism_challenge.evaluator.dataset import (
    FINEWEB_EDU_DATASET,
    FINEWEB_EDU_LICENSE,
    FINEWEB_EDU_PIN_SHA,
    LOCKED_MANIFEST_FILENAME,
    LOCKED_SPLITS,
    LockedDatasetError,
    assign_split,
    bucket_to_split,
    iter_locked_documents,
    load_locked_manifest,
    load_locked_train_texts,
    locked_shard_paths,
    partition_bucket,
    shard_offset_trace,
    verify_locked_manifest,
    verify_locked_manifest_or_raise,
)

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
EXPECTED_PIN = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"


class _CountingTokenizer:
    """Deterministic, offline token counter used to stand in for gpt2 in unit tests."""

    name = "test-tokenizer"
    fingerprint = "a" * 64

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def _docs(n: int) -> list[tuple[str, str]]:
    return [
        (f"doc-{i:06d}", f"Document number {i} about prisms, light, and learning curves.")
        for i in range(n)
    ]


def _prep(tmp_path: Path, n: int = 500, docs_per_shard: int = 64) -> Path:
    prepare_locked_dataset(
        _docs(n),
        tmp_path,
        token_counter=_CountingTokenizer(),
        docs_per_shard=docs_per_shard,
        source_doc_limit=n,
    )
    return tmp_path


def test_prism_dataset_partition_boundaries_are_fixed() -> None:
    # VAL-DATA-006: fixed/pinned bucket ranges train 0-949 / val 950-974 / test 975-999.
    assert bucket_to_split(0) == "train"
    assert bucket_to_split(949) == "train"
    assert bucket_to_split(950) == "val"
    assert bucket_to_split(974) == "val"
    assert bucket_to_split(975) == "test"
    assert bucket_to_split(999) == "test"
    with pytest.raises(LockedDatasetError):
        bucket_to_split(1000)
    with pytest.raises(LockedDatasetError):
        bucket_to_split(-1)

    bucket = partition_bucket("doc-000123")
    assert 0 <= bucket < 1000
    assert partition_bucket("doc-000123") == bucket  # deterministic
    assert assign_split("doc-000123") == bucket_to_split(bucket)


def test_prism_dataset_prep_three_disjoint_splits(tmp_path: Path) -> None:
    # VAL-DATA-002 + VAL-DATA-005: three non-empty splits, mutually disjoint by doc id.
    root = _prep(tmp_path, n=2000, docs_per_shard=64)

    ids: dict[str, set[str]] = {}
    for split in LOCKED_SPLITS:
        split_dir = root / split
        assert split_dir.is_dir()
        shards = list(split_dir.glob(f"{split}-*.jsonl"))
        assert shards, f"{split} produced no shards"
        assert all(shard.stat().st_size > 0 for shard in shards)
        ids[split] = {doc.doc_id for doc in iter_locked_documents(root, split)}

    assert set(ids) == set(LOCKED_SPLITS)
    assert ids["train"] and ids["val"] and ids["test"]
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])

    for split, id_set in ids.items():
        for doc_id in id_set:
            assert bucket_to_split(partition_bucket(doc_id)) == split


def test_prism_dataset_manifest_pin_and_provenance(tmp_path: Path) -> None:
    # VAL-DATA-001 + VAL-DATA-006 + VAL-DATA-016.
    root = _prep(tmp_path, n=600, docs_per_shard=64)
    payload = json.loads((root / LOCKED_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert FINEWEB_EDU_PIN_SHA == EXPECTED_PIN
    assert payload["dataset"]["pin_sha"] == EXPECTED_PIN
    assert payload["dataset"]["name"] == FINEWEB_EDU_DATASET
    assert payload["dataset"]["license"] == FINEWEB_EDU_LICENSE == "odc-by"

    partition = payload["partition"]
    assert partition["hash"] == "sha256"
    assert partition["key"] == "doc.id"
    assert partition["modulus"] == 1000
    assert partition["buckets"] == {
        "train": [0, 949],
        "val": [950, 974],
        "test": [975, 999],
    }

    assert payload["tokenizer"]["id"] == "test-tokenizer"
    assert SHA256_RE.match(payload["tokenizer"]["fingerprint"])

    for split in LOCKED_SPLITS:
        split_meta = payload["splits"][split]
        assert SHA256_RE.match(split_meta["fingerprint"])
        assert split_meta["token_count"] == sum(s["token_count"] for s in split_meta["shards"])
        assert split_meta["doc_count"] == sum(s["doc_count"] for s in split_meta["shards"])
        assert split_meta["byte_count"] == sum(s["bytes"] for s in split_meta["shards"])
        assert split_meta["token_count"] > 0


def test_prism_dataset_manifest_shard_sha256_matches_on_disk(tmp_path: Path) -> None:
    # VAL-DATA-003: every shard listed with sha256 that matches on-disk; set equality.
    root = _prep(tmp_path, n=600, docs_per_shard=64)
    manifest = load_locked_manifest(root)
    assert verify_locked_manifest(root, manifest) == []

    payload = json.loads((root / LOCKED_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    listed: set[Path] = set()
    for split in LOCKED_SPLITS:
        for shard in payload["splits"][split]["shards"]:
            path = root / shard["path"]
            assert path.exists()
            assert SHA256_RE.match(shard["sha256"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == shard["sha256"]
            listed.add(path)

    on_disk = {path for split in LOCKED_SPLITS for path in (root / split).glob(f"{split}-*.jsonl")}
    assert on_disk == listed


def test_prism_dataset_prep_deterministic_across_reruns(tmp_path: Path) -> None:
    # VAL-DATA-004: byte-identical shards (same sha256s) across re-runs, order-independent.
    docs = _docs(800)
    shuffled = list(docs)
    random.Random(7).shuffle(shuffled)

    run1 = tmp_path / "run1"
    run2 = tmp_path / "run2"
    prepare_locked_dataset(docs, run1, token_counter=_CountingTokenizer(), docs_per_shard=64)
    prepare_locked_dataset(shuffled, run2, token_counter=_CountingTokenizer(), docs_per_shard=64)

    p1 = json.loads((run1 / LOCKED_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    p2 = json.loads((run2 / LOCKED_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert p1["dataset"]["pin_sha"] == p2["dataset"]["pin_sha"]
    assert p1["partition"] == p2["partition"]
    assert p1["splits"] == p2["splits"]  # shards, sha256s, and fingerprints all identical


def test_prism_dataset_tamper_detected_via_sha256(tmp_path: Path) -> None:
    # VAL-DATA-015: a single-byte shard mutation is reported as a sha256 mismatch.
    root = _prep(tmp_path, n=300, docs_per_shard=64)
    manifest = load_locked_manifest(root)
    assert verify_locked_manifest(root, manifest) == []

    shard = sorted((root / "train").glob("train-*.jsonl"))[0]
    mutated = bytearray(shard.read_bytes())
    mutated[0] ^= 0xFF
    shard.write_bytes(bytes(mutated))

    problems = verify_locked_manifest(root, manifest)
    assert problems
    assert any(shard.name in problem for problem in problems)
    with pytest.raises(LockedDatasetError):
        verify_locked_manifest_or_raise(root, manifest)


def test_prism_dataset_verify_detects_missing_and_extra_shards(tmp_path: Path) -> None:
    # VAL-DATA-003: missing listed shard and unlisted on-disk shard are both flagged.
    root = _prep(tmp_path, n=300, docs_per_shard=64)
    manifest = load_locked_manifest(root)

    extra = root / "train" / "train-99999.jsonl"
    extra.write_text('{"id":"x","text":"y"}\n', encoding="utf-8")
    problems = verify_locked_manifest(root, manifest)
    assert any("train-99999.jsonl" in p for p in problems)
    extra.unlink()
    assert verify_locked_manifest(root, manifest) == []

    removed = sorted((root / "train").glob("train-*.jsonl"))[0]
    removed.unlink()
    assert any(removed.name in p for p in verify_locked_manifest(root, manifest))


def test_prism_dataset_multishard_iteration_deterministic(tmp_path: Path) -> None:
    # VAL-DATA-018: multi-shard iteration order is deterministic + challenge-controlled.
    root = _prep(tmp_path, n=2000, docs_per_shard=16)
    manifest = load_locked_manifest(root)
    assert len(locked_shard_paths(manifest, "train")) > 1

    trace1 = shard_offset_trace(root, "train")
    trace2 = shard_offset_trace(root, "train")
    assert trace1 == trace2

    docs1 = [doc.doc_id for doc in iter_locked_documents(root, "train")]
    docs2 = [doc.doc_id for doc in iter_locked_documents(root, "train")]
    assert docs1 == docs2

    ordered_shards: list[str] = []
    for shard, _offset in trace1:
        if not ordered_shards or ordered_shards[-1] != shard:
            ordered_shards.append(shard)
    assert ordered_shards == locked_shard_paths(manifest, "train")

    # offsets are contiguous 0..n-1 within each shard
    per_shard: dict[str, list[int]] = {}
    for shard, offset in trace1:
        per_shard.setdefault(shard, []).append(offset)
    for offsets in per_shard.values():
        assert offsets == list(range(len(offsets)))


def test_prism_dataset_loader_reads_train_texts(tmp_path: Path) -> None:
    root = _prep(tmp_path, n=400, docs_per_shard=64)
    texts = load_locked_train_texts(str(root / "train"))
    assert texts
    assert all(isinstance(text, str) and text for text in texts)
    # iterating the same dir yields the same ordered text stream
    again = [doc.text for doc in iter_locked_documents(root, "train")]
    assert texts == again


def test_prism_dataset_load_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(LockedDatasetError):
        load_locked_manifest(tmp_path)


def test_prism_dataset_download_requires_datasets_package() -> None:
    # The HF download layer is network/prep-only; without `datasets` it must fail clearly.
    with pytest.raises(LockedDatasetError, match="datasets"):
        list(download_fineweb_edu_documents(limit=1))
