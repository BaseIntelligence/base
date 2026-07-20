from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from prism_challenge.evaluator.data_prep import prepare_locked_dataset
from prism_challenge.evaluator.dataset import (
    FINEWEB_EDU_PIN_SHA,
    LOCKED_MANIFEST_FILENAME,
    LockedDatasetError,
    load_locked_manifest,
    partition_spec,
    verify_locked_manifest,
    verify_locked_manifest_or_raise,
)

# A submission/operator that tampers ONLY the manifest's pin SHA or partition spec while leaving
# every shard byte-identical previously slipped past ``verify_locked_manifest`` (it checked shard
# sha256/bytes/missing/extra/fingerprint but never the pin or the canonical partition boundaries).
# These tests pin that hole shut: an altered pin/partition with intact shards must be REFUSED
# (reinforces VAL-CHEAT-019 / VAL-DATA-015 tamper-evidence; architecture.md sections 3, 6, 12).


class _CountingTokenizer:
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


def _rewrite_manifest(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Mutate MANIFEST.json on disk WITHOUT touching any shard (intact-shards tamper)."""
    path = root / LOCKED_MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def test_prism_dataset_verify_passes_on_canonical_pin_and_partition(tmp_path: Path) -> None:
    root = _prep(tmp_path)
    manifest = load_locked_manifest(root)
    assert manifest.pin_sha == FINEWEB_EDU_PIN_SHA
    assert manifest.partition == partition_spec()
    assert verify_locked_manifest(root, manifest) == []


def test_prism_dataset_verify_refuses_altered_pin_with_intact_shards(tmp_path: Path) -> None:
    root = _prep(tmp_path)
    assert verify_locked_manifest(root, load_locked_manifest(root)) == []

    _rewrite_manifest(root, lambda p: p["dataset"].__setitem__("pin_sha", "0" * 40))
    manifest = load_locked_manifest(root)
    # Shards are byte-identical, only the pin changed -> the shard checks alone would pass.
    assert manifest.pin_sha != FINEWEB_EDU_PIN_SHA
    problems = verify_locked_manifest(root, manifest)
    assert any("pin" in problem.lower() for problem in problems), problems
    with pytest.raises(LockedDatasetError):
        verify_locked_manifest_or_raise(root, manifest)


def test_prism_dataset_verify_refuses_altered_partition_buckets(tmp_path: Path) -> None:
    root = _prep(tmp_path)
    assert verify_locked_manifest(root, load_locked_manifest(root)) == []

    def _widen_train(payload: dict[str, Any]) -> None:
        payload["partition"]["buckets"]["train"] = [0, 998]

    _rewrite_manifest(root, _widen_train)
    manifest = load_locked_manifest(root)
    assert manifest.partition != partition_spec()
    problems = verify_locked_manifest(root, manifest)
    assert any("partition" in problem.lower() for problem in problems), problems
    with pytest.raises(LockedDatasetError):
        verify_locked_manifest_or_raise(root, manifest)


def test_prism_dataset_verify_refuses_altered_partition_modulus(tmp_path: Path) -> None:
    root = _prep(tmp_path)
    _rewrite_manifest(root, lambda p: p["partition"].__setitem__("modulus", 500))
    manifest = load_locked_manifest(root)
    problems = verify_locked_manifest(root, manifest)
    assert any("partition" in problem.lower() for problem in problems), problems


def test_prism_dataset_verify_still_flags_pin_when_only_a_subset_split_checked(
    tmp_path: Path,
) -> None:
    # The pin/partition are manifest-global tamper-evidence; checking a single split must not let a
    # tampered pin slip through.
    root = _prep(tmp_path)
    _rewrite_manifest(root, lambda p: p["dataset"].__setitem__("pin_sha", "f" * 40))
    manifest = load_locked_manifest(root)
    problems = verify_locked_manifest(root, manifest, splits=("train",))
    assert any("pin" in problem.lower() for problem in problems), problems
