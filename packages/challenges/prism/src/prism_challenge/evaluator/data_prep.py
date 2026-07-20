"""One-time, network-enabled FineWeb-Edu prep job (architecture.md section 3).

This module downloads a PINNED FineWeb-Edu subset OUTSIDE the eval sandbox (network is
allowed for prep only) and writes three fixed, mutually-disjoint splits plus a
``MANIFEST.json`` (pin SHA + provenance + per-shard sha256). The output is byte-identical
across re-runs: documents are partitioned by ``sha256(doc.id) % 1000`` with fixed boundaries,
sorted by doc id before sharding, and serialized with stable JSON. The HuggingFace download is
a thin, lazily-imported layer so the deterministic core stays unit-testable offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from .dataset import (
    DEFAULT_DOCS_PER_SHARD,
    FINEWEB_EDU_DATASET,
    FINEWEB_EDU_PIN_SHA,
    LOCKED_SPLITS,
    LockedDatasetError,
    LockedManifest,
    LockedShard,
    assign_split,
    build_locked_manifest,
    write_locked_manifest,
)

# A document source yields ``(doc_id, text)`` pairs. ``doc_id`` drives the hash-partition.
DocumentSource = Iterable[tuple[str, str]]

FINEWEB_EDU_DEFAULT_CONFIG = "default"
FINEWEB_EDU_DEFAULT_SPLIT = "train"


class TokenCounter(Protocol):
    """Provenance token counter: ``name`` + ``fingerprint`` + a deterministic ``count_tokens``."""

    @property
    def name(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...

    def count_tokens(self, text: str) -> int: ...


class Gpt2TokenCounter:
    """Default provenance token counter backed by the offline gpt2 tiktoken encoding."""

    name = "gpt2"

    def __init__(self) -> None:
        self._encoding: object | None = None

    def _load(self) -> object:
        if self._encoding is None:
            try:
                import tiktoken
            except ImportError as exc:  # pragma: no cover - exercised only without tiktoken
                raise LockedDatasetError("tiktoken is required for the gpt2 token counter") from exc
            self._encoding = tiktoken.get_encoding("gpt2")
        return self._encoding

    @property
    def fingerprint(self) -> str:
        encoding = self._load()
        n_vocab = getattr(encoding, "n_vocab", 0)
        return hashlib.sha256(f"tiktoken:gpt2:n_vocab={n_vocab}".encode()).hexdigest()

    def count_tokens(self, text: str) -> int:
        return len(self._load().encode(text))  # type: ignore[attr-defined]


def _render_shard(
    doc_ids: list[str], texts: dict[str, str], counter: TokenCounter
) -> tuple[bytes, int]:
    lines: list[str] = []
    token_count = 0
    for doc_id in doc_ids:
        text = texts[doc_id]
        token_count += counter.count_tokens(text)
        lines.append(
            json.dumps(
                {"id": doc_id, "text": text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    content = ("\n".join(lines) + "\n") if lines else ""
    return content.encode("utf-8"), token_count


def prepare_locked_dataset(
    documents: DocumentSource,
    output_dir: Path | str,
    *,
    token_counter: TokenCounter | None = None,
    pin_sha: str = FINEWEB_EDU_PIN_SHA,
    dataset_name: str = FINEWEB_EDU_DATASET,
    source_doc_limit: int | None = None,
    docs_per_shard: int = DEFAULT_DOCS_PER_SHARD,
) -> LockedManifest:
    """Partition ``documents`` into locked train/val/test shards and author MANIFEST.json.

    Deterministic across re-runs: per-split documents are de-duplicated by id, sorted by id,
    and chunked into fixed-size shards written with stable JSON, so identical inputs (in any
    order) yield byte-identical shards and identical sha256s.
    """
    if docs_per_shard < 1:
        raise LockedDatasetError("docs_per_shard must be >= 1")
    counter: TokenCounter = token_counter or Gpt2TokenCounter()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    buckets: dict[str, dict[str, str]] = {name: {} for name in LOCKED_SPLITS}
    for doc_id, text in documents:
        if not isinstance(doc_id, str) or not doc_id:
            raise LockedDatasetError(f"invalid document id: {doc_id!r}")
        if not isinstance(text, str):
            raise LockedDatasetError(f"document {doc_id!r} has non-string text")
        buckets[assign_split(doc_id)].setdefault(doc_id, text)

    splits: dict[str, list[LockedShard]] = {}
    for name in LOCKED_SPLITS:
        split_dir = root / name
        split_dir.mkdir(parents=True, exist_ok=True)
        for stale in split_dir.glob(f"{name}-*.jsonl"):
            stale.unlink()
        ordered_ids = sorted(buckets[name])
        shards: list[LockedShard] = []
        for shard_index, start in enumerate(range(0, len(ordered_ids), docs_per_shard)):
            chunk = ordered_ids[start : start + docs_per_shard]
            shard_name = f"{name}-{shard_index:05d}.jsonl"
            data, token_count = _render_shard(chunk, buckets[name], counter)
            (split_dir / shard_name).write_bytes(data)
            shards.append(
                LockedShard(
                    path=f"{name}/{shard_name}",
                    sha256=hashlib.sha256(data).hexdigest(),
                    bytes=len(data),
                    doc_count=len(chunk),
                    token_count=token_count,
                )
            )
        splits[name] = shards

    manifest = build_locked_manifest(
        splits=splits,
        tokenizer_id=counter.name,
        tokenizer_fingerprint=counter.fingerprint,
        pin_sha=pin_sha,
        dataset_name=dataset_name,
        source_doc_limit=source_doc_limit,
    )
    write_locked_manifest(root, manifest)
    return manifest


def download_fineweb_edu_documents(
    *,
    limit: int,
    pin_sha: str = FINEWEB_EDU_PIN_SHA,
    dataset_name: str = FINEWEB_EDU_DATASET,
    config_name: str = FINEWEB_EDU_DEFAULT_CONFIG,
    split: str = FINEWEB_EDU_DEFAULT_SPLIT,
    token: str | None = None,
) -> Iterator[tuple[str, str]]:
    """Stream ``(doc_id, text)`` from the PINNED FineWeb-Edu commit (network/prep-only).

    Lazily imports ``datasets`` so the rest of this module stays importable offline. The
    revision is pinned to the immutable commit SHA, never a moving tag. ``token`` is an
    OPTIONAL HuggingFace credential (FineWeb-Edu is public, so anonymous access works);
    when supplied it is sourced from a Docker secret file, never a plaintext literal.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise LockedDatasetError(
            "the 'datasets' package is required for the FineWeb-Edu prep download; "
            "install it in the network-enabled prep environment"
        ) from exc

    load_kwargs: dict[str, object] = {
        "name": config_name,
        "split": split,
        "revision": pin_sha,
        "streaming": True,
    }
    if token:
        load_kwargs["token"] = token
    stream = load_dataset(dataset_name, **load_kwargs)
    count = 0
    for record in stream:
        if count >= limit:
            break
        doc_id = record.get("id")
        text = record.get("text")
        if not isinstance(doc_id, str) or not doc_id:
            continue
        if not isinstance(text, str) or not text:
            continue
        yield doc_id, text
        count += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare the locked FineWeb-Edu dataset (one-time network prep job).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int, help="number of source documents")
    parser.add_argument("--docs-per-shard", type=int, default=DEFAULT_DOCS_PER_SHARD)
    parser.add_argument("--pin-sha", default=FINEWEB_EDU_PIN_SHA)
    args = parser.parse_args(argv)

    from ..config import PrismSettings

    documents = download_fineweb_edu_documents(
        limit=args.limit,
        pin_sha=args.pin_sha,
        token=PrismSettings().hf_token_value(),
    )
    manifest = prepare_locked_dataset(
        documents,
        args.output_dir,
        pin_sha=args.pin_sha,
        source_doc_limit=args.limit,
        docs_per_shard=args.docs_per_shard,
    )
    summary = {
        "output_dir": str(args.output_dir),
        "pin_sha": manifest.pin_sha,
        "split_doc_counts": {name: manifest.splits[name].doc_count for name in LOCKED_SPLITS},
        "split_token_counts": {name: manifest.splits[name].token_count for name in LOCKED_SPLITS},
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
