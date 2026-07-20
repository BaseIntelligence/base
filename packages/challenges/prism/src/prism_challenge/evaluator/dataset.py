from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FINEWEB_EDU_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_SUBSETS: dict[str, dict[str, int | str]] = {
    "sample-10BT": {"token_count": 10_000_000_000, "official_mode": "gpu_proxy_eval"},
    "sample-100BT": {"token_count": 100_000_000_000, "official_mode": "full_scale_eval"},
    "sample-350BT": {"token_count": 350_000_000_000, "official_mode": "phase_2_scale"},
}


def _fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Locked FineWeb-Edu data plane (architecture.md section 3)
#
# A one-time, network-enabled prep job (see ``data_prep.py``) downloads a PINNED
# FineWeb-Edu subset and writes three fixed, mutually-disjoint splits plus a
# ``MANIFEST.json`` carrying the pin SHA, dataset provenance, the partition spec,
# and the sha256 of every shard. The loader below reads those locked shards in a
# deterministic, challenge-controlled order and verifies shard integrity against
# the manifest (tamper-evidence). The miner only ever sees the ``train`` split.
# ---------------------------------------------------------------------------

# Immutable HuggingFace commit (NOT a moving tag). Recorded verbatim in MANIFEST.json.
FINEWEB_EDU_PIN_SHA = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
FINEWEB_EDU_LICENSE = "odc-by"
LOCKED_MANIFEST_FILENAME = "MANIFEST.json"
LOCKED_MANIFEST_SCHEMA = "prism_locked_data.v1"
LOCKED_SPLITS: tuple[str, ...] = ("train", "val", "test")

# Deterministic hash-partition over ``doc.id``: bucket = int(sha256(id), 16) % 1000.
PARTITION_HASH = "sha256"
PARTITION_KEY = "doc.id"
PARTITION_MODULUS = 1000
PARTITION_BUCKETS: dict[str, tuple[int, int]] = {
    "train": (0, 949),
    "val": (950, 974),
    "test": (975, 999),
}
DEFAULT_DOCS_PER_SHARD = 1024


class LockedDatasetError(RuntimeError):
    """Raised on a malformed/missing manifest or a failed locked-dataset integrity check."""


def partition_bucket(doc_id: str) -> int:
    """Return the fixed/pinned partition bucket in ``[0, PARTITION_MODULUS)`` for a doc id."""
    digest = hashlib.sha256(doc_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % PARTITION_MODULUS


def bucket_to_split(bucket: int) -> str:
    """Map a partition bucket to its split using the fixed/pinned boundaries."""
    if not 0 <= bucket < PARTITION_MODULUS:
        raise LockedDatasetError(f"partition bucket {bucket} out of range [0, {PARTITION_MODULUS})")
    for split, (low, high) in PARTITION_BUCKETS.items():
        if low <= bucket <= high:
            return split
    raise LockedDatasetError(f"partition bucket {bucket} maps to no split")  # pragma: no cover


def assign_split(doc_id: str) -> str:
    """Deterministically assign a document to its split by hash-partition over ``doc.id``."""
    return bucket_to_split(partition_bucket(doc_id))


def partition_spec() -> dict[str, Any]:
    """The fixed/pinned partition specification recorded in MANIFEST.json."""
    return {
        "hash": PARTITION_HASH,
        "key": PARTITION_KEY,
        "modulus": PARTITION_MODULUS,
        "buckets": {split: [low, high] for split, (low, high) in PARTITION_BUCKETS.items()},
    }


@dataclass(frozen=True)
class LockedShard:
    path: str
    sha256: str
    bytes: int
    doc_count: int
    token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "doc_count": self.doc_count,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LockedShard:
        try:
            return cls(
                path=str(payload["path"]),
                sha256=str(payload["sha256"]),
                bytes=int(payload["bytes"]),
                doc_count=int(payload["doc_count"]),
                token_count=int(payload["token_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LockedDatasetError(f"invalid shard entry in manifest: {payload!r}") from exc


@dataclass(frozen=True)
class LockedSplit:
    name: str
    shards: tuple[LockedShard, ...]
    doc_count: int
    token_count: int
    byte_count: int
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_count": self.doc_count,
            "token_count": self.token_count,
            "byte_count": self.byte_count,
            "fingerprint": self.fingerprint,
            "shards": [shard.to_dict() for shard in self.shards],
        }

    @classmethod
    def from_dict(cls, name: str, payload: dict[str, Any]) -> LockedSplit:
        try:
            shards = tuple(LockedShard.from_dict(entry) for entry in payload["shards"])
            return cls(
                name=name,
                shards=shards,
                doc_count=int(payload["doc_count"]),
                token_count=int(payload["token_count"]),
                byte_count=int(payload["byte_count"]),
                fingerprint=str(payload["fingerprint"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LockedDatasetError(f"invalid split {name!r} in manifest") from exc


@dataclass(frozen=True)
class LockedManifest:
    schema_version: str
    dataset: dict[str, Any]
    partition: dict[str, Any]
    tokenizer: dict[str, Any]
    splits: dict[str, LockedSplit]

    @property
    def pin_sha(self) -> str:
        return str(self.dataset["pin_sha"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": dict(self.dataset),
            "partition": self.partition,
            "tokenizer": dict(self.tokenizer),
            "splits": {
                name: self.splits[name].to_dict() for name in LOCKED_SPLITS if name in self.splits
            },
        }


@dataclass(frozen=True)
class LockedDocument:
    doc_id: str
    text: str
    shard: str
    offset: int
    index: int


def _split_fingerprint(name: str, shards: tuple[LockedShard, ...]) -> str:
    return _fingerprint({"split": name, "shard_sha256": [shard.sha256 for shard in shards]})


def build_locked_manifest(
    *,
    splits: dict[str, list[LockedShard]],
    tokenizer_id: str,
    tokenizer_fingerprint: str,
    pin_sha: str = FINEWEB_EDU_PIN_SHA,
    dataset_name: str = FINEWEB_EDU_DATASET,
    license_id: str = FINEWEB_EDU_LICENSE,
    source_doc_limit: int | None = None,
) -> LockedManifest:
    """Author a :class:`LockedManifest` from per-split shard lists."""
    locked_splits: dict[str, LockedSplit] = {}
    for name in LOCKED_SPLITS:
        shards = tuple(splits.get(name, ()))
        locked_splits[name] = LockedSplit(
            name=name,
            shards=shards,
            doc_count=sum(shard.doc_count for shard in shards),
            token_count=sum(shard.token_count for shard in shards),
            byte_count=sum(shard.bytes for shard in shards),
            fingerprint=_split_fingerprint(name, shards),
        )
    dataset: dict[str, Any] = {
        "name": dataset_name,
        "pin_sha": pin_sha,
        "license": license_id,
    }
    if source_doc_limit is not None:
        dataset["source_doc_limit"] = source_doc_limit
    return LockedManifest(
        schema_version=LOCKED_MANIFEST_SCHEMA,
        dataset=dataset,
        partition=partition_spec(),
        tokenizer={"id": tokenizer_id, "fingerprint": tokenizer_fingerprint},
        splits=locked_splits,
    )


def write_locked_manifest(root: Path | str, manifest: LockedManifest) -> Path:
    """Write MANIFEST.json deterministically (sorted keys) and return its path."""
    path = Path(root) / LOCKED_MANIFEST_FILENAME
    payload = json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path


def parse_locked_manifest(payload: dict[str, Any]) -> LockedManifest:
    if not isinstance(payload, dict):
        raise LockedDatasetError("locked manifest must be a JSON object")
    schema = payload.get("schema_version")
    if schema != LOCKED_MANIFEST_SCHEMA:
        raise LockedDatasetError(
            f"unexpected manifest schema_version {schema!r}; expected {LOCKED_MANIFEST_SCHEMA!r}"
        )
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or "pin_sha" not in dataset:
        raise LockedDatasetError("locked manifest is missing dataset.pin_sha")
    splits_payload = payload.get("splits")
    if not isinstance(splits_payload, dict):
        raise LockedDatasetError("locked manifest is missing splits")
    splits = {
        name: LockedSplit.from_dict(name, splits_payload[name])
        for name in LOCKED_SPLITS
        if name in splits_payload
    }
    return LockedManifest(
        schema_version=str(schema),
        dataset=dataset,
        partition=payload.get("partition", {}),
        tokenizer=payload.get("tokenizer", {}),
        splits=splits,
    )


def load_locked_manifest(root: Path | str) -> LockedManifest:
    path = Path(root) / LOCKED_MANIFEST_FILENAME
    if not path.exists():
        raise LockedDatasetError(f"locked dataset manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LockedDatasetError(f"locked dataset manifest is not valid JSON: {path}") from exc
    return parse_locked_manifest(payload)


def verify_locked_manifest(
    root: Path | str,
    manifest: LockedManifest,
    *,
    splits: tuple[str, ...] | None = None,
) -> list[str]:
    """Recompute on-disk sha256s and return a list of integrity problems (empty == OK).

    Detects: a tampered dataset pin SHA, a tampered partition spec (boundaries/hash/modulus),
    listed-but-missing shards, sha256/byte mismatches (tampering), unlisted on-disk shards, and
    split-fingerprint drift. The pin/partition checks refuse an altered pin/partition even when the
    shards are left byte-identical (architecture.md sections 3, 6, 12).
    """
    base = Path(root)
    names = splits if splits is not None else LOCKED_SPLITS
    problems: list[str] = []
    actual_pin = manifest.dataset.get("pin_sha")
    if actual_pin != FINEWEB_EDU_PIN_SHA:
        problems.append(
            f"dataset pin_sha mismatch: manifest {actual_pin!r} != expected {FINEWEB_EDU_PIN_SHA!r}"
        )
    expected_partition = partition_spec()
    if manifest.partition != expected_partition:
        problems.append(
            f"partition spec mismatch: manifest {manifest.partition!r} "
            f"!= canonical {expected_partition!r}"
        )
    for name in names:
        split = manifest.splits.get(name)
        if split is None:
            problems.append(f"split {name!r} missing from manifest")
            continue
        listed: set[Path] = set()
        for shard in split.shards:
            shard_path = base / shard.path
            listed.add(shard_path)
            if not shard_path.exists():
                problems.append(f"listed shard missing on disk: {shard.path}")
                continue
            data = shard_path.read_bytes()
            actual = hashlib.sha256(data).hexdigest()
            if actual != shard.sha256:
                problems.append(
                    f"sha256 mismatch for {shard.path}: manifest {shard.sha256} != on-disk {actual}"
                )
            if len(data) != shard.bytes:
                problems.append(
                    f"byte-count mismatch for {shard.path}: "
                    f"manifest {shard.bytes} != on-disk {len(data)}"
                )
        split_dir = base / name
        if split_dir.is_dir():
            for found in split_dir.glob(f"{name}-*.jsonl"):
                if found not in listed:
                    problems.append(f"unlisted shard on disk: {name}/{found.name}")
        if _split_fingerprint(name, split.shards) != split.fingerprint:
            problems.append(f"split fingerprint mismatch for {name}")
    return problems


def verify_locked_manifest_or_raise(
    root: Path | str,
    manifest: LockedManifest,
    *,
    splits: tuple[str, ...] | None = None,
) -> None:
    problems = verify_locked_manifest(root, manifest, splits=splits)
    if problems:
        raise LockedDatasetError("locked dataset integrity check failed: " + "; ".join(problems))


def locked_shard_paths(manifest: LockedManifest, split: str) -> list[str]:
    """Ordered, challenge-controlled relative shard paths for a split (manifest order)."""
    split_meta = manifest.splits.get(split)
    if split_meta is None:
        raise LockedDatasetError(f"split {split!r} not present in manifest")
    return [shard.path for shard in split_meta.shards]


def _resolve_split_dir(root: Path, split: str) -> Path:
    candidate = root / split
    if candidate.is_dir():
        return candidate
    # ``root`` may itself be the split directory (e.g. the miner's train-only mount).
    if any(root.glob(f"{split}-*.jsonl")):
        return root
    raise LockedDatasetError(f"locked split {split!r} not found under {root}")


def _locked_shard_files(split_dir: Path, split: str) -> list[Path]:
    # Sorted by zero-padded filename => fixed, challenge-controlled order across runs.
    return sorted(split_dir.glob(f"{split}-*.jsonl"), key=lambda path: path.name)


def _relative_shard_path(root: Path, shard_path: Path, split: str) -> str:
    try:
        return shard_path.relative_to(root).as_posix()
    except ValueError:
        return f"{split}/{shard_path.name}"


def _iter_shard_lines(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield stripped


def iter_locked_documents(root: Path | str, split: str) -> Iterator[LockedDocument]:
    """Iterate a locked split's documents in a deterministic, challenge-controlled order.

    Shards are consumed in sorted-filename order and lines within a shard in file order,
    so two iterations of the same on-disk split yield byte-identical traces.
    """
    base = Path(root)
    split_dir = _resolve_split_dir(base, split)
    index = 0
    for shard_path in _locked_shard_files(split_dir, split):
        rel = _relative_shard_path(base, shard_path, split)
        for offset, line in enumerate(_iter_shard_lines(shard_path)):
            try:
                record = json.loads(line)
                doc_id = str(record["id"])
                text = record["text"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise LockedDatasetError(f"malformed locked shard line {rel}:{offset}") from exc
            if not isinstance(text, str):
                raise LockedDatasetError(f"locked shard line {rel}:{offset} has non-string text")
            yield LockedDocument(doc_id=doc_id, text=text, shard=rel, offset=offset, index=index)
            index += 1


def shard_offset_trace(root: Path | str, split: str) -> list[tuple[str, int]]:
    """The ordered (shard, offset) trace a single-pass harness consumes for a split."""
    return [(doc.shard, doc.offset) for doc in iter_locked_documents(root, split)]


def load_locked_train_texts(data_dir: Path | str) -> list[str]:
    """Convenience loader: the ordered raw-text stream of the read-only locked train split."""
    return [doc.text for doc in iter_locked_documents(Path(data_dir), "train")]
