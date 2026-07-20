"""Explicit CPU re-exec test-mode seam (architecture.md 3.4/3.5; VAL-PRISM-013/015).

This installs the repo's OWN CPU re-exec seam (:func:`mock_reexec.cpu_reexec_run`) as an
explicit, opt-in configuration so a local mission harness can stand up a faithful worker /
audit-replay path on a CPU-only host: a real, deterministic ``prism_run_manifest.v2`` is authored
by the challenge runner on CPU with NO GPU, Docker, or broker contacted. It is OFF by default and
never used by a production deploy (which always dispatches to the real broker).

Enable it by setting ``PRISM_WORKER_PLANE__CPU_REEXEC_TEST_MODE=true``: :func:`create_app` then
calls :func:`configure_cpu_reexec_test_mode`, which stages a tiny locked train shard (unless a dir
is supplied) and replaces ``DockerExecutor.run`` with the CPU seam for the whole process.

The two-script bundle below is the smallest real deterministic run: a byte-level ``TinyLM`` trained
one step at a time over the challenge instrument. Two honest re-execs of the SAME submission author
byte-identical manifests EXCEPT the host-timing fields in the ``compute`` block, so
:func:`normalize_manifest_for_replication` drops those before hashing, letting independent worker
replicas converge on one ``manifest_sha256`` (the base worker plane's reconciliation acceptance
rule).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import WorkerPlaneConfig
from ..proof import compute_manifest_sha256
from . import container as _container
from .container import PrismContainerEvaluator
from .interface import PrismContext
from .mock_reexec import cpu_reexec_run
from .source_similarity import SourceFile

# A tiny CPU-torch two-script bundle: a byte-level next-token TinyLM trained one step at a time over
# the challenge instrument. No GPU, no tokenizer (byte basis), deterministic under the forced seed.
TINY_ARCHITECTURE = """
import torch
from torch import nn


class TinyLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 8)
        self.head = nn.Linear(8, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return TinyLM(ctx.vocab_size)
"""

TINY_TRAINING = """
import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        nv = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, nv), batch.tokens[:, 1:].reshape(-1) % nv
        )
        loss.backward()
        opt.step()
"""

#: Fields in the manifest ``compute`` block that legitimately differ between two honest re-execs of
#: the SAME submission (host wall-clock + resident memory). They are normalized to a fixed value
#: before hashing so deterministic replicas produce one ``manifest_sha256``.
VOLATILE_COMPUTE_FIELDS: tuple[str, ...] = (
    "wall_clock_seconds",
    "peak_rss_bytes",
    "peak_vram_bytes",
)

_SHARD_LINE = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample number {i} '
    'has enough bytes to cover several challenge instrument batches deterministically"}}\n'
)


def stage_tiny_train_data(root: Path | str, *, lines: int = 64) -> Path:
    """Stage a tiny locked train shard under ``root`` and return its directory.

    Mirrors the locked FineWeb-Edu train mount the real broker provides, sized so a byte-level run
    covers several instrument batches deterministically.
    """

    data_dir = Path(root) / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shard = data_dir / "train-00000.jsonl"
    if not shard.exists():
        shard.write_text("".join(_SHARD_LINE.format(i=i) for i in range(lines)), encoding="utf-8")
    return data_dir


def install_cpu_reexec_seam(train_data_dir: Path | str) -> None:
    """Replace ``DockerExecutor.run`` with the CPU re-exec seam for this process.

    Idempotent: a repeated call re-binds the same CPU runner. No broker/Docker is ever contacted
    after this is installed.
    """

    _container.DockerExecutor.run = cpu_reexec_run(  # type: ignore[assignment,method-assign]
        train_data_dir=Path(train_data_dir)
    )


def cpu_test_context(config: WorkerPlaneConfig) -> PrismContext:
    """Build the tiny deterministic :class:`PrismContext` for CPU re-exec test mode."""

    return PrismContext(
        vocab_size=config.cpu_reexec_vocab_size,
        sequence_length=config.cpu_reexec_sequence_length,
        seed=config.cpu_reexec_seed,
        step_budget=config.cpu_reexec_step_budget,
    )


def configure_cpu_reexec_test_mode(settings: Any) -> Path:
    """Stage tiny train data (unless supplied) + install the CPU seam. Returns the train data dir.

    Called by :func:`create_app` when ``worker_plane.cpu_reexec_test_mode`` is on so the prism
    service's own worker (e.g. the audit-replay path) re-executes on CPU, and reusable by a local
    worker/validator agent process to install the exact same seam.
    """

    config = settings.worker_plane
    if config.cpu_reexec_train_data_dir:
        data_dir = Path(config.cpu_reexec_train_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        data_dir = stage_tiny_train_data(
            settings.base_eval_artifact_root, lines=config.cpu_reexec_train_lines
        )
    install_cpu_reexec_seam(data_dir)
    return data_dir


def normalize_manifest_for_replication(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy with volatile host-timing fields fixed to a constant.

    Two honest CPU re-execs of the SAME submission differ in host timing: the
    classical ``compute`` wall/RSS fields plus (after telemetry-rt) per-step
    ``metrics.train_series.points[*].wall_s`` and the linked
    ``train_series_sha256`` digests. Fix timing so canonical manifest bytes
    (and thus :func:`compute_manifest_sha256`) match across replicas.
    """

    from .train_series import train_series_sha256

    normalized = copy.deepcopy(manifest)
    compute = normalized.get("compute")
    if isinstance(compute, dict):
        for field in VOLATILE_COMPUTE_FIELDS:
            if field in compute:
                compute[field] = 0

    metrics = normalized.get("metrics")
    series_digest: str | None = None
    if isinstance(metrics, dict):
        raw_series = metrics.get("train_series")
        if isinstance(raw_series, dict):
            points = raw_series.get("points")
            if isinstance(points, list):
                for point in points:
                    if isinstance(point, dict) and "wall_s" in point:
                        point["wall_s"] = 0.0
            series_digest = train_series_sha256(raw_series)
            if "train_series_sha256" in metrics:
                metrics["train_series_sha256"] = series_digest
            if "sha256" in raw_series:
                raw_series["sha256"] = series_digest

    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, dict) and "train_series_sha256" in artifacts:
        artifacts["train_series_sha256"] = series_digest or ("0" * 64)

    return normalized


@dataclass(frozen=True)
class CpuReexecOutcome:
    """A CPU re-exec result: the (optionally normalized) manifest, its hash, and artifact dir."""

    manifest: dict[str, Any]
    manifest_sha256: str
    artifact_dir: Path


def evaluate_cpu_reexec(
    settings: Any,
    *,
    submission_id: str,
    arch_code: str = TINY_ARCHITECTURE,
    train_code: str = TINY_TRAINING,
    normalize: bool = True,
) -> CpuReexecOutcome:
    """Run one deterministic CPU re-exec for ``submission_id`` and hash its manifest.

    The CPU seam must already be installed (:func:`configure_cpu_reexec_test_mode`). Returns the
    normalized-for-replication manifest and its ``manifest_sha256`` so a worker can sign a proof or
    an auditor can compute the authoritative hash.
    """

    ctx = cpu_test_context(settings.worker_plane)
    evaluator = PrismContainerEvaluator(settings=settings, ctx=ctx)
    files = (
        SourceFile("architecture.py", arch_code, _sha256(arch_code)),
        SourceFile("training.py", train_code, _sha256(train_code)),
    )
    result = evaluator.evaluate(
        submission_id=submission_id,
        code=arch_code,
        code_hash=files[0].sha256,
        arch_hash=files[0].sha256,
        backend=settings.execution_backend,
        files=files,
    )
    manifest = result.run_manifest
    if manifest is None:
        raise RuntimeError(f"CPU re-exec for {submission_id!r} produced no manifest")
    if normalize:
        manifest = normalize_manifest_for_replication(manifest)
    return CpuReexecOutcome(
        manifest=manifest,
        manifest_sha256=compute_manifest_sha256(manifest),
        artifact_dir=Path(result.artifact_output_path or ""),
    )


def _sha256(text: str) -> str:
    from hashlib import sha256

    return sha256(text.encode()).hexdigest()


__all__ = [
    "TINY_ARCHITECTURE",
    "TINY_TRAINING",
    "VOLATILE_COMPUTE_FIELDS",
    "CpuReexecOutcome",
    "configure_cpu_reexec_test_mode",
    "cpu_test_context",
    "evaluate_cpu_reexec",
    "install_cpu_reexec_seam",
    "normalize_manifest_for_replication",
    "stage_tiny_train_data",
]
