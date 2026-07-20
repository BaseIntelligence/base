"""Host-side held-out delta + anti-memorization gap (architecture.md sections 5, 6).

The held-out delta-over-random-init is the scoring TIE-BREAKER:
``delta = bpb(random-init twin on val) - bpb(trained model on val)`` where a LARGER improvement
ranks better. The challenge also records the train-vs-held-out gap
(``gap = bpb(trained on val) - prequential train bpb``) as an ANTI-MEMORIZATION flag and penalizes
an excessive gap.

The secret ``val`` split is NEVER bind-mounted into the network=none eval container (VAL-HARNESS-015
/ VAL-CHEAT-007) and is never exposed via ``PrismContext``; only the CHALLENGE SCORER reads it, here
on the worker host. To compute the held-out bpb the in-container runner persists the trained
weights (``trained_state.pt``); this module rebuilds the miner architecture under the FORCED seed
(both a random-init twin and a fresh model loaded with the trained weights) and evaluates both on
val. The factory + forward run in a short-lived, time/memory-bounded ``forkserver`` child (mirroring
``static_instantiation``) so a hostile model cannot stall or balloon the worker.

To complete LIVE within a bounded compute budget (the full secret-val eval is single-threaded on
the manager CPU and overruns a tight timeout), the held-out eval is capped to a FIXED, DETERMINISTIC
val byte budget: only a stable prefix of the val split is scored, and the SAME prefix is used for
both the random-init twin and the trained model so the delta stays comparable. The timeout is
configurable (raised for the scorer). The byte denominator keeps the delta tokenizer-agnostic and
the fixed prefix keeps it deterministic (same submission + seed + budget => same delta).
"""

from __future__ import annotations

import importlib.util
import math
import multiprocessing as mp
import os
import resource
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import LockedDatasetError, iter_locked_documents
from .interface import PrismContext
from .scoring import MEMORIZATION_GAP_THRESHOLD_BPB

# Raised from the original 120s: a promote-scale (~350M) model forward over the bounded val on the
# manager CPU needs headroom above 120s, while the byte budget keeps the absolute compute small.
# Configurable per-deploy via ``PrismSettings.base_eval_heldout_timeout_seconds``.
DEFAULT_TIMEOUT_SECONDS = 600.0
DEFAULT_MEMORY_HEADROOM_BYTES = 8 * 1024**3
DEFAULT_HELDOUT_BATCH_SIZE = 8
# Fixed, deterministic val compute budget: the held-out eval scores only a stable PREFIX of the
# secret val split covering at most this many raw UTF-8 bytes (plus at most one boundary-crossing
# document). Both the random-init twin and the trained model see the IDENTICAL prefix, so the delta
# stays comparable and directionally correct while the host-side eval completes within budget. A
# value <= 0 disables the cap (scores the entire val split). Configurable per-deploy via
# ``PrismSettings.base_eval_heldout_val_byte_budget``.
DEFAULT_HELDOUT_VAL_BYTE_BUDGET = 65536
# A NaN/Inf held-out batch loss is sanitized to a worst-case (high) per-batch code-length so a
# degenerate model can never collapse into a finite, advantageous held-out bpb.
WORST_CASE_LOSS_MULTIPLIER = 2.0


# The host always measures the held-out val bpb on raw UTF-8 BYTES (tokenizer-agnostic), so the
# anti-memorization GAP comparison is only valid when the run's prequential TRAIN bpb was ALSO
# measured on the byte basis (architecture.md sections 5, 6; AGENTS.md held-out invariant).
VAL_BPB_BASIS = "bytes"


@dataclass(frozen=True)
class HeldoutResult:
    """Challenge-computed held-out delta + anti-memorization gap on the SECRET val split."""

    val_bpb_trained: float
    val_bpb_random_init: float
    heldout_delta: float
    train_heldout_gap: float | None
    memorization_flag: bool
    train_bpb_basis: str | None = None
    val_bpb_basis: str = VAL_BPB_BASIS
    val_covered_bytes: int = 0
    # The CONVERGED (final-checkpoint) train bpb: the trained model re-evaluated byte-level over a
    # fixed prefix of the exposed train split. When present it is the train side of the gap (a
    # like-for-like, byte-basis comparison with ``val_bpb_trained``); ``gap_basis`` records which
    # train reference the gap used ("converged" vs the curve-averaged "prequential" fallback).
    train_bpb_converged: float | None = None
    gap_basis: str | None = None

    def as_metrics(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "val_bpb_trained": self.val_bpb_trained,
            "val_bpb_random_init": self.val_bpb_random_init,
            "heldout_delta": self.heldout_delta,
            "held_out_delta": self.heldout_delta,
            "memorization_flag": self.memorization_flag,
            "val_bpb_basis": self.val_bpb_basis,
            "val_covered_bytes": self.val_covered_bytes,
        }
        if self.train_bpb_basis is not None:
            payload["train_bpb_basis"] = self.train_bpb_basis
        if self.train_bpb_converged is not None:
            payload["train_bpb_converged"] = self.train_bpb_converged
        if self.gap_basis is not None:
            payload["gap_basis"] = self.gap_basis
        if self.train_heldout_gap is not None:
            payload["train_heldout_gap"] = self.train_heldout_gap
            payload["train_val_gap"] = self.train_heldout_gap
        return payload


def _build_context() -> Any:
    try:
        context = mp.get_context("forkserver")
        context.set_forkserver_preload(["torch"])
        return context
    except (ValueError, OSError):  # pragma: no cover - platform without forkserver
        return mp.get_context("fork")


_MP_CONTEXT = _build_context()


def val_split_present(val_data_dir: str | os.PathLike[str] | None) -> bool:
    """True when a non-empty ``val`` split is resolvable at ``val_data_dir`` (host-only)."""
    if not val_data_dir:
        return False
    base = Path(val_data_dir)
    if not base.is_dir():
        return False
    if any(base.glob("val-*.jsonl")):
        return True
    nested = base / "val"
    return nested.is_dir() and any(nested.glob("val-*.jsonl"))


def compute_heldout_metrics(
    *,
    files: Mapping[str, str],
    entrypoint: str,
    ctx: PrismContext,
    trained_state_path: str | os.PathLike[str] | None,
    val_data_dir: str | os.PathLike[str] | None,
    train_bpb: float | None,
    train_bpb_basis: str | None = None,
    train_data_dir: str | os.PathLike[str] | None = None,
    build_model_symbol: str = "build_model",
    batch_size: int = DEFAULT_HELDOUT_BATCH_SIZE,
    val_byte_budget: int = DEFAULT_HELDOUT_VAL_BYTE_BUDGET,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_headroom_bytes: int = DEFAULT_MEMORY_HEADROOM_BYTES,
    gap_threshold_bpb: float = MEMORIZATION_GAP_THRESHOLD_BPB,
) -> HeldoutResult | None:
    """Compute the held-out delta + anti-memorization gap, or ``None`` to skip gracefully.

    Returns ``None`` (held-out skipped, the run still scores on prequential bpb) when the trained
    weights are absent, the secret val split is unavailable, or the bounded child cannot produce a
    finite/positive held-out bpb for both models.

    ``val_byte_budget`` caps the host-side compute to a FIXED, DETERMINISTIC prefix of the secret
    val split (at most this many raw UTF-8 bytes, plus one boundary-crossing document). The twin and
    the trained model are scored over the IDENTICAL prefix so the delta stays comparable and
    directionally correct, while the eval completes LIVE within budget. A value <= 0 scores the
    whole val split.

    ``train_data_dir`` (when resolvable host-side) switches the anti-memorization GAP to use the
    CONVERGED (final-checkpoint) train bpb as the train reference: the trained model is re-evaluated
    byte-level over a fixed prefix of the exposed train split (the SAME byte-level method as val),
    so the gap is a like-for-like comparison of the converged model's train vs held-out performance.
    This closes a false-negative hole in the prequential reference: the curve-averaged prequential
    train bpb is inflated by early high-loss steps, which SHRINKS the gap and can MISS a genuine
    memorizer. Because both sides are byte-level on the same trained model, the converged gap is
    basis-consistent by construction (no tokenizer-basis false-flag).

    ``train_bpb_basis`` is the tokenizer basis the run's prequential TRAIN bpb was measured on. It
    only governs the PREQUENTIAL fallback used when ``train_data_dir`` is unavailable: the held-out
    val bpb is always byte-level (``VAL_BPB_BASIS``), so the prequential-reference gap is applied
    ONLY when the two bases are like-for-like, sparing a benign tokenizer-using learner from a false
    flag (VAL-SCORE-009 / VAL-SCORE-004). The byte-denominator delta tie-breaker (random twin vs
    trained, both byte-level on val) stays valid regardless of the train basis.
    """
    if trained_state_path is None:
        return None
    state_path = Path(trained_state_path)
    if not state_path.is_file():
        return None
    if not val_split_present(val_data_dir):
        return None

    parent_conn, child_conn = _MP_CONTEXT.Pipe(duplex=False)
    proc = _MP_CONTEXT.Process(
        target=_child_heldout,
        args=(
            dict(files),
            entrypoint,
            build_model_symbol,
            ctx,
            str(state_path),
            str(val_data_dir),
            str(train_data_dir) if train_data_dir else "",
            int(batch_size),
            int(val_byte_budget),
            int(memory_headroom_bytes),
            child_conn,
        ),
    )
    proc.start()
    child_conn.close()
    result: tuple[str, object] | None = None
    try:
        if parent_conn.poll(timeout_seconds):
            try:
                result = parent_conn.recv()
            except EOFError:
                result = None
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()
        else:
            proc.join()
        parent_conn.close()

    if result is None:
        return None
    kind, detail = result
    if kind != "ok" or not isinstance(detail, Mapping):
        return None
    val_bpb_random = detail.get("val_bpb_random_init")
    val_bpb_trained = detail.get("val_bpb_trained")
    if not _is_sane_bpb(val_bpb_random) or not _is_sane_bpb(val_bpb_trained):
        return None
    val_bpb_random = float(val_bpb_random)  # type: ignore[arg-type]
    val_bpb_trained = float(val_bpb_trained)  # type: ignore[arg-type]
    heldout_delta = val_bpb_random - val_bpb_trained
    covered = detail.get("val_covered_bytes")
    val_covered_bytes = int(covered) if isinstance(covered, int | float) else 0
    train_converged_raw = detail.get("train_bpb_converged")
    train_bpb_converged = (
        float(train_converged_raw)  # type: ignore[arg-type]
        if _is_sane_bpb(train_converged_raw)
        else None
    )
    gap: float | None = None
    flag = False
    gap_basis: str | None = None
    if train_bpb_converged is not None:
        # Prefer the CONVERGED (final-checkpoint) train bpb: the trained model byte-level on the
        # exposed train split, like-for-like with the byte-level held-out val bpb. It reflects the
        # converged model (not the inflated curve-averaged AUC), so a genuine memorizer is reliably
        # flagged; the symmetric byte basis means no tokenizer-basis gating is needed.
        gap = val_bpb_trained - train_bpb_converged
        flag = gap > gap_threshold_bpb
        gap_basis = "converged"
    else:
        # Fallback: the curve-averaged prequential train bpb, gated to a like-for-like basis so a
        # benign tokenizer-using learner is never false-flagged on a cross-basis gap.
        bases_comparable = train_bpb_basis is None or train_bpb_basis == VAL_BPB_BASIS
        if bases_comparable and train_bpb is not None and math.isfinite(float(train_bpb)):
            gap = val_bpb_trained - float(train_bpb)
            flag = gap > gap_threshold_bpb
            gap_basis = "prequential"
    return HeldoutResult(
        val_bpb_trained=val_bpb_trained,
        val_bpb_random_init=val_bpb_random,
        heldout_delta=heldout_delta,
        train_heldout_gap=gap,
        memorization_flag=flag,
        train_bpb_basis=train_bpb_basis,
        val_covered_bytes=val_covered_bytes,
        train_bpb_converged=train_bpb_converged,
        gap_basis=gap_basis,
    )


def _is_sane_bpb(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _child_heldout(
    files: Mapping[str, str],
    entrypoint: str,
    symbol: str,
    ctx: PrismContext,
    trained_state_path: str,
    val_data_dir: str,
    train_data_dir: str,
    batch_size: int,
    val_byte_budget: int,
    memory_headroom_bytes: int,
    conn: Any,
) -> None:
    try:
        import torch

        torch.set_num_threads(1)
        if memory_headroom_bytes > 0:
            try:
                cap = _vmsize_bytes() + memory_headroom_bytes
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            except (ValueError, OSError):
                pass

        workdir = tempfile.mkdtemp(prefix="prism-heldout-")
        for rel_path, content in files.items():
            dest = os.path.join(workdir, rel_path)
            parent = os.path.dirname(dest)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as handle:
                handle.write(content)
        entry_path = os.path.join(workdir, entrypoint)
        sys.path.insert(0, os.path.dirname(entry_path))
        sys.path.insert(0, workdir)

        spec = importlib.util.spec_from_file_location("prism_heldout_arch", entry_path)
        if spec is None or spec.loader is None:
            conn.send(("error", f"cannot load architecture entrypoint {entrypoint}"))
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules["prism_heldout_arch"] = module
        spec.loader.exec_module(module)
        factory = getattr(module, symbol, None)
        if factory is None or not callable(factory):
            conn.send(("error", f"architecture is missing callable {symbol}"))
            return

        # A fixed, deterministic prefix of the val split (same prefix for twin + trained), bounding
        # the host-side compute so the held-out delta completes LIVE within budget.
        texts = _load_split_texts(val_data_dir, "val", val_byte_budget)
        if not texts:
            conn.send(("skip", "val split empty"))
            return
        val_covered_bytes = sum(len(text.encode("utf-8")) for text in texts)

        vocab_size = max(int(ctx.vocab_size), 2)
        seq_len = max(int(ctx.sequence_length), 2)
        baseline_nats = math.log(vocab_size)

        # Random-init baseline: a forced-seed CPU instantiation of the SAME architecture. This is a
        # REPRESENTATIVE random-init twin for the byte-level held-out delta tie-breaker, NOT a
        # faithful reproduction of the in-container runner's step-0 weights (the runner seeds CUDA +
        # random + sets deterministic flags on the GPU device; here we only seed CPU torch). The
        # delta stays a valid "trained vs a random-init twin of the same architecture" comparison
        # and is directionally correct (a genuine learner has strictly lower val bpb than this
        # baseline; a no-op coincides with it => delta ~ 0).
        torch.manual_seed(int(ctx.seed))
        twin = factory(ctx)
        if not isinstance(twin, torch.nn.Module):
            conn.send(("error", "build_model returned a non-nn.Module value"))
            return
        val_bpb_random = _bpb_over_texts(
            twin,
            texts,
            vocab_size=vocab_size,
            seq_len=seq_len,
            batch_size=batch_size,
            baseline_nats=baseline_nats,
        )

        # Trained model: a fresh instance loaded with the persisted trained weights.
        torch.manual_seed(int(ctx.seed))
        trained = factory(ctx)
        if not isinstance(trained, torch.nn.Module):
            conn.send(("error", "build_model returned a non-nn.Module value"))
            return
        # weights_only=True REFUSES arbitrary pickle reconstruction: trained_state.pt lives in the
        # miner-writable artifacts_dir, so an unguarded torch.load is a host RCE sink. A hostile
        # pickle raises here and the held-out step fails safe (the run still scores on prequential
        # bpb). Caller-side, only the manifest-recorded artifact path is ever passed in.
        state = torch.load(trained_state_path, map_location="cpu", weights_only=True)
        trained.load_state_dict(_strip_module_prefix(state), strict=False)
        val_bpb_trained = _bpb_over_texts(
            trained,
            texts,
            vocab_size=vocab_size,
            seq_len=seq_len,
            batch_size=batch_size,
            baseline_nats=baseline_nats,
        )

        # CONVERGED (final-checkpoint) train bpb: the SAME trained model re-evaluated byte-level
        # over a fixed deterministic prefix of the EXPOSED train split (the data it actually
        # learned), the SAME byte-level method used for val. The train side of the memorization gap;
        # measuring it on the converged model (not the curve-averaged AUC) reliably exposes a
        # memorizer's train-vs-held-out divergence. Absent/unresolvable train data => omitted, and
        # the parent falls back to the prequential train reference.
        train_bpb_converged: float | None = None
        if train_data_dir:
            train_texts = _load_split_texts(train_data_dir, "train", val_byte_budget)
            if train_texts:
                train_bpb_converged = _bpb_over_texts(
                    trained,
                    train_texts,
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    baseline_nats=baseline_nats,
                )

        conn.send(
            (
                "ok",
                {
                    "val_bpb_random_init": val_bpb_random,
                    "val_bpb_trained": val_bpb_trained,
                    "val_covered_bytes": val_covered_bytes,
                    "train_bpb_converged": train_bpb_converged,
                },
            )
        )
    except BaseException as exc:  # noqa: BLE001 - report any failure cleanly to the parent
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass
        os._exit(0)


def _load_split_texts(data_dir: str, split: str, byte_budget: int) -> list[str]:
    """A deterministic prefix of ``split`` bounded by ``byte_budget`` raw UTF-8 bytes.

    Documents are consumed in the locked, challenge-controlled order (``iter_locked_documents``) and
    accumulated until the cumulative byte count reaches ``byte_budget`` (the boundary-crossing
    document is included, so at least one document is always scored). A ``byte_budget`` <= 0 returns
    the entire split. The prefix is identical across runs and is the SAME for both the twin and the
    trained model (val) and for the converged train reference, so the held-out metrics stay
    deterministic and comparable.
    """
    try:
        texts: list[str] = []
        total = 0
        for doc in iter_locked_documents(data_dir, split):
            texts.append(doc.text)
            if byte_budget > 0:
                total += len(doc.text.encode("utf-8"))
                if total >= byte_budget:
                    break
        return texts
    except LockedDatasetError:
        return []


def _strip_module_prefix(state: Any) -> Any:
    if not isinstance(state, Mapping):
        return state
    prefix = "module."
    if not any(isinstance(key, str) and key.startswith(prefix) for key in state):
        return dict(state)
    stripped: dict[Any, Any] = {}
    for key, value in state.items():
        new_key = key[len(prefix) :] if isinstance(key, str) and key.startswith(prefix) else key
        stripped[new_key] = value
    return stripped


def _bpb_over_texts(
    model: Any,
    texts: list[str],
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    baseline_nats: float,
) -> float | None:
    """Single-pass byte-level bits-per-byte of a FIXED model over the held-out texts (no training).

    The denominator is raw UTF-8 BYTES covered (tokenizer-agnostic, mirroring the prequential train
    metric); byte-level tokenization matches the runner's default instrument. Returns ``None`` for
    an empty / non-finite measurement so it never collapses into a fabricated score.
    """
    import torch
    import torch.nn.functional as functional

    vocab = max(int(vocab_size), 2)
    seq = max(int(seq_len), 2)
    bs = max(int(batch_size), 1)
    needed = seq * bs
    worst = baseline_nats * WORST_CASE_LOSS_MULTIPLIER
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    was_training = model.training
    model.eval()
    sum_nll_nats = 0.0
    covered_bytes = 0
    samples = 0

    def _flush(chunk: list[int]) -> None:
        nonlocal sum_nll_nats, covered_bytes, samples
        count = len(chunk)
        if count < 2:
            return
        if count == needed:
            rows, cols = bs, seq
        else:
            rows, cols = 1, count
        tokens = (
            torch.tensor(chunk, dtype=torch.long).remainder(vocab).view(rows, cols).to(model_device)
        )
        with torch.no_grad():
            logits = _logits_tensor(model(tokens))
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        predictions = logits[:, :-1, :].reshape(-1, vocab)
        targets = (tokens[:, 1:].reshape(-1)) % vocab
        loss = float(functional.cross_entropy(predictions, targets).detach().item())
        if not math.isfinite(loss):
            loss = worst
        predicted = rows * max(cols - 1, 0)
        sum_nll_nats += loss * predicted
        covered_bytes += count
        samples += 1

    buffer: list[int] = []
    for text in texts:
        for byte in text.encode("utf-8"):
            buffer.append(int(byte))
            if len(buffer) >= needed:
                _flush(buffer[:needed])
                buffer = buffer[needed:]
    if buffer:
        _flush(buffer)

    if was_training:
        model.train()
    if covered_bytes <= 0 or samples == 0:
        return None
    bits = sum_nll_nats / math.log(2.0)
    if not math.isfinite(bits):
        return None
    bpb = bits / covered_bytes
    return bpb if math.isfinite(bpb) and bpb > 0.0 else None


def _logits_tensor(logits: Any) -> Any:
    import torch

    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(logits, tuple | list) and logits and isinstance(logits[0], torch.Tensor):
        return logits[0]
    if hasattr(logits, "logits"):
        return logits.logits
    raise RuntimeError("miner model forward did not return a logits tensor")


def _vmsize_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0
