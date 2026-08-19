"""Prism-shaped AutoModel entry for LoopMoE (recipe 2.0).

Exposes build_model / train for the operator harness seams. Uses FineWeb
stream + prism_telemetry from ctx. Real single-node DDP (one process per
GPU) + NVFP4 TE recipe when the class exists.

submission_nonce: loopmoe-chunkwy-1h-4x5090-20260818T0530Z
"""

from __future__ import annotations

import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import torch

from nemo_automodel.components.models.loopmoe import kernels as loopmoe_kernels
from nemo_automodel.components.models.loopmoe.model import build_loopmoe

try:
    import prism_telemetry
except ImportError:

    class _TelemetryFallback:
        @staticmethod
        def report(**_kwargs):
            return None

        @staticmethod
        def finish_evaluation():
            return None

    prism_telemetry = _TelemetryFallback()


PEAK_LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
EPS = 1e-8
WARMUP_FRAC = 0.02
MIN_LR_FRAC = 0.10
GRAD_CLIP = 1.0
AUX_LOSS_COEF = 0.01
REPORT_EVERY = 10
WALL_MARGIN_S = 90.0
# Per-GPU microbatch. Harness default is 8 *then DataParallel-sharded*.
# DDP keeps this whole batch on every rank (× world_size global tokens).
# Factored WY drops the 5-D decay tensor; mb=8 feeds GEMMs (seq stays 512).
DEFAULT_MICRO_BATCH = 8
PEAK_FLOPS_PER_GPU = 209.5e12
PAYLOAD_NAME = "loopmoe_ddp_payload.pt"
METRICS_NAME = "loopmoe_ddp_metrics.json"
WEIGHTS_NAME = "loopmoe_ddp_weights.pt"


def build_model(ctx):
    """CPU module; harness moves it to ctx['device'] after param-cap check."""
    return build_loopmoe(ctx)


def _param_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in name or "loop_bias" in name or "inject_scale" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _lr_at(step, total_steps):
    warmup = max(1, int(WARMUP_FRAC * total_steps))
    if step < warmup:
        return PEAK_LR * float(step + 1) / float(warmup)
    t = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return PEAK_LR * (MIN_LR_FRAC + (1.0 - MIN_LR_FRAC) * cos)


def _maybe_te_recipe():
    """Prefer NVFP4. SM120 (consumer Blackwell) disables RHT / stochastic rounding."""
    try:
        from transformer_engine.common import recipe as te_recipe  # type: ignore
    except Exception as exc:  # noqa: BLE001
        print(f"[loopmoe] TE recipe import failed ({exc})", flush=True)
        return None, "none"
    sm = None
    if torch.cuda.is_available():
        sm = torch.cuda.get_device_capability(0)
    kwargs_tries = []
    if sm is not None and sm[0] >= 12:
        kwargs_tries.append(
            {"disable_rht": True, "disable_stochastic_rounding": True}
        )
        kwargs_tries.append({"disable_rht": True})
    kwargs_tries.append({})
    for name in ("NVFP4BlockScaling", "Float4BlockScaling", "MXFP4BlockScaling"):
        cls = getattr(te_recipe, name, None)
        if cls is None:
            continue
        for kw in kwargs_tries:
            try:
                rec = cls(**kw)
                print(
                    f"[loopmoe] NVFP4 recipe class={name} kwargs={kw} sm={sm}",
                    flush=True,
                )
                return rec, "nvfp4"
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001
                print(f"[loopmoe] {name}({kw}) failed ({exc})", flush=True)
                continue
    delayed = getattr(te_recipe, "DelayedScaling", None)
    if delayed is not None:
        try:
            print("[loopmoe] NVFP4 class missing; DelayedScaling FP8 fallback", flush=True)
            return delayed(), "fp8"
        except Exception as exc:  # noqa: BLE001
            print(f"[loopmoe] DelayedScaling failed ({exc})", flush=True)
    return None, "none"


def _fp8_ctx(enabled, rec):
    """TE 2.16+ uses `autocast`; older wheels still export `fp8_autocast`."""
    if not enabled or rec is None:
        from contextlib import nullcontext

        return nullcontext()
    try:
        from transformer_engine.pytorch import autocast as te_autocast  # type: ignore

        try:
            ctx = te_autocast(enabled=True, recipe=rec)
        except TypeError:
            ctx = te_autocast(recipe=rec)
        if not getattr(_fp8_ctx, "_logged", False):
            print("[loopmoe] using te.autocast for NVFP4/FP8 recipe (fwd+bwd)", flush=True)
            _fp8_ctx._logged = True
        return ctx
    except Exception as exc:  # noqa: BLE001
        print(f"[loopmoe] te.autocast unavailable ({exc}); trying fp8_autocast", flush=True)
    try:
        from transformer_engine.pytorch import fp8_autocast  # type: ignore

        if not getattr(_fp8_ctx, "_logged", False):
            print("[loopmoe] using te.fp8_autocast for NVFP4/FP8 recipe", flush=True)
            _fp8_ctx._logged = True
        return fp8_autocast(enabled=True, fp8_recipe=rec)
    except Exception as exc:  # noqa: BLE001
        print(f"[loopmoe] fp8_autocast unavailable ({exc}); BF16", flush=True)
        from contextlib import nullcontext

        return nullcontext()


def _unwrap(model):
    return model.module if hasattr(model, "module") else model


def _te_version():
    try:
        import transformer_engine as te  # type: ignore

        return str(getattr(te, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "missing"


def _rendezvous_port():
    """High IPv4 port derived from pid — never resolve hostname localhost."""
    return 29511 + (os.getpid() % 487)


def _set_dist_env(port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_SOCKET_FAMILY", "AF_INET")
    os.environ.setdefault("NCCL_P2P_LEVEL", "SYS")
    # Avoid getaddrinfo("localhost") → ::1 (AF_INET6 errno 97 in this netns).
    os.environ["TORCH_DIST_INIT_BARRIER"] = "1"


def _make_adam(model, *, zero=False):
    kwargs = dict(lr=PEAK_LR, betas=BETAS, eps=EPS)
    groups = _param_groups(model)
    if zero:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        try:
            opt = ZeroRedundancyOptimizer(
                groups,
                optimizer_class=torch.optim.AdamW,
                parameters_as_bucket_view=True,
                fused=True,
                **kwargs,
            )
            print("[loopmoe] ZeRO-1 ZeroRedundancyOptimizer fused AdamW", flush=True)
            return opt
        except (TypeError, RuntimeError) as exc:
            print(f"[loopmoe] ZeRO-1 fused failed ({exc}); plain AdamW", flush=True)
            try:
                return ZeroRedundancyOptimizer(
                    groups,
                    optimizer_class=torch.optim.AdamW,
                    parameters_as_bucket_view=True,
                    **kwargs,
                )
            except Exception as exc2:  # noqa: BLE001
                print(f"[loopmoe] ZeRO-1 unavailable ({exc2}); DDP AdamW", flush=True)
    try:
        return torch.optim.AdamW(groups, fused=True, **kwargs)
    except (TypeError, RuntimeError) as exc:
        print(f"[loopmoe] fused AdamW unavailable ({exc}); foreach", flush=True)
        try:
            return torch.optim.AdamW(groups, foreach=True, **kwargs)
        except TypeError:
            return torch.optim.AdamW(groups, **kwargs)


def _maybe_compile(model):
    # TE NVFP4 + dynamo OOMed the first 8-GPU smoke; enable only when
    # LOOPMOE_COMPILE=1 after a saturated eager run.
    if os.environ.get("LOOPMOE_COMPILE", "").strip() not in {"1", "true", "yes"}:
        return model, False
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model, False
    try:
        compiled = compile_fn(model, mode="default", fullgraph=False, dynamic=False)
        return compiled, True
    except Exception as exc:  # noqa: BLE001
        print(f"[loopmoe] torch.compile skipped ({exc})", flush=True)
        return model, False


def _release_parent_cuda(model, stream=None):
    """Parent FLOPs probe leaves ~30GiB on GPU 0; workers cannot spawn until it is gone."""
    import gc

    def _cpu_tensors(obj):
        for child in obj.modules() if hasattr(obj, "modules") else []:
            for name, val in list(vars(child).items()):
                if torch.is_tensor(val) and val.is_cuda:
                    setattr(child, name, val.detach().cpu())
        for p in obj.parameters():
            p.grad = None
            if p.data.is_cuda:
                p.data = p.data.cpu()
        for b in obj.buffers():
            if b.is_cuda:
                b.data = b.data.cpu()

    model.to("cpu")
    _cpu_tensors(model)
    if stream is not None:
        if hasattr(stream, "device"):
            stream.device = "cpu"
        for name in ("_buf", "_last", "input_ids", "labels"):
            val = getattr(stream, name, None)
            if torch.is_tensor(val) and val.is_cuda:
                setattr(stream, name, val.detach().cpu())
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:  # noqa: BLE001
            pass
        freed = []
        for i in range(torch.cuda.device_count()):
            try:
                free, total = torch.cuda.mem_get_info(i)
                freed.append(f"{i}:{free/1e9:.2f}/{total/1e9:.2f}GiB")
            except Exception:  # noqa: BLE001
                continue
        print(f"[loopmoe] parent CUDA released {freed}", flush=True)


class _LocalStream:
    """Rank-local FineWeb stream (same contract as harness SeededTrainStream)."""

    def __init__(self, texts, tok, device, seq_len, batch_size, seed, rank=0):
        self._texts = list(texts)
        if not self._texts:
            raise ValueError("empty train text pool")
        self._tok = tok
        self.device = device
        self.seq_len = max(8, int(seq_len))
        self.batch_size = max(1, int(batch_size))
        self.seed = int(seed) + 10007 * int(rank)
        self.tokens_seen = 0
        self._epoch = 0
        self._order = self._perm(0)
        self._pos = 0
        self._buf = []
        self._eos = getattr(tok, "eos_token_id", None)

    def _perm(self, epoch):
        import random

        order = list(range(len(self._texts)))
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def _encode(self, text):
        return self._tok(text, add_special_tokens=False)["input_ids"]

    def _fill(self):
        need = self.batch_size * (self.seq_len + 1)
        while len(self._buf) < need:
            if self._pos >= len(self._order):
                self._epoch += 1
                self._order = self._perm(self._epoch)
                self._pos = 0
            text = self._texts[self._order[self._pos]]
            self._pos += 1
            ids = self._encode(text)
            if not ids:
                continue
            self._buf.extend(ids)
            if self._eos is not None:
                self._buf.append(self._eos)

    def next_batch(self):
        self._fill()
        need = self.batch_size * (self.seq_len + 1)
        window = self._buf[:need]
        del self._buf[:need]
        ids = torch.tensor(window, dtype=torch.long).view(self.batch_size, self.seq_len + 1)
        input_ids = ids[:, :-1].contiguous().to(self.device, non_blocking=True)
        labels = ids[:, 1:].contiguous().to(self.device, non_blocking=True)
        self.tokens_seen += int(labels.numel())
        return input_ids, labels


def _enable_fast_matmul():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:  # noqa: BLE001
        pass


def _write_parent_sidecar(out_dir, reports, probe_curve):
    """Rank-0 file the harness parent ingests (`ingest_ddp_sidecar`)."""
    if not out_dir:
        return
    path = Path(out_dir) / "telemetry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "report_count": len(reports),
                "reports": reports,
                "probe_curve": probe_curve,
            }
        ),
        encoding="utf-8",
    )


def _worker_probe(model, tok, texts, device, seq_len, fallback_loss):
    """Teacher-forced CE on payload texts, else the train loss (still a curve)."""
    if tok is not None and texts:
        try:
            from prismlib.probes import teacher_forced_ce

            return float(teacher_forced_ce(model, tok, texts[:8], device, seq_len))
        except Exception:  # noqa: BLE001
            pass
    return float(fallback_loss)


def _train_loop(
    train_model,
    stream,
    *,
    device,
    max_steps,
    stop_s,
    rec,
    te_mode,
    guard,
    rank=0,
    world=1,
    zero=False,
    telemetry_bag=None,
    probe_tok=None,
    probe_texts=None,
    probe_seq_len=512,
):
    """Single backward: CE + local MoE aux. No second backward, no DP gather."""
    core = _unwrap(train_model)
    if hasattr(core, "grad_checkpoint"):
        # TE NVFP4 Linear cannot recompute under torch.utils.checkpoint
        # (saved-tensor count 94 vs 45). Aux is already in the same loss.
        core.grad_checkpoint = False
    opt = _make_adam(core, zero=zero)
    use_amp = device == "cuda"
    use_te = rec is not None
    t0 = time.time()
    step = 0
    last_loss = 0.0
    last_aux = 0.0
    grad_norm = 0.0
    tokens_this = 0
    reports, probes = [], []
    train_model.train()
    while step < max_steps and (time.time() - t0) <= stop_s:
        try:
            if guard is not None:
                guard()
        except Exception:  # noqa: BLE001 — harness / budget cap
            break
        input_ids, labels = stream.next_batch() if hasattr(stream, "next_batch") else next(stream)
        tokens_this += int(input_ids.numel())
        # TE recipe MUST wrap backward — closing autocast after forward
        # makes NVFP4 wgrad pick a cublasLt algo that SM120 rejects.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            with _fp8_ctx(use_te, rec):
                logits = train_model(input_ids)
                if hasattr(logits, "logits"):
                    logits = logits.logits
                loss = loopmoe_kernels.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
                )
                aux = getattr(core, "aux_loss", None)
                if aux is not None and torch.is_tensor(aux) and aux.requires_grad:
                    last_aux = float(aux.detach().float().item())
                    loss = loss + AUX_LOSS_COEF * aux.float()
                elif aux is not None and torch.is_tensor(aux):
                    last_aux = float(aux.detach().float().item())
                opt.zero_grad(set_to_none=True)
                loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(core.parameters(), GRAD_CLIP))
        lr = _lr_at(step, max_steps)
        for group in opt.param_groups:
            group["lr"] = lr
        opt.step()
        last_loss = float(loss.detach().float().item())
        step += 1
        if rank == 0 and step == 1 and torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                km = loopmoe_kernels.kernel_map()
                print(
                    f"[loopmoe] step1 mem_free={free/1e9:.2f}/{total/1e9:.2f}GiB "
                    f"ckpt={getattr(core, 'grad_checkpoint', None)} te_mode={te_mode} "
                    f"delta_kernel={km.get('delta_kernel')} attn_kernel={km.get('attn_kernel')} "
                    f"ce_kernel={km.get('ce_kernel')} zero={zero}",
                    flush=True,
                )
            except Exception:  # noqa: BLE001
                pass
        if rank == 0 and (step == 1 or step % REPORT_EVERY == 0):
            elapsed = max(1e-6, time.time() - t0)
            tps_local = tokens_this / elapsed
            tps_global = tps_local * world
            km = loopmoe_kernels.kernel_map()
            print(
                f"[loopmoe] step={step} loss={last_loss:.4f} aux={last_aux:.4f} "
                f"tok/s_local={tps_local:.1f} tok/s_global={tps_global:.1f} "
                f"world={world} te_mode={te_mode} rank={rank} "
                f"delta_kernel={km.get('delta_kernel')} attn_kernel={km.get('attn_kernel')}",
                flush=True,
            )
            prism_telemetry.report(loss=last_loss, step=step, grad_norm=grad_norm)
            tokens_seen = int(getattr(stream, "tokens_seen", 0) or tokens_this)
            rec_pt = {
                "step": step,
                "loss": last_loss,
                "grad_norm": grad_norm,
                "tokens_seen": tokens_seen,
                "at_secs": round(time.time() - t0, 3),
            }
            reports.append(rec_pt)
            probe_loss = _worker_probe(
                _unwrap(train_model),
                probe_tok or getattr(stream, "_tok", None),
                probe_texts or list(getattr(stream, "_texts", []) or [])[:8],
                device,
                probe_seq_len or int(getattr(stream, "seq_len", 512) or 512),
                last_loss,
            )
            probes.append(
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "bytes_seen": max(1, tokens_seen * 4),
                    "flops_spent": 0.0,
                    "wall_s": rec_pt["at_secs"],
                    "probe_loss": probe_loss,
                }
            )
    if telemetry_bag is not None:
        telemetry_bag["reports"] = reports
        telemetry_bag["probe_curve"] = probes
    elapsed = time.time() - t0
    tps_local = tokens_this / max(1e-6, elapsed)
    return {
        "train_loss": last_loss,
        "train_steps": step,
        "train_seconds": elapsed,
        "moe_aux_loss": last_aux,
        "tokens_local": tokens_this,
        "tokens_per_sec_local": tps_local,
        "tokens_per_sec": tps_local * world,
        "final_lr": _lr_at(max(step - 1, 0), max_steps),
        "peak_lr": PEAK_LR,
    }


def ddp_worker_main(payload_path=None, rank=None, world=None, port=None):
    """One process per GPU. Called from ddp_worker.py via mp.spawn."""
    payload_path = payload_path or os.environ.get("LOOPMOE_PAYLOAD")
    rank = int(os.environ["RANK"] if rank is None else rank)
    world = int(os.environ["WORLD_SIZE"] if world is None else world)
    port = int(os.environ["MASTER_PORT"] if port is None else port)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    _set_dist_env(port)
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    backend = "nccl"
    # Explicit IPv4 store — never resolve hostname localhost (AF_INET6 errno 97).
    store_kwargs = dict(
        host_name="127.0.0.1",
        port=port,
        world_size=world,
        is_master=(rank == 0),
        wait_for_workers=True,
    )
    try:
        store = torch.distributed.TCPStore(**store_kwargs, use_libuv=False)
    except TypeError:
        store = torch.distributed.TCPStore("127.0.0.1", port, world, rank == 0, True)
    torch.distributed.init_process_group(
        backend=backend,
        store=store,
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=15),
    )
    print(
        f"[loopmoe] ddp init rank={rank}/{world} local_rank={local_rank} "
        f"backend={backend} master=127.0.0.1:{port} "
        f"device={torch.cuda.get_device_name(local_rank)} "
        f"sm={torch.cuda.get_device_capability(local_rank)} "
        f"nccl={getattr(torch.cuda.nccl, 'version', lambda: '?')()}",
        flush=True,
    )
    print(f"[loopmoe] rank={rank} loading payload", flush=True)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    ctx = dict(payload["ctx"])
    ctx["device"] = device
    ctx["te_available"] = True
    texts = list(payload.get("texts") or [])
    texts_path = payload.get("texts_path")
    if not texts and texts_path:
        texts = [json.loads(line) for line in open(texts_path, encoding="utf-8") if line.strip()]
    if not texts:
        raise RuntimeError("DDP worker missing train texts")
    print(f"[loopmoe] rank={rank} texts={len(texts)} building model", flush=True)
    model = build_loopmoe(ctx)
    # TE Linear writes `_extra_state` during the parent FLOPs probe; a
    # freshly constructed worker module does not declare those keys yet.
    model.load_state_dict(payload["state_dict"], strict=False)
    model = model.to(device)
    micro = int(payload["micro_batch"])
    seq_len = int(payload["seq_len"])
    rec, te_mode = _maybe_te_recipe()
    parallel = str(payload.get("parallel") or os.environ.get("LOOPMOE_PARALLEL", "ddp")).strip().lower()
    if parallel not in {"ddp", "zero1", "fsdp"}:
        parallel = "ddp"
    print(
        f"[loopmoe] worker te_version={_te_version()} te_mode={te_mode} "
        f"use_te_linear={getattr(model, 'use_te', None)} parallel={parallel}",
        flush=True,
    )
    loopmoe_kernels.enable_attn_backends()
    loopmoe_kernels.log_kernel_banner()
    use_zero = parallel == "zero1"
    if parallel == "fsdp":
        train_wrap, parallel_used = _wrap_fsdp(model, local_rank)
    else:
        train_wrap = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        parallel_used = "zero1" if use_zero else "ddp"
    compiled, did_compile = _maybe_compile(train_wrap)
    _enable_fast_matmul()
    stream = _LocalStream(
        texts,
        payload["tokenizer"],
        device,
        seq_len=seq_len,
        batch_size=micro,
        seed=int(payload["seed"]),
        rank=rank,
    )
    cap_s = float(payload["cap_s"])
    stop_s = max(30.0, cap_s - float(payload.get("wall_margin_s", WALL_MARGIN_S)))
    t_limit = time.time() + stop_s

    def guard():
        if time.time() >= t_limit:
            raise RuntimeError("wall")

    telemetry_bag = {"reports": [], "probe_curve": []}
    metrics = _train_loop(
        compiled,
        stream,
        device="cuda",
        max_steps=int(payload["max_steps"]),
        stop_s=stop_s,
        rec=rec,
        te_mode=te_mode,
        guard=guard,
        rank=rank,
        world=world,
        zero=use_zero,
        telemetry_bag=telemetry_bag if rank == 0 else None,
        probe_tok=payload.get("tokenizer"),
        probe_texts=texts[:8],
        probe_seq_len=seq_len,
    )
    km = loopmoe_kernels.kernel_map()
    metrics.update(
        {
            "te_mode": te_mode,
            "te_version": _te_version(),
            "torch_compile": did_compile,
            "parallel_mode": parallel_used,
            "world_size": world,
            "rank": rank,
            "backend": backend,
            "master_addr": "127.0.0.1",
            "micro_batch": micro,
            "seq_len": seq_len,
            "gpu_count": world,
            "te_available": True,
            **km,
        }
    )
    tokens_t = torch.tensor([float(metrics["tokens_local"])], device=device)
    torch.distributed.all_reduce(tokens_t, op=torch.distributed.ReduceOp.SUM)
    metrics["tokens_seen"] = int(tokens_t.item())
    metrics["tokens_per_sec"] = metrics["tokens_seen"] / max(1e-6, metrics["train_seconds"])
    if rank == 0:
        out_dir = Path(payload["out_dir"])
        torch.save({k: v.detach().cpu() for k, v in _unwrap(compiled).state_dict().items()}, out_dir / WEIGHTS_NAME)
        (out_dir / METRICS_NAME).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_parent_sidecar(
            out_dir,
            telemetry_bag.get("reports") or [],
            telemetry_bag.get("probe_curve") or [],
        )
        print(
            f"[loopmoe] train done steps={metrics['train_steps']} "
            f"seconds={metrics['train_seconds']:.1f} loss={metrics['train_loss']:.4f} "
            f"tokens={metrics['tokens_seen']} tok/s={metrics['tokens_per_sec']:.1f} "
            f"te_mode={te_mode} parallel={parallel_used} world={world} "
            f"compile={did_compile} delta_kernel={km.get('delta_kernel')}",
            flush=True,
        )
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    return metrics


def _wrap_fsdp(model, local_rank):
    """FSDP2 fully_shard when available; else FSDP1. TE Linear extra-state is sticky."""
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for child in list(model.children()):
            try:
                fully_shard(child, mp_policy=mp)
            except Exception:  # noqa: BLE001
                continue
        fully_shard(model, mp_policy=mp)
        print(f"[loopmoe] FSDP2 fully_shard rank={local_rank}", flush=True)
        return model, "fsdp2"
    except Exception as exc:  # noqa: BLE001
        print(f"[loopmoe] FSDP2 unavailable ({exc}); trying FSDP1", flush=True)
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        )
        wrapped = FSDP(
            model,
            mixed_precision=mp,
            use_orig_params=True,
            device_id=local_rank,
        )
        print(f"[loopmoe] FSDP1 wrap rank={local_rank}", flush=True)
        return wrapped, "fsdp1"
    except Exception as exc:  # noqa: BLE001
        print(f"[loopmoe] FSDP failed ({exc}); falling back to DDP", flush=True)
        ddp = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        return ddp, "ddp"


def _launch_ddp(model, ctx, gpu_count):
    workdir = Path(ctx.get("workdir") or os.environ.get("PRISM_WORKDIR") or "/tmp")
    out_dir = workdir / "loopmoe_ddp"
    out_dir.mkdir(parents=True, exist_ok=True)
    stream = ctx.get("train_stream")
    if stream is None:
        raise RuntimeError("train_stream required for DDP LoopMoE")
    texts = list(getattr(stream, "_texts", []) or [])
    tok = ctx.get("tokenizer") or getattr(stream, "_tok", None)
    if not texts or tok is None:
        raise RuntimeError("DDP payload needs stream texts + tokenizer")
    # Free parent CUDA so workers own the devices (probe left ~30GiB on GPU 0).
    cpu_sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    _release_parent_cuda(model, stream)
    seq_len = int(ctx.get("seq_len") or getattr(stream, "seq_len", 512) or 512)
    harness_bs = int(ctx.get("batch_size") or getattr(stream, "batch_size", 8) or 8)
    env_micro = os.environ.get("LOOPMOE_MICRO_BATCH", "").strip()
    # Do not inherit harness batch_size (that was DP-sharded). LoopMoE
    # activations at seq=512 need a small per-GPU microbatch.
    micro = int(env_micro) if env_micro.isdigit() else DEFAULT_MICRO_BATCH
    _ = harness_bs  # kept for payload logs / MFU context
    cap_s = float(ctx.get("train_hours_cap", 1.0)) * 3600.0
    texts_path = out_dir / "train_texts.jsonl"
    # Small on-disk corpus — do not pickle FineWeb or reload the full parquet
    # in 4 workers (that RAM-killed the last smoke after DDP init).
    with open(texts_path, "w", encoding="utf-8") as fh:
        for text in texts[:4096]:
            fh.write(json.dumps(text, ensure_ascii=False) + "\n")
    payload = {
        "state_dict": cpu_sd,
        "texts": [],
        "texts_path": str(texts_path),
        "tokenizer": tok,
        "ctx": {
            "seed": int(ctx.get("seed", 0)),
            "vocab_size": int(ctx.get("vocab_size") or 50257),
            "te_available": True,
            "arch": ctx.get("arch"),
            "prism_width_multiplier": ctx.get("prism_width_multiplier", 1.0),
        },
        "seed": int(ctx.get("seed", 0)),
        "seq_len": seq_len,
        "micro_batch": micro,
        "max_steps": int(ctx.get("max_train_steps", 20000)),
        "cap_s": cap_s,
        "wall_margin_s": WALL_MARGIN_S,
        "out_dir": str(out_dir),
        "parallel": os.environ.get("LOOPMOE_PARALLEL", "ddp").strip().lower(),
    }
    payload_path = out_dir / PAYLOAD_NAME
    torch.save(payload, payload_path)
    port = _rendezvous_port()
    _set_dist_env(port)
    print(
        f"[loopmoe] launching dist spawn world={gpu_count} master=127.0.0.1:{port} "
        f"micro_batch={micro} seq={seq_len} parallel={payload['parallel']}",
        flush=True,
    )
    from nemo_automodel.components.models.loopmoe.ddp_worker import spawn_workers

    spawn_workers(gpu_count, port, str(payload_path))
    metrics_path = out_dir / METRICS_NAME
    weights_path = out_dir / WEIGHTS_NAME
    if not metrics_path.is_file() or not weights_path.is_file():
        raise RuntimeError("DDP workers did not write metrics/weights")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trained = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(trained, strict=False)
    device = ctx.get("device") or "cuda"
    if device != "cpu":
        model.to(device)
    # Authoritative harness counter + FLOPs spend so MFU is real.
    tokens = int(metrics.get("tokens_seen") or 0)
    if hasattr(stream, "tokens_seen"):
        stream.tokens_seen = int(getattr(stream, "tokens_seen", 0)) + tokens
        if getattr(stream, "flops_per_token", 0.0):
            stream.flops_spent = float(stream.flops_per_token) * float(stream.tokens_seen)
        stream.batches_yielded = int(getattr(stream, "batches_yielded", 0)) + int(
            metrics.get("train_steps") or 0
        )
    return metrics


def train(model, ctx):
    device = ctx.get("device") or "cuda"
    seed = int(ctx.get("seed", 0))
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _enable_fast_matmul()
    guard = ctx.get("guard")
    gpu_count = int(ctx.get("gpu_count") or 0)
    if gpu_count <= 0 and torch.cuda.is_available():
        gpu_count = int(torch.cuda.device_count())
    te_available = bool(ctx.get("te_available", False))
    if not te_available:
        try:
            te_available = __import__("importlib").util.find_spec("transformer_engine") is not None
        except Exception:  # noqa: BLE001
            te_available = False

    rec, te_mode = _maybe_te_recipe() if te_available else (None, "none")
    print(
        f"[loopmoe] train start gpu_count={gpu_count} te_available={te_available} "
        f"te_mode={te_mode} te_version={_te_version()} "
        f"cuda_devices={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
        f"use_te_linear={getattr(model, 'use_te', None)}",
        flush=True,
    )

    # Marketplace often only lists 8×5090 hosts (no GPU splitting). Cap at 4
    # so the proof matches the 4-GPU contract and leaves headroom on GPU 0.
    max_gpus = int(os.environ.get("LOOPMOE_MAX_GPUS", "4") or 4)
    if gpu_count > max_gpus:
        print(f"[loopmoe] capping visible GPUs {gpu_count} -> {max_gpus}", flush=True)
        gpu_count = max_gpus
    if gpu_count > 1 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
        metrics = _launch_ddp(model, ctx, gpu_count)
        fpt = float(ctx.get("flops_per_token_probe") or 0.0)
        elapsed = float(metrics.get("train_seconds") or 1.0)
        tokens = float(metrics.get("tokens_seen") or 0.0)
        n_params = float(sum(p.numel() for p in model.parameters()))
        loop_f = float(getattr(model, "prism_loop_factor", 1.0) or 1.0)
        fpt_analytic = 6.0 * n_params * loop_f
        fpt_source = "probe"
        if fpt <= 0.0:
            fpt = fpt_analytic
            fpt_source = "analytic_6n_loops"
        mfu = (tokens * fpt) / (PEAK_FLOPS_PER_GPU * gpu_count * elapsed) if fpt > 0 else 0.0
        metrics["mfu_est"] = mfu
        metrics["flops_per_token_probe"] = fpt
        metrics["flops_per_token_analytic"] = fpt_analytic
        metrics["flops_per_token_source"] = fpt_source
        print(
            f"[loopmoe] ddp parent metrics world={metrics.get('world_size')} "
            f"te_mode={metrics.get('te_mode')} tok/s={metrics.get('tokens_per_sec')} "
            f"mfu_est={mfu*100:.2f}% fpt_src={fpt_source} compile={metrics.get('torch_compile')}",
            flush=True,
        )
        prism_telemetry.finish_evaluation()
        return metrics

    stream = ctx.get("train_stream")
    if stream is None:
        raise RuntimeError("train_stream required for live AutoModel LoopMoE")
    compiled, did_compile = _maybe_compile(model)
    max_steps = int(ctx.get("max_train_steps", 20000))
    cap_s = float(ctx.get("train_hours_cap", 1.0)) * 3600.0
    stop_s = max(60.0, cap_s - WALL_MARGIN_S)
    metrics = _train_loop(
        compiled,
        stream,
        device=device,
        max_steps=max_steps,
        stop_s=stop_s,
        rec=rec,
        te_mode=te_mode,
        guard=guard,
        rank=0,
        world=1,
    )
    metrics.update(
        {
            "te_mode": te_mode,
            "te_version": _te_version(),
            "te_available": te_available,
            "torch_compile": did_compile,
            "parallel_mode": "single",
            "world_size": 1,
            "backend": "none",
            "gpu_count": gpu_count,
            "tokens_seen": int(getattr(stream, "tokens_seen", 0)) or int(metrics["tokens_local"]),
        }
    )
    print(
        f"[loopmoe] train done steps={metrics['train_steps']} "
        f"seconds={metrics['train_seconds']:.1f} te_mode={te_mode} parallel=single",
        flush=True,
    )
    prism_telemetry.finish_evaluation()
    return metrics
