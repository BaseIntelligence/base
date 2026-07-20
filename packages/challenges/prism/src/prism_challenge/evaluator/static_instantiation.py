"""Bounded forced-seed ``build_model`` instantiation gate (architecture.md section 4.1).

The static phase instantiates the miner's ``build_model(ctx)`` under the challenge-forced seed
BEFORE any GPU work so that contract/instantiation problems are caught cleanly and hostile
construction is time/resource-bounded. The factory runs in a short-lived child process with a
wall-clock budget and an address-space cap, so an infinite loop or a memory balloon at
construction is killed there instead of stalling the worker. The AST sandbox has already vetted
every script, so the child loads the architecture entrypoint with normal builtins to mirror the
GPU runner and avoid false rejections of legitimate PyTorch code.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import resource
import sys
import tempfile
from collections.abc import Mapping
from typing import Any

from .interface import PrismContext
from .sandbox import SandboxViolation, _synthetic_evidence

RETURN_TYPE_RULE = "prism:build-model-return-type"
INSTANTIATION_RULE = "prism:build-model-instantiation"
RESOURCE_RULE = "prism:build-model-resource"
PARAM_CAP_RULE = "prism:param-cap"

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MEMORY_HEADROOM_BYTES = 8 * 1024**3


def _build_context() -> Any:
    """Prefer ``forkserver`` so the bounded child is forked from a clean, single-threaded server.

    The worker is multi-threaded (event loop + thread pool); forking directly from it risks
    inheriting a held lock and deadlocking the child. ``forkserver`` (with torch preloaded for
    speed) sidesteps that and also keeps the child from inheriting the worker's open file handles.
    """
    try:
        context = mp.get_context("forkserver")
        context.set_forkserver_preload(["torch"])
        return context
    except (ValueError, OSError):  # pragma: no cover - platform without forkserver
        return mp.get_context("fork")


_MP_CONTEXT = _build_context()

_MEMORY_MARKERS = (
    "memoryerror",
    "out of memory",
    "outofmemory",
    "defaultcpuallocator",
    "cannot allocate",
    "alloc_cpu",
    "bad_alloc",
)


def check_build_model_static(
    files: Mapping[str, str],
    entrypoint: str,
    *,
    ctx: PrismContext,
    build_model_symbol: str = "build_model",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    memory_headroom_bytes: int = DEFAULT_MEMORY_HEADROOM_BYTES,
    max_parameters: int | None = None,
) -> int:
    """Instantiate ``build_model`` under the forced seed in a bounded child process.

    Returns the model's parameter count on success. Raises :class:`SandboxViolation` when the
    factory returns a non-``nn.Module``, raises during construction, exceeds the time/resource
    budget, or yields more than the parameter cap. The cap defaults to ``ctx.max_parameters``
    (dual ladder: explore 124M / promote 350M) and may be overridden via ``max_parameters``; it is
    architecture-agnostic because it counts the actual instantiated tensors regardless of the
    model family. The whole check happens before any GPU lease/job is created.
    """
    cap = ctx.max_parameters if max_parameters is None else max_parameters
    result: tuple[str, object] | None = None
    # The PARENT owns the scratch workdir so it is always removed even when the child is
    # terminated/killed (timeout / over-budget) before its own cleanup could run; the child only
    # writes the miner files into it. This is what prevents a leaked /tmp dir per static check.
    with tempfile.TemporaryDirectory(
        prefix="prism-static-instantiate-", ignore_cleanup_errors=True
    ) as workdir:
        parent_conn, child_conn = _MP_CONTEXT.Pipe(duplex=False)
        proc = _MP_CONTEXT.Process(
            target=_child_instantiate,
            args=(
                dict(files),
                entrypoint,
                build_model_symbol,
                ctx,
                int(memory_headroom_bytes),
                workdir,
                child_conn,
            ),
        )
        proc.start()
        child_conn.close()
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
        raise SandboxViolation(
            "build_model construction exceeded the static time/resource budget "
            f"({timeout_seconds:g}s) and was terminated",
            _evidence(RESOURCE_RULE, entrypoint, "build_model construction was time-bounded"),
        )

    kind, detail = result
    if kind == "ok":
        param_count = int(detail)  # type: ignore[call-overload]
        if cap is not None and param_count > cap:
            raise SandboxViolation(
                f"model has {param_count:,} parameters, exceeding the {cap:,} parameter cap",
                _evidence(
                    PARAM_CAP_RULE,
                    entrypoint,
                    f"build_model instantiated {param_count} parameters over the {cap} cap",
                ),
            )
        return param_count
    if kind == "not_module":
        raise SandboxViolation(
            f"build_model must return a torch.nn.Module, got {detail}",
            _evidence(RETURN_TYPE_RULE, entrypoint, "build_model returned a non-nn.Module value"),
        )
    text = str(detail)
    if any(marker in text.lower() for marker in _MEMORY_MARKERS):
        raise SandboxViolation(
            f"build_model exceeded the static memory budget during construction: {text}",
            _evidence(RESOURCE_RULE, entrypoint, "build_model construction was memory-bounded"),
        )
    raise SandboxViolation(
        f"build_model raised during forced-seed instantiation: {text}",
        _evidence(INSTANTIATION_RULE, entrypoint, "build_model raised during instantiation"),
    )


def _evidence(rule_id: str, entrypoint: str, explanation: str):
    return _synthetic_evidence(
        rule_id=rule_id,
        artifact_path=entrypoint,
        ast_node="Call",
        basis=f"{entrypoint}:build_model",
        explanation=explanation,
    )


def _child_instantiate(
    files: Mapping[str, str],
    entrypoint: str,
    symbol: str,
    ctx: PrismContext,
    memory_headroom_bytes: int,
    workdir: str,
    conn,
) -> None:
    try:
        import torch

        # Single-threaded keeps the bounded child lightweight and the forced-seed run
        # deterministic; the address-space cap below is what hard-bounds a memory balloon.
        torch.set_num_threads(1)

        if memory_headroom_bytes > 0:
            try:
                cap = _vmsize_bytes() + memory_headroom_bytes
                resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            except (ValueError, OSError):
                pass

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

        torch.manual_seed(int(ctx.seed))

        spec = importlib.util.spec_from_file_location("prism_static_arch", entry_path)
        if spec is None or spec.loader is None:
            conn.send(("error", f"cannot load architecture entrypoint {entrypoint}"))
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules["prism_static_arch"] = module
        spec.loader.exec_module(module)

        factory = getattr(module, symbol, None)
        if factory is None or not callable(factory):
            conn.send(("error", f"architecture is missing callable {symbol}"))
            return

        model = factory(ctx)
        if not isinstance(model, torch.nn.Module):
            conn.send(("not_module", type(model).__name__))
            return
        param_count = _realized_param_count(model, torch, ctx)
        conn.send(("ok", param_count))
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


def _realized_param_count(model: Any, torch: Any, ctx: PrismContext) -> int:
    """Count the REALIZED parameters, materializing lazy/dynamic params first.

    ``LazyLinear``, params created in the first ``forward``, and config-driven widths are not yet
    allocated after construction (and ``numel()`` even raises on a lazy/uninitialized parameter),
    so a naive count would let an over-cap model hide behind deferred construction. A best-effort
    dummy token-id forward under the already-applied forced seed materializes them; the realized
    total is then counted so the cap binds the model that would actually train. The forward is
    best-effort (a model that cannot run the bare token-id contract is counted as-constructed);
    a parameter that is still uninitialized afterwards surfaces as a clean instantiation error.
    """
    try:
        _run_materialization_forward(model, torch, ctx)
    except Exception:
        pass
    return int(sum(int(param.numel()) for param in model.parameters()))


def _first_param_device(model: Any, torch: Any) -> Any:
    for param in model.parameters():
        try:
            return param.device
        except (ValueError, RuntimeError):
            continue
    return torch.device("cpu")


def _run_materialization_forward(model: Any, torch: Any, ctx: PrismContext) -> None:
    """Run a dummy token-id forward so lazy/dynamic params take their realized shape.

    Parameter counts do not depend on the sequence length, so a bounded dummy sequence is enough;
    ``eval()`` + ``no_grad`` keeps it side-effect free and the forced seed (already applied by the
    caller) keeps any deferred initialization deterministic.
    """
    seq_len = min(max(int(getattr(ctx, "sequence_length", 16) or 16), 2), 1024)
    device = _first_param_device(model, torch)
    tokens = torch.zeros((1, seq_len), dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(tokens)
    finally:
        if was_training:
            model.train()


def _vmsize_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return 0
    return 0
