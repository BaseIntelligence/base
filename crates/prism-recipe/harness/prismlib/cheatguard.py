"""Harness-side anti-cheat battery (Prism v3, E5).

Hook interface called by main.py (missing module / raising hook is a silent
no-op there — every hook here is additionally self-guarding):

    pre_train(ctx)                  snapshot threads/streams + static AST audit
    pre_eval(ctx)                   thread diff, secret-seed read+unset, JIT reset
    post_eval(report) -> report     append results under report["cheatguard"]

The helper functions (deep_clone, guard buffers, pointer poisoning,
determinism double-run, timing-ratio check, jit_cache_reset) are pure and
importable by the eval-side modules (prismlib.eval_v3 / train_v3) so the
correctness gate of spike 08 §7.4 can run inside the eval subprocess; the
parent-side hooks only record what the parent process can observe.

Banned-pattern rules are shared with the Rust intake scanner
(crates/prism-recipe/src/zip_submit.rs) via
harness/cheatguard_patterns.json (uploaded to the pod beside main.py); a
builtin fallback copy keeps the audit functional if the file is absent.

Pure stdlib + optional torch. Never raise.
"""

import ast
import copy
import hashlib
import json
import os
import re
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_PATTERNS_PATH = os.path.join(os.path.dirname(_HERE), "cheatguard_patterns.json")

# Keep in sync with harness/cheatguard_patterns.json (fallback when the
# uploaded JSON is absent, e.g. older HARNESS_FILES manifest).
_FALLBACK_PATTERNS = {
    "banned_extensions": [".so", ".pyd", ".cubin", ".dll", ".pyc", ".pyo"],
    "banned_substrings": ["ctypes", "cuModuleLoadData"],
    "banned_call_strings": ["torch.utils.cpp_extension.load"],
    "banned_call_allowed_suffixes": ["_inline"],
    "base64_decode_marker": "base64.b64decode(",
    "base64_blob_min_len": 256,
    "ast_banned_import_roots": ["ctypes"],
    "ast_forbidden_print_markers": ["METRICS_JSON", "EVAL_OK", "EVAL_FAIL"],
}

_STATE = {
    "threads_before": None,
    "secret_seed_hex": None,
    "combined_seed": None,
    "ast_findings": None,
}


def _patterns():
    try:
        with open(_PATTERNS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            merged = dict(_FALLBACK_PATTERNS)
            merged.update({k: v for k, v in data.items() if isinstance(v, type(_FALLBACK_PATTERNS.get(k)))})
            return merged
    except Exception:  # noqa: BLE001
        pass
    return dict(_FALLBACK_PATTERNS)


# ---------------------------------------------------------------- seeds


def cantor(a, b):
    """Cantor pairing of two non-negative ints."""
    a, b = int(a), int(b)
    s = a + b
    return (s * (s + 1)) // 2 + b


def combine_with_recipe(secret_hex, public_seed=None):
    """Fold the 128-bit secret seed into the public recipe seed via Cantor
    pairing over 64-bit chunks (spike 08 §7.4: fresh secret-seeded data,
    Cantor-combined with public seeds)."""
    if public_seed is None:
        try:
            public_seed = int(os.environ.get("PRISM_EVAL_PUBLIC_SEED", "7"))
        except ValueError:
            public_seed = 7
    seed = int(public_seed) & ((1 << 63) - 1)
    raw = bytes.fromhex(re.sub(r"[^0-9a-fA-F]", "", str(secret_hex)) or "0")
    for off in range(0, len(raw), 8):
        chunk = int.from_bytes(raw[off : off + 8], "big")
        seed = cantor(seed, chunk) & ((1 << 63) - 1)
    return seed


def read_secret_seed(ctx=None):
    """Locate the per-run secret seed without exposing it.

    Order: (1) `eval-assets/SECRET_SEED` file staged post-train by the Lium
    client (E5 protocol — file delivery so the train-phase child never sees
    it), (2) `PRISM_EVAL_SECRET_SEED` env (read once and unset immediately,
    per spike 08 §7.4). Returns hex string or None; never logs the value.
    """
    if _STATE["secret_seed_hex"]:
        return _STATE["secret_seed_hex"]
    seed = None
    assets_dir = (ctx or {}).get("eval_assets_dir") or os.environ.get("PRISM_EVAL_ASSETS_DIR", "").strip()
    if assets_dir:
        try:
            with open(os.path.join(assets_dir, "SECRET_SEED"), "r", encoding="utf-8") as f:
                seed = f.read().strip() or None
        except OSError:
            seed = None
    if seed is None and "PRISM_EVAL_SECRET_SEED" in os.environ:
        seed = os.environ.pop("PRISM_EVAL_SECRET_SEED", "").strip() or None
    if seed and re.fullmatch(r"[0-9a-fA-F]{16,64}", seed):
        _STATE["secret_seed_hex"] = seed
        return seed
    return None


# ------------------------------------------------------- tensor helpers


def deep_clone(obj):
    """Deep-clone eval inputs: torch tensors via .clone(), everything else
    via copy.deepcopy (spike 08 §7.4: miner forward must not alias inputs)."""
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.clone()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(obj, (list, tuple)):
        return type(obj)(deep_clone(x) for x in obj)
    if isinstance(obj, dict):
        return {k: deep_clone(v) for k, v in obj.items()}
    return copy.deepcopy(obj)


def poison_tensor_(t, sentinel=12345.6789):
    """Pointer-poisoning: pre-fill a tensor with the sentinel pattern before
    miner code runs (detect lazy/deferred writes and aliasing)."""
    t.fill_(sentinel)
    return t


def verify_poison(t, sentinel=12345.6789, atol=1e-6):
    """Post-forward poison check; returns a report dict."""
    try:
        import torch

        remaining = int((t - sentinel).abs().le(atol).sum().item())
        return {
            "poison_sentinel_remaining": remaining,
            "poison_ok": remaining == 0,
            "is_tensor": type(t) is torch.Tensor,
        }
    except Exception as exc:  # noqa: BLE001
        return {"poison_ok": None, "error": str(exc)[:200]}


def finite_check(t, name="tensor"):
    """NaN/Inf guard report for an output tensor (or nested structure)."""
    try:
        import math

        try:
            import torch

            if isinstance(t, torch.Tensor):
                arr = t.detach()
                n_nan = int(torch.isnan(arr).sum().item())
                n_inf = int(torch.isinf(arr).sum().item())
                return {"name": name, "nan": n_nan, "inf": n_inf, "finite_ok": n_nan == 0 and n_inf == 0}
        except ImportError:
            pass
        if isinstance(t, (list, tuple)):
            checks = [finite_check(x, f"{name}[{i}]") for i, x in enumerate(t)]
            return {
                "name": name,
                "nan": sum(c["nan"] for c in checks),
                "inf": sum(c["inf"] for c in checks),
                "finite_ok": all(c["finite_ok"] for c in checks),
            }
        if isinstance(t, float) and (math.isnan(t) or math.isinf(t)):
            return {
                "name": name,
                "nan": int(math.isnan(t)),
                "inf": int(math.isinf(t)),
                "finite_ok": False,
            }
        return {"name": name, "nan": 0, "inf": 0, "finite_ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"name": name, "nan": -1, "inf": -1, "finite_ok": None, "error": str(exc)[:200]}


def determinism_double_run(fn, *args, tol=0.0, **kwargs):
    """Re-run one probe and compare outputs (bitwise when tol=0); returns a
    report dict with `determinism_ok` (spike 08 §7.4 correctness gate)."""
    try:
        first = fn(*args, **kwargs)
        second = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return {"determinism_ok": None, "error": str(exc)[:200]}
    try:
        import torch

        if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
            diff = float((first - second).abs().max().item()) if first.numel() else 0.0
            same = bool(torch.equal(first, second)) if tol == 0.0 else diff <= tol
            return {"determinism_ok": same, "max_abs_diff": diff}
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        return {"determinism_ok": None, "error": str(exc)[:200]}
    try:
        return {"determinism_ok": bool(first == second), "max_abs_diff": None}
    except Exception as exc:  # noqa: BLE001
        return {"determinism_ok": None, "error": str(exc)[:200]}


def timing_ratio_check(event_seconds, sync_seconds, threshold=1.5):
    """CUDA-event vs full-sync timing ratio; >threshold flags hidden async
    work escaping the timed window (spike 08 §7.4 hybrid timing)."""
    try:
        if not event_seconds or event_seconds <= 0:
            return {"timing_ratio": None, "timing_flag": False}
        ratio = float(sync_seconds) / float(event_seconds)
        return {"timing_ratio": ratio, "timing_flag": ratio > float(threshold)}
    except Exception as exc:  # noqa: BLE001
        return {"timing_ratio": None, "timing_flag": None, "error": str(exc)[:200]}


def jit_cache_reset():
    """Drop torch/JIT caches between phases (best-effort)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            import torch._dynamo

            torch._dynamo.reset()
        except Exception:  # noqa: BLE001
            pass
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------- thread audit


def snapshot_threads():
    """Spawn-time snapshot: thread names + count (timers included)."""
    try:
        return sorted(t.name for t in threading.enumerate())
    except Exception:  # noqa: BLE001
        return []


def diff_threads(before):
    """Post-train thread diff vs the spawn-time snapshot."""
    after = snapshot_threads()
    before = before or []
    return sorted(set(after) - set(before))


# --------------------------------------------------------- AST audit


class _AuditVisitor(ast.NodeVisitor):
    def __init__(self, patterns, rel):
        self.patterns = patterns
        self.rel = rel
        self.findings = []

    def _flag(self, rule, node, detail):
        self.findings.append(
            {"file": self.rel, "rule": rule, "line": getattr(node, "lineno", 0), "detail": detail[:200]}
        )

    def visit_Import(self, node):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in self.patterns["ast_banned_import_roots"]:
                self._flag("banned_import", node, f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        root = (node.module or "").split(".")[0]
        if root in self.patterns["ast_banned_import_roots"]:
            self._flag("banned_import", node, f"from {node.module} import ...")
        self.generic_visit(node)

    def visit_Call(self, node):
        # eval_short_circuit pattern: printing harness result markers.
        func = node.func
        is_print = isinstance(func, ast.Name) and func.id == "print"
        if is_print:
            for arg in node.args:
                text = None
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    text = arg.value
                elif isinstance(arg, ast.JoinedStr):
                    text = "".join(
                        v.value for v in arg.values if isinstance(v, ast.Constant) and isinstance(v.value, str)
                    )
                for marker in self.patterns["ast_forbidden_print_markers"]:
                    if text and marker in text:
                        self._flag("forbidden_print_marker", node, f"print(...{marker}...)")
        self.generic_visit(node)


def _scan_text(rel, text, patterns, findings):
    """Raw-text rules shared with zip_submit.rs scan_source semantics."""
    for sub in patterns["banned_substrings"]:
        if sub and sub in text:
            findings.append({"file": rel, "rule": "banned_substring", "line": 0, "detail": sub})
    for call in patterns["banned_call_strings"]:
        if not call or call not in text:
            continue
        for m in re.finditer(re.escape(call), text):
            tail = text[m.end() : m.end() + 16]
            if any(tail.startswith(sfx) for sfx in patterns["banned_call_allowed_suffixes"]):
                continue
            findings.append({"file": rel, "rule": "banned_call", "line": 0, "detail": call})
            break
    marker = patterns["base64_decode_marker"]
    min_len = int(patterns["base64_blob_min_len"])
    if marker and marker in text:
        for m in re.finditer(r"[A-Za-z0-9+/=]{%d,}" % max(min_len, 8), text):
            findings.append(
                {"file": rel, "rule": "base64_blob", "line": 0, "detail": f"{marker} near {len(m.group(0))}-char blob"}
            )
            break


def ast_audit_sources(paths):
    """Static audit of miner sources pre-exec (spike 08 §7.4 step 1). Best
    effort: unreadable/unparseable files are flagged, never fatal."""
    patterns = _patterns()
    findings = []
    files = []
    for p in paths:
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = [d for d in dirs if d not in ("prismlib", "eval", "__pycache__", ".git")]
                files.extend(os.path.join(root, n) for n in names)
    for path in sorted(set(files)):
        rel = os.path.basename(path)
        if os.path.splitext(path)[1].lower() in patterns["banned_extensions"]:
            findings.append({"file": rel, "rule": "banned_extension", "line": 0, "detail": path[-120:]})
            continue
        if not path.endswith(".py"):
            continue
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError as exc:
            findings.append({"file": rel, "rule": "unreadable", "line": 0, "detail": str(exc)[:120]})
            continue
        _scan_text(rel, text, patterns, findings)
        try:
            tree = ast.parse(text, filename=rel)
        except SyntaxError as exc:
            findings.append({"file": rel, "rule": "syntax_error", "line": exc.lineno or 0, "detail": str(exc)[:120]})
            continue
        visitor = _AuditVisitor(patterns, rel)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    return findings


# ------------------------------------------------------------- hooks


def pre_train(ctx):
    """Spawn-time snapshot + static AST audit of the miner sources."""
    try:
        _STATE["threads_before"] = snapshot_threads()
        paths = []
        for key in ("arch_path", "train_path"):
            p = (ctx or {}).get(key)
            if p and os.path.isfile(p):
                paths.append(p)
        workdir = (ctx or {}).get("workdir")
        if workdir and os.path.isdir(workdir):
            paths.append(workdir)
        _STATE["ast_findings"] = ast_audit_sources(paths)
        return {"ast_findings": len(_STATE["ast_findings"])}
    except Exception:  # noqa: BLE001
        return None


def pre_eval(ctx):
    """Post-train: thread diff, secret-seed capture (file or env+unset),
    JIT cache reset. All facts are stashed for post_eval."""
    try:
        extra = diff_threads(_STATE.get("threads_before"))
        seed_hex = read_secret_seed(ctx)
        if seed_hex:
            _STATE["combined_seed"] = combine_with_recipe(seed_hex, (ctx or {}).get("seed"))
        jit_cache_reset()
        _STATE["extra_threads"] = extra
        return {"extra_threads": extra, "secret_seed": bool(seed_hex)}
    except Exception:  # noqa: BLE001
        return None


def post_eval(report):
    """Append the cheatguard results into the METRICS report dict (mutated
    in place AND returned — main.py ignores the return value)."""
    try:
        combined = _STATE.get("combined_seed")
        findings = _STATE.get("ast_findings") or []
        cg = {
            "ast_ok": not any(f["rule"] != "syntax_error" for f in findings),
            "ast_findings": findings[:50],
            "extra_threads": _STATE.get("extra_threads") or [],
            "secret_seed": "present" if _STATE.get("secret_seed_hex") else "absent",
            "combined_seed_sha256_16": hashlib.sha256(str(combined).encode()).hexdigest()[:16]
            if combined is not None
            else None,
            # Determinism/poison/timing helpers run eval-side; parent records
            # that they were offered (spike 08 §7.4 correctness gate).
            "determinism_ok": _STATE.get("determinism_ok"),
            "pointer_poison_ok": _STATE.get("pointer_poison_ok"),
            "timing_flag": _STATE.get("timing_flag"),
        }
        report["cheatguard"] = cg
    except Exception as exc:  # noqa: BLE001
        try:
            report["cheatguard"] = {"error": str(exc)[:200]}
        except Exception:  # noqa: BLE001
            pass
    return report
