"""Recipe 2.0 AutoModel staging + thin train adapter (operator-owned).

The miner submits `automodel.base` + `automodel.patch` (+ optional
`prism.toml`). Master intake applies the patch (crate `prism-automodel`);
this harness module stages the pin + applied tree on the pod and invokes
the AutoModel train entry under Prism gates (netns / telemetry / caps /
dataset ownership stay in the parent harness).

# Env / layout contract (keep in sync with `prism_automodel::env`)

| Env | Role |
|-----|------|
| `PRISM_AUTOMODEL_APPLIED_DIR` | Preferred: already-applied tree from intake |
| `PRISM_AUTOMODEL_PIN_DIR` | Clean pin checkout (when harness applies) |
| `PRISM_AUTOMODEL_PATCH_PATH` | Unified diff path (with pin dir) |
| `PRISM_AUTOMODEL_BASE` | Pin id (`automodel@v0.5.0` / `automodel@fixture-v1`) |
| `PRISM_AUTOMODEL_PRISM_TOML` | Optional miner `prism.toml` |
| `PRISM_AUTOMODEL_FIXTURE=1` | CI/Sim: FIXTURE_PIN + happy.patch — **not** real-train proof |
| `PRISM_AUTOMODEL_FIXTURE_ROOT` | Override fixtures root (`automodel-pin/`, `patches/`) |

Pod layout after [`stage`]:

```text
$PRISM_WORKDIR/automodel/           applied tree (nemo_automodel/…)
$PRISM_WORKDIR/automodel.base       pin id
$PRISM_WORKDIR/prism.toml           optional knobs (when present)
$PRISM_WORKDIR/.prism_automodel.json  stage manifest
```

Fixture / host-sim stub runs must set `PRISM_AUTOMODEL_FIXTURE=1` (or pin id
`automodel@fixture-v1`). Markers `AUTOMODEL_FIXTURE_STAGED` /
`AUTOMODEL_FIXTURE_ENTRY_OK` mean staging+entry gates passed — they are
**not** evidence of a real NVIDIA AutoModel train (same rule as Design
`sim-install-ok` / `DESIGN_FORCE_SIM` stubs).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Mirror prism_automodel::env (Rust) — do not rename without updating both.
ENV_PIN_DIR = "PRISM_AUTOMODEL_PIN_DIR"
ENV_APPLIED_DIR = "PRISM_AUTOMODEL_APPLIED_DIR"
ENV_PATCH_PATH = "PRISM_AUTOMODEL_PATCH_PATH"
ENV_BASE = "PRISM_AUTOMODEL_BASE"
ENV_PRISM_TOML = "PRISM_AUTOMODEL_PRISM_TOML"
ENV_FIXTURE = "PRISM_AUTOMODEL_FIXTURE"
ENV_FIXTURE_ROOT = "PRISM_AUTOMODEL_FIXTURE_ROOT"

POD_APPLIED_DIRNAME = "automodel"
POD_STAGE_MANIFEST = ".prism_automodel.json"
DEFAULT_TRAIN_ENTRY = "nemo_automodel/recipes/llm/train_ft.py"
FIXTURE_PIN_ID = "automodel@fixture-v1"
LIVE_PIN_ID = "automodel@v0.5.0"

_HARNESS_CTX_KEYS = ("arch_path", "train_path", "workdir", "automodel")


def _log(msg):
    print(f"[automodel] {msg}", flush=True)


def _truthy(raw):
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def fixture_requested(base_id=None):
    """Whether this run is the CI/Sim fixture path (not real-train proof)."""
    if _truthy(os.environ.get(ENV_FIXTURE)):
        return True
    bid = (base_id or os.environ.get(ENV_BASE) or "").strip()
    return bid == FIXTURE_PIN_ID


def layout_requested(workdir=None):
    """True when env / workdir indicates a recipe 2.0 AutoModel submission."""
    if os.environ.get(ENV_APPLIED_DIR) or os.environ.get(ENV_PIN_DIR):
        return True
    if _truthy(os.environ.get(ENV_FIXTURE)):
        return True
    if workdir:
        wd = Path(workdir)
        if (wd / "automodel.base").is_file() and (wd / "automodel.patch").is_file():
            return True
        if (wd / POD_APPLIED_DIRNAME / "nemo_automodel").is_dir():
            return True
        # Intake packs MaterializedAutomodel as tree_blob → submission/
        # (nemo_automodel/… + .prism/automodel.{base,patch}).
        sub = wd / "submission"
        if (sub / "nemo_automodel").is_dir() and (
            (sub / ".prism" / "automodel.base").is_file()
            or (sub / "automodel.base").is_file()
        ):
            return True
    return False


def _intake_submission_applied(workdir):
    """Return submission/ path when it looks like an intake-materialised tree."""
    sub = Path(workdir) / "submission"
    if (sub / "nemo_automodel").is_dir() and (
        (sub / ".prism" / "automodel.base").is_file()
        or (sub / "automodel.base").is_file()
    ):
        return sub
    return None


def discover_fixture_root():
    """Locate `crates/prism-automodel/fixtures` (local/CI only; not on Lium pods)."""
    override = os.environ.get(ENV_FIXTURE_ROOT, "").strip()
    if override:
        root = Path(override)
        if (root / "automodel-pin").is_dir():
            return root
        raise FileNotFoundError(f"{ENV_FIXTURE_ROOT} missing automodel-pin/: {root}")

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "crates" / "prism-automodel" / "fixtures"
        if (candidate / "automodel-pin").is_dir():
            return candidate
        # Worktree may place fixtures next to a checked-out harness-only tree.
        candidate = parent / "prism-automodel" / "fixtures"
        if (candidate / "automodel-pin").is_dir():
            return candidate
    raise FileNotFoundError(
        "automodel fixtures not found; set PRISM_AUTOMODEL_FIXTURE_ROOT "
        "or PRISM_AUTOMODEL_APPLIED_DIR (fixture mode is local/CI only)"
    )


def parse_prism_toml(text):
    """Minimal prism.toml knobs: entry / train_script / model_config.

    Accepts either bare `entry = "..."` lines or a `[prism]` table. Unknown
    keys are ignored (intake may be stricter).
    """
    out = {}
    if not text:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip().lower()
        val = val.strip().strip("\"'")
        if key in ("entry", "train_script", "train_entry"):
            out["entry"] = val
        elif key in ("model_config", "config"):
            out["model_config"] = val
    return out


def resolve_entry(applied_dir, prism_toml_path=None, knobs=None):
    """Resolve train entry path relative to the applied tree."""
    knobs = dict(knobs or {})
    if prism_toml_path and Path(prism_toml_path).is_file():
        knobs.update(parse_prism_toml(Path(prism_toml_path).read_text(encoding="utf-8")))
    entry = (knobs.get("entry") or DEFAULT_TRAIN_ENTRY).lstrip("/")
    if ".." in entry.split("/"):
        raise ValueError(f"entry path escape rejected: {entry}")
    abs_entry = Path(applied_dir) / entry
    if not abs_entry.is_file():
        raise FileNotFoundError(f"AutoModel train entry missing: {abs_entry}")
    return entry, abs_entry, knobs


def _copytree(src, dst):
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=False)


def _apply_patch(pin_dir, patch_path):
    """Apply unified diff with `git apply` (fail-closed)."""
    pin_dir = Path(pin_dir)
    patch_path = Path(patch_path)
    if not patch_path.is_file():
        raise FileNotFoundError(f"patch not found: {patch_path}")
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git required to apply automodel.patch on the pod")
    for check in (True, False):
        cmd = [git, "-C", str(pin_dir), "apply", "--whitespace=nowarn"]
        if check:
            cmd.append("--check")
        cmd.append(str(patch_path))
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if r.returncode != 0:
            detail = (r.stderr or r.stdout or "git apply failed").strip()[:400]
            raise RuntimeError(f"git apply failed: {detail}")


def stage(workdir, *, force_fixture=False):
    """Stage pin + applied miner tree under `$workdir/automodel`.

    Returns a manifest dict written to `.prism_automodel.json`.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / POD_APPLIED_DIRNAME
    base_id = (os.environ.get(ENV_BASE) or "").strip()
    prism_toml_src = os.environ.get(ENV_PRISM_TOML, "").strip() or None
    mode = "live"
    stub = False
    source = None

    # Workdir-local submit artifacts (intake may drop these next to the harness).
    local_base = workdir / "automodel.base"
    local_patch = workdir / "automodel.patch"
    local_toml = workdir / "prism.toml"
    intake_sub = _intake_submission_applied(workdir)
    if intake_sub is not None:
        meta_base = intake_sub / ".prism" / "automodel.base"
        if not meta_base.is_file():
            meta_base = intake_sub / "automodel.base"
        if meta_base.is_file() and not base_id:
            base_id = meta_base.read_text(encoding="utf-8").strip()
        sub_toml = intake_sub / "prism.toml"
        if sub_toml.is_file() and not prism_toml_src:
            prism_toml_src = str(sub_toml)
    if local_base.is_file() and not base_id:
        base_id = local_base.read_text(encoding="utf-8").strip()
    if local_toml.is_file() and not prism_toml_src:
        prism_toml_src = str(local_toml)

    use_fixture = force_fixture or fixture_requested(base_id)

    applied_env = os.environ.get(ENV_APPLIED_DIR, "").strip()
    pin_env = os.environ.get(ENV_PIN_DIR, "").strip()
    patch_env = os.environ.get(ENV_PATCH_PATH, "").strip()
    if local_patch.is_file() and not patch_env:
        patch_env = str(local_patch)

    if applied_env:
        src = Path(applied_env)
        if not src.is_dir():
            raise FileNotFoundError(f"{ENV_APPLIED_DIR} not a directory: {src}")
        _log(f"staging applied tree from {src}")
        _copytree(src, dest)
        source = "applied_dir"
        mode = "fixture" if use_fixture else "live"
        stub = use_fixture
    elif intake_sub is not None:
        _log(f"staging applied tree from intake submission/ ({intake_sub})")
        _copytree(intake_sub, dest)
        source = "submission_tree_blob"
        mode = "fixture" if use_fixture else "live"
        stub = use_fixture
    elif use_fixture:
        root = discover_fixture_root()
        pin = root / "automodel-pin"
        patch = root / "patches" / "happy.patch"
        if patch_env:
            patch = Path(patch_env)
        _log(f"fixture stage: pin={pin} patch={patch}")
        _copytree(pin, dest)
        _apply_patch(dest, patch)
        base_id = base_id or FIXTURE_PIN_ID
        mode = "fixture"
        stub = True
        source = "fixture"
    elif pin_env:
        pin = Path(pin_env)
        if not pin.is_dir():
            raise FileNotFoundError(f"{ENV_PIN_DIR} not a directory: {pin}")
        if not patch_env:
            raise FileNotFoundError(
                f"{ENV_PATCH_PATH} required when staging from {ENV_PIN_DIR}"
            )
        _log(f"staging pin+patch: pin={pin} patch={patch_env}")
        _copytree(pin, dest)
        _apply_patch(dest, patch_env)
        source = "pin_patch"
        mode = "live"
        stub = False
    else:
        raise RuntimeError(
            "AutoModel stage: set PRISM_AUTOMODEL_APPLIED_DIR, or "
            "PRISM_AUTOMODEL_PIN_DIR+PRISM_AUTOMODEL_PATCH_PATH, or "
            "PRISM_AUTOMODEL_FIXTURE=1 for CI fixtures"
        )

    if not base_id:
        base_id = FIXTURE_PIN_ID if stub else LIVE_PIN_ID
    (workdir / "automodel.base").write_text(base_id + "\n", encoding="utf-8")

    if prism_toml_src and Path(prism_toml_src).is_file():
        shutil.copy2(prism_toml_src, workdir / "prism.toml")
        prism_toml_path = str(workdir / "prism.toml")
    elif (workdir / "prism.toml").is_file():
        prism_toml_path = str(workdir / "prism.toml")
    else:
        prism_toml_path = None

    entry_rel, entry_abs, knobs = resolve_entry(dest, prism_toml_path)
    manifest = {
        "layout": "automodel",
        "mode": mode,
        "stub": stub,
        "source": source,
        "base": base_id,
        "applied_dir": str(dest),
        "entry": entry_rel,
        "entry_abs": str(entry_abs),
        "model_config": knobs.get("model_config"),
        "real_train": not stub,
        "note": (
            "fixture/stub — not proof of real AutoModel train"
            if stub
            else "live applied tree"
        ),
    }
    (workdir / POD_STAGE_MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if stub:
        print("AUTOMODEL_FIXTURE_STAGED", flush=True)
    _log(
        f"staged mode={mode} stub={stub} base={base_id} entry={entry_rel} "
        f"real_train={not stub}"
    )
    return manifest


def load_manifest(workdir):
    path = Path(workdir) / POD_STAGE_MANIFEST
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def invoke_entry(manifest, *, ctx=None):
    """Run the resolved AutoModel train entry under the applied tree.

    Fixture entries typically expose `main()` only. Live trees may expose
    Prism-shaped `build_model` / `train`, or a `main()` CLI. Returns a
    result dict; fixture runs set `stub=True` and must not be scored as
    real trains.
    """
    applied = Path(manifest["applied_dir"])
    entry_abs = Path(manifest["entry_abs"])
    stub = bool(manifest.get("stub"))

    # Applied tree first so miner AutoModel packages resolve; harness stays
    # importable via the parent-configured PYTHONPATH prefix.
    applied_s = str(applied)
    if applied_s not in sys.path:
        sys.path.insert(0, applied_s)

    import importlib.util

    spec = importlib.util.spec_from_file_location("prism_automodel_entry", entry_abs)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load entry {entry_abs}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    result = {
        "status": "ok",
        "stub": stub,
        "real_train": not stub,
        "entry": manifest.get("entry"),
        "mode": manifest.get("mode"),
    }

    if hasattr(mod, "train") and callable(mod.train) and ctx is not None:
        # Prism-shaped entry inside the applied tree (optional).
        model = None
        if hasattr(mod, "build_model") and callable(mod.build_model):
            model = mod.build_model(ctx)
        metrics = mod.train(model, ctx) if model is not None else mod.train(ctx)
        if not isinstance(metrics, dict):
            metrics = {"result": metrics}
        result["train_metrics"] = metrics
        result["via"] = "train"
    elif hasattr(mod, "main") and callable(mod.main):
        mod.main()
        result["via"] = "main"
        result["train_metrics"] = {"entry_main": True}
    else:
        raise RuntimeError(
            f"entry {entry_abs} needs main() or train() (got {dir(mod)[:20]})"
        )

    if stub:
        print("AUTOMODEL_FIXTURE_ENTRY_OK", flush=True)
        _log("fixture entry returned — not a scored real AutoModel train")
    return result


def build_model(ctx):
    """Harness-owned `build_model` seam for miner_entry / train_v3 / eval_v3.

    Live: load `build_model` from the resolved entry (or a model_config
    module). Fixture/stub: tiny torch module so gates can run locally —
    still flagged `automodel_stub` in train metrics (not real-train proof).
    """
    workdir = ctx.get("workdir") or os.environ.get("PRISM_WORKDIR", ".")
    manifest = ctx.get("automodel") or load_manifest(workdir)
    if not manifest:
        raise RuntimeError("automodel stage manifest missing; call stage() first")

    applied = Path(manifest["applied_dir"])
    if str(applied) not in sys.path:
        sys.path.insert(0, str(applied))

    entry_abs = Path(manifest["entry_abs"])
    import importlib.util

    spec = importlib.util.spec_from_file_location("prism_automodel_entry", entry_abs)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "build_model") and callable(mod.build_model):
        model = mod.build_model(ctx)
        return model

    if manifest.get("stub") or fixture_requested(manifest.get("base")):
        import torch
        import torch.nn as nn

        class _FixtureStub(nn.Module):
            """CI fixture stand-in — not a real AutoModel architecture."""

            def __init__(self, vocab=256, d=16, block=64):
                super().__init__()
                self.block = block
                self.emb = nn.Embedding(vocab, d)
                self.pos = nn.Embedding(block, d)
                self.head = nn.Linear(d, vocab, bias=False)
                self.head.weight = self.emb.weight

            def forward(self, ids):
                b, t = ids.shape
                t = min(t, self.block)
                ids = ids[:, -t:]
                pos = torch.arange(t, device=ids.device)
                return self.head(self.emb(ids) + self.pos(pos)[None, :, :])

        _log(
            "WARNING: fixture stub build_model — not proof of real AutoModel train"
        )
        vs = int(ctx.get("vocab_size") or 256)
        return _FixtureStub(vocab=max(vs, 32))

    raise RuntimeError(
        f"AutoModel entry {entry_abs} has no build_model(); "
        "live trees must expose build_model or a Prism train adapter"
    )


def train(model, ctx):
    """Harness-owned `train` seam: prefer entry.train, else short fixture loop."""
    workdir = ctx.get("workdir") or os.environ.get("PRISM_WORKDIR", ".")
    manifest = ctx.get("automodel") or load_manifest(workdir)
    if not manifest:
        raise RuntimeError("automodel stage manifest missing")

    applied = Path(manifest["applied_dir"])
    if str(applied) not in sys.path:
        sys.path.insert(0, str(applied))

    telemetry = ctx.get("telemetry")
    guard = ctx.get("guard")
    stream = ctx.get("train_stream")
    max_steps = int(ctx.get("max_train_steps") or 8)

    entry_abs = Path(manifest["entry_abs"])
    import importlib.util

    spec = importlib.util.spec_from_file_location("prism_automodel_entry", entry_abs)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "train") and callable(mod.train):
        metrics = mod.train(model, ctx)
        if not isinstance(metrics, dict):
            metrics = {"result": metrics}
        metrics.setdefault("automodel_stub", bool(manifest.get("stub")))
        metrics.setdefault("real_train", not bool(manifest.get("stub")))
        return metrics

    # Fixture / stub path: invoke main() for gate coverage, then a tiny
    # stream loop so telemetry + caps still exercise when torch is present.
    if hasattr(mod, "main") and callable(mod.main):
        mod.main()

    import torch

    if stream is None:
        return {
            "automodel_stub": True,
            "real_train": False,
            "entry_main": True,
            "train_steps": 0,
            "note": "fixture stub — not proof of real AutoModel train",
        }

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    steps = 0
    last = 0.0
    for input_ids, labels in stream:
        if guard is not None:
            guard()
        out = model(input_ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.item())
        steps += 1
        if telemetry is not None:
            telemetry.report(loss=last, step=steps)
        if steps >= min(max_steps, 8):
            break
    return {
        "train_loss": last,
        "train_steps": steps,
        "automodel_stub": True,
        "real_train": False,
        "note": "fixture stub — not proof of real AutoModel train",
    }


def write_seams(workdir, manifest):
    """Write harness-owned architecture.py / training.py seams at workdir root.

    Keeps miner_entry / train_v3 / eval_v3 on the existing import path while
    the real logic lives in this module (miner never owns the adapter).
    """
    workdir = Path(workdir)
    arch = '''\
"""Harness-owned AutoModel seam — not miner code."""
from prismlib.automodel import build_model as build_model
'''
    train_py = '''\
"""Harness-owned AutoModel seam — not miner code."""
from prismlib.automodel import train as train
'''
    (workdir / "architecture.py").write_text(arch, encoding="utf-8")
    (workdir / "training.py").write_text(train_py, encoding="utf-8")
    # Ensure legacy path resolution finds seams (not under submission/).
    return {
        "arch_path": str(workdir / "architecture.py"),
        "train_path": str(workdir / "training.py"),
        "automodel": manifest,
    }


def stage_and_prepare(workdir):
    """Stage applied tree + write seams; return paths for harness ctx."""
    manifest = stage(workdir)
    return write_seams(workdir, manifest)


def fixture_smoke(workdir=None):
    """Local/CI smoke: stage FIXTURE_PIN + happy.patch, invoke entry.

    Prints fixture markers; returns manifest + invoke result. Does **not**
    emit EVAL_OK / scored METRICS_JSON.
    """
    os.environ[ENV_FIXTURE] = "1"
    td = workdir or tempfile.mkdtemp(prefix="prism-automodel-fixture-")
    manifest = stage(td, force_fixture=True)
    # Prove patch applied (happy.patch widens ToyModel + adds layers.py).
    model_py = Path(manifest["applied_dir"]) / (
        "nemo_automodel/components/models/toy/model.py"
    )
    text = model_py.read_text(encoding="utf-8")
    if "width: int = 16" not in text:
        raise AssertionError("happy.patch did not apply to toy model")
    layers = Path(manifest["applied_dir"]) / (
        "nemo_automodel/components/models/toy/layers.py"
    )
    if not layers.is_file():
        raise AssertionError("happy.patch did not add layers.py")
    result = invoke_entry(manifest, ctx=None)
    if not result.get("stub"):
        raise AssertionError("fixture smoke must be stub/non-scored")
    return {"workdir": td, "manifest": manifest, "result": result}


if __name__ == "__main__":
    out = fixture_smoke()
    print(json.dumps({"ok": True, "stub": True, "entry": out["manifest"]["entry"]}))
    sys.exit(0)
