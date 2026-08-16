"""Pod manifest (METRICS_JSON v2 `pod_manifest`).

Provenance + environment facts: nvidia-smi -q snapshot, python/torch/
transformers versions, `unshare` availability, netns on/off, eval battery
discovery status, harness file-set hash. Collection never fails the run —
missing tools record null fields. Only non-secret env knobs are recorded.
"""

import hashlib
import os
import platform
import subprocess
import time

from . import RECIPE_VERSION

# Non-secret knobs only — never add credential-bearing vars here.
#
# Completeness matters: the manifest is the run's attestation of the budget
# that was in force. It previously recorded neither the parameter cap nor any
# eval budget, so two runs under different caps were indistinguishable after
# the fact. Every knob that can move a cap belongs here.
_ENV_KNOBS = (
    "PRISM_MAX_TRAIN_STEPS",
    "PRISM_TRAIN_HOURS_CAP",
    # Operator seed-variance mode. Recorded because a run trained off the
    # recipe seed lattice is NOT comparable to a scored submission, and that
    # must be visible in the run's own attestation rather than inferred.
    "PRISM_SEED_OVERRIDE",
    "PRISM_TEST_TRAIN_MINUTES",
    "PRISM_TEST_MAX_PARAMS",
    "PRISM_MAX_PARAMS",
    "PRISM_SEQ_LEN",
    "PRISM_TRAIN_BATCH_SIZE",
    "PRISM_PROBE_EVERY",
    "PRISM_PROBE_TIME_BUDGET_S",
    "PRISM_BUILD_TIMEOUT_S",
    "PRISM_SCORE_TIMEOUT_S",
    "PRISM_ALLOW_CPU",
    "PRISM_GPU_TYPE",
    # Dual-cap budget: the FLOPs currency and its probe knobs.
    "PRISM_TRAIN_FLOPS_CAP",
    "PRISM_TEST_TRAIN_FLOPS",
    "PRISM_MIN_SPEND_FRACTION",
    "PRISM_FLOPS_PROBE_SAMPLES",
    "PRISM_FLOPS_PROBE_CV_MAX",
    "PRISM_FLOPS_ANALYTIC_GAP_MAX",
    # Eval budgets: the battery ceiling in force at score time.
    "PRISM_EVAL_TIMEOUT_S",
    "PRISM_EVAL_BATTERY_BUDGET_S",
)


def _run_text(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    if r.returncode != 0:
        return None
    return r.stdout


def nvidia_smi_snapshot():
    out = _run_text(["nvidia-smi", "-q"])
    if out is None:
        return None
    return out[:16000]


def torch_facts():
    facts = {
        "torch": None,
        "torch_cuda": None,
        "cuda_available": False,
        "transformers": None,
    }
    try:
        import torch

        facts["torch"] = torch.__version__
        facts["torch_cuda"] = getattr(torch.version, "cuda", None)
        facts["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    try:
        import transformers

        facts["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    return facts


def harness_files_sha256():
    """SHA-256 hex over the uploaded harness file set.

    Env-first: `prism-lium` sets PRISM_HARNESS_FILES_SHA256 from
    `prism_recipe::harness_files_sha256()` (the authoritative Rust-side hash
    of the embedded set). Fallback for local dev: hash the on-disk package
    with the same algorithm — for each (path, contents) sorted by path:
    `path bytes || 0x00 || contents || 0xff` into one SHA-256.
    """
    env = os.environ.get("PRISM_HARNESS_FILES_SHA256", "").strip()
    if env:
        return env
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = ["main.py"]
    for sub in ("prismlib", "eval"):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".py"):
                paths.append(f"{sub}/{name}")
    h = hashlib.sha256()
    for rel in sorted(paths):
        with open(os.path.join(root, rel), "rb") as f:
            data = f.read()
        h.update(rel.encode())
        h.update(b"\x00")
        h.update(data)
        h.update(b"\xff")
    return h.hexdigest()


def collect_manifest(*, netns, unshare, eval_battery, started_ts):
    facts = torch_facts()
    return {
        "recipe_version": RECIPE_VERSION,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_ts)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": facts["torch"],
        "torch_cuda": facts["torch_cuda"],
        "cuda_available": facts["cuda_available"],
        "transformers": facts["transformers"],
        "nvidia_smi_q": nvidia_smi_snapshot(),
        "unshare": unshare,
        "netns": bool(netns),
        "netns_disabled_env": os.environ.get("PRISM_DISABLE_NETNS") == "1",
        # Cheap heuristic per recipe 1.3.0: the legacy ctx["dataset_path"]
        # key stays; whether miner code read it directly is not tracked
        # here — the agentic review remains the enforcer.
        "dataset_path_usage": "unknown",
        "eval_battery": eval_battery,
        "harness_files_sha256": harness_files_sha256(),
        "env_knobs": {k: os.environ.get(k) for k in _ENV_KNOBS},
    }
