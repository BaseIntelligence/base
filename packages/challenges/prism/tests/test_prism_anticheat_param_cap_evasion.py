from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from prism_challenge.evaluator.container import (
    _CONTAINER_EVAL_SCRIPT,
    _classify_failure,
    _ensure_compute_block,
)
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.sandbox import SandboxViolation
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME, ComputeBlock
from prism_challenge.evaluator.static_instantiation import (
    PARAM_CAP_RULE,
    check_build_model_static,
)

# Param-cap evasion hardening (architecture.md sections 4.1, 6; VAL-CHEAT-016, VAL-CHEAT-022):
# lazy/dynamic over-cap construction is materialized + rejected at the static gate, the cap binds
# the model ACTUALLY trained/scored in the runner, and the static-instantiate workdir is cleaned up.

CTX = PrismContext(vocab_size=256, sequence_length=16)


# --- static_instantiation.py workdir leak fix (architecture.md section 4.1) ---------------------


def test_static_instantiation_cleans_up_tempdir() -> None:
    tmp_root = Path(tempfile.gettempdir())
    pattern = "prism-static-instantiate-*"
    before = set(tmp_root.glob(pattern))
    arch = "import torch\n\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n"
    for _ in range(3):
        check_build_model_static({"architecture.py": arch}, "architecture.py", ctx=CTX)
    leaked = set(tmp_root.glob(pattern)) - before
    assert not leaked, f"static-instantiate workdirs leaked into {tmp_root}: {sorted(leaked)}"


# --- VAL-CHEAT-016: lazy / dynamic over-cap construction is materialized then rejected ----------

LAZY_OVER_CAP_ARCH = (
    "import torch\n"
    "from torch import nn\n\n"
    "class LazyBig(nn.Module):\n"
    "    def __init__(self, vocab, dim=8, out=2000):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.head = nn.LazyLinear(out)\n"
    "    def forward(self, tokens):\n"
    "        return self.head(self.emb(tokens))\n\n"
    "def build_model(ctx):\n"
    "    return LazyBig(ctx.vocab_size)\n"
)

DYNAMIC_OVER_CAP_ARCH = (
    "import torch\n"
    "from torch import nn\n\n"
    "class DynamicBig(nn.Module):\n"
    "    def __init__(self, vocab, dim=8):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.vocab = vocab\n"
    "        self.dim = dim\n"
    "        self._built = False\n"
    "    def forward(self, tokens):\n"
    "        if not self._built:\n"
    "            self.head = nn.Linear(self.dim, 4000)\n"
    "            self._built = True\n"
    "        return self.head(self.emb(tokens))\n\n"
    "def build_model(ctx):\n"
    "    return DynamicBig(ctx.vocab_size)\n"
)


@pytest.mark.parametrize("arch", [LAZY_OVER_CAP_ARCH, DYNAMIC_OVER_CAP_ARCH])
def test_static_lazy_dynamic_over_cap_rejected_param_cap(arch: str) -> None:
    with pytest.raises(SandboxViolation) as raised:
        check_build_model_static(
            {"architecture.py": arch}, "architecture.py", ctx=CTX, max_parameters=10_000
        )
    assert raised.value.evidence[0].rule_id == PARAM_CAP_RULE


def test_static_lazy_under_cap_materialized_and_counted() -> None:
    arch = (
        "import torch\n"
        "from torch import nn\n\n"
        "class LazySmall(nn.Module):\n"
        "    def __init__(self, vocab, dim=8, out=500):\n"
        "        super().__init__()\n"
        "        self.emb = nn.Embedding(vocab, dim)\n"
        "        self.head = nn.LazyLinear(out)\n"
        "    def forward(self, tokens):\n"
        "        return self.head(self.emb(tokens))\n\n"
        "def build_model(ctx):\n"
        "    return LazySmall(ctx.vocab_size)\n"
    )
    count = check_build_model_static(
        {"architecture.py": arch}, "architecture.py", ctx=CTX, max_parameters=1_000_000
    )
    # Materialized realized count = emb(256*8) + head(500*8 + 500).
    assert count == 256 * 8 + (500 * 8 + 500)


# --- VAL-CHEAT-022: the cap binds the model ACTUALLY trained/scored in the runner ---------------

ARCH_SMALL = (
    "import torch\n"
    "from torch import nn\n\n"
    "class TinyLM(nn.Module):\n"
    "    def __init__(self, vocab):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, 8)\n"
    "        self.head = nn.Linear(8, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        return self.head(self.emb(tokens))\n\n"
    "def build_model(ctx):\n"
    "    return TinyLM(ctx.vocab_size)\n"
)

# training.py ignores the sub-cap build_model and trains a DIFFERENT, over-cap model inline.
TRAIN_OVER_CAP_INLINE = (
    "import torch\n"
    "from torch import nn\n\n"
    "class Oversized(nn.Module):\n"
    "    def __init__(self, vocab, dim=600):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.head = nn.Linear(dim, vocab)\n\n"
    "    def forward(self, tokens):\n"
    "        return self.head(self.emb(tokens))\n\n"
    "def train(ctx):\n"
    "    model = Oversized(ctx.vocab_size)\n"
    "    opt = torch.optim.SGD(model.parameters(), lr=0.01)\n"
    "    for batch in ctx.iter_train_batches(model, batch_size=1):\n"
    "        opt.zero_grad()\n"
    "        logits = model(batch.tokens)\n"
    "        v = logits.shape[-1]\n"
    "        import torch.nn.functional as F\n"
    "        loss = F.cross_entropy(\n"
    "            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v\n"
    "        )\n"
    "        loss.backward()\n"
    "        opt.step()\n"
)

# training.py trains a model whose over-cap weights are DEFERRED until the first forward (the
# lazy / first-forward construction vector). At construction only the small embedding exists, so the
# pre-loop param-cap check passes; the first instrumented forward realizes the over-cap projection
# and the runner must re-check + reject BEFORE recording any ranking bpb.
TRAIN_DEFERRED_OVER_CAP = (
    "import torch\n"
    "from torch import nn\n\n"
    "class DeferredBig(nn.Module):\n"
    "    def __init__(self, vocab, dim=8):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, dim)\n"
    "        self.vocab = vocab\n"
    "        self.dim = dim\n"
    "        self.proj = None\n"
    "    def forward(self, tokens):\n"
    "        if self.proj is None:\n"
    "            self.proj = nn.Linear(self.dim, 5000)\n"
    "            self.out = nn.Linear(5000, self.vocab)\n"
    "        return self.out(self.proj(self.emb(tokens)))\n\n"
    "def train(ctx):\n"
    "    model = DeferredBig(ctx.vocab_size)\n"
    "    for _ in ctx.iter_train_batches(model, batch_size=1):\n"
    "        pass\n"
)

# An honest miner that builds + trains the sub-cap build_model.
TRAIN_HONEST = (
    "import torch\n\n"
    "def train(ctx):\n"
    "    model = ctx.build_model()\n"
    "    opt = torch.optim.SGD(model.parameters(), lr=0.01)\n"
    "    for batch in ctx.iter_train_batches(model, batch_size=1):\n"
    "        opt.zero_grad()\n"
    "        logits = model(batch.tokens)\n"
    "        v = logits.shape[-1]\n"
    "        import torch.nn.functional as F\n"
    "        loss = F.cross_entropy(\n"
    "            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v\n"
    "        )\n"
    "        loss.backward()\n"
    "        opt.step()\n"
)

LOCKED_LINE = '{"id": "doc-%d", "text": "the locked fineweb-edu train split sample sentence %d"}\n'


def _run_runner(
    tmp_path: Path,
    *,
    run_name: str,
    arch_code: str,
    train_code: str,
    max_parameters: int,
    vocab_size: int = 128,
    sequence_length: int = 16,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    root = tmp_path / run_name
    project = root / "project"
    project.mkdir(parents=True)
    (project / "architecture.py").write_text(arch_code, encoding="utf-8")
    (project / "training.py").write_text(train_code, encoding="utf-8")

    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "train-00000.jsonl").write_text(
        "".join(LOCKED_LINE % (i, i) for i in range(40)), encoding="utf-8"
    )
    artifacts = root / "artifacts"
    artifacts.mkdir()

    payload = {
        "submission_id": run_name,
        "architecture_entrypoint": "architecture.py",
        "training_entrypoint": "training.py",
        "build_model_symbol": "build_model",
        "train_symbol": "train",
        "execution_mode": "gpu_proxy_eval",
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "context": {
            "vocab_size": vocab_size,
            "sequence_length": sequence_length,
            "max_layers": 2,
            "max_parameters": max_parameters,
            "seed": 1337,
            "data_dir": str(data_dir),
            "artifacts_dir": str(artifacts),
            "reference_tokenizer_dir": str(root / "tok"),
            "token_budget": None,
            "step_budget": None,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "distributed_backend": None,
        },
    }
    runner = root / "runner.py"
    runner.write_text(_CONTAINER_EVAL_SCRIPT, encoding="utf-8")
    payload_path = root / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    env = dict(os.environ)
    env["PRISM_PROJECT_ROOT"] = str(project)
    env["PRISM_DATA_DIR"] = str(data_dir)
    env["PRISM_ARTIFACT_OUTPUT_PATH"] = str(artifacts)
    proc = subprocess.run(
        [sys.executable, str(runner), str(payload_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc, artifacts


def _read_manifest(artifacts: Path) -> dict:
    return json.loads((artifacts / RUN_MANIFEST_V2_FILENAME).read_text(encoding="utf-8"))


def test_runner_scored_over_cap_model_rejected_no_ranking_bpb(tmp_path: Path) -> None:
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="scored-overcap",
        arch_code=ARCH_SMALL,
        train_code=TRAIN_OVER_CAP_INLINE,
        max_parameters=50_000,  # inline Oversized model has >150k params
    )
    # The over-cap model that training.py actually trains is rejected before producing a score.
    assert proc.returncode != 0, proc.stdout
    assert "PRISM_RUNNER_PARAM_CAP" in proc.stderr, proc.stderr
    # No ranking bpb is attributable to the over-cap model: the run fails before authoring a
    # scored manifest (and if any manifest is present its final_score must be null).
    manifest_path = artifacts / RUN_MANIFEST_V2_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest.get("score", {}).get("final_score") is None, manifest.get("score")


def test_runner_deferred_lazy_over_cap_rejected_no_ranking_bpb(tmp_path: Path) -> None:
    # The over-cap weights are created only on the first forward (lazy / first-forward deferred):
    # the runner re-checks the realized param count after that forward and fails with
    # prism:param-cap before any ranking bpb is captured (VAL-CHEAT-022, the runtime twin of the
    # static lazy/dynamic gate).
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="deferred-overcap",
        arch_code=ARCH_SMALL,
        train_code=TRAIN_DEFERRED_OVER_CAP,
        max_parameters=50_000,  # only the deferred proj/out (>600k params) breach the cap
    )
    assert proc.returncode != 0, proc.stdout
    assert "PRISM_RUNNER_PARAM_CAP" in proc.stderr, proc.stderr
    manifest_path = artifacts / RUN_MANIFEST_V2_FILENAME
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest.get("score", {}).get("final_score") is None, manifest.get("score")


def test_runner_scored_over_cap_classified_as_param_cap() -> None:
    rule_id, _ = _classify_failure("PRISM_RUNNER_PARAM_CAP: scored model has 153728 parameters", 1)
    assert rule_id == "prism:param-cap"


def test_runner_honest_under_cap_records_realized_model_params(tmp_path: Path) -> None:
    proc, artifacts = _run_runner(
        tmp_path,
        run_name="scored-honest",
        arch_code=ARCH_SMALL,
        train_code=TRAIN_HONEST,
        max_parameters=5_000_000,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    # The challenge records the realized scored-model parameter count; it equals the build_model
    # model (TinyLM: emb 128*8 + head 8*128 + 128 bias) and is within the cap.
    expected = 128 * 8 + (8 * 128 + 128)
    assert manifest["metrics"]["model_params"] == expected
    assert manifest["metrics"]["model_params"] <= 5_000_000
    assert manifest["score"]["final_score"] is not None


# --- Host reconciliation: the realized scored-model param count surfaces in compute.model_params -


def test_ensure_compute_block_carries_model_params(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "prism_run_manifest.v2",
        "run": {"world_size": 1, "nproc_per_node": 1, "device": "cuda:0"},
        "metrics": {"model_params": 123456},
    }
    _ensure_compute_block(manifest, {"actual_gpu_count": 1, "max_gpu_count": 8}, tmp_path)
    compute = manifest["compute"]
    parsed = ComputeBlock.model_validate(compute)
    assert parsed.model_params == 123456
    assert compute["model_params"] == 123456


def test_compute_block_model_params_optional() -> None:
    block = ComputeBlock(gpu_count=1, world_size=1, nproc_per_node=1, device="cpu")
    assert block.model_params is None
