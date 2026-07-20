from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import _CONTAINER_EVAL_SCRIPT
from prism_challenge.evaluator.sandbox import SandboxViolation, inspect_code
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME

# Explicit negative-test suite for the anti-metric-manipulation / determinism anti-cheat class
# (architecture.md section 6.3). The challenge RE-EXECUTES the miner loop under a forced random
# init + deterministic flags, owns the data order, and computes the score ITSELF from the
# instrumented predict-then-train online-loss capture; any miner-reported metric/manifest is
# ignored. Covers VAL-CHEAT-010..014, 020, 021. The runtime cases drive the real challenge
# runner.py on a tiny CPU model + a tiny locked-data fixture (single pass); the static
# reject-determinism case drives the AST sandbox + the submission pipeline.

ARCH_LM = """
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

# Honest learner: real optimizer steps on the challenge instrument (predict-then-train).
TRAIN_LEARN = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        opt.step()
    flat = [round(float(x), 6) for p in model.parameters() for x in p.detach().flatten().tolist()]
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"params": flat}), encoding="utf-8"
    )
"""

# VAL-CHEAT-010: a miner that toggles the determinism flags + re-seeds before building the model.
# The challenge re-applies the forced init at build time so the toggles confer nothing.
TRAIN_TOGGLE_FLAGS = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    torch.manual_seed(13572468)
    torch.use_deterministic_algorithms(False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        opt.step()
    flat = [round(float(x), 6) for p in model.parameters() for x in p.detach().flatten().tolist()]
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"params": flat}), encoding="utf-8"
    )
"""

# VAL-CHEAT-011: a miner that consumes uncontrolled RNG and branches on it -- the forced init +
# deterministic data order make two re-executions produce the same score (no bankable luck).
TRAIN_NONDETERMINISTIC = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    _noise = torch.randn(32).sum()  # extra RNG draws (would diverge if RNG were uncontrolled)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens) + 0.0 * torch.randn_like(model(batch.tokens))
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        opt.step()
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"done": True}), encoding="utf-8"
    )
"""

# VAL-CHEAT-012: a miner that writes its own manifest-shaped files into artifacts_dir.
TRAIN_FAKE_MANIFEST = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        opt.step()
    for name in ("prism_run_manifest.v2.json", "prism_run_manifest.custom.json"):
        pathlib.Path(ctx.artifacts_dir, name).write_text(
            json.dumps(
                {
                    "schema_version": "prism_run_manifest.v2",
                    "submission_id": "miner-evil",
                    "metrics": {"prequential_bpb": 0.0000123, "bits_per_byte": 0.0000123},
                    "score": {"final_score": 0.999999},
                }
            ),
            encoding="utf-8",
        )
"""

# VAL-CHEAT-013: a miner that reports a fabricated low loss through stdout (the only reporting
# channel) -- the score is recomputed from the instrumented capture, never from the print.
TRAIN_FAKE_LOSS = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        v = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
        )
        loss.backward()
        opt.step()
        print("loss=0.0000001")
        print("PRISM_METRICS_JSON=" + json.dumps({"bpb": 0.0000001, "final_score": 1.0}))
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"done": True}), encoding="utf-8"
    )
"""

# VAL-CHEAT-014: a dead loop that iterates the instrument but never updates the model (no learning).
TRAIN_DEAD_LOOP = """
import json
import pathlib


def train(ctx):
    model = ctx.build_model()
    for _batch in ctx.iter_train_batches(model, batch_size=1):
        pass
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"done": True}), encoding="utf-8"
    )
"""

# VAL-CHEAT-021: a miner that probes the evaluation context (env vars / ctx fields / device) to
# special-case the "scored" run. The forced context is consistent, so it gains no advantage.
TRAIN_PROBING = """
import json
import pathlib

import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    scored = (
        ctx.world_size == 1
        and int(ctx.rank) == 0
        and str(ctx.device) == "cpu"
        and not torch.cuda.is_available()
    )
    opt = torch.optim.AdamW(model.parameters(), lr=0.05)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        if scored:
            opt.zero_grad()
            logits = model(batch.tokens)
            v = logits.shape[-1]
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, v), batch.tokens[:, 1:].reshape(-1) % v
            )
            loss.backward()
            opt.step()
        # else: the "unscored" branch would skip learning -- never reached under the forced ctx.
    pathlib.Path(ctx.artifacts_dir, "miner_probe.json").write_text(
        json.dumps({"scored_branch": scored}), encoding="utf-8"
    )
"""

LOCKED_TEXT_LINE = (
    '{"id": "doc-%d", "text": "the locked fineweb-edu train split sample sentence %d"}\n'
)


def _locked_shard(lines: int) -> str:
    return "".join(LOCKED_TEXT_LINE % (i, i) for i in range(lines))


def _run_runner(
    tmp_path: Path,
    *,
    run_name: str,
    train_code: str,
    arch_code: str = ARCH_LM,
    seed: int = 1337,
    submission_id: str = "sub-anticheat",
) -> tuple[subprocess.CompletedProcess[str], Path]:
    root = tmp_path / run_name
    project = root / "project"
    project.mkdir(parents=True)
    (project / "architecture.py").write_text(arch_code, encoding="utf-8")
    (project / "training.py").write_text(train_code, encoding="utf-8")

    data_dir = root / "data"
    data_dir.mkdir()
    (data_dir / "train-00000.jsonl").write_text(_locked_shard(48), encoding="utf-8")

    artifacts = root / "artifacts"
    artifacts.mkdir()

    payload = {
        "submission_id": submission_id,
        "architecture_entrypoint": "architecture.py",
        "training_entrypoint": "training.py",
        "build_model_symbol": "build_model",
        "train_symbol": "train",
        "execution_mode": "gpu_proxy_eval",
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "context": {
            "vocab_size": 64,
            "sequence_length": 16,
            "max_layers": 2,
            "max_parameters": 5_000_000,
            "seed": seed,
            "data_dir": str(data_dir),
            "artifacts_dir": str(artifacts),
            "reference_tokenizer_dir": str(root / "tok"),
            "token_budget": 512,
            "step_budget": 24,
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


def _probe(artifacts: Path) -> dict:
    return json.loads((artifacts / "miner_probe.json").read_text(encoding="utf-8"))


# --- VAL-CHEAT-010: miner cannot change the forced seed / determinism flags ---------------------


def test_anticheat_miner_cannot_change_forced_seed_or_flags(tmp_path: Path) -> None:
    toggle, toggle_art = _run_runner(tmp_path, run_name="toggle", train_code=TRAIN_TOGGLE_FLAGS)
    benign, benign_art = _run_runner(tmp_path, run_name="benign10", train_code=TRAIN_LEARN)
    assert toggle.returncode == 0, toggle.stderr
    assert benign.returncode == 0, benign.stderr
    toggle_manifest = _read_manifest(toggle_art)
    benign_manifest = _read_manifest(benign_art)
    # The challenge-authored manifest records the FORCED seed + deterministic flags regardless of
    # the miner toggling them.
    assert toggle_manifest["run"]["seed"] == 1337
    assert toggle_manifest["run"]["forced_init"] is True
    assert toggle_manifest["run"]["deterministic_algorithms"] is True
    # Forced init wins: identical trained params + identical step-0 online loss vs the benign twin.
    assert _probe(toggle_art)["params"] == _probe(benign_art)["params"]
    assert toggle_manifest["metrics"]["step0_loss"] == benign_manifest["metrics"]["step0_loss"]


def test_anticheat_forced_seed_step0_identical_across_repeats(tmp_path: Path) -> None:
    first, first_art = _run_runner(tmp_path, run_name="rep1", train_code=TRAIN_TOGGLE_FLAGS)
    second, second_art = _run_runner(tmp_path, run_name="rep2", train_code=TRAIN_TOGGLE_FLAGS)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert (
        _read_manifest(first_art)["metrics"]["step0_loss"]
        == _read_manifest(second_art)["metrics"]["step0_loss"]
    )


# --- VAL-CHEAT-011: non-deterministic tricks do not yield reproducibly inflated scores ----------


def test_anticheat_nondeterministic_run_is_reproducible(tmp_path: Path) -> None:
    first, first_art = _run_runner(tmp_path, run_name="nd1", train_code=TRAIN_NONDETERMINISTIC)
    second, second_art = _run_runner(tmp_path, run_name="nd2", train_code=TRAIN_NONDETERMINISTIC)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    m1 = _read_manifest(first_art)
    m2 = _read_manifest(second_art)
    # Two re-executions of the SAME bundle at the SAME forced seed/data order give the same score.
    assert m1["metrics"]["online_loss"] == m2["metrics"]["online_loss"]
    assert m1["metrics"]["prequential_bpb"] == m2["metrics"]["prequential_bpb"]
    assert m1["score"]["final_score"] == m2["score"]["final_score"]


# --- VAL-CHEAT-012: miner-written manifest is discarded -----------------------------------------


def test_anticheat_miner_written_manifest_discarded(tmp_path: Path) -> None:
    proc, artifacts = _run_runner(tmp_path, run_name="fakeman", train_code=TRAIN_FAKE_MANIFEST)
    assert proc.returncode == 0, proc.stderr
    manifest = _read_manifest(artifacts)
    assert manifest["schema_version"] == "prism_run_manifest.v2"
    assert manifest["submission_id"] == "sub-anticheat"
    assert manifest["miner_reported_ignored"] is True
    # The fabricated low bpb / score is not adopted; metrics come from the instrument.
    assert manifest["metrics"]["prequential_bpb"] != 0.0000123
    assert "online_loss" in manifest["metrics"]
    # Any other miner-written manifest-shaped file is quarantined (removed).
    leftover = sorted(p.name for p in artifacts.glob("prism_run_manifest*.json"))
    assert leftover == [RUN_MANIFEST_V2_FILENAME]


# --- VAL-CHEAT-013: fake low loss reported by the miner does not change the score ---------------


def test_anticheat_fake_reported_loss_does_not_change_score(tmp_path: Path) -> None:
    fake, fake_art = _run_runner(tmp_path, run_name="fakeloss", train_code=TRAIN_FAKE_LOSS)
    benign, benign_art = _run_runner(tmp_path, run_name="benign13", train_code=TRAIN_LEARN)
    assert fake.returncode == 0, fake.stderr
    assert benign.returncode == 0, benign.stderr
    fake_manifest = _read_manifest(fake_art)
    benign_manifest = _read_manifest(benign_art)
    # The fabricated stdout value never reaches the score; the instrumented capture is identical
    # to the benign twin that ran the same loop without the bogus print.
    assert fake_manifest["metrics"]["online_loss"] == benign_manifest["metrics"]["online_loss"]
    assert (
        fake_manifest["metrics"]["prequential_bpb"] == benign_manifest["metrics"]["prequential_bpb"]
    )
    assert fake_manifest["score"]["final_score"] != 1.0
    assert fake_manifest["metrics"]["prequential_bpb"] != 0.0000001


# --- VAL-CHEAT-014: single-point gaming / dead loop earns no advantageous score -----------------


def test_anticheat_dead_loop_no_advantage_over_honest_learner(tmp_path: Path) -> None:
    dead, dead_art = _run_runner(tmp_path, run_name="dead", train_code=TRAIN_DEAD_LOOP)
    learner, learner_art = _run_runner(tmp_path, run_name="learner14", train_code=TRAIN_LEARN)
    assert dead.returncode == 0, dead.stderr
    assert learner.returncode == 0, learner.stderr
    dead_manifest = _read_manifest(dead_art)
    learner_manifest = _read_manifest(learner_art)
    dead_bpb = dead_manifest["metrics"]["prequential_bpb"]
    learner_bpb = learner_manifest["metrics"]["prequential_bpb"]
    # The whole prequential curve is integrated: a dead loop never improves so its bpb is no better
    # (strictly worse) than the honest learner -- single-point gaming confers no advantage.
    assert dead_bpb > learner_bpb
    # The honest learner actually reduces the online loss across the single pass; the frozen dead
    # loop does not (both consume the SAME challenge-ordered batches, so a like-for-like compare at
    # the final batch shows the trained model strictly beats the frozen one).
    learner_online = learner_manifest["metrics"]["online_loss"]
    dead_online = dead_manifest["metrics"]["online_loss"]
    assert learner_online[-1] < learner_online[0]
    assert dead_online[-1] >= learner_online[-1]


# --- VAL-CHEAT-021: evaluation-context probing yields no advantage ------------------------------


def test_anticheat_context_probing_no_advantage(tmp_path: Path) -> None:
    probing, probing_art = _run_runner(tmp_path, run_name="probe", train_code=TRAIN_PROBING)
    honest, honest_art = _run_runner(tmp_path, run_name="honest21", train_code=TRAIN_LEARN)
    assert probing.returncode == 0, probing.stderr
    assert honest.returncode == 0, honest.stderr
    probing_manifest = _read_manifest(probing_art)
    honest_manifest = _read_manifest(honest_art)
    # The probe resolves to the "scored" branch and behaves like the honest twin: same score,
    # no advantage from special-casing the evaluation context.
    assert _probe(probing_art)["scored_branch"] is True
    assert probing_manifest["metrics"]["online_loss"] == honest_manifest["metrics"]["online_loss"]
    assert (
        probing_manifest["metrics"]["prequential_bpb"]
        == honest_manifest["metrics"]["prequential_bpb"]
    )


def test_anticheat_context_probing_consistent_across_repeats(tmp_path: Path) -> None:
    first, first_art = _run_runner(tmp_path, run_name="probe-a", train_code=TRAIN_PROBING)
    second, second_art = _run_runner(tmp_path, run_name="probe-b", train_code=TRAIN_PROBING)
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    # The loop behaves identically across re-executions (no context-dependent divergence).
    assert (
        _read_manifest(first_art)["metrics"]["online_loss"]
        == _read_manifest(second_art)["metrics"]["online_loss"]
    )


# --- VAL-CHEAT-020: rejection reason is deterministic and reproducible across resubmissions -----

VIOLATING_TRAIN = (
    "import torch\n"
    "from architecture import build_model\n\n"
    "def train(ctx):\n"
    "    build_model(ctx)\n"
    "    return torch.hub.load('pytorch/vision', 'resnet18')\n"
)


def test_anticheat_rejection_reason_deterministic_inspect(tmp_path: Path) -> None:
    rule_ids = []
    for _ in range(4):
        try:
            inspect_code(VIOLATING_TRAIN, require_contract=False)
        except SandboxViolation as exc:
            rule_ids.append(exc.evidence[0].rule_id)
        else:  # pragma: no cover - the vector must be rejected
            rule_ids.append("NOT-REJECTED")
    assert len(set(rule_ids)) == 1, rule_ids
    assert rule_ids[0] != "NOT-REJECTED"


def _submit(client, code: str, *, nonce: str) -> str:
    payload = {"code": code, "filename": "bundle.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, hotkey="hk", nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _process(client) -> None:
    response = client.post(
        "/internal/v1/worker/process-next",
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200, response.text


def _submission_row(client, submission_id: str) -> dict:
    repository = client.app.state.repository

    async def fetch() -> dict:
        async with repository.database.connect() as conn:
            rows = await conn.execute_fetchall(
                "SELECT status, error FROM submissions WHERE id=?", (submission_id,)
            )
        return dict(rows[0])

    return anyio.run(fetch)


def test_anticheat_rejection_reason_reproducible_across_resubmissions(tmp_path: Path) -> None:
    # Plagiarism/exact-hash dedup is a SEPARATE early gate that would reject an identical
    # resubmission for a duplicate reason; disable it so the determinism of the STATIC sandbox
    # reject reason itself is what is exercised across resubmissions.
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'resub.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        # No OpenRouter key in the unit env; disable the gate (covered in test_*llm*).
        distributed_contract_policy="off",
        plagiarism_enabled=False,
    )
    code = two_script_bundle(train_code=VIOLATING_TRAIN)
    outcomes = []
    with TestClient(create_app(settings)) as client:
        for i in range(3):
            submission_id = _submit(client, code, nonce=f"anticheat-det-{i}")
            _process(client)
            outcomes.append(_submission_row(client, submission_id))
    statuses = {row["status"] for row in outcomes}
    errors = {str(row["error"]) for row in outcomes}
    assert statuses == {"rejected"}, outcomes
    # The same violating bundle rejects for the SAME reason every time (not flaky/order-dependent).
    assert len(errors) == 1, outcomes
    assert "torch.hub.load" in errors.pop()
