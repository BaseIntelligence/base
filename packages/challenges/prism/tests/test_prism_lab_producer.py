"""Architecture-lab producer layer: family / variant / curve / timing writers.

Drives the full submit -> process pipeline with a faked container run (a hand-authored v2 manifest)
and asserts the finalize path now populates ``architecture_families`` + ``training_variants`` +
``submission_curves`` + ``eval_jobs`` timing + ``submissions.name`` (none of which existed before).
"""

from __future__ import annotations

import json
import math
import sqlite3

from base.challenge_sdk.executor import DockerRunResult
from conftest import signed_headers, two_script_bundle
from fastapi.testclient import TestClient

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME

ARCH_A = """
ARCHITECTURE_NAME = "Rotary MoE v3!!"
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

ARCH_B = """
ARCHITECTURE_NAME = "Another / Arch [v2]"
import torch
from torch import nn


class WideLM(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, 16)
        self.head = nn.Linear(16, vocab)

    def forward(self, tokens):
        return self.head(self.emb(tokens))


def build_model(ctx):
    return WideLM(ctx.vocab_size)
"""


def _train(variant: int) -> str:
    # Distinct AST constant per variant -> distinct training_hash (comments alone would normalize
    # away under the AST-dump fingerprint).
    return (
        "def train(ctx):\n"
        "    model = ctx.build_model()\n"
        f"    _variant = {variant}\n"
        "    return None\n"
    )


def _v2_manifest(submission_id: str, *, sum_nll_nats: float, covered_bytes: int) -> dict:
    bits = sum_nll_nats / math.log(2.0)
    bpb = bits / covered_bytes
    # Synthetic held-out primary so master-side finalize can crown (VAL-RESLAB-006). Keep ordering
    # consistent with lower-bpb-is-better by mapping a larger heldout_delta for lower bpb.
    heldout_delta = 1.0 / (1.0 + bpb)
    online_loss = [2.5, 2.0, 1.5]
    cumulative = [covered_bytes // 3, (covered_bytes * 2) // 3, covered_bytes]
    return {
        "schema_version": "prism_run_manifest.v2",
        "submission_id": submission_id,
        "run_id": "prism-reexec-" + submission_id,
        "mode": "gpu_proxy_eval",
        "run": {
            "seed": 1337,
            "forced_init": True,
            "world_size": 1,
            "rank": 0,
            "local_rank": 0,
            "device": "cuda:0",
            "nproc_per_node": 1,
        },
        "data": {
            "covered_bytes": covered_bytes,
            "covered_bytes_cumulative": cumulative,
            "single_pass": True,
        },
        "metrics": {
            "online_loss": online_loss,
            "sum_neg_log_likelihood_nats": sum_nll_nats,
            "sum_neg_log2_likelihood_bits": bits,
            "cumulative_codelength_bits": bits,
            "covered_bytes": covered_bytes,
            "total_bytes_covered": covered_bytes,
            "predicted_tokens": 96,
            "tokens_seen": 96,
            "prequential_bpb": bpb,
            "bits_per_byte": bpb,
            "heldout_delta": heldout_delta,
            "held_out_delta": heldout_delta,
            "val_bpb_trained": max(0.1, bpb),
            "val_bpb_random_init": max(0.1, bpb) + heldout_delta,
            "step0_loss": online_loss[0],
            "consumed_batches": 3,
            "random_init_baseline_nats": math.log(128),
            "nan_inf_batches": 0,
            "model_params": 4096,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
        "miner_reported_ignored": True,
    }


def _final_score(sum_nll_nats: float, covered_bytes: int) -> float:
    from prism_challenge.evaluator.scoring import score_prequential_bpb

    manifest = _v2_manifest("score-probe", sum_nll_nats=sum_nll_nats, covered_bytes=covered_bytes)
    return float(score_prequential_bpb(manifest).final_score)


def _settings(tmp_path) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'lab.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        base_eval_artifact_root=tmp_path / "artifacts",
        plagiarism_enabled=False,
        distributed_contract_policy="off",
    )


def _submit(client: TestClient, bundle: str, *, hotkey: str, nonce: str) -> str:
    payload = {"code": bundle, "filename": "project.zip"}
    body = json.dumps(payload, separators=(",", ":")).encode()
    response = client.post(
        "/v1/submissions",
        content=body,
        headers={
            **signed_headers("secret", body, hotkey=hotkey, nonce=nonce),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 200, response.text
    return str(response.json()["id"])


def _process(client: TestClient) -> None:
    response = client.post(
        "/internal/v1/worker/process-next", headers={"Authorization": "Bearer secret"}
    )
    assert response.status_code == 200, response.text


def test_finalize_populates_lab_tables(tmp_path, monkeypatch) -> None:
    # call index -> (sum_nll_nats, covered_bytes); lower bpb => higher final_score.
    run_params = [
        (50.0 * math.log(2.0), 250),  # #1 archA/T1  bpb 0.2   final 0.8333
        (200.0 * math.log(2.0), 250),  # #2 archA/T2  bpb 0.8   final 0.5556
        (10.0 * math.log(2.0), 250),  # #3 archA/T3  bpb 0.04  final 0.9615 (new best)
        (100.0 * math.log(2.0), 250),  # #4 archB/T1  bpb 0.4   final 0.7143
    ]
    counter = {"i": 0}

    def fake_run(self, spec, timeout_seconds):
        payload = json.loads((spec.mounts[0].source / "payload.json").read_text())
        artifact_dir = spec.mounts[1].source
        artifact_dir.mkdir(parents=True, exist_ok=True)
        sum_nll, covered = run_params[counter["i"]]
        counter["i"] += 1
        manifest = _v2_manifest(
            str(payload["submission_id"]), sum_nll_nats=sum_nll, covered_bytes=covered
        )
        (artifact_dir / RUN_MANIFEST_V2_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
        return DockerRunResult(
            container_name="prism-eval",
            stdout='PRISM_METRICS_JSON={"covered_bytes":250.0}\n',
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    db_path = tmp_path / "lab.sqlite3"

    with TestClient(create_app(_settings(tmp_path))) as client:
        sub1 = _submit(
            client,
            two_script_bundle(arch_code=ARCH_A, train_code=_train(1)),
            hotkey="hkA",
            nonce="n1",
        )
        _process(client)
        sub2 = _submit(
            client,
            two_script_bundle(arch_code=ARCH_A, train_code=_train(2)),
            hotkey="hkB",
            nonce="n2",
        )
        _process(client)
        sub3 = _submit(
            client,
            two_script_bundle(arch_code=ARCH_A, train_code=_train(3)),
            hotkey="hkC",
            nonce="n3",
        )
        _process(client)
        sub4 = _submit(
            client,
            two_script_bundle(arch_code=ARCH_B, train_code=_train(1)),
            hotkey="hkD",
            nonce="n4",
        )
        _process(client)
        for sid in (sub1, sub2, sub3, sub4):
            assert client.get(f"/v1/submissions/{sid}").json()["status"] == "completed"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # --- submissions.name: parsed + moderated (the "!!" is dropped). ---
        name_row = conn.execute("SELECT name FROM submissions WHERE id=?", (sub1,)).fetchone()
        assert name_row["name"] == "Rotary MoE v3"

        # --- architecture_families: archA family (3 submissions) + archB family. ---
        families = conn.execute(
            "SELECT * FROM architecture_families ORDER BY q_arch_best DESC"
        ).fetchall()
        assert len(families) == 2
        fam_a = next(f for f in families if f["owner_submission_id"] == sub1)
        # Owner + display_name stay STABLE to the family-creating submission (#1, hkA), even though
        # the canonical/best submission advanced to #3.
        assert fam_a["owner_hotkey"] == "hkA"
        assert fam_a["display_name"] == "Rotary MoE v3"
        assert fam_a["canonical_submission_id"] == sub3
        assert fam_a["q_arch_best"] == _final_score(*run_params[2])
        architecture_id = fam_a["id"]

        fam_b = next(f for f in families if f["owner_submission_id"] == sub4)
        assert fam_b["owner_hotkey"] == "hkD"
        assert fam_b["display_name"] == "Another / Arch [v2]"
        assert fam_b["canonical_submission_id"] == sub4

        # --- training_variants: 3 distinct variants under archA, exactly one current best. ---
        variants = conn.execute(
            "SELECT * FROM training_variants WHERE architecture_id=?", (architecture_id,)
        ).fetchall()
        assert len(variants) == 3
        best = [v for v in variants if v["is_current_best"] == 1]
        assert len(best) == 1
        assert best[0]["submission_id"] == sub3
        assert best[0]["owner_hotkey"] == "hkC"
        assert best[0]["q_recipe"] == _final_score(*run_params[2])
        # Distinct training hashes (one per training script variant).
        assert len({v["training_hash"] for v in variants}) == 3

        # archB has its own single variant, current-best within its (single-variant) family.
        variants_b = conn.execute(
            "SELECT * FROM training_variants WHERE architecture_id=?", (fam_b["id"],)
        ).fetchall()
        assert len(variants_b) == 1
        assert variants_b[0]["is_current_best"] == 1

        # --- submission_curves: loss curve + reconciled compute persisted per submission. ---
        curve = conn.execute(
            "SELECT * FROM submission_curves WHERE submission_id=?", (sub1,)
        ).fetchone()
        assert curve is not None
        assert json.loads(curve["online_loss"]) == [2.5, 2.0, 1.5]
        assert json.loads(curve["covered_bytes_cumulative"]) == [83, 166, 250]
        assert curve["step0_loss"] == 2.5
        assert curve["baseline_nats"] == math.log(128)
        compute = json.loads(curve["compute"])
        assert compute["gpu_count"] == 1
        assert compute["model_params"] == 4096
        # estimated_flops = 6 * model_params * tokens_consumed (host-computed at reconciliation).
        assert compute["estimated_flops"] == 6.0 * 4096 * 96
        curve_count = conn.execute("SELECT COUNT(*) FROM submission_curves").fetchone()[0]
        assert curve_count == 4

        # --- eval_jobs timing: the scored job carries host-side started_at / ended_at. ---
        job = conn.execute(
            "SELECT started_at, ended_at FROM eval_jobs WHERE submission_id=? AND level!='l1'",
            (sub1,),
        ).fetchone()
        assert job is not None
        assert job["started_at"] is not None
        assert job["ended_at"] is not None
    finally:
        conn.close()
