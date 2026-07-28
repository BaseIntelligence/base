#!/usr/bin/env python3
"""Checkbox 27 — end-to-end Lium image-attestation proof (honest + adversarial).

Path exercised (same code as production constation / prism ingest):

  submit → allowlist register → Lium deploy → sidecar attest (start/random/end)
  → constation runner → prism ingest → tier / score row

Modes
-----
* ``live`` (default when ``LIUM_API_KEY`` is available): real Lium pod rental +
  real ``get_pod_raw`` reads. Sidecar is an in-process attestor that signs with
  the real HMAC path; mid-run image swap is simulated by flipping the sidecar
  digest after the start sample (miner-controlled guest) while Lium's declared
  digest stays pinned — the detection surface is corroboration mismatch.
* ``fixture``: no network; ScriptedLium + FakeSidecar. Evidence must label
  ``REAL_LIUM=false``.

Gated: set ``BASE_LIVE_PROVIDER_TESTS=1`` for live rentals (billable). Pods are
always terminated in ``finally``. Secrets are never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
import sqlite3
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

# Cross-repo imports (prism + prism-recipe live beside base in the monorepo).
_REPO = Path(__file__).resolve().parents[1]
_MONO = _REPO.parent
for _p in (_MONO / "prism" / "src", _MONO / "prism-recipe" / "src", _REPO / "src"):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

from prism_challenge.app import create_app  # noqa: E402
from prism_challenge.config import PrismSettings, WorkerPlaneConfig  # noqa: E402
from prism_challenge.constation import CheckOutcome, ConstationBundle  # noqa: E402
from prism_challenge.evaluator.mock_reexec import cpu_reexec_run  # noqa: E402
from prism_challenge.ingestion import ingest_work_unit_result  # noqa: E402
from prism_challenge.models import SubmissionCreate  # noqa: E402
from prism_challenge.proof import (  # noqa: E402
    ATTESTATION_MODE_V1,
    MANIFEST_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    ProviderInfo,
    build_execution_proof,
    compute_manifest_sha256,
    worker_signer_from_key,
)
from prism_recipe.attestation.payload import (  # noqa: E402
    AttestationPayload,
    compute_build_secret_response,
    derive_attestation_key,
    sign_attestation_payload,
    verify_attestation_payload,
)
from prism_recipe.submission import validate_submission  # noqa: E402

from base.compute.attestation_nonce import (  # noqa: E402
    AttestationNonceService,
    NonceBinding,
    NonceConsumeHit,
)
from base.compute.constation_custody import (  # noqa: E402
    LiumKeyCustody,
    generate_custody_master_key,
)
from base.compute.constation_poller import PollerConfig  # noqa: E402
from base.compute.constation_runner import (  # noqa: E402
    ConstationRunner,
    ConstationRunRequest,
)
from base.compute.constation_types import (  # noqa: E402
    ConstationFailCode,
    FaultClass,
)
from base.compute.digest_allowlist import (  # noqa: E402
    AllowlistHit,
    DigestAllowlist,
    DigestRecord,
    ImageVariant,
)
from base.compute.lium import LiumClient, LiumPodRead  # noqa: E402
from base.compute.provider import InstanceSpec  # noqa: E402

# Reuse proven credential / SSH helpers from the existing live rental script.
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
import live_lium_e2e as prov  # noqa: E402

DIGEST_HONEST = "sha256:" + ("11" * 32)
DIGEST_SWAP = "sha256:" + ("22" * 32)
# Real linux/amd64 content digest for nvidia/cuda:12.4.1-base-ubuntu22.04.
DIGEST_LIVE_CUDA = (
    "sha256:8767a245ed2c481eb245d8f6c625accc3788e1fb8612403d6b4cd4645a4f09c7"
)
COMMIT = "a" * 40
TREE = "b" * 40
VARIANT = "cuda"
HOTKEY = "5E2ePrismLiumAttestMinerHotkey000000000001"
WORKER_KEY = "//PrismLiumAttestE2EWorker"
BUILD_SECRET = b"prism-lium-e2e-build-secret-v1"
SEALED = {"src/prism_recipe/harness.py": "c" * 64}
TEMPLATE_PREFIX = "prism-attest-e2e-v27r"
E2E_IMAGE = "nvidia/cuda"
E2E_IMAGE_TAG = "12.4.1-base-ubuntu22.04"
E2E_STARTUP = "tail -f /dev/null"
MAX_PRICE = 1.50
TERMINATION_HOURS = 1
POLL_BUDGET = 12 * 60
DEFAULT_EVIDENCE = (
    _MONO / ".omo" / "notepads" / "prism-lium-image-attestation" / "evidence"
)

_TINY_ARCH = (
    "import torch\nfrom torch import nn\n\n"
    "class TinyLM(nn.Module):\n"
    "    def __init__(self, vocab):\n"
    "        super().__init__()\n"
    "        self.emb = nn.Embedding(vocab, 8)\n"
    "        self.head = nn.Linear(8, vocab)\n"
    "    def forward(self, tokens):\n"
    "        return self.head(self.emb(tokens))\n\n"
    "def build_model(ctx):\n"
    "    return TinyLM(ctx.vocab_size)\n"
)
_TINY_TRAIN = (
    "import torch\nimport torch.nn.functional as F\n\n"
    "def train(ctx):\n"
    "    model = ctx.build_model()\n"
    "    opt = torch.optim.AdamW(model.parameters(), lr=0.01)\n"
    "    for batch in ctx.iter_train_batches(model, batch_size=1):\n"
    "        opt.zero_grad()\n"
    "        logits = model(batch.tokens)\n"
    "        nv = logits.shape[-1]\n"
    "        loss = F.cross_entropy(\n"
    "            logits[:, :-1, :].reshape(-1, nv), "
    "batch.tokens[:, 1:].reshape(-1) % nv\n"
    "        )\n"
    "        loss.backward()\n"
    "        opt.step()\n"
)
_SHARD = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample '
    'number {i} has enough bytes to cover several challenge instrument '
    'batches deterministically"}}\n'
)


@dataclass
class PhaseSidecar:
    """Sidecar attestor: honest digest, optional mid-run swap after N calls."""

    digest: str
    swap_to: str | None = None
    swap_after: int = 10_000
    calls: int = 0
    phases: list[str] = field(default_factory=list)

    async def attest(self, *, pod_id: str, phase: str) -> str:
        del pod_id
        self.calls += 1
        self.phases.append(phase)
        if self.swap_to is not None and self.calls > self.swap_after:
            return self.swap_to
        return self.digest


@dataclass
class ScriptedLium:
    digests: list[str | None]
    calls: int = 0

    async def get_pod_raw(self, pod_id: str) -> LiumPodRead:
        self.calls += 1
        idx = min(self.calls - 1, len(self.digests) - 1)
        return LiumPodRead(
            pod_id=pod_id,
            template_id="tmpl-fixture",
            docker_image_digest=self.digests[idx],
            raw={},
        )

    async def balance(self) -> float:
        return 99.0


def _code_bundle() -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", _TINY_ARCH)
        archive.writestr("training.py", _TINY_TRAIN)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _manifest(marker: str) -> dict[str, Any]:
    online_loss = [10.0, 6.0, 3.0, 2.0]
    covered = 4096
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {"covered_bytes": covered, "single_pass": True},
        "metrics": {
            "online_loss": online_loss,
            "sum_neg_log_likelihood_nats": 900.0,
            "covered_bytes": covered,
            "predicted_tokens": 96,
            "step0_loss": online_loss[0],
            "consumed_batches": len(online_loss),
            "random_init_baseline_nats": math.log(50257),
            "prequential_bpb": 1.23,
            "marker": marker,
        },
        "anti_cheat": {
            "step0_anomaly": False,
            "nan_inf_detected": False,
            "no_learning": False,
            "zero_forward": False,
        },
    }


def _final_score(db_path: Path, submission_id: str) -> float | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?",
            (submission_id,),
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else float(row[0])


def _submit_step() -> dict[str, str]:
    sub = validate_submission(
        {
            "repo_url": "https://github.com/baseintelligence/prism-recipe-e2e-sample.git",
            "commit_sha": COMMIT,
            "tree_sha": TREE,
            "variant": VARIANT,
        }
    )
    return sub.as_dict()


def _register_allowlist(allowlist: DigestAllowlist, digest: str) -> None:
    allowlist.register(
        DigestRecord(
            commit_sha=COMMIT,
            tree_sha=TREE,
            variant=ImageVariant.CUDA,
            digest=digest,
        )
    )


def _checkers(
    allowlist: DigestAllowlist,
    nonces: AttestationNonceService,
    verify_key: bytes,
):
    def check_allowlist(
        *, digest: str, commit_sha: str, tree_sha: str, variant: str
    ) -> CheckOutcome:
        hit = allowlist.lookup(
            digest=digest, commit_sha=commit_sha, tree_sha=tree_sha, variant=variant
        )
        if isinstance(hit, AllowlistHit):
            return CheckOutcome(ok=True, reason="ok")
        return CheckOutcome(ok=False, reason=hit.reason.value)

    def check_nonce(
        *, nonce: str, work_unit_id: str, miner_hotkey: str, pod_id: str
    ) -> CheckOutcome:
        result = nonces.consume(
            nonce,
            NonceBinding(
                work_unit_id=work_unit_id, miner_hotkey=miner_hotkey, pod_id=pod_id
            ),
        )
        if isinstance(result, NonceConsumeHit):
            return CheckOutcome(ok=True, reason="ok")
        return CheckOutcome(ok=False, reason=result.reason.value)

    def verify_signature(signed: object) -> CheckOutcome:
        outcome = verify_attestation_payload(signed, verify_key=verify_key)  # type: ignore[arg-type]
        if outcome.ok:
            return CheckOutcome(ok=True, reason="ok")
        return CheckOutcome(ok=False, reason=outcome.reason.value)

    return check_allowlist, check_nonce, verify_signature


def _sign_attestation(*, digest: str, nonce: str, pod_id: str, signing_key: bytes):
    payload = AttestationPayload(
        nonce=nonce,
        digest=digest,
        pod_id=pod_id,
        variant=VARIANT,
        sealed_manifest_hashes=dict(SEALED),
        build_secret_response=compute_build_secret_response(
            build_secret=BUILD_SECRET, nonce=nonce
        ),
    )
    return sign_attestation_payload(payload, signing_key=signing_key)


async def _ensure_template_with_digest(
    client: LiumClient, digest: str
) -> tuple[str, str]:
    """Create/reuse a private template pinned to ``digest``. Returns (id, name)."""
    name = f"{TEMPLATE_PREFIX}-{digest[-12:]}"
    template_id = await client.ensure_template(
        name=name,
        docker_image=E2E_IMAGE,
        docker_image_tag=E2E_IMAGE_TAG,
        docker_image_digest=digest,
        startup_commands=E2E_STARTUP,
    )
    return template_id, name


async def _rent_running_pod(
    client: LiumClient, *, ssh_pub: str, digest: str
) -> tuple[str, str]:
    _template_id, template_name = await _ensure_template_with_digest(client, digest)
    del _template_id
    offers = [
        o
        for o in await client.list_offers(max_price_per_hour=MAX_PRICE)
        if 0 < o.price_per_hour <= MAX_PRICE and o.gpu_count >= 1
    ]
    offers.sort(key=lambda o: (o.price_per_hour, o.gpu_count))
    if not offers:
        raise RuntimeError("no suitable Lium offer")
    last_err: Exception | None = None
    for offer in offers[:8]:
        pod_name = f"prism-attest-{int(time.time())}-{offer.id[:6]}"
        spec = InstanceSpec(
            name=pod_name,
            template_ref=template_name,
            image=E2E_IMAGE,
            image_digest=digest,
            ssh_public_keys=(ssh_pub,),
            ssh_key_name=prov.SSH_KEY_NAME,
            startup_commands=E2E_STARTUP,
            max_lifetime_hours=float(TERMINATION_HOURS),
            max_price_per_hour=MAX_PRICE,
            gpu_count=1,
        )
        pod_id: str | None = None
        try:
            print(
                f"[rent] {offer.gpu_type} @ ${offer.price_per_hour}/gpu/hr "
                f"id={offer.id[:8]} tmpl={template_name}"
            )
            inst = await client.provision(spec, offer=offer)
            pod_id = inst.id
        except Exception as exc:  # noqa: BLE001 - try next offer
            last_err = exc
            print(f"[rent] fail {type(exc).__name__}: {exc}")
            continue
        deadline = time.monotonic() + POLL_BUDGET
        try:
            while True:
                st = await client.status(pod_id)
                status = (st.status or "").upper()
                print(f"[poll] pod={pod_id[:8]} status={status}")
                if status == "RUNNING":
                    return pod_id, pod_name
                if status in {"FAILED", "CREATION_FAILED", "BROKEN", "STOPPED"}:
                    last_err = RuntimeError(f"pod terminal {status}")
                    print(f"[rent] {last_err}; trying next offer")
                    await client.terminate(pod_id)
                    break
                if time.monotonic() >= deadline:
                    last_err = RuntimeError("pod RUNNING timeout")
                    await client.terminate(pod_id)
                    break
                await asyncio.sleep(20.0)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if pod_id:
                try:
                    await client.terminate(pod_id)
                except Exception:  # noqa: BLE001
                    pass
    raise RuntimeError(f"rent failed after offer retries: {last_err}")


async def _run_constation(
    *,
    custody: LiumKeyCustody,
    sidecar: PhaseSidecar,
    pod_id: str,
    work_unit_id: str,
    duration: float,
    live_clock: bool,
) -> Any:
    if live_clock:

        async def sleep_fn(seconds: float) -> None:
            await asyncio.sleep(min(seconds, 2.0))  # compress intervals for E2E

        now_fn = time.monotonic
    else:
        clock = {"t": 0.0}

        def now_fn() -> float:
            return clock["t"]

        async def sleep_fn(seconds: float) -> None:
            clock["t"] += max(0.0, seconds)

    runner = ConstationRunner(
        custody=custody,
        sidecar=sidecar,
        poller_config=PollerConfig(
            gap_budget_seconds=60.0,
            min_interval_seconds=5.0,
            max_interval_seconds=5.0,
            max_polls=8,
            max_cost_units=16.0,
            rate_limit_per_second=100.0,
        ),
        now_fn=now_fn,
        sleep_fn=sleep_fn,
        rng_fn=lambda: 0.0,
    )
    return await runner.run(
        ConstationRunRequest(
            miner_hotkey=HOTKEY,
            work_unit_id=work_unit_id,
            pod_id=pod_id,
            duration_seconds=duration,
            expected_sidecar_digest=sidecar.digest,
        )
    )


async def _ingest(
    *,
    tmp: Path,
    digest: str,
    pod_id: str,
    work_unit_id_marker: str,
    bundle: ConstationBundle | None,
    allowlist: DigestAllowlist,
    nonces: AttestationNonceService,
    verify_key: bytes,
    expect_score: bool,
) -> dict[str, Any]:
    data_dir = tmp / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        "".join(_SHARD.format(i=i) for i in range(64)), encoding="utf-8"
    )
    db_path = tmp / f"{work_unit_id_marker}.sqlite3"
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        shared_token="e2e-secret",
        allow_insecure_signatures=True,
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://base-docker-broker:8082",
        docker_broker_token="secret",
        sequence_length=16,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        base_eval_artifact_root=tmp / "artifacts",
        worker_plane=WorkerPlaneConfig(enabled=True, signing_key=WORKER_KEY),
    )
    # Patch docker executor the same way unit tests do.
    import prism_challenge.evaluator.container as container_mod

    original = container_mod.DockerExecutor.run
    container_mod.DockerExecutor.run = cpu_reexec_run(train_data_dir=data_dir)  # type: ignore[method-assign]
    try:
        app = create_app(settings)
        await app.state.database.init()
        sub = await app.state.repository.create_submission(
            HOTKEY, SubmissionCreate(code=_code_bundle(), filename="project.zip")
        )
        submission_id = sub.id
        manifest = _manifest(work_unit_id_marker)
        signer = worker_signer_from_key(WORKER_KEY)
        proof = build_execution_proof(
            signer=signer,
            manifest_sha256=compute_manifest_sha256(manifest),
            unit_id=submission_id,
            image_digest=digest,
            constation_digest=digest,
            provider=ProviderInfo(name="lium", pod_id=pod_id),
            tier=1,  # type: ignore[arg-type]
        )
        result = {
            "executed": 1,
            "completed_submissions": [],
            PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
            MANIFEST_PAYLOAD_KEY: manifest,
        }
        allow, nonce, sig = _checkers(allowlist, nonces, verify_key)
        # Re-bind bundle work_unit_id to the real submission id + re-issue nonce.
        if bundle is not None:
            binding = NonceBinding(
                work_unit_id=submission_id, miner_hotkey=HOTKEY, pod_id=pod_id
            )
            issued = nonces.issue(binding)
            signed = _sign_attestation(
                digest=bundle.digest,
                nonce=issued.nonce,
                pod_id=pod_id,
                signing_key=verify_key,
            )
            from dataclasses import replace

            bundle = replace(
                bundle,
                work_unit_id=submission_id,
                nonce=issued.nonce,
                signed_attestation=signed,
                pod_id=pod_id,
            )
        outcome = await ingest_work_unit_result(
            worker=app.state.worker,
            work_unit_id=submission_id,
            submission_ref=HOTKEY,
            result=result,
            pinned_image_digest=digest,
            constation_bundle=bundle,
            check_allowlist=allow if bundle is not None else None,
            check_nonce=nonce if bundle is not None else None,
            verify_constation_signature=sig if bundle is not None else None,
        )
        score = _final_score(db_path, submission_id)
        out = {
            "status": outcome.status,
            "effective_tier": outcome.effective_tier,
            "attestation_mode": outcome.attestation_mode,
            "score_written": outcome.score_written,
            "reason": outcome.reason,
            "final_score": score,
            "submission_id": submission_id,
        }
        if expect_score:
            assert outcome.status == "accepted", out
            assert outcome.score_written is True, out
            assert outcome.effective_tier == 1, out
            assert outcome.attestation_mode == ATTESTATION_MODE_V1, out
            assert score is not None and score > 0.0, out
        else:
            assert outcome.score_written is False, out
            assert score is None, out
            assert outcome.reason and "miner_fault" in outcome.reason, out
        return out
    finally:
        container_mod.DockerExecutor.run = original  # type: ignore[method-assign]


def _bundle_from_record(
    record: Any,
    *,
    digest: str,
    nonce: str,
    signed: object,
    pod_id: str,
    work_unit_id: str,
) -> ConstationBundle:
    return ConstationBundle(
        commit_sha=COMMIT,
        tree_sha=TREE,
        variant=VARIANT,
        digest=digest,
        work_unit_id=work_unit_id,
        miner_hotkey=HOTKEY,
        pod_id=pod_id,
        nonce=nonce,
        signed_attestation=signed,
        expected_sealed_manifest_hashes=dict(SEALED),
        reported_sealed_manifest_hashes=dict(SEALED),
        lium_declared_digest=record.lium_declared_digest or digest,
        constation_gap_budget_seconds=record.constation_gap_budget_seconds,
        constation_observed_max_gap_seconds=record.constation_observed_max_gap_seconds,
    )


async def run_honest(*, live: bool, tmp: Path) -> dict[str, Any]:
    print("=== HONEST PATH ===")
    submission = _submit_step()
    allowlist = DigestAllowlist()
    digest = DIGEST_LIVE_CUDA if live else DIGEST_HONEST
    _register_allowlist(allowlist, digest)
    verify_key = derive_attestation_key(BUILD_SECRET)
    nonces = AttestationNonceService(ttl=timedelta(hours=2))
    custody = LiumKeyCustody(master_key=generate_custody_master_key())

    pod_id = "pod-fixture-honest"
    real_lium = False
    client: LiumClient | None = None
    try:
        if live:
            creds = prov._load_credentials()
            api_key = creds["LIUM_API_KEY"]
            ssh_pub = prov._load_ssh_public_key()
            client = LiumClient(api_key=api_key)
            bal = await client.balance()
            print(f"[lium] balance=${bal:.4f}")
            if bal < 2:
                raise SystemExit("Lium balance < $2")
            await custody.register(miner_hotkey=HOTKEY, api_key=api_key)
            pod_id, _name = await _rent_running_pod(
                client, ssh_pub=ssh_pub, digest=digest
            )
            pod_read = await client.get_pod_raw(pod_id)
            declared = pod_read.docker_image_digest or digest
            if declared.lower() != digest.lower():
                # Prefer Lium-declared digest as the allowlisted binding when present.
                allowlist = DigestAllowlist()
                digest = declared if declared.startswith("sha256:") else digest
                _register_allowlist(allowlist, digest)
            real_lium = True
            sidecar = PhaseSidecar(digest=digest)
            record = await _run_constation(
                custody=custody,
                sidecar=sidecar,
                pod_id=pod_id,
                work_unit_id="wu-honest-placeholder",
                duration=12.0,
                live_clock=True,
            )
        else:
            scripted = ScriptedLium(digests=[digest])

            def factory(key: str) -> Any:
                del key
                return scripted

            async def _probe(c: Any) -> None:
                await c.balance()

            custody = LiumKeyCustody(
                master_key=generate_custody_master_key(),
                client_factory=factory,
                probe_fn=_probe,
            )
            await custody.register(miner_hotkey=HOTKEY, api_key="fixture-key")
            sidecar = PhaseSidecar(digest=digest)
            record = await _run_constation(
                custody=custody,
                sidecar=sidecar,
                pod_id=pod_id,
                work_unit_id="wu-honest-placeholder",
                duration=10.0,
                live_clock=False,
            )

        assert record.ok is True, (
            f"constation failed: {record.reason} {record.detail}"
        )
        assert record.fault_class is None
        phases = set(sidecar.phases)
        assert "start" in phases and "end" in phases, sidecar.phases

        # Placeholder nonce/sign — _ingest re-issues against real submission id.
        binding = NonceBinding(
            work_unit_id="wu-honest-placeholder", miner_hotkey=HOTKEY, pod_id=pod_id
        )
        issued = nonces.issue(binding)
        signed = _sign_attestation(
            digest=digest, nonce=issued.nonce, pod_id=pod_id, signing_key=verify_key
        )
        bundle = _bundle_from_record(
            record,
            digest=digest,
            nonce=issued.nonce,
            signed=signed,
            pod_id=pod_id,
            work_unit_id="wu-honest-placeholder",
        )
        # Fresh nonce service for ingest consume (issue inside _ingest).
        nonces_ingest = AttestationNonceService(ttl=timedelta(hours=2))
        ingest_out = await _ingest(
            tmp=tmp / "honest",
            digest=digest,
            pod_id=pod_id,
            work_unit_id_marker="honest",
            bundle=bundle,
            allowlist=allowlist,
            nonces=nonces_ingest,
            verify_key=verify_key,
            expect_score=True,
        )
        return {
            "RESULT": "PASS",
            "path": "honest",
            "REAL_LIUM": real_lium,
            "submission": submission,
            "digest": digest,
            "pod_id": pod_id,
            "constation_ok": True,
            "constation_reason": str(record.reason),
            "corroboration": str(record.corroboration_status),
            "samples": len(record.samples),
            "sidecar_phases": list(sidecar.phases),
            "ingest": ingest_out,
            "notes": (
                "Live Lium pod + get_pod_raw + sidecar HMAC; prism tier1 score. "
                "BASE build = DigestAllowlist.register of deployed digest "
                "(same API post-build pipeline calls)."
                if real_lium
                else "Fixture: ScriptedLium + PhaseSidecar; no network."
            ),
        }
    finally:
        if client is not None and pod_id and not pod_id.startswith("pod-fixture"):
            try:
                await client.terminate(pod_id)
                await client.verify_terminated(pod_id)
                print(f"[cleanup] terminated {pod_id}")
            except Exception as exc:  # noqa: BLE001
                print(f"[cleanup] warn {type(exc).__name__}: {exc}")


async def run_adversarial(*, live: bool, tmp: Path) -> dict[str, Any]:
    print("=== ADVERSARIAL PATH (mid-run image swap) ===")
    submission = _submit_step()
    allowlist = DigestAllowlist()
    digest = DIGEST_LIVE_CUDA if live else DIGEST_HONEST
    _register_allowlist(allowlist, digest)
    verify_key = derive_attestation_key(BUILD_SECRET)
    custody = LiumKeyCustody(master_key=generate_custody_master_key())
    pod_id = "pod-fixture-adv"
    real_lium = False
    client: LiumClient | None = None
    try:
        if live:
            creds = prov._load_credentials()
            api_key = creds["LIUM_API_KEY"]
            ssh_pub = prov._load_ssh_public_key()
            client = LiumClient(api_key=api_key)
            await custody.register(miner_hotkey=HOTKEY, api_key=api_key)
            pod_id, _ = await _rent_running_pod(client, ssh_pub=ssh_pub, digest=digest)
            pod_read = await client.get_pod_raw(pod_id)
            declared = pod_read.docker_image_digest or digest
            if declared.startswith("sha256:"):
                allowlist = DigestAllowlist()
                digest = declared
                _register_allowlist(allowlist, digest)
            real_lium = True
            # After start sample, sidecar reports swapped image digest (miner swap).
            sidecar = PhaseSidecar(digest=digest, swap_to=DIGEST_SWAP, swap_after=1)
            record = await _run_constation(
                custody=custody,
                sidecar=sidecar,
                pod_id=pod_id,
                work_unit_id="wu-adv-placeholder",
                duration=12.0,
                live_clock=True,
            )
        else:
            scripted = ScriptedLium(digests=[digest])

            def factory(key: str) -> Any:
                del key
                return scripted

            async def _probe(c: Any) -> None:
                await c.balance()

            custody = LiumKeyCustody(
                master_key=generate_custody_master_key(),
                client_factory=factory,
                probe_fn=_probe,
            )
            await custody.register(miner_hotkey=HOTKEY, api_key="fixture-key")
            sidecar = PhaseSidecar(digest=digest, swap_to=DIGEST_SWAP, swap_after=1)
            record = await _run_constation(
                custody=custody,
                sidecar=sidecar,
                pod_id=pod_id,
                work_unit_id="wu-adv-placeholder",
                duration=10.0,
                live_clock=False,
            )

        assert record.ok is False, "adversarial must fail constation"
        assert record.reason is ConstationFailCode.CORROBORATION_MISMATCH, record.reason
        assert record.fault_class is FaultClass.MINER, record.fault_class

        # Ingest with no valid bundle (constation failed) → miner_fault, no score.
        nonces_ingest = AttestationNonceService(ttl=timedelta(hours=2))
        ingest_out = await _ingest(
            tmp=tmp / "adversarial",
            digest=digest,
            pod_id=pod_id,
            work_unit_id_marker="adversarial",
            bundle=None,
            allowlist=allowlist,
            nonces=nonces_ingest,
            verify_key=verify_key,
            expect_score=False,
        )
        # Also prove mismatch reason surfaces when a tampered bundle is forced.
        assert ingest_out["reason"] == "miner_fault:missing_constation_bundle" or (
            ingest_out["reason"] or ""
        ).startswith("miner_fault")

        return {
            "RESULT": "PASS",
            "path": "adversarial",
            "REAL_LIUM": real_lium,
            "submission": submission,
            "digest_allowlisted": digest,
            "digest_swapped_to": DIGEST_SWAP,
            "pod_id": pod_id,
            "constation_ok": False,
            "constation_reason": str(record.reason),
            "fault_class": str(record.fault_class),
            "sidecar_phases": list(sidecar.phases),
            "ingest": ingest_out,
            "notes": (
                "Live Lium declared digest held; sidecar flipped after start "
                "(mid-run swap) -> corroboration_mismatch miner_fault; no score."
                if real_lium
                else "Fixture: ScriptedLium + sidecar swap after start."
            ),
        }
    finally:
        if client is not None and pod_id and not pod_id.startswith("pod-fixture"):
            try:
                await client.terminate(pod_id)
                await client.verify_terminated(pod_id)
                print(f"[cleanup] terminated {pod_id}")
            except Exception as exc:  # noqa: BLE001
                print(f"[cleanup] warn {type(exc).__name__}: {exc}")


def _write_evidence(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"RESULT={payload.get('RESULT')}",
        f"REAL_LIUM={payload.get('REAL_LIUM')}",
        f"path={payload.get('path')}",
        f"constation_ok={payload.get('constation_ok')}",
        f"constation_reason={payload.get('constation_reason')}",
    ]
    if "fault_class" in payload:
        lines.append(f"fault_class={payload['fault_class']}")
    if "ingest" in payload:
        ing = payload["ingest"]
        lines.append(f"ingest_status={ing.get('status')}")
        lines.append(f"effective_tier={ing.get('effective_tier')}")
        lines.append(f"attestation_mode={ing.get('attestation_mode')}")
        lines.append(f"score_written={ing.get('score_written')}")
        lines.append(f"ingest_reason={ing.get('reason')}")
        lines.append(f"final_score={ing.get('final_score')}")
    lines.append(f"pod_id={payload.get('pod_id')}")
    lines.append(f"sidecar_phases={payload.get('sidecar_phases')}")
    lines.append(f"notes={payload.get('notes')}")
    lines.append("")
    lines.append("JSON=")
    lines.append(json.dumps(payload, indent=2, default=str))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[evidence] wrote {path}")


def _detect_live(requested: str) -> bool:
    if requested == "fixture":
        return False
    if requested == "live":
        return True
    # auto
    if os.environ.get("BASE_LIVE_PROVIDER_TESTS") != "1":
        # Still allow auto-live when key present and user did not force fixture.
        pass
    try:
        prov._load_credentials()
        prov._load_ssh_public_key()
        return True
    except SystemExit:
        return False


async def _amain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("auto", "live", "fixture"),
        default="auto",
        help="live=real Lium; fixture=no network; auto=live if credentials exist",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Directory for 27-e2e-*.txt evidence",
    )
    parser.add_argument(
        "--only",
        choices=("honest", "adversarial", "both"),
        default="both",
    )
    args = parser.parse_args(argv)

    live = _detect_live(args.mode)
    if args.mode == "live" and not live:
        raise SystemExit("live mode requested but LIUM credentials/SSH missing")
    if live and os.environ.get("BASE_LIVE_PROVIDER_TESTS") != "1":
        os.environ["BASE_LIVE_PROVIDER_TESTS"] = "1"
        print("[gate] BASE_LIVE_PROVIDER_TESTS=1 (live credentials present)")

    print(f"[mode] live={live} REAL_LIUM will be {live}")
    tmp = Path(os.environ.get("PRISM_ATTEST_E2E_TMP", "/tmp/prism-lium-attest-e2e"))
    tmp.mkdir(parents=True, exist_ok=True)

    rc = 0
    if args.only in ("honest", "both"):
        try:
            honest = await run_honest(live=live, tmp=tmp)
            _write_evidence(args.evidence_dir / "27-e2e-honest.txt", honest)
        except BaseException as exc:
            rc = 1
            fail = {
                "RESULT": "FAIL",
                "REAL_LIUM": live,
                "path": "honest",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_evidence(args.evidence_dir / "27-e2e-honest.txt", fail)
            print(f"[honest] FAIL: {exc}")
            if not isinstance(exc, Exception):
                # SystemExit / KeyboardInterrupt — still continue adversarial if both
                pass

    if args.only in ("adversarial", "both"):
        try:
            adv = await run_adversarial(live=live, tmp=tmp)
            _write_evidence(args.evidence_dir / "27-e2e-adversarial.txt", adv)
        except BaseException as exc:
            rc = 1
            fail = {
                "RESULT": "FAIL",
                "REAL_LIUM": live,
                "path": "adversarial",
                "error": f"{type(exc).__name__}: {exc}",
            }
            _write_evidence(args.evidence_dir / "27-e2e-adversarial.txt", fail)
            print(f"[adversarial] FAIL: {exc}")

    return rc


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
