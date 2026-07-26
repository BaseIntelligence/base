#!/usr/bin/env python3
"""End-to-end Lium image-attestation path (plan checkbox 27).

Wires the REAL product modules:

  submission identity → DigestAllowlist → ConstationRunner/poller
  → prism ``constation_ok`` → fail-closed ``ingest_work_unit_result``
  → tier / ``attestation_mode`` / score-row assertions

Modes
-----
* ``--mode honest`` — matching digests across start/interval/end; expect
  ``constation_ok`` True, score row written, ``effective_tier == 1``,
  ``attestation_mode == miner_rent_image_pin_evidence_v1``.
* ``--mode adversarial`` — mid-run sidecar digest swap (image swap TOCTOU);
  expect runner/constation fail, **no** score row, ``miner_fault:*`` reason.

Live vs offline
---------------
* Live requires ``LIUM_API_KEY`` (and optional ``BASE_LIVE_PROVIDER_TESTS=1``).
  When the key is absent the script runs **offline fixture mode** that still
  exercises the real runner, allowlist, nonce service, ``constation_ok``, and
  prism ingestion — not toy stubs that skip those modules.
* Offline always prints ``REAL_LIUM=false`` and never claims live success.

Evidence header (every run stdout starts with)::

    REAL_LIUM=true|false
    REASON=...
    MODE=honest|adversarial
    RESULT=PASS|FAIL

Exit code 0 on PASS, 1 on FAIL.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import math
import os
import sqlite3
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Path bootstrap so `uv run python scripts/e2e_lium_attestation.py` works
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_ROOT = _SCRIPT_DIR.parent
if str(_BASE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_BASE_ROOT / "src"))
_PRISM_SRC = _BASE_ROOT / "packages" / "challenges" / "prism" / "src"
if _PRISM_SRC.is_dir() and str(_PRISM_SRC) not in sys.path:
    sys.path.insert(0, str(_PRISM_SRC))

from base.compute.attestation_nonce import (  # noqa: E402
    AttestationNonceService,
    NonceBinding,
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
    DigestAllowlist,
    DigestRecord,
    ImageVariant,
)
from base.compute.lium import LiumPodRead  # noqa: E402
from prism_challenge.app import create_app  # noqa: E402
from prism_challenge.config import PrismSettings, WorkerPlaneConfig  # noqa: E402
from prism_challenge.constation import (  # noqa: E402
    ConstationBundle,
    adapt_allowlist_lookup,
    adapt_nonce_consume,
    constation_ok,
)
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

Mode = Literal["honest", "adversarial"]

DIGEST_HONEST = "sha256:" + ("11" * 32)
DIGEST_SWAPPED = "sha256:" + ("22" * 32)
COMMIT_SHA = "a" * 40
TREE_SHA = "b" * 40
VARIANT = ImageVariant.CUDA
HOTKEY = "5E2EMinerHotkeyLiumAttestation000000000001"
POD_ID = "pod-e2e-constation-001"
WORKER_KEY = "//WorkerE2ELiumAttestation"
FIXTURE_API_KEY = "lium-fixture-key-NEVER-LOG-OR-CLAIM-AS-LIVE"

SEALED_MANIFEST = {"src/prism_recipe/harness.py": "c" * 64}

TINY_ARCH = """
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

TINY_TRAIN = """
import torch
import torch.nn.functional as F


def train(ctx):
    model = ctx.build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    for batch in ctx.iter_train_batches(model, batch_size=1):
        opt.zero_grad()
        logits = model(batch.tokens)
        nv = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, nv), batch.tokens[:, 1:].reshape(-1) % nv
        )
        loss.backward()
        opt.step()
"""

_SHARD_LINE = (
    '{{"id": "doc-{i}", "text": "the locked fineweb edu training sample number {i} '
    'has enough bytes to cover several challenge instrument batches deterministically"}}\n'
)


# ---------------------------------------------------------------------------
# Offline Lium / sidecar fixtures (drive REAL ConstationRunner)
# ---------------------------------------------------------------------------


@dataclass
class FakeClock:
    t: float = 0.0

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


@dataclass
class SequenceRng:
    values: list[float]
    i: int = 0

    def __call__(self) -> float:
        if not self.values:
            return 0.0
        v = self.values[self.i % len(self.values)]
        self.i += 1
        return v


@dataclass
class ScriptedLium:
    """Minimal LiumClient stand-in — same surface ConstationRunner polls."""

    digests: list[str | None] = field(default_factory=lambda: [DIGEST_HONEST])
    calls: int = 0

    async def get_pod_raw(self, pod_id: str) -> LiumPodRead:
        self.calls += 1
        idx = min(self.calls - 1, len(self.digests) - 1)
        digest = self.digests[idx]
        return LiumPodRead(
            pod_id=pod_id,
            template_id="tmpl-e2e-1",
            docker_image_digest=digest,
            raw={"id": pod_id, "template": {"docker_image_digest": digest}},
        )

    async def balance(self) -> float:
        return 1.0


@dataclass
class PhaseSidecar:
    """Sidecar attestor; adversarial mode swaps digest after start phase."""

    honest_digest: str = DIGEST_HONEST
    swapped_digest: str = DIGEST_SWAPPED
    adversarial: bool = False
    calls: list[str] = field(default_factory=list)

    async def attest(self, *, pod_id: str, phase: str) -> str:
        del pod_id
        self.calls.append(phase)
        if self.adversarial and phase != "start":
            return self.swapped_digest
        return self.honest_digest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _code_bundle() -> str:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", TINY_ARCH)
        archive.writestr("training.py", TINY_TRAIN)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _manifest(marker: str) -> dict[str, Any]:
    covered_bytes = 4096
    online_loss = [10.0, 6.0, 3.0, 2.0]
    return {
        "schema_version": "prism_run_manifest.v2",
        "data": {"covered_bytes": covered_bytes, "single_pass": True},
        "metrics": {
            "online_loss": online_loss,
            "sum_neg_log_likelihood_nats": 900.0,
            "covered_bytes": covered_bytes,
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


def _stage_train(root: Path, *, lines: int = 64) -> Path:
    data_dir = root / "train-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train-00000.jsonl").write_text(
        "".join(_SHARD_LINE.format(i=i) for i in range(lines)), encoding="utf-8"
    )
    return data_dir


def _final_score(db_path: Path, submission_id: str) -> float | None:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT final_score FROM scores WHERE submission_id=?", (submission_id,)
        ).fetchone()
    finally:
        conn.close()
    return None if row is None else float(row[0])


def _settings(tmp: Path) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp / 'coord.sqlite3'}",
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


def _emit_header(
    *,
    real_lium: bool,
    reason: str,
    mode: Mode,
    result: str,
) -> None:
    print(f"REAL_LIUM={'true' if real_lium else 'false'}")
    print(f"REASON={reason}")
    print(f"MODE={mode}")
    print(f"RESULT={result}")


def _live_available() -> bool:
    return bool(os.environ.get("LIUM_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Core path: allowlist → runner → constation_ok → ingest
# ---------------------------------------------------------------------------


async def _run_constation_runner(*, adversarial: bool) -> Any:
    """Drive real ConstationRunner with fixture Lium + sidecar."""
    scripted = ScriptedLium(digests=[DIGEST_HONEST])

    def factory(key: str) -> Any:
        del key
        return scripted

    async def _probe(client: Any) -> None:
        del client
        await scripted.balance()

    custody = LiumKeyCustody(
        master_key=generate_custody_master_key(),
        client_factory=factory,
    )
    custody.probe_fn = _probe
    verdict = await custody.register(miner_hotkey=HOTKEY, api_key=FIXTURE_API_KEY)
    if not verdict.ok:
        raise RuntimeError(f"fixture custody register failed: {verdict.reason}")

    clock = FakeClock()
    runner = ConstationRunner(
        custody=custody,
        sidecar=PhaseSidecar(adversarial=adversarial),
        poller_config=PollerConfig(
            gap_budget_seconds=60.0,
            min_interval_seconds=5.0,
            max_interval_seconds=5.0,
            max_polls=10,
            max_cost_units=10.0,
            rate_limit_per_second=100.0,
        ),
        now_fn=clock.now,
        sleep_fn=clock.sleep,
        rng_fn=SequenceRng([0.0]),
    )
    return await runner.run(
        ConstationRunRequest(
            miner_hotkey=HOTKEY,
            work_unit_id="wu-placeholder",  # overwritten after submission seed
            pod_id=POD_ID,
            duration_seconds=10.0 if not adversarial else 10.0,
        )
    )


async def run_offline(mode: Mode) -> dict[str, Any]:
    """Full offline path through real product modules. Returns assertion bag."""
    adversarial = mode == "adversarial"
    tmp = Path(tempfile.mkdtemp(prefix=f"e2e-lium-{mode}-"))
    data_dir = _stage_train(tmp)

    # --- 1. Allowlist: BASE-produced digest registration (submission identity) ---
    allowlist = DigestAllowlist()
    allowlist.register(
        DigestRecord(
            commit_sha=COMMIT_SHA,
            tree_sha=TREE_SHA,
            variant=VARIANT,
            digest=DIGEST_HONEST,
        )
    )

    # --- 2. Continuous constation runner/poller (fixture Lium, real runner) ---
    # First pass uses placeholder work_unit; we re-bind nonce to real submission id.
    run_record = await _run_constation_runner(adversarial=adversarial)

    # --- 3. Prism app + submission seed ---
    # Patch DockerExecutor before create_app workers touch it.
    import prism_challenge.evaluator.container as container_mod

    container_mod.DockerExecutor.run = cpu_reexec_run(train_data_dir=data_dir)  # type: ignore[method-assign, assignment]

    settings = _settings(tmp)
    app = create_app(settings)
    await app.state.database.init()
    submission = await app.state.repository.create_submission(
        HOTKEY, SubmissionCreate(code=_code_bundle(), filename="project.zip")
    )
    submission_id = submission.id
    db_path = tmp / "coord.sqlite3"

    # --- 4. Nonce service (real single-use consume) ---
    nonce_svc = AttestationNonceService(ttl=timedelta(hours=1))
    issued = nonce_svc.issue(
        NonceBinding(work_unit_id=submission_id, miner_hotkey=HOTKEY, pod_id=POD_ID)
    )

    def check_allowlist(**kwargs: Any) -> Any:
        return adapt_allowlist_lookup(allowlist.lookup(**kwargs))

    def check_nonce(**kwargs: Any) -> Any:
        return adapt_nonce_consume(
            nonce_svc.consume(
                kwargs["nonce"],
                NonceBinding(
                    work_unit_id=kwargs["work_unit_id"],
                    miner_hotkey=kwargs["miner_hotkey"],
                    pod_id=kwargs["pod_id"],
                ),
            )
        )

    def verify_signature(_signed: object) -> Any:
        from prism_challenge.constation import CheckOutcome

        return CheckOutcome(ok=True, reason="ok")

    # Bundle digest: honest uses runner sidecar; adversarial may have mismatch.
    sidecar_digest = run_record.sidecar_digest or DIGEST_HONEST
    lium_declared = run_record.lium_declared_digest
    # For adversarial mid-run swap the runner fails closed before a clean
    # record; still build a bundle that reflects the swap for constation_ok /
    # ingest fail-closed (corroboration or allowlist miss).
    if adversarial:
        bundle_digest = DIGEST_SWAPPED
        lium_for_bundle: str | None = DIGEST_HONEST  # Lium still declares original
        gap_obs = run_record.constation_observed_max_gap_seconds
        gap_budget = run_record.constation_gap_budget_seconds
    else:
        bundle_digest = sidecar_digest
        lium_for_bundle = lium_declared
        gap_obs = run_record.constation_observed_max_gap_seconds
        gap_budget = run_record.constation_gap_budget_seconds

    bundle = ConstationBundle(
        commit_sha=COMMIT_SHA,
        tree_sha=TREE_SHA,
        variant=VARIANT.value,
        digest=bundle_digest,
        work_unit_id=submission_id,
        miner_hotkey=HOTKEY,
        pod_id=POD_ID,
        nonce=issued.nonce,
        signed_attestation={"sig": "fixture-ok"},
        expected_sealed_manifest_hashes=dict(SEALED_MANIFEST),
        reported_sealed_manifest_hashes=dict(SEALED_MANIFEST),
        lium_declared_digest=lium_for_bundle,
        constation_gap_budget_seconds=gap_budget,
        constation_observed_max_gap_seconds=gap_obs,
    )

    # --- 5. constation_ok (sole elevation predicate) ---
    constation_result = constation_ok(
        bundle,
        check_allowlist=check_allowlist,
        check_nonce=check_nonce,
        verify_signature=verify_signature,
    )

    # --- 6. Execution proof + fail-closed ingest ---
    signer = worker_signer_from_key(WORKER_KEY)
    manifest = _manifest(mode)
    proof = build_execution_proof(
        signer=signer,
        manifest_sha256=compute_manifest_sha256(manifest),
        unit_id=submission_id,
        image_digest=DIGEST_HONEST,
        constation_digest=DIGEST_HONEST if not adversarial else DIGEST_SWAPPED,
        provider=ProviderInfo(name="lium", pod_id=POD_ID),
        tier=1,  # type: ignore[arg-type]
    )
    result_payload = {
        "executed": 1,
        "completed_submissions": [],
        PROOF_PAYLOAD_KEY: proof.model_dump(mode="json"),
        MANIFEST_PAYLOAD_KEY: manifest,
    }

    # Re-issue nonce for ingest (constation_ok already consumed the first).
    issued2 = nonce_svc.issue(
        NonceBinding(work_unit_id=submission_id, miner_hotkey=HOTKEY, pod_id=POD_ID)
    )
    bundle_for_ingest = ConstationBundle(
        commit_sha=bundle.commit_sha,
        tree_sha=bundle.tree_sha,
        variant=bundle.variant,
        digest=bundle.digest,
        work_unit_id=bundle.work_unit_id,
        miner_hotkey=bundle.miner_hotkey,
        pod_id=bundle.pod_id,
        nonce=issued2.nonce,
        signed_attestation=bundle.signed_attestation,
        expected_sealed_manifest_hashes=dict(bundle.expected_sealed_manifest_hashes),
        reported_sealed_manifest_hashes=dict(bundle.reported_sealed_manifest_hashes),
        lium_declared_digest=bundle.lium_declared_digest,
        constation_gap_budget_seconds=bundle.constation_gap_budget_seconds,
        constation_observed_max_gap_seconds=bundle.constation_observed_max_gap_seconds,
    )

    outcome = await ingest_work_unit_result(
        worker=app.state.worker,
        work_unit_id=submission_id,
        submission_ref=HOTKEY,
        result=result_payload,
        pinned_image_digest=DIGEST_HONEST,
        constation_bundle=bundle_for_ingest,
        check_allowlist=check_allowlist,
        check_nonce=check_nonce,
        verify_constation_signature=verify_signature,
    )
    score = _final_score(db_path, submission_id)

    return {
        "mode": mode,
        "run_record_ok": bool(run_record.ok),
        "run_record_reason": str(run_record.reason.value),
        "run_record_fault_class": (
            None if run_record.fault_class is None else str(run_record.fault_class.value)
        ),
        "run_sample_count": len(run_record.samples),
        "constation_ok": bool(constation_result.ok),
        "constation_reason": str(constation_result.reason.value),
        "ingest_status": outcome.status,
        "ingest_reason": outcome.reason,
        "score_written": outcome.score_written,
        "finalized": outcome.finalized,
        "effective_tier": outcome.effective_tier,
        "attestation_mode": outcome.attestation_mode,
        "score_row": score,
        "submission_id": submission_id,
        "tmp": str(tmp),
    }


def _assert_honest(bag: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not bag["run_record_ok"]:
        errors.append(f"runner expected ok, got reason={bag['run_record_reason']}")
    if not bag["constation_ok"]:
        errors.append(f"constation_ok expected True, reason={bag['constation_reason']}")
    if bag["ingest_status"] != "accepted":
        errors.append(f"ingest status={bag['ingest_status']} reason={bag['ingest_reason']}")
    if not bag["score_written"]:
        errors.append("score_written expected True")
    if bag["score_row"] is None:
        errors.append("scores.final_score row missing")
    if bag["effective_tier"] != 1:
        errors.append(f"effective_tier expected 1 got {bag['effective_tier']}")
    if bag["attestation_mode"] != ATTESTATION_MODE_V1:
        errors.append(
            f"attestation_mode expected {ATTESTATION_MODE_V1!r} got {bag['attestation_mode']!r}"
        )
    return errors


def _assert_adversarial(bag: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    # Runner must fail closed on mid-run swap (corroboration mismatch).
    if bag["run_record_ok"]:
        errors.append("runner expected fail on mid-run digest swap")
    if bag["run_record_reason"] != ConstationFailCode.CORROBORATION_MISMATCH.value:
        errors.append(
            f"runner reason expected corroboration_mismatch got {bag['run_record_reason']}"
        )
    if bag["run_record_fault_class"] != FaultClass.MINER.value:
        errors.append(
            f"fault_class expected miner_fault got {bag['run_record_fault_class']}"
        )
    if bag["constation_ok"]:
        errors.append("constation_ok expected False after digest swap")
    if bag["score_written"]:
        errors.append("score_written must be False")
    if bag["score_row"] is not None:
        errors.append(f"score row must be absent, got {bag['score_row']}")
    if bag["ingest_status"] != "rejected":
        errors.append(f"ingest status expected rejected got {bag['ingest_status']}")
    reason = bag["ingest_reason"] or ""
    if not str(reason).startswith("miner_fault:"):
        errors.append(f"ingest reason expected miner_fault:* got {reason!r}")
    return errors


async def run_live(mode: Mode) -> dict[str, Any]:
    """Placeholder live path — only entered when LIUM_API_KEY is set.

    Live pod rent + mid-run image swap is intentionally not fabricated here.
    When credentials exist, operators should extend this to call real LiumClient
    rent/poll/delete (see scripts/live_lium_e2e.py) and feed the same
    allowlist → runner → constation_ok → ingest chain.
    """
    raise NotImplementedError(
        "Live Lium E2E pod cycle not implemented in this environment; "
        "offline fixture path is the supported proof without LIUM_API_KEY. "
        f"mode={mode}"
    )


async def async_main(mode: Mode, *, force_offline: bool) -> int:
    live = _live_available() and not force_offline
    if live:
        try:
            bag = await run_live(mode)
            real_lium = True
            reason = "LIUM_API_KEY present; live path executed"
        except NotImplementedError as exc:
            # Fall back to offline rather than inventing live success.
            print(f"# live path unavailable: {exc}", file=sys.stderr)
            bag = await run_offline(mode)
            real_lium = False
            reason = "no live pod cycle implemented; offline fixtures used"
    else:
        bag = await run_offline(mode)
        real_lium = False
        reason = "no LIUM_API_KEY" if not _live_available() else "force_offline"

    errors = _assert_honest(bag) if mode == "honest" else _assert_adversarial(bag)
    result = "PASS" if not errors else "FAIL"
    _emit_header(real_lium=real_lium, reason=reason, mode=mode, result=result)
    print("---")
    for key in (
        "run_record_ok",
        "run_record_reason",
        "run_record_fault_class",
        "run_sample_count",
        "constation_ok",
        "constation_reason",
        "ingest_status",
        "ingest_reason",
        "score_written",
        "effective_tier",
        "attestation_mode",
        "score_row",
        "submission_id",
    ):
        print(f"{key}={bag.get(key)!r}")
    if errors:
        print("---")
        print("ASSERTION_ERRORS:")
        for err in errors:
            print(f"  - {err}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("honest", "adversarial"),
        required=True,
        help="honest: matching digests → tier 1 + score; adversarial: mid-run swap → miner_fault",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force offline fixture mode even if LIUM_API_KEY is set",
    )
    args = parser.parse_args(argv)
    return asyncio.run(async_main(args.mode, force_offline=args.offline))


if __name__ == "__main__":
    raise SystemExit(main())
