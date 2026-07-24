#!/usr/bin/env python
"""In-pod worker agent runner for live Lium miner attestation.

Runs INSIDE a rented Lium pod and enrolls the miner-funded
:class:`base.worker.runtime.WorkerAgent` against a public master
(``master_url``). The miner-signed binding is pre-signed on the host so the
pod never holds the miner private key. Every emitted ExecutionProof carries
LIUM provider provenance (provider name + the REAL pod id + optional pinned
image digest).

By default this agent uses a Prism CPU-reexec full-envelope executor that
returns ``{execution_proof, manifest_sha256, run_manifest}`` with a scoreable
``prism_run_manifest.v2``. That satisfies Prism worker-plane finalization
(``POST /internal/v1/work_units/result`` must include ``run_manifest``; without
it Prism returns 422 ``manifest_missing``).

When ``prism_recipe`` is installed in-pod (or ``recipe_gate=required``), the
executor also runs the recipe LLM rules gate **before** cpu reexec and attaches
``attestation.llm_gate`` metadata (rules_digest, model, decision, checked_at)
to the result payload / run_manifest extras. Scoring path must not silently
skip the gate without metadata.

Set ``executor_mode`` in the config JSON:

- ``prism_cpu_reexec`` (default when prism_challenge is importable): full envelope
- ``stub``: legacy proof-only path (tests / fallbacks; will 422 on Prism finalize)

CONFIG-DRIVEN (JSON path in ``argv[1]``). Signing uses torch-free
``bittensor_wallet`` when available. NOT for production image packaging alone;
the live path still installs platform + prism_challenge (+ optional prism_recipe)
in-pod.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

try:  # light, torch-free keypair (Rust wheel) preferred inside the pod
    from bittensor_wallet import Keypair
except ImportError:  # pragma: no cover - host dry-run may only have full bittensor
    from bittensor import Keypair  # type: ignore[no-redef]

from base.validator.agent import BrokerConfig
from base.validator.agent.signing import KeypairRequestSigner
from base.worker.coordination_client import WorkerCoordinationClient
from base.worker.executor import (
    StubManifestExecutor,
    WorkerProofExecutor,
    WorkerProvenance,
)
from base.worker.runtime import WorkerAgent, WorkerBinding


def _load_config() -> dict[str, Any]:
    if len(sys.argv) < 2:
        raise SystemExit("usage: pod_worker_agent.py <config.json>")
    return json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))


def _run_recipe_llm_gate(config: dict[str, Any]) -> dict[str, Any] | None:
    """Run prism-recipe LLM rules gate and return attestation metadata.

    Scoring path must never silently skip the gate. Prefer real OpenRouter
    via ``OPENROUTER_API_KEY``. When the key is missing and
    ``recipe_gate_fallback=deterministic_rules_hash`` is set (smoke constraint
    only), return an attested deterministic rules-hash pass so train/envelope
    can continue while still proving the rules files were loaded.
    """
    import os
    from datetime import datetime, timezone

    enabled = str(config.get("recipe_gate") or "").strip().lower()
    if enabled in {"0", "false", "off", "no"}:
        return None
    # Default on when prism_recipe is importable or explicitly requested.
    force = enabled in {"1", "true", "on", "yes", "required"}
    try:
        from prism_recipe.llm_gate import (  # type: ignore[import-not-found]
            run_rules_gate,
            rules_digest,
        )
        from prism_recipe.smoke_train import (  # type: ignore[import-not-found]
            read_sealed_sources,
        )
    except Exception as exc:  # noqa: BLE001
        if force:
            raise SystemExit(
                f"recipe_gate required but prism_recipe import failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        print(
            f"[pod-agent] recipe gate skipped (prism_recipe unavailable: "
            f"{type(exc).__name__})",
            flush=True,
        )
        return None

    arch_src, train_src = read_sealed_sources()
    has_key = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    fallback = str(config.get("recipe_gate_fallback") or "").strip().lower()

    if not has_key and fallback in {
        "deterministic_rules_hash",
        "deterministic",
        "rules_hash",
    }:
        # Smoke constraint only: load rules, hash them, mark ok with attested kind.
        digest = rules_digest()
        checked_at = datetime.now(timezone.utc).isoformat()
        gate_meta = {
            "llm_gate": {
                "rules_digest": digest,
                "model": "deterministic_rules_hash",
                "decision": "pass",
                "ok": True,
                "checked_at": checked_at,
                "reason": "smoke_deterministic_rules_hash_fallback",
                "prompt_hash": digest,
                "prompt_version": "prism-recipe-rules-gate-v1",
                "reason_codes": ["deterministic_rules_hash", "openrouter_key_absent"],
                "kind": "deterministic_rules_hash",
            }
        }
        print(
            f"[pod-agent] recipe gate deterministic fallback ok "
            f"digest={digest[:16]}…",
            flush=True,
        )
        return gate_meta

    gate = run_rules_gate(
        architecture_source=arch_src,
        training_source=train_src,
    )
    meta = gate.attestation_metadata()
    llm = meta.setdefault("llm_gate", {})
    if isinstance(llm, dict) and "kind" not in llm:
        llm["kind"] = "openrouter" if has_key else "missing_key_structured"
    print(
        f"[pod-agent] recipe gate decision={gate.decision} ok={gate.ok} "
        f"digest={gate.rules_digest[:16]}… reason={gate.reason}",
        flush=True,
    )
    if not gate.ok and str(config.get("recipe_gate_block_on_fail") or "1") not in {
        "0",
        "false",
        "off",
        "no",
    }:
        raise RuntimeError(f"recipe_llm_gate_failed:{gate.reason}")
    return meta


class PrismCpuReexecEnvelopeExecutor:
    """AssignmentExecutor that posts a full Prism ExternalResultEnvelope body.

    Mirrors ``packages/challenges/prism/scripts/mission/mission_worker.py``
    ``CpuReexecWorkerExecutor``: run ``evaluate_cpu_reexec``, hash the normalized
    ``prism_run_manifest.v2``, and return payload keys Prism ingestion needs:

    - ``run_manifest``
    - ``manifest_sha256``
    - (optionally pre-built ``execution_proof``; ``WorkerProofExecutor`` fills it
      when missing using worker provenance including real Lium ``pod_id``)
    - optional ``attestation`` (recipe LLM gate metadata when prism-recipe is
      installed in-pod)
    """

    def __init__(self, settings: Any, *, config: dict[str, Any] | None = None) -> None:
        self._settings = settings
        self._config = config or {}

    async def execute(self, context: Any, *, progress: Any) -> Any:
        from base.validator.agent.executor import ExecutionResult
        from base.worker.proof import MANIFEST_SHA256_PAYLOAD_KEY
        from prism_challenge.evaluator.cpu_test_mode import evaluate_cpu_reexec

        unit_id = context.assignment.work_unit_id

        # Recipe LLM gate first (before train/reexec) when enabled/available.
        recipe_attestation = await asyncio.to_thread(
            _run_recipe_llm_gate, self._config
        )

        outcome = await asyncio.to_thread(
            evaluate_cpu_reexec, self._settings, submission_id=unit_id
        )
        print(
            f"[pod-agent] cpu_reexec unit={unit_id} "
            f"schema={outcome.manifest.get('schema_version')} "
            f"sha={outcome.manifest_sha256[:16]}…",
            flush=True,
        )
        payload: dict[str, Any] = {
            MANIFEST_SHA256_PAYLOAD_KEY: outcome.manifest_sha256,
            "run_manifest": outcome.manifest,
        }
        # Keep run_manifest identical to the hashed object; gate metadata
        # lives in sibling payload keys (attestation/metadata) for REPORT.
        if recipe_attestation:
            payload["attestation"] = recipe_attestation
            payload["metadata"] = {
                "recipe_path": "prism-recipe+cpu_reexec",
                **recipe_attestation,
            }
        return ExecutionResult(
            success=True,
            payload=payload,
        )


def _prism_cpu_settings(config: dict[str, Any]) -> Any:
    """Build PrismSettings for in-pod CPU reexec test mode."""
    from prism_challenge.config import PrismSettings
    from prism_challenge.evaluator.cpu_test_mode import configure_cpu_reexec_test_mode

    artifact_root = Path(
        str(config.get("artifact_root") or "/tmp/prism-pod-worker-artifacts")
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    train_data_dir = config.get("cpu_reexec_train_data_dir")
    worker_plane: dict[str, Any] = {
        "enabled": True,
        "cpu_reexec_test_mode": True,
        "cpu_reexec_sequence_length": int(config.get("cpu_reexec_sequence_length", 16)),
        "cpu_reexec_step_budget": int(config.get("cpu_reexec_step_budget", 24)),
        "cpu_reexec_vocab_size": int(config.get("cpu_reexec_vocab_size", 64)),
        "cpu_reexec_seed": int(config.get("cpu_reexec_seed", 1234)),
        "cpu_reexec_train_lines": int(config.get("cpu_reexec_train_lines", 64)),
    }
    if train_data_dir:
        worker_plane["cpu_reexec_train_data_dir"] = str(train_data_dir)
    token = str(config.get("prism_shared_token") or "pod-worker-local-token")
    settings = PrismSettings(
        database_url=f"sqlite+aiosqlite:///{artifact_root}/worker.sqlite3",
        shared_token=token,
        shared_token_file=None,
        allow_insecure_signatures=True,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url=str(
            config.get("docker_broker_url") or "http://127.0.0.1:0"
        ),
        docker_broker_token=token,
        base_eval_artifact_root=artifact_root,
        worker_plane=worker_plane,
    )
    configure_cpu_reexec_test_mode(settings)
    return settings


def _resolve_inner_executor(config: dict[str, Any]) -> Any:
    """Pick full-envelope Prism CPU reexec, or stub if forced / unavailable."""
    mode = str(config.get("executor_mode") or "prism_cpu_reexec").strip().lower()
    if mode in {"stub", "stub_manifest", "proof_only"}:
        print("[pod-agent] executor_mode=stub (proof-only; Prism may 422)", flush=True)
        return StubManifestExecutor()

    try:
        settings = _prism_cpu_settings(config)
    except Exception as exc:  # noqa: BLE001 - fall back only when mode is auto
        if mode in {"auto", ""}:
            print(
                f"[pod-agent] prism_cpu_reexec unavailable ({type(exc).__name__}: {exc}); "
                "falling back to stub",
                flush=True,
            )
            return StubManifestExecutor()
        raise SystemExit(
            f"prism_cpu_reexec executor required but setup failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    print("[pod-agent] executor_mode=prism_cpu_reexec (full envelope)", flush=True)
    return PrismCpuReexecEnvelopeExecutor(settings, config=config)


def _build_agent(config: dict[str, Any]) -> WorkerAgent:
    worker_keypair = Keypair.create_from_uri(config["worker_uri"])
    signer = KeypairRequestSigner(worker_keypair)
    binding = WorkerBinding(
        miner_hotkey=str(config["miner_hotkey"]),
        signature=str(config["binding_signature"]),
        nonce=str(config["binding_nonce"]),
    )
    provenance = WorkerProvenance(
        provider_name=str(config.get("provider", "lium")),
        miner_hotkey=str(config["miner_hotkey"]),
        executor_id=config.get("executor_id"),
        pod_id=str(config["pod_id"]),
        image_digest=config.get("image_digest"),
    )
    inner = _resolve_inner_executor(config)
    # WorkerProofExecutor adds execution_proof with real Lium pod_id provenance
    # when the inner executor does not already attach one.
    executor = WorkerProofExecutor(inner, signer=signer, provenance=provenance)
    master_url = str(config["master_url"])
    client = WorkerCoordinationClient(
        master_url,
        signer,
        timeout_seconds=float(config.get("request_timeout_seconds", 15.0)),
    )
    return WorkerAgent(
        client=client,
        executor=executor,
        broker=BrokerConfig(
            broker_url=str(config.get("broker_url", "http://127.0.0.1:0"))
        ),
        binding=binding,
        provider=str(config.get("provider", "lium")),
        provider_instance_ref=str(config["pod_id"]),
        capabilities=list(config.get("capabilities", ["gpu"])),
        heartbeat_interval_seconds=int(config.get("heartbeat_interval_seconds", 5)),
        poll_interval_seconds=float(config.get("poll_interval_seconds", 2.0)),
    )


async def _run(config: dict[str, Any]) -> None:
    agent = _build_agent(config)
    print(f"[pod-agent] worker_pubkey={agent.worker_pubkey}", flush=True)
    shutdown = asyncio.Event()

    run_seconds = float(config.get("run_seconds", 120.0))

    async def _deadline() -> None:
        await asyncio.sleep(run_seconds)
        shutdown.set()

    worker_id = await agent.register(shutdown)
    print(f"[pod-agent] registered worker_id={worker_id}", flush=True)

    async def _heartbeat() -> None:
        while not shutdown.is_set():
            try:
                await agent.heartbeat_once()
            except Exception as exc:  # noqa: BLE001 - keep heartbeating
                print(f"[pod-agent] heartbeat error: {exc}", flush=True)
            await asyncio.sleep(agent.heartbeat_interval)

    async def _work() -> None:
        while not shutdown.is_set():
            try:
                summary = await agent.process_pending_assignments()
                if summary.pulled:
                    print(
                        f"[pod-agent] cycle pulled={summary.pulled} "
                        f"completed={summary.completed} failed={summary.failed}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001 - keep polling
                print(f"[pod-agent] assignment error: {exc}", flush=True)
            await asyncio.sleep(float(config.get("poll_interval_seconds", 2.0)))

    await asyncio.gather(_deadline(), _heartbeat(), _work())
    print("[pod-agent] shutdown", flush=True)


def main() -> None:
    config = _load_config()
    asyncio.run(_run(config))


if __name__ == "__main__":
    main()
