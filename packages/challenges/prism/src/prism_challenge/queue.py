from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from base.challenge_sdk.executor import DockerExecutor, DockerLimits, DockerMount, DockerRunSpec

from .config import PrismSettings
from .db import dumps
from .evaluator import source_similarity
from .evaluator.anti_cheat import evaluate_anti_cheat
from .evaluator.checkpoint_publisher import CheckpointPublisher
from .evaluator.component_signatures import (
    ComponentSemanticSignature,
    build_semantic_signature,
)
from .evaluator.components import (
    PrismComponentFingerprints,
    PrismProjectComponents,
    architecture_name,
    component_fingerprints,
    project_components,
)
from .evaluator.container import InfrastructureEvaluationError, PrismContainerEvaluator
from .evaluator.distributed_contract import (
    check_distributed_contract,
    enforce_single_node_bound,
)
from .evaluator.interface import DEFAULT_TRAINING_ENTRYPOINT, PrismContext
from .evaluator.modes import execution_mode_from_value
from .evaluator.review_rules import ReviewRule, load_review_rules
from .evaluator.sandbox import SandboxViolation, inspect_code
from .evaluator.scoring import ScoreValidationError, score_prequential_bpb
from .evaluator.static_instantiation import check_build_model_static
from .gpu_scheduler import (
    GpuLease,
    GpuLeaseScheduler,
    lease_request_from_runtime,
    targets_from_settings,
)
from .models import SubmissionStatus
from .proof import compute_manifest_sha256, read_manifest_sha256
from .repository import PrismRepository, now_iso

DEFAULT_REVIEW_RULES = (
    ReviewRule("prism:no-secret-exfiltration", "Do not read, infer, print, or transmit secrets."),
    ReviewRule("prism:no-escape", "Do not use filesystem, process, or network escapes."),
    ReviewRule("prism:model-contract", "Only implement the Prism model and recipe contract."),
)
CONTAINER_EXECUTION_BACKENDS = frozenset(
    {"base_container", "base_gpu", "container_gpu", "docker_gpu"}
)
SUPPORTED_EXECUTION_BACKENDS = CONTAINER_EXECUTION_BACKENDS

logger = logging.getLogger(__name__)


class EvalWallTimeExceeded(RuntimeError):
    """Raised when an eval exceeds the orchestration wall-time backstop and is force-killed.

    The inner docker run has its own ``base_eval_hard_timeout_seconds``; this backstop guards the
    orchestration layer so a thread that never returns (hung CUDA call, wedged docker daemon)
    cannot hold its GPU lease forever. On this error the container is reaped and the lease released.
    """


class WorkerFinalizationError(RuntimeError):
    """Raised when worker-plane finalization cannot derive its score inputs from the submission.

    Deriving the deterministic source-static tail (snapshot -> component review -> anti-cheat) from
    the submission SOURCE is an internal prism step, not a worker fault; a failure here is transient
    and MUST NOT look like a clean finalize. The claimed submission is reverted to ``pending``
    (retryable) and this signals ingestion to report the forwarded result as un-finalized rather
    than terminally failed, so a transient derivation error is retried instead of silently sealed.
    """


EvaluatorFactory = Callable[[PrismSettings, PrismContext], PrismContainerEvaluator]


def _default_evaluator_factory(
    settings: PrismSettings, ctx: PrismContext
) -> PrismContainerEvaluator:
    return PrismContainerEvaluator(settings=settings, ctx=ctx)


def _is_v2_run_manifest(manifest: Any) -> bool:
    """True when the container returned a challenge-authored prism_run_manifest.v2 with metrics."""
    return (
        isinstance(manifest, dict)
        and manifest.get("schema_version") == "prism_run_manifest.v2"
        and isinstance(manifest.get("metrics"), dict)
    )


def _metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    """Return the first present, non-None submission metadata value among ``keys``."""
    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


def _optional_real(value: Any) -> float | None:
    """Coerce a manifest scalar to ``float`` for a nullable REAL column, else ``None``."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


@dataclass(frozen=True)
class StaticReviewOutcome:
    code: str
    rejected: bool
    reason: str | None = None
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentReview:
    components: PrismProjectComponents
    fingerprints: PrismComponentFingerprints
    semantic_signature: ComponentSemanticSignature


class PrismWorker:
    def __init__(
        self,
        repository: PrismRepository,
        ctx: PrismContext,
        *,
        execution_backend: str = "base_gpu",
        settings: PrismSettings | None = None,
        evaluator_factory: EvaluatorFactory | None = None,
        checkpoint_publisher: CheckpointPublisher | None = None,
    ) -> None:
        if execution_backend not in SUPPORTED_EXECUTION_BACKENDS:
            raise ValueError(f"Unsupported execution backend: {execution_backend}")
        self.repository = repository
        self.ctx = ctx
        self.execution_backend = execution_backend
        self.settings = settings or PrismSettings()
        self._evaluator_factory = evaluator_factory or _default_evaluator_factory
        self._checkpoint_publisher = checkpoint_publisher

    async def process_next(self) -> str | None:
        submission = await self.repository.claim_next()
        if submission is None:
            return None
        return await self._process_claimed(submission)

    async def process_submission(
        self, submission_id: str, *, resume_checkpoint_ref: str | None = None
    ) -> str | None:
        """Process exactly the submission assigned by the coordination plane.

        Claims the SPECIFIC pending submission (CAS on status) and runs the same re-execution path
        as :meth:`process_next`. A submission that is not pending (already terminal, or in-flight on
        another validator) is a no-op returning ``None``, so a busy validator never starts a second
        run and re-posting a completed assignment never re-dispatches the broker or mutates the
        recorded result. ``resume_checkpoint_ref`` (set on a reassignment) resumes the re-execution
        from the last public HF checkpoint instead of from scratch.
        """
        submission = await self.repository.claim_submission(submission_id)
        if submission is None:
            return None
        return await self._process_claimed(submission, resume_checkpoint_ref=resume_checkpoint_ref)

    async def finalize_worker_result(
        self,
        submission_id: str,
        manifest: dict[str, Any],
    ) -> str | None:
        """Finalize a submission from a forwarded worker manifest WITHOUT re-executing.

        The heavy GPU evaluation already ran on the miner-funded worker; ingestion has already
        verified+reconciled the run (proof + plausibility). This claims the pending submission (a
        CAS, so a duplicate delivery or an already-finalized submission is a no-op returning
        ``None``) and finalizes it from the forwarded ``prism_run_manifest.v2`` alone: the
        challenge-owned prequential bpb (``score_prequential_bpb``) with the deterministic
        source-static tail -- the AST anti-cheat multiplier (``evaluate_anti_cheat`` over the
        submission SOURCE) and the static fingerprints/arch_hash/name from the component review. It
        takes NO GPU lease and NEVER constructs the evaluator (no ``evaluator.evaluate`` /
        ``_evaluate_within_wall_time`` / ``_augment_with_heldout``). The held-out delta is SKIPPED
        (``skip_heldout=True``): score finalizes on the degraded secondary-only band with
        ``emission_crown_eligible=False`` so the path cannot crown emission against honest
        master-side held-out winners, and the master-only secret val split
        (``base_eval_val_data_dir``) is never read (architecture.md 4; VAL-RESLAB-006).

        A failure while deriving the source-static tail is transient and internal (not a worker
        fault), so it does NOT terminally fail the submission: it reverts the claim to ``pending``
        and raises :class:`WorkerFinalizationError` so ingestion reports the result as un-finalized
        and retryable rather than a clean finalize with ``status=failed``.

        Scoring is never gated on TEE evidence (Prism NO TEE residual).
        """
        submission = await self.repository.claim_submission(submission_id)
        if submission is None:
            return None
        code = str(submission["code"])
        filename = str(submission.get("filename") or "model.py")
        raw_metadata = submission.get("metadata")
        metadata = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
        hotkey = str(submission.get("hotkey") or "")
        try:
            snapshot = self._snapshot_from_submission(code, filename, metadata)
            component_review = self._component_review(snapshot)
            code_for_eval = self._entrypoint_code(snapshot, component_review.components.entrypoint)
            arch_hash = (
                component_review.fingerprints.family_hash
                or sha256(code_for_eval.encode()).hexdigest()
            )
            arch_name = architecture_name(component_review.components)
            previous = await self.repository.previous_codes(submission_id)
            anti = evaluate_anti_cheat(
                code_for_eval,
                previous,
                allowed_import_roots=self._local_import_roots(snapshot),
            )
        except Exception as exc:
            # Deriving the score inputs from the submission SOURCE failed. This is an internal,
            # transient prism error (not a worker fault), so it MUST NOT masquerade as a clean
            # finalize: revert the CAS claim to pending so the forwarded result stays retryable and
            # signal ingestion instead of terminally sealing the submission as failed.
            logger.warning(
                "worker-plane finalization could not derive score inputs for %s: %s",
                submission_id,
                exc,
            )
            await self._revert_submission_to_pending(submission_id, str(exc))
            raise WorkerFinalizationError(str(exc)) from exc
        await self._finalize_container_score(
            submission_id=submission_id,
            arch_hash=arch_hash,
            anti=anti,
            manifest=manifest,
            hotkey=hotkey,
            fingerprints=component_review.fingerprints,
            name=arch_name,
            skip_heldout=True,
        )
        return submission_id

    async def replay_audit_manifest_sha256(
        self, submission_id: str, *, resume_checkpoint_ref: str | None = None
    ) -> str | None:
        """Re-execute a finalized submission's evaluation for an audit; return its manifest sha.

        Audits are the sampled minority the validator bears (architecture.md 3.5): the whole
        evaluation is replayed on the validator's OWN broker to obtain an authoritative
        ``prism_run_manifest.v2`` to compare against the audited worker manifest. This path is
        VERIFY-ONLY -- it never claims the submission, writes a score, records an eval job, or
        changes the submission status -- so a passing audit leaves the finalized result untouched.
        The already-passed static / LLM gates are skipped; only the container re-execution is
        repeated (the honest run is deterministic, so an honest worker's hash reproduces). Returns
        ``None`` on any replay failure, resolving the audit inconclusive rather than confirming it.
        """
        if self.execution_backend not in CONTAINER_EXECUTION_BACKENDS:
            return None
        submission = await self.repository.submission_execution_row(submission_id)
        if submission is None:
            return None
        code = str(submission["code"])
        filename = str(submission.get("filename") or "model.py")
        metadata = cast(dict[str, Any], submission["metadata"])
        code_hash = str(submission.get("code_hash") or sha256(code.encode()).hexdigest())
        try:
            snapshot = self._snapshot_from_submission(code, filename, metadata)
            component_review = self._component_review(snapshot)
            code_for_eval = self._entrypoint_code(snapshot, component_review.components.entrypoint)
            arch_hash = (
                component_review.fingerprints.family_hash
                or sha256(code_for_eval.encode()).hexdigest()
            )
            execution_mode = execution_mode_from_value(metadata.get("execution_mode"))
        except Exception:
            return None
        runtime_config = await self.repository.runtime_config(self.settings, official=True)
        score_eligible = metadata.get("score_eligible")
        scheduler = GpuLeaseScheduler(
            self.repository.database, targets_from_settings(self.settings, runtime_config)
        )
        lease = await scheduler.enqueue_or_allocate(
            lease_request_from_runtime(
                submission_id=submission_id,
                job_id=None,
                runtime_policy=runtime_config,
                mode=execution_mode.value,
                score_eligible=bool(score_eligible) if score_eligible is not None else None,
            )
        )
        if not lease.active:
            return None
        effective_settings = self.settings.model_copy(
            update={
                "base_eval_gpu_count": lease.gpu_count,
                "base_eval_gpu_type": runtime_config.gpu_policy.gpu_type,
                "base_eval_gpu_server": lease.target_server,
                "base_eval_gpu_device_ids": lease.device_ids,
            }
        )
        evaluator = self._evaluator_factory(effective_settings, self.ctx)
        if self._checkpoint_publisher is not None and evaluator._checkpoint_publisher is None:
            evaluator._checkpoint_publisher = self._checkpoint_publisher
        attempt = (
            await self.repository.container_job_attempt_count(submission_id, self.execution_backend)
            + 1
        )
        try:
            result = await self._evaluate_within_wall_time(
                evaluator,
                submission_id=submission_id,
                code=code_for_eval,
                code_hash=code_hash,
                arch_hash=arch_hash,
                files=snapshot.files,
                components=component_review.components,
                gpu_lease=lease,
                execution_mode=execution_mode,
                attempt=attempt,
                resume_checkpoint_ref=resume_checkpoint_ref,
            )
        except Exception:
            return None
        finally:
            await asyncio.to_thread(evaluator.reap_job, submission_id)
            await scheduler.release_for_submission(submission_id, "audit replay finished")
        if not _is_v2_run_manifest(result.run_manifest):
            return None
        if result.run_manifest_path:
            try:
                return read_manifest_sha256(result.run_manifest_path)
            except OSError:
                pass
        return compute_manifest_sha256(cast(dict[str, Any], result.run_manifest))

    async def _process_claimed(
        self, submission: dict[str, Any], *, resume_checkpoint_ref: str | None = None
    ) -> str:
        submission_id = str(submission["id"])
        code = str(submission["code"])
        filename = str(submission.get("filename") or "model.py")
        raw_metadata = submission.get("metadata")
        metadata = cast(dict[str, Any], raw_metadata) if isinstance(raw_metadata, dict) else {}
        hotkey = str(submission.get("hotkey") or "")
        code_hash = str(submission.get("code_hash") or sha256(code.encode()).hexdigest())
        if self.execution_backend in CONTAINER_EXECUTION_BACKENDS:
            return await self._process_container(
                submission_id,
                code,
                filename,
                metadata,
                hotkey,
                code_hash,
                resume_checkpoint_ref=resume_checkpoint_ref,
            )
        raise ValueError(f"Unsupported execution backend: {self.execution_backend}")

    async def _process_container(
        self,
        submission_id: str,
        code: str,
        filename: str,
        metadata: dict[str, Any],
        hotkey: str,
        code_hash: str,
        *,
        resume_checkpoint_ref: str | None = None,
    ) -> str:
        # Static gates run FIRST: a sandbox / param-cap / distributed-contract rejection precedes
        # and SKIPS the LLM review entirely -- no llm_reviews/llm_review_events row and no GPU
        # work for a statically-rejected bundle (VAL-LLM-020, VAL-CONTRACT-018).
        try:
            snapshot = self._snapshot_from_submission(code, filename, metadata)
            component_review = self._component_review(snapshot)
            code_for_eval = self._entrypoint_code(snapshot, component_review.components.entrypoint)
            if not code_for_eval.strip():
                await self._reject_submission(submission_id, "submission contains no Python source")
                return submission_id
            report = self._inspect_project_snapshot(snapshot, code_for_eval)
            await self._static_model_instantiation_check(snapshot, component_review)
            self._distributed_contract_check(snapshot, component_review, metadata)
        except (SandboxViolation, SyntaxError) as exc:
            await self._reject_submission(submission_id, str(exc))
            return submission_id
        except Exception as exc:
            await self._reject_submission(submission_id, str(exc))
            return submission_id

        # Deterministic similarity/admission runs only AFTER the static gates have passed.
        # LLM hard-gate approval is removed: no gateway/provider call, no held quarantine.
        try:
            review = await self._review_static_submission(
                submission_id=submission_id,
                snapshot=snapshot,
                component_review=component_review,
                code_for_eval=code_for_eval,
                filename=filename,
                hotkey=hotkey,
                code_hash=code_hash,
            )
            if review.rejected:
                await self._reject_submission(submission_id, review.reason or "review rejected")
                return submission_id
            code = review.code
        except Exception as exc:
            await self._reject_submission(submission_id, str(exc))
            return submission_id

        code_hash = sha256(code.encode()).hexdigest()
        arch_hash = component_review.fingerprints.family_hash
        if not arch_hash:
            arch_hash = sha256(":".join(sorted(report.ast_fingerprint)).encode()).hexdigest()
        # Parsed + moderated miner-declared architecture name (deterministic; consensus-critical).
        arch_name = architecture_name(component_review.components)
        previous = await self.repository.previous_codes(submission_id)
        anti = evaluate_anti_cheat(
            code,
            previous,
            allowed_import_roots=self._local_import_roots(snapshot),
        )
        runtime_config = await self.repository.runtime_config(self.settings, official=True)
        try:
            execution_mode = execution_mode_from_value(metadata.get("execution_mode"))
        except ValueError as exc:
            await self._reject_submission(submission_id, str(exc))
            return submission_id
        score_eligible = metadata.get("score_eligible")
        scheduler = GpuLeaseScheduler(
            self.repository.database, targets_from_settings(self.settings, runtime_config)
        )
        lease = await scheduler.enqueue_or_allocate(
            lease_request_from_runtime(
                submission_id=submission_id,
                job_id=None,
                runtime_policy=runtime_config,
                mode=execution_mode.value,
                score_eligible=bool(score_eligible) if score_eligible is not None else None,
            )
        )
        if not lease.active:
            async with self.repository.database.connect() as conn:
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.PENDING.value, lease.reason, now_iso(), submission_id),
                )
            return submission_id
        effective_settings = self.settings.model_copy(
            update={
                "base_eval_gpu_count": lease.gpu_count,
                "base_eval_gpu_type": runtime_config.gpu_policy.gpu_type,
                "base_eval_gpu_server": lease.target_server,
                "base_eval_gpu_device_ids": lease.device_ids,
            }
        )
        evaluator = self._evaluator_factory(effective_settings, self.ctx)
        if self._checkpoint_publisher is not None and evaluator._checkpoint_publisher is None:
            # The validator downloads a resume checkpoint via the same publisher the master uses;
            # the default factory leaves it unset (lazy real HF client at deploy).
            evaluator._checkpoint_publisher = self._checkpoint_publisher
        attempt = (
            await self.repository.container_job_attempt_count(submission_id, self.execution_backend)
            + 1
        )
        components = component_review.components
        try:
            result = await self._evaluate_within_wall_time(
                evaluator,
                submission_id=submission_id,
                code=code,
                code_hash=code_hash,
                arch_hash=arch_hash,
                files=snapshot.files,
                components=components,
                gpu_lease=lease,
                execution_mode=execution_mode,
                attempt=attempt,
                resume_checkpoint_ref=resume_checkpoint_ref,
            )
            await self._record_container_job(
                submission_id=submission_id,
                status="completed",
                container_name=result.container_name,
                metrics=result.metrics,
                lease=lease,
                artifact_output_path=result.artifact_output_path,
                run_manifest_path=result.run_manifest_path,
                attempt=attempt,
                started_at=result.started_at,
                ended_at=result.ended_at,
            )
            if not _is_v2_run_manifest(result.run_manifest):
                await self._fail_submission(
                    submission_id,
                    "container run produced no challenge-authored prism_run_manifest.v2",
                )
                return submission_id
            await self._finalize_container_score(
                submission_id=submission_id,
                arch_hash=arch_hash,
                anti=anti,
                manifest=cast(dict[str, Any], result.run_manifest),
                hotkey=hotkey,
                fingerprints=component_review.fingerprints,
                name=arch_name,
                artifact_output_path=result.artifact_output_path,
                run_manifest_path=result.run_manifest_path,
            )
            return submission_id
        except InfrastructureEvaluationError as exc:
            await self._record_container_job(
                submission_id=submission_id,
                status="infra_failed",
                container_name=None,
                metrics={},
                error=str(exc),
                lease=lease,
                infra_retryable=True,
                artifact_output_path=exc.artifact_output_path,
                run_manifest_path=exc.run_manifest_path,
                attempt=attempt,
            )
            async with self.repository.database.connect() as conn:
                await conn.execute(
                    "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                    (SubmissionStatus.PENDING.value, str(exc), now_iso(), submission_id),
                )
            return submission_id
        except EvalWallTimeExceeded as exc:
            await self._record_container_job(
                submission_id=submission_id,
                status="failed",
                container_name=None,
                metrics={},
                error=str(exc),
                lease=lease,
                attempt=attempt,
            )
            await self._fail_submission(submission_id, str(exc))
            return submission_id
        except Exception as exc:
            await self._record_container_job(
                submission_id=submission_id,
                status="failed",
                container_name=None,
                metrics={},
                error=str(exc),
                lease=lease,
                attempt=attempt,
            )
            await self._fail_submission(submission_id, str(exc))
            return submission_id
        finally:
            # Reap (force-kill) the eval container FIRST so an overrunning or wedged job stops
            # consuming the GPU, THEN release the lease. Ordering matters: releasing before the
            # kill could hand the device to the next eval while the old process is still resident
            # (architecture.md 4.3, 10; VAL-HARNESS-027). Both steps are best-effort.
            await asyncio.to_thread(evaluator.reap_job, submission_id)
            await scheduler.release_for_submission(submission_id, "container job finished")

    async def _evaluate_within_wall_time(
        self,
        evaluator: PrismContainerEvaluator,
        *,
        submission_id: str,
        code: str,
        code_hash: str,
        arch_hash: str,
        files: tuple[source_similarity.SourceFile, ...],
        components: PrismProjectComponents,
        gpu_lease: GpuLease,
        execution_mode: Any,
        attempt: int,
        resume_checkpoint_ref: str | None = None,
    ) -> Any:
        eval_call = asyncio.to_thread(
            evaluator.evaluate,
            submission_id=submission_id,
            code=code,
            code_hash=code_hash,
            arch_hash=arch_hash,
            backend=self.execution_backend,
            files=files,
            architecture_entrypoint=components.architecture_entrypoint,
            training_entrypoint=components.training_entrypoint,
            build_model_symbol=components.build_model_symbol,
            train_symbol=components.train_symbol,
            gpu_lease=gpu_lease,
            execution_mode=execution_mode,
            attempt=attempt,
            resume_checkpoint_ref=resume_checkpoint_ref,
        )
        timeout = self.settings.resolved_orchestration_timeout_seconds
        try:
            return await asyncio.wait_for(eval_call, timeout=timeout)
        except TimeoutError as exc:
            raise EvalWallTimeExceeded(
                f"eval exceeded orchestration wall-time of {timeout:g}s; container force-killed"
            ) from exc

    def _inspect_project_snapshot(
        self, snapshot: source_similarity.SourceSnapshot, primary_code: str
    ):
        local_imports = self._local_import_roots(snapshot)
        primary_file = next(
            (file for file in snapshot.python_files if file.content == primary_code),
            None,
        )
        report = inspect_code(
            primary_code,
            require_contract=False,
            allowed_import_roots=local_imports,
            artifact_path=primary_file.path if primary_file else "model.py",
        )
        for file in snapshot.python_files:
            if file.content == primary_code:
                continue
            inspect_code(
                file.content,
                require_contract=False,
                allowed_import_roots=local_imports,
                artifact_path=file.path,
            )
        return report

    async def _static_model_instantiation_check(
        self,
        snapshot: source_similarity.SourceSnapshot,
        component_review: ComponentReview,
    ) -> None:
        """Instantiate build_model under the forced seed before any GPU work.

        Rejects non-nn.Module returns, surfaces construction errors cleanly, and bounds hostile
        construction (infinite loop / memory balloon) at the static phase.
        """
        components = component_review.components
        entrypoint = components.architecture_entrypoint or components.entrypoint
        files = {file.path: file.content for file in snapshot.python_files}
        await asyncio.to_thread(
            check_build_model_static,
            files,
            entrypoint,
            ctx=self.ctx,
            build_model_symbol=components.build_model_symbol,
            timeout_seconds=self.settings.static_instantiation_timeout_seconds,
            memory_headroom_bytes=self.settings.static_instantiation_memory_headroom_bytes,
        )

    def _distributed_contract_check(
        self,
        snapshot: source_similarity.SourceSnapshot,
        component_review: ComponentReview,
        metadata: dict[str, Any],
    ) -> None:
        """Multi-GPU static contract (architecture.md section 8), before any GPU work.

        Statically verifies the training script uses the distributed primitives + a rank-0 write
        guard (per ``distributed_contract_policy``) and enforces the single-node bound (reject a
        ``gpu_count > 8`` / multi-node request). Raises SandboxViolation on a violation, which the
        caller converts into a clean ``rejected`` outcome with no GPU lease/job.
        """
        components = component_review.components
        training_entry = components.training_entrypoint or DEFAULT_TRAINING_ENTRYPOINT
        training_code = next(
            (file.content for file in snapshot.python_files if file.path == training_entry),
            "",
        )
        if training_code:
            check_distributed_contract(
                training_code,
                artifact_path=training_entry,
                policy=self.settings.distributed_contract_policy,
            )
        enforce_single_node_bound(
            _metadata_value(metadata, "gpu_count", "num_gpus", "requested_gpu_count", "gpus"),
            num_nodes=_metadata_value(metadata, "num_nodes", "nnodes", "nodes"),
            max_gpu_count=self.settings.base_eval_max_gpu_count,
        )

    def _local_import_roots(self, snapshot: source_similarity.SourceSnapshot) -> set[str]:
        return {
            Path(file.path).stem
            for file in snapshot.python_files
            if Path(file.path).stem != "__init__"
        }

    def _snapshot_from_submission(
        self,
        code: str,
        filename: str,
        metadata: dict[str, Any],
    ) -> source_similarity.SourceSnapshot:
        return source_similarity.snapshot_from_submission(
            code,
            filename,
            metadata,
            max_files=self.settings.plagiarism_storage_max_files,
            max_bytes=self.settings.plagiarism_storage_max_bytes,
        )

    def _component_review(self, snapshot: source_similarity.SourceSnapshot) -> ComponentReview:
        components = project_components(snapshot)
        fingerprints = component_fingerprints(components)
        return ComponentReview(
            components=components,
            fingerprints=fingerprints,
            semantic_signature=build_semantic_signature(components, fingerprints),
        )

    def _entrypoint_code(self, snapshot: source_similarity.SourceSnapshot, entrypoint: str) -> str:
        match = next((file for file in snapshot.files if file.path == entrypoint), None)
        if match is None:
            raise ValueError(f"Prism project entrypoint not found: {entrypoint}")
        return match.content

    async def _review_static_submission(
        self,
        *,
        submission_id: str,
        snapshot: source_similarity.SourceSnapshot,
        component_review: ComponentReview,
        code_for_eval: str,
        filename: str,
        hotkey: str,
        code_hash: str,
    ) -> StaticReviewOutcome:
        # Invoked ONLY after the static AST sandbox / param-cap / distributed-contract gates have
        # passed. Deterministic similarity replaces the removed LLM hard-gate and quarantine hold.
        await self.repository.store_source_snapshot(
            submission_id=submission_id,
            hotkey=hotkey,
            code_hash=code_hash,
            payload=snapshot.to_payload(),
        )
        if not self.settings.plagiarism_enabled:
            return StaticReviewOutcome(code_for_eval, False)
        runtime_config = await self.repository.runtime_config(self.settings, official=True)
        history = await self.repository.source_similarity_candidates(
            exclude_submission_id=submission_id
        )
        duplicate = source_similarity.classify_duplicate(
            submission_id=submission_id,
            code_hash=code_hash,
            snapshot=snapshot,
            architecture_graph=component_review.semantic_signature.architecture_graph,
            rows=history,
            thresholds=runtime_config.duplicate_thresholds.model_dump(),
            top_k=self.settings.plagiarism_top_k,
        )
        if duplicate.candidate is not None:
            # Borderline duplicate formerly became HELD/quarantine. After gateway removal that
            # band is terminally rejected (never held) so no submission needs LLM review.
            rejected = duplicate.rejected or duplicate.held
            violations = ["duplicate_similarity"] if rejected else []
            await self.repository.store_plagiarism_review(
                submission_id=submission_id,
                candidate_submission_id=duplicate.candidate.submission_id,
                similarity=float(duplicate.report["source_similarity"]),
                verdict=rejected,
                reason=duplicate.reason,
                violations=violations,
                report=duplicate.report,
            )
            if rejected:
                return StaticReviewOutcome(
                    code_for_eval,
                    True,
                    reason=duplicate.reason,
                    violations=tuple(violations),
                )
            return StaticReviewOutcome(code_for_eval, False)

        return StaticReviewOutcome(code_for_eval, False)

    async def _reject_submission(self, submission_id: str, reason: str) -> None:
        logger.warning("submission %s rejected: %s", submission_id, reason)
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                (SubmissionStatus.REJECTED.value, reason, now_iso(), submission_id),
            )

    async def _fail_submission(self, submission_id: str, reason: str) -> None:
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                (SubmissionStatus.FAILED.value, reason, now_iso(), submission_id),
            )

    async def _revert_submission_to_pending(self, submission_id: str, reason: str) -> None:
        """Return a claimed (running) submission to ``pending`` so its work unit stays retryable.

        Used when a worker-plane finalization fails for an internal/transient reason: the CAS claim
        set the submission ``running``, and reverting it to ``pending`` (rather than terminally
        ``failed``) lets a redelivered forwarded result re-claim and finalize it.
        """
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "UPDATE submissions SET status=?, error=?, updated_at=? WHERE id=?",
                (SubmissionStatus.PENDING.value, reason, now_iso(), submission_id),
            )

    def _review_rules(self) -> tuple[ReviewRule, ...]:
        return load_review_rules(
            defaults=DEFAULT_REVIEW_RULES,
            rules_json=self.settings.subnet_rules_json,
            rules_file=self.settings.subnet_rules_file,
        )

    def _pair_sandbox_runner(self, submission_id: str) -> source_similarity.SandboxRunner:
        executor = DockerExecutor(
            challenge=self.settings.slug,
            docker_bin=self.settings.docker_bin,
            allowed_images=(self.settings.plagiarism_sandbox_image,),
            backend=self.settings.docker_backend,
            broker_url=self.settings.docker_broker_url,
            broker_token=self.settings.docker_broker_token,
            broker_token_file=str(self.settings.docker_broker_token_file)
            if self.settings.docker_broker_token_file
            else None,
        )

        def run(left: Path, right: Path, script: Path) -> str:
            result = executor.run(
                DockerRunSpec(
                    image=self.settings.plagiarism_sandbox_image,
                    command=("python", "/compare.py"),
                    mounts=(
                        DockerMount(left, "/current"),
                        DockerMount(right, "/candidate"),
                        DockerMount(script, "/compare.py"),
                    ),
                    labels={"base.job": submission_id, "base.task": "plagiarism"},
                    limits=DockerLimits(
                        cpus=min(self.settings.docker_cpus, 1.0),
                        memory="512m",
                        memory_swap="512m",
                        pids_limit=128,
                        network="none",
                        read_only=True,
                    ),
                ),
                self.settings.plagiarism_sandbox_timeout_seconds,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr or result.stdout or "pair sandbox failed")
            return result.stdout

        return run

    async def _record_container_job(
        self,
        *,
        submission_id: str,
        status: str,
        container_name: str | None,
        metrics: dict[str, float],
        error: str | None = None,
        lease: GpuLease | None = None,
        artifact_output_path: str | None = None,
        run_manifest_path: str | None = None,
        infra_retryable: bool = False,
        attempt: int = 0,
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> None:
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "INSERT INTO eval_jobs("
                "id, submission_id, level, status, attempts, external_job_id, metrics, error, "
                "created_at, "
                "updated_at, gpu_lease_id, target_id, target_server, gpu_device_ids, "
                "requested_gpu_count, actual_gpu_count, gpu_mode, gpu_tier, "
                "artifact_output_path, run_manifest_path, started_at, ended_at, infra_retryable) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    submission_id,
                    self.execution_backend,
                    status,
                    attempt,
                    container_name,
                    dumps(metrics),
                    error,
                    now_iso(),
                    now_iso(),
                    lease.id if lease else None,
                    lease.target_id if lease else None,
                    lease.target_server if lease else None,
                    dumps(list(lease.device_ids)) if lease else dumps([]),
                    lease.requested_gpu_count if lease else 0,
                    lease.gpu_count if lease else 0,
                    lease.mode if lease else "",
                    lease.tier if lease else "",
                    artifact_output_path,
                    run_manifest_path,
                    started_at,
                    ended_at,
                    int(infra_retryable),
                ),
            )

    async def _finalize_container_score(
        self,
        *,
        submission_id: str,
        arch_hash: str,
        anti: Any,
        manifest: dict[str, Any],
        hotkey: str = "",
        fingerprints: PrismComponentFingerprints | None = None,
        name: str | None = None,
        skip_heldout: bool = False,
        artifact_output_path: str | None = None,
        run_manifest_path: str | None = None,
    ) -> None:
        """Finalize a container run using the CHALLENGE-OWNED prequential bits-per-byte score.

        The authoritative score is recomputed by ``scoring.score_prequential_bpb`` from the
        challenge-authored ``prism_run_manifest.v2`` (the legacy NAS q_arch/q_recipe derivation and
        the component-reward branching are NOT on this path, so they no longer affect the score).
        A degenerate run that cannot yield a finite/positive bpb is failed rather than scored.

        Alongside the ``scores`` / ``submissions`` rows this also populates the architecture-lab
        tables: it upserts the ``architecture_families`` row (keyed by ``family_hash`` == arch_hash)
        and the ``training_variants`` row (keyed by ``(architecture_id, training_hash)``), and
        persists the loss curve + reconciled compute block into ``submission_curves`` so the data is
        centrally queryable (none of these are inputs to the score).

        ``skip_heldout`` forces the degraded secondary-only scoring path (no held-out primary,
        ``emission_crown_eligible=False``). The worker plane sets it so a forwarded worker manifest
        is graded without ever needing the master-only secret val split and cannot advance
        architecture/training emission crowns (architecture.md 4; VAL-RESLAB-006).

        Scoring is never gated on TEE evidence (Prism NO TEE residual).
        """
        try:
            score = score_prequential_bpb(manifest, skip_heldout=skip_heldout)
        except ScoreValidationError as exc:
            await self._fail_submission(submission_id, f"prequential scoring failed: {exc}")
            return
        anti_multiplier = max(0.0, min(1.0, float(getattr(anti, "multiplier", 1.0))))
        final_score_value = max(0.0, score.final_score * anti_multiplier)
        # Emission crown requires challenge-owned held-out primary (VAL-RESLAB-006). Worker
        # skip_heldout / missing held-out degrades the scalar and must not advance q_arch_best /
        # q_recipe crowns (fail-closed; never invent held-out).
        crown_eligible = bool(getattr(score, "emission_crown_eligible", True)) and (
            final_score_value > 0.0 and not bool(getattr(score, "anomaly", False))
        )
        if skip_heldout or not bool(getattr(score, "emission_crown_eligible", True)):
            crown_eligible = False
        metrics_payload = score.metrics_payload()
        metrics_payload["arch_hash"] = arch_hash
        metrics_payload["emission_crown_eligible"] = crown_eligible
        # Dual ladder stage labels on the scores row (VAL-RESLAB-003); never enter final_score.
        from .evaluator.param_ladder import (
            STAGE_EXPLORE,
            ladder_labels,
            normalize_param_ladder_stage,
            resolve_package_pin,
        )

        stage_raw = None
        model_params_metric = None
        package_pin_metric = None
        for block_key in ("compute", "metrics", "score"):
            block = manifest.get(block_key) if isinstance(manifest, dict) else None
            if not isinstance(block, dict):
                continue
            if stage_raw is None and isinstance(block.get("param_ladder_stage"), str):
                stage_raw = block.get("param_ladder_stage")
            if model_params_metric is None:
                mp = block.get("model_params")
                if isinstance(mp, int) and not isinstance(mp, bool) and mp >= 0:
                    model_params_metric = mp
            if package_pin_metric is None:
                for pin_key in ("package_pin", "architecture_source_hash", "source_hash"):
                    pin_val = block.get(pin_key)
                    if isinstance(pin_val, str) and pin_val.strip():
                        package_pin_metric = pin_val.strip()
                        break
        ctx_stage = getattr(self.ctx, "param_ladder_stage", None)
        stage = normalize_param_ladder_stage(stage_raw or ctx_stage or STAGE_EXPLORE)
        ctx_cap = getattr(self.ctx, "max_parameters", None)
        stage_cap = int(ctx_cap) if isinstance(ctx_cap, int) and ctx_cap > 0 else None
        labels = ladder_labels(
            stage,
            param_count=model_params_metric,
            score_valid=crown_eligible,
            max_parameters=stage_cap,
        )
        package_pin = resolve_package_pin(
            family_hash=arch_hash,
            package_pin=package_pin_metric,
        )
        metrics_payload.update(
            {
                "param_ladder_stage": labels["param_ladder_stage"],
                "param_ladder_cap": labels["param_ladder_cap"],
                "provisional_crown_eligible": bool(
                    labels["provisional_crown_eligible"] and crown_eligible
                ),
                "package_pin": package_pin,
            }
        )
        if model_params_metric is not None:
            metrics_payload.setdefault("model_params", model_params_metric)
        training_hash = fingerprints.training_hash if fingerprints else ""
        arch_fingerprint = (fingerprints.arch_fingerprint if fingerprints else "") or arch_hash
        behavior_fingerprint = (
            fingerprints.behavior_fingerprint if fingerprints else ""
        ) or arch_hash
        now = now_iso()
        # Crown input: only held-out-primary eligible scores advance q_arch_best / q_recipe.
        # Degraded skip_heldout runs still persist scores/metrics for inventory, but use 0.0 as
        # the architecture-family / training-variant crown key so they cannot displace crowns.
        crown_score_value = final_score_value if crown_eligible else 0.0
        async with self.repository.database.connect() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO scores("
                "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus,"
                "penalty, final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    submission_id,
                    final_score_value,
                    0.0,
                    score.anti_cheat_multiplier * anti_multiplier,
                    0.0,
                    0.0,
                    final_score_value,
                    dumps(metrics_payload),
                    now,
                ),
            )
            await conn.execute(
                "UPDATE submissions SET status=?, arch_hash=?, name=?, updated_at=? WHERE id=?",
                (SubmissionStatus.COMPLETED.value, arch_hash, name, now, submission_id),
            )
            architecture_id = await self._upsert_architecture_family(
                conn,
                family_hash=arch_hash,
                arch_fingerprint=arch_fingerprint,
                behavior_fingerprint=behavior_fingerprint,
                owner_hotkey=hotkey,
                submission_id=submission_id,
                final_score=crown_score_value,
                display_name=name,
                now=now,
                param_ladder_stage=stage,
                package_pin=package_pin,
                crown_score_valid=crown_eligible,
            )
            await self._upsert_training_variant(
                conn,
                architecture_id=architecture_id,
                training_hash=training_hash or arch_hash,
                owner_hotkey=hotkey,
                submission_id=submission_id,
                final_score=crown_score_value,
                now=now,
                param_ladder_stage=stage,
                package_pin=package_pin,
                crown_score_valid=crown_eligible,
            )
            await self._persist_submission_curve(
                conn,
                submission_id=submission_id,
                manifest=manifest,
                now=now,
                artifact_output_path=artifact_output_path,
                run_manifest_path=run_manifest_path,
            )

    async def _upsert_architecture_family(
        self,
        conn: Any,
        *,
        family_hash: str,
        arch_fingerprint: str,
        behavior_fingerprint: str,
        owner_hotkey: str,
        submission_id: str,
        final_score: float,
        display_name: str | None,
        now: str,
        param_ladder_stage: str = "explore",
        package_pin: str = "",
        crown_score_valid: bool = True,
    ) -> str:
        """Upsert the family keyed by ``family_hash``; returns the stable ``architecture_id``.

        Owner / owner_submission_id / display_name stay stable to the family-creating submission;
        the canonical (best) submission + ``q_arch_best`` advance only when a higher final_score
        arrives. Crown status follows the dual ladder
        (explore provisional → promote confirm/revoke).
        """
        from .evaluator.param_ladder import (
            CROWN_STATUS_NONE,
            resolve_crown_status_transition,
            resolve_package_pin,
        )

        pin = resolve_package_pin(family_hash=family_hash, package_pin=package_pin or None)
        rows = await conn.execute_fetchall(
            "SELECT id, q_arch_best, "
            "COALESCE(crown_status, 'none') AS crown_status, "
            "COALESCE(param_ladder_stage, 'explore') AS param_ladder_stage, "
            "COALESCE(package_pin, '') AS package_pin "
            "FROM architecture_families WHERE family_hash=?",
            (family_hash,),
        )
        if rows:
            row = list(rows)[0]
            architecture_id = str(row[0])
            previous_score = float(row[1])
            previous_status = str(row[2] or CROWN_STATUS_NONE)
            previous_stage = str(row[3] or "explore")
            previous_pin = str(row[4] or "") or None
            score_beats = None
            if crown_score_valid and previous_score > 0.0:
                score_beats = final_score >= previous_score
            elif crown_score_valid:
                score_beats = True
            else:
                score_beats = False
            new_status = resolve_crown_status_transition(
                previous_status=previous_status,
                previous_stage=previous_stage,
                previous_pin=previous_pin,
                incoming_stage=param_ladder_stage,
                incoming_pin=pin,
                score_valid=bool(crown_score_valid and final_score > 0.0),
                score_beats_previous=score_beats,
            )
            if final_score > previous_score:
                await conn.execute(
                    "UPDATE architecture_families SET canonical_submission_id=?, q_arch_best=?, "
                    "crown_status=?, param_ladder_stage=?, package_pin=?, "
                    "updated_at=? WHERE id=?",
                    (
                        submission_id,
                        final_score,
                        new_status,
                        param_ladder_stage,
                        pin,
                        now,
                        architecture_id,
                    ),
                )
            else:
                await conn.execute(
                    "UPDATE architecture_families SET crown_status=?, param_ladder_stage=?, "
                    "package_pin=CASE WHEN ? != '' THEN ? ELSE package_pin END, "
                    "updated_at=? WHERE id=?",
                    (
                        new_status,
                        param_ladder_stage,
                        pin,
                        pin,
                        now,
                        architecture_id,
                    ),
                )
            return architecture_id
        architecture_id = str(uuid4())
        new_status = resolve_crown_status_transition(
            previous_status=CROWN_STATUS_NONE,
            previous_stage=None,
            previous_pin=None,
            incoming_stage=param_ladder_stage,
            incoming_pin=pin,
            score_valid=bool(crown_score_valid and final_score > 0.0),
            score_beats_previous=True,
        )
        await conn.execute(
            "INSERT INTO architecture_families("
            "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey,"
            "owner_submission_id, canonical_submission_id, q_arch_best, display_name,"
            "created_at, updated_at, crown_status, param_ladder_stage, package_pin) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                architecture_id,
                family_hash,
                arch_fingerprint,
                behavior_fingerprint,
                owner_hotkey,
                submission_id,
                submission_id,
                final_score,
                display_name,
                now,
                now,
                new_status,
                param_ladder_stage,
                pin,
            ),
        )
        return architecture_id

    async def _upsert_training_variant(
        self,
        conn: Any,
        *,
        architecture_id: str,
        training_hash: str,
        owner_hotkey: str,
        submission_id: str,
        final_score: float,
        now: str,
        param_ladder_stage: str = "explore",
        package_pin: str = "",
        crown_score_valid: bool = True,
    ) -> None:
        """Upsert the variant keyed by ``(architecture_id, training_hash)``.

        The representative submission / owner / score advance only when a better final_score
        arrives, then ``is_current_best`` is recomputed across the architecture's variants
        (highest score wins, ties broken by earliest creation then id) so exactly one is flagged.
        Crown status follows explore provisional / promote confirm-revoke (VAL-RESLAB-004/005).
        """
        from .evaluator.param_ladder import (
            CROWN_STATUS_NONE,
            resolve_crown_status_transition,
            resolve_package_pin,
        )

        pin = resolve_package_pin(
            family_hash=f"{architecture_id}:{training_hash}",
            package_pin=package_pin or None,
        )
        rows = await conn.execute_fetchall(
            "SELECT id, q_recipe, "
            "COALESCE(crown_status, 'none') AS crown_status, "
            "COALESCE(param_ladder_stage, 'explore') AS param_ladder_stage, "
            "COALESCE(package_pin, '') AS package_pin "
            "FROM training_variants "
            "WHERE architecture_id=? AND training_hash=?",
            (architecture_id, training_hash),
        )
        if rows:
            row = list(rows)[0]
            variant_id = str(row[0])
            previous_score = float(row[1])
            previous_status = str(row[2] or CROWN_STATUS_NONE)
            previous_stage = str(row[3] or "explore")
            previous_pin = str(row[4] or "") or None
            score_beats = None
            if crown_score_valid and previous_score > 0.0:
                score_beats = final_score >= previous_score
            elif crown_score_valid:
                score_beats = True
            else:
                score_beats = False
            new_status = resolve_crown_status_transition(
                previous_status=previous_status,
                previous_stage=previous_stage,
                previous_pin=previous_pin,
                incoming_stage=param_ladder_stage,
                incoming_pin=pin,
                score_valid=bool(crown_score_valid and final_score > 0.0),
                score_beats_previous=score_beats,
            )
            if final_score > previous_score:
                await conn.execute(
                    "UPDATE training_variants SET owner_hotkey=?, submission_id=?, q_recipe=?, "
                    "metric_mean=?, metric_std=?, crown_status=?, param_ladder_stage=?, "
                    "package_pin=?, updated_at=? WHERE id=?",
                    (
                        owner_hotkey,
                        submission_id,
                        final_score,
                        final_score,
                        0.0,
                        new_status,
                        param_ladder_stage,
                        pin,
                        now,
                        variant_id,
                    ),
                )
            else:
                await conn.execute(
                    "UPDATE training_variants SET crown_status=?, param_ladder_stage=?, "
                    "package_pin=CASE WHEN ? != '' THEN ? ELSE package_pin END, "
                    "updated_at=? WHERE id=?",
                    (
                        new_status,
                        param_ladder_stage,
                        pin,
                        pin,
                        now,
                        variant_id,
                    ),
                )
        else:
            new_status = resolve_crown_status_transition(
                previous_status=CROWN_STATUS_NONE,
                previous_stage=None,
                previous_pin=None,
                incoming_stage=param_ladder_stage,
                incoming_pin=pin,
                score_valid=bool(crown_score_valid and final_score > 0.0),
                score_beats_previous=True,
            )
            await conn.execute(
                "INSERT INTO training_variants("
                "id, architecture_id, training_hash, owner_hotkey, submission_id, q_recipe,"
                "metric_mean, metric_std, is_current_best, created_at, updated_at, "
                "crown_status, param_ladder_stage, package_pin) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    architecture_id,
                    training_hash,
                    owner_hotkey,
                    submission_id,
                    final_score,
                    final_score,
                    0.0,
                    0,
                    now,
                    now,
                    new_status,
                    param_ladder_stage,
                    pin,
                ),
            )
        await conn.execute(
            "UPDATE training_variants SET is_current_best=0 WHERE architecture_id=?",
            (architecture_id,),
        )
        await conn.execute(
            "UPDATE training_variants SET is_current_best=1 WHERE id=("
            "SELECT id FROM training_variants WHERE architecture_id=? "
            "AND COALESCE(crown_status, 'none') IN ('none', 'provisional', 'confirmed') "
            "ORDER BY q_recipe DESC, created_at ASC, id ASC LIMIT 1)",
            (architecture_id,),
        )

    async def _persist_submission_curve(
        self,
        conn: Any,
        *,
        submission_id: str,
        manifest: dict[str, Any],
        now: str,
        artifact_output_path: str | None = None,
        run_manifest_path: str | None = None,
    ) -> None:
        """Persist the loss curve + train series + reconciled compute block centrally.

        ``online_loss`` remains the legacy CE series. ``train_series`` stores the full
        challenge-owned ``prism_train_series.v1`` document (grad_norm / clip / wall) when the
        runner authored it; miner self-reports never enter this column (VAL-TELE-002..006).
        """
        metrics_block = manifest.get("metrics")
        metrics = metrics_block if isinstance(metrics_block, dict) else {}
        data_block = manifest.get("data")
        data = data_block if isinstance(data_block, dict) else {}
        compute_block = manifest.get("compute")
        compute = compute_block if isinstance(compute_block, dict) else {}
        online_loss = metrics.get("online_loss") or []
        covered_bytes_cumulative = data.get("covered_bytes_cumulative") or []
        train_series_payload = self._load_train_series_for_persist(
            manifest=manifest,
            metrics=metrics,
            artifact_output_path=artifact_output_path,
            run_manifest_path=run_manifest_path,
        )
        await conn.execute(
            "INSERT OR REPLACE INTO submission_curves("
            "submission_id, online_loss, covered_bytes_cumulative, step0_loss, baseline_nats,"
            "compute, train_series, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                submission_id,
                dumps(online_loss),
                dumps(covered_bytes_cumulative),
                _optional_real(metrics.get("step0_loss")),
                _optional_real(metrics.get("random_init_baseline_nats")),
                dumps(compute),
                dumps(train_series_payload) if train_series_payload is not None else None,
                now,
            ),
        )

    def _load_train_series_for_persist(
        self,
        *,
        manifest: dict[str, Any],
        metrics: dict[str, Any],
        artifact_output_path: str | None = None,
        run_manifest_path: str | None = None,
    ) -> dict[str, Any] | None:
        """Load challenge-owned train series for ``submission_curves.train_series``.

        Resolution order (worker-plane safe):

        1. Inline ``metrics.train_series`` when ``authority=challenge`` and (if present)
           ``metrics.train_series_sha256`` matches. Worker-plane finalize has no local
           ``artifact_output_path`` / ``run_manifest_path``; the challenge container embeds
           the series body in the forwarded ``prism_run_manifest.v2`` so this path is the
           primary one for queue/worker persist (online_loss already travels the same way).
        2. On-disk side-car referenced by ``metrics.train_series_path`` / artifacts, verified
           by sha256 when a digest pointer exists (local/container re-exec path).

        Miner-authored series that fail authority or digest checks are ignored (VAL-TELE-006).
        """
        import json

        from .evaluator.train_series import (
            load_challenge_series,
            series_is_challenge_owned,
            train_series_sha256,
        )

        artifacts = manifest.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, dict) else {}
        expected = metrics.get("train_series_sha256") or artifacts.get("train_series_sha256")

        inline = metrics.get("train_series")
        if isinstance(inline, dict) and series_is_challenge_owned(inline):
            if isinstance(expected, str) and expected:
                if train_series_sha256(inline) == expected:
                    return inline
                # Pointer disagrees with embedded body: defense-in-depth, try files next.
            else:
                # Challenge-owned embed without a digest pointer (older or test fixtures).
                return inline

        path_name = metrics.get("train_series_path") or artifacts.get("train_series")

        candidate_dirs: list[Path] = []
        if isinstance(run_manifest_path, str) and run_manifest_path:
            candidate_dirs.append(Path(run_manifest_path).parent)
        if isinstance(artifact_output_path, str) and artifact_output_path:
            candidate_dirs.append(Path(artifact_output_path))

        for root in candidate_dirs:
            if isinstance(path_name, str) and path_name:
                series_path = root / path_name
                if series_path.is_file():
                    try:
                        raw = series_path.read_bytes()
                        payload = json.loads(raw.decode("utf-8"))
                    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
                        payload = None
                    if isinstance(payload, dict):
                        if isinstance(expected, str) and expected:
                            if train_series_sha256(raw) != expected:
                                # Miner plant that does not match the challenge digest: ignore.
                                payload = None
                        if payload is not None and series_is_challenge_owned(payload):
                            return payload
            loaded = load_challenge_series(
                root,
                expected_sha256=expected if isinstance(expected, str) else None,
            )
            if loaded is not None:
                return loaded
        return None
