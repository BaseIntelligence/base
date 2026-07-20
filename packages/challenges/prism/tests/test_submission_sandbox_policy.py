from __future__ import annotations

import pytest
from base.challenge_sdk.executor import DockerRunResult

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator.container import ContainerEvaluationError, PrismContainerEvaluator
from prism_challenge.evaluator.interface import PrismContext
from prism_challenge.evaluator.sandbox import SandboxViolation, inspect_code
from prism_challenge.evaluator.source_similarity import SourceFile

CONTRACT = "\ndef build_model(ctx):\n    pass\n\ndef get_recipe(ctx):\n    return {}\n"


def _evidence_for(code: str, *, artifact_path: str = "model.py"):
    with pytest.raises(SandboxViolation) as raised:
        inspect_code(code, artifact_path=artifact_path)
    assert raised.value.evidence_payload() == [item.model_dump() for item in raised.value.evidence]
    return raised.value.evidence[0]


def test_network_access_violation_has_deterministic_evidence() -> None:
    code = "import socket" + CONTRACT

    first = _evidence_for(code, artifact_path="src/model.py")
    second = _evidence_for(code, artifact_path="src/model.py")

    assert first == second
    assert first.rule_id == "prism:no-network"
    assert first.artifact_path == "src/model.py"
    assert first.line == 1
    assert first.ast_node == "Import"
    assert first.snippet_hash == "7773c884b8cd524d37b6b377f43cb5f05899db17d58442afb8508841ccc88991"
    assert "socket" in first.explanation


def test_filesystem_call_violation_has_ast_location_and_hash_only() -> None:
    code = (
        "def build_model(ctx):\n"
        "    return open('/etc/passwd').read()\n\n"
        "def get_recipe(ctx):\n"
        "    return {}\n"
    )

    evidence = _evidence_for(code)

    assert evidence.rule_id == "prism:no-filesystem"
    assert evidence.artifact_path == "model.py"
    assert evidence.line == 2
    assert evidence.ast_node == "Call"
    assert len(evidence.snippet_hash) == 64
    assert "/etc/passwd" not in evidence.model_dump_json()


def test_forbidden_import_and_process_import_rule_ids_are_specific() -> None:
    generic = _evidence_for("import numpy" + CONTRACT, artifact_path="train.py")
    process = _evidence_for("import os" + CONTRACT, artifact_path="model.py")

    assert generic.rule_id == "prism:no-forbidden-import"
    assert generic.artifact_path == "train.py"
    assert process.rule_id == "prism:no-process"


def test_dynamic_process_action_violation_uses_attribute_rule() -> None:
    code = "def build_model(ctx):\n    return ctx.system\n\ndef get_recipe(ctx):\n    return {}\n"

    evidence = _evidence_for(code)

    assert evidence.rule_id == "prism:no-process"
    assert evidence.ast_node == "Attribute"
    assert evidence.line == 2


def test_artifact_size_limit_fails_before_container_run(monkeypatch) -> None:
    def fail_run(self, spec, timeout_seconds):
        raise AssertionError("container should not run for oversized artifacts")

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fail_run)
    evaluator = PrismContainerEvaluator(
        settings=PrismSettings(
            shared_token="secret",
            docker_backend="broker",
            docker_broker_url="http://broker",
            docker_broker_token="token",
            plagiarism_storage_max_bytes=10,
        ),
        ctx=PrismContext(sequence_length=16),
    )
    oversized = SourceFile(
        "artifacts/large.txt",
        "x" * 11,
        "4fc82b26aecb47d2868c4efbe3581732a3e7cbcc6c2efb0f0de03d9367475a7b",
    )

    with pytest.raises(ContainerEvaluationError, match="artifact payload exceeds") as raised:
        evaluator.evaluate(
            submission_id="sub-artifact",
            code="",
            code_hash="code",
            arch_hash="arch",
            backend="base_gpu",
            files=(oversized,),
        )

    evidence = raised.value.evidence[0]
    assert evidence.rule_id == "prism:artifact-size"
    assert evidence.artifact_path == "artifacts/large.txt"
    assert evidence.ast_node == "ArtifactReference.bytes"
    assert raised.value.evidence_payload()[0]["snippet_hash"] == evidence.snippet_hash


def test_timeout_resource_violation_has_deterministic_evidence(monkeypatch) -> None:
    def fake_run(self, spec, timeout_seconds):
        return DockerRunResult("container", "", "", 124, timed_out=True)

    monkeypatch.setattr("prism_challenge.evaluator.container.DockerExecutor.run", fake_run)
    evaluator = PrismContainerEvaluator(
        settings=PrismSettings(
            shared_token="secret",
            docker_backend="broker",
            docker_broker_url="http://broker",
            docker_broker_token="token",
            base_eval_budget_seconds=2,
            base_eval_watchdog_grace_seconds=1,
            base_eval_timeout_seconds=7,
        ),
        ctx=PrismContext(sequence_length=16),
    )

    # The outer docker/broker cap firing means the loop hung past every inner budget; it lands as
    # the wall-clock budget safety cap being exceeded (architecture.md 4.3, 9).
    with pytest.raises(ContainerEvaluationError, match="safety cap") as raised:
        evaluator.evaluate(
            submission_id="sub-timeout",
            code="def build_model(ctx): pass\ndef get_recipe(ctx): return {}",
            code_hash="code",
            arch_hash="arch",
            backend="base_gpu",
        )

    evidence = raised.value.evidence[0]
    assert evidence.rule_id == "prism:budget-exceeded"
    assert evidence.artifact_path == "container://prism-eval"
    assert evidence.ast_node == "DockerRunSpec.timeout_seconds"
