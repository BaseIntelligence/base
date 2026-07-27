"""Dual-gate plagiarism tests: deterministic ranker + OpenRouter LLM sole verdict."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from prism_challenge.config import PrismSettings
from prism_challenge.evaluator import plagiarism_adjudicator as adj
from prism_challenge.evaluator import source_similarity
from prism_challenge.evaluator.plagiarism_adjudicator import (
    PlagiarismAdjudication,
    PlagiarismLlmConfig,
    adjudicate_plagiarism,
    build_comparison_prompt,
    config_from_settings,
)
from prism_challenge.evaluator.source_similarity import (
    DuplicatePolicyDecision,
    SimilarityCandidate,
    SourceSnapshot,
)


class _FakeClient:
    def __init__(self, payload: dict[str, Any] | Exception) -> None:
        self.payload = payload
        self.calls = 0
        self.last_kwargs: dict[str, Any] | None = None

    def complete(self, **kwargs: Any) -> dict[str, Any]:
        self.calls += 1
        self.last_kwargs = kwargs
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def _tool_payload(*, plagiarized: bool, reason: str, confidence: float = 0.9) -> dict[str, Any]:
    import json

    args = json.dumps(
        {
            "reason": reason,
            "plagiarized": plagiarized,
            "confidence": confidence,
            "violations": ["same_architecture_no_change"] if plagiarized else [],
        }
    )
    return {
        "model": "openai/gpt-4o",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "SubmitPlagiarismVerdict",
                                "arguments": args,
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {"total_tokens": 100},
    }


ARCH_A = """
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 32)
        self.out = nn.Linear(32, vocab_size)

    def forward(self, x):
        return self.out(self.emb(x))

def build_model(ctx):
    return Model(ctx.vocab_size)
"""

TRAIN_A = """
def train(ctx):
    model = build_model(ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for batch in ctx.train_batches():
        loss = model(batch).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return None
"""


def test_adjudicator_copy_rejects_via_llm() -> None:
    client = _FakeClient(
        _tool_payload(plagiarized=True, reason="identical architecture and training loop")
    )
    result = adjudicate_plagiarism(
        current_code=ARCH_A + "\n" + TRAIN_A,
        candidate_code=ARCH_A + "\n" + TRAIN_A,
        comparison_report={"source_similarity": 0.99, "graph_similarity": 1.0},
        deterministic_reason="borderline",
        deterministic_outcome="quarantine",
        candidate_submission_id="cand-1",
        config=PlagiarismLlmConfig(enabled=True, required=True, api_key="sk-test"),
        client=client,
    )
    assert result.used_llm is True
    assert result.plagiarized is True
    assert result.rejected is True
    assert client.calls == 1


def test_adjudicator_novel_allows_via_llm() -> None:
    client = _FakeClient(_tool_payload(plagiarized=False, reason="independent design"))
    result = adjudicate_plagiarism(
        current_code="novel arch",
        candidate_code=ARCH_A,
        comparison_report={"source_similarity": 0.87},
        deterministic_reason="borderline",
        deterministic_outcome="quarantine",
        config=PlagiarismLlmConfig(enabled=True, required=True, api_key="sk-test"),
        client=client,
    )
    assert result.used_llm is True
    assert result.plagiarized is False


def test_adjudicator_fail_closed_without_key() -> None:
    result = adjudicate_plagiarism(
        current_code="a",
        candidate_code="b",
        comparison_report={},
        deterministic_reason="borderline",
        deterministic_outcome="quarantine",
        config=PlagiarismLlmConfig(
            enabled=True, required=True, api_key=None, api_key_file="/nonexistent/key"
        ),
    )
    assert result.plagiarized is True
    assert result.used_llm is False
    assert "api key" in result.reason.lower()


def test_adjudicator_fail_closed_on_provider_error() -> None:
    client = _FakeClient(RuntimeError("429 rate limit"))
    result = adjudicate_plagiarism(
        current_code="a",
        candidate_code="b",
        comparison_report={},
        deterministic_reason="borderline",
        deterministic_outcome="quarantine",
        config=PlagiarismLlmConfig(enabled=True, required=True, api_key="sk-test", max_retries=0),
        client=client,
    )
    assert result.plagiarized is True
    assert "fail-closed" in result.reason.lower()
    assert "plagiarism_llm_failed" in result.violations


def test_prompt_includes_both_sources_and_report() -> None:
    prompt = build_comparison_prompt(
        current_code="CURRENT_MARKER_XYZ",
        candidate_code="CANDIDATE_MARKER_ABC",
        comparison_report={"score": 0.91},
        deterministic_reason="borderline source",
        deterministic_outcome="quarantine",
        max_chars=10_000,
    )
    assert "CURRENT_MARKER_XYZ" in prompt
    assert "CANDIDATE_MARKER_ABC" in prompt
    assert "quarantine" in prompt
    assert "0.91" in prompt


def test_config_from_settings_reads_openrouter_fields(tmp_path: Path) -> None:
    key_file = tmp_path / "openrouter_api_key"
    key_file.write_text("sk-or-test-key\n", encoding="utf-8")
    settings = PrismSettings(
        shared_token="token",
        allow_insecure_signatures=True,
        plagiarism_llm_enabled=True,
        plagiarism_llm_required=True,
        openrouter_api_key_file=str(key_file),
        openrouter_model="openai/gpt-4o-mini",
        openrouter_base_url="https://openrouter.ai/api/v1",
    )
    cfg = config_from_settings(settings)
    assert cfg.enabled is True
    assert cfg.required is True
    assert cfg.model == "openai/gpt-4o-mini"
    assert adj.resolve_api_key(cfg) == "sk-or-test-key"


def test_exact_hash_still_hard_rejects_without_llm() -> None:
    code = ARCH_A + "\n" + TRAIN_A
    code_hash = sha256(code.encode()).hexdigest()
    snap_payload = {
        "files": [{"path": "architecture.py", "content": code, "sha256": code_hash}],
        "ast_features": ["Module", "ClassDef:Model"],
        "token_shingles": ["import torch", "class Model"],
        "fingerprint": "fp1",
    }
    snap = SourceSnapshot.from_payload(snap_payload)
    history = [
        {
            "submission_id": "prior-1",
            "hotkey": "hk-prior",
            "code_hash": code_hash,
            "files": snap_payload["files"],
            "ast_features": snap_payload["ast_features"],
            "token_shingles": snap_payload["token_shingles"],
            "fingerprint": "fp1",
            "architecture_graph": {"nodes": ["Model"], "edges": []},
            "architecture_graph_hash": "g1",
        }
    ]
    decision = source_similarity.classify_duplicate(
        submission_id="new-1",
        code_hash=code_hash,
        snapshot=snap,
        architecture_graph={"nodes": ["Model"], "edges": []},
        rows=history,
        thresholds=None,
        top_k=2,
    )
    assert decision.rejected is True
    assert "exact source hash" in decision.reason


@pytest.mark.asyncio
async def test_worker_quarantine_defers_to_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prism_challenge.db import Database
    from prism_challenge.queue import PrismWorker
    from prism_challenge.repository import PrismRepository

    calls: list[str] = []

    def _fake_adj(**kwargs: Any) -> PlagiarismAdjudication:
        calls.append("llm")
        return PlagiarismAdjudication(
            plagiarized=False,
            reason="llm says independent",
            confidence=0.8,
            used_llm=True,
            candidate_submission_id=kwargs.get("candidate_submission_id"),
        )

    monkeypatch.setattr(
        "prism_challenge.queue.adjudicate_plagiarism",
        _fake_adj,
    )

    settings = PrismSettings(
        database_path=tmp_path / "p.sqlite3",
        shared_token="secret",
        allow_insecure_signatures=True,
        plagiarism_enabled=True,
        plagiarism_llm_enabled=True,
        plagiarism_llm_required=True,
        openrouter_api_key="sk-test",
        distributed_contract_policy="off",
    )
    db = Database(tmp_path / "p.sqlite3")
    await db.init()
    repo = PrismRepository(db, epoch_seconds=settings.epoch_seconds)

    # PrismContext import retained for type surface; process path uses bare worker.

    # PrismContext construction may need many fields - avoid process path and call review only.
    # Build worker with a bare context via object.__new__ if needed.
    worker = object.__new__(PrismWorker)
    worker.repository = repo
    worker.settings = settings
    worker.ctx = None  # type: ignore[assignment]
    worker.execution_backend = "base_gpu"

    snap = SourceSnapshot.from_payload(
        {
            "files": [
                {"path": "architecture.py", "content": ARCH_A, "sha256": "a" * 64},
                {"path": "training.py", "content": TRAIN_A, "sha256": "b" * 64},
            ],
            "ast_features": ["A"],
            "token_shingles": ["t1"],
            "fingerprint": "f",
        }
    )
    cand_snap = SourceSnapshot.from_payload(
        {
            "files": [
                {"path": "architecture.py", "content": ARCH_A + "\n# n", "sha256": "c" * 64},
                {"path": "training.py", "content": TRAIN_A, "sha256": "d" * 64},
            ],
            "ast_features": ["A"],
            "token_shingles": ["t1"],
            "fingerprint": "f2",
        }
    )
    forced = DuplicatePolicyDecision(
        outcome="quarantine",
        reason="borderline source or semantic graph similarity requires review",
        candidate=SimilarityCandidate(
            submission_id="prior-x",
            hotkey="hk2",
            code_hash="c" * 64,
            score=0.9,
            ast_similarity=0.9,
            token_similarity=0.88,
            file_similarity=0.5,
            snapshot=cand_snap,
            graph_similarity=0.85,
        ),
        report={
            "source_similarity": 0.9,
            "graph_similarity": 0.85,
            "outcome": "quarantine",
        },
    )
    monkeypatch.setattr(source_similarity, "classify_duplicate", lambda **kwargs: forced)

    class _FP:
        family_hash = "fam1"

    class _Comp:
        entrypoint = "architecture.py"

    class _Sem:
        architecture_graph = {"nodes": ["x"], "edges": []}

    class _CR:
        fingerprints = _FP()
        components = _Comp()
        semantic_signature = _Sem()

    outcome = await worker._review_static_submission(  # noqa: SLF001
        submission_id="sub-new",
        snapshot=snap,
        component_review=_CR(),  # type: ignore[arg-type]
        code_for_eval=ARCH_A,
        filename="project.zip",
        hotkey="miner-1",
        code_hash="e" * 64,
    )
    assert calls == ["llm"]
    assert outcome.rejected is False, outcome.reason


@pytest.mark.asyncio
async def test_worker_quarantine_llm_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prism_challenge.db import Database
    from prism_challenge.queue import PrismWorker
    from prism_challenge.repository import PrismRepository

    def _fake_adj(**kwargs: Any) -> PlagiarismAdjudication:
        return PlagiarismAdjudication(
            plagiarized=True,
            reason="same architecture with no material change",
            confidence=0.95,
            violations=["same_architecture_no_change"],
            used_llm=True,
        )

    monkeypatch.setattr("prism_challenge.queue.adjudicate_plagiarism", _fake_adj)

    settings = PrismSettings(
        database_path=tmp_path / "p2.sqlite3",
        shared_token="secret",
        allow_insecure_signatures=True,
        plagiarism_enabled=True,
        plagiarism_llm_enabled=True,
        plagiarism_llm_required=True,
        openrouter_api_key="sk-test",
        distributed_contract_policy="off",
    )
    db = Database(tmp_path / "p2.sqlite3")
    await db.init()
    repo = PrismRepository(db, epoch_seconds=settings.epoch_seconds)
    worker = object.__new__(PrismWorker)
    worker.repository = repo
    worker.settings = settings

    snap = SourceSnapshot.from_payload(
        {
            "files": [{"path": "architecture.py", "content": ARCH_A, "sha256": "a" * 64}],
            "ast_features": ["A"],
            "token_shingles": ["t1"],
            "fingerprint": "f",
        }
    )
    cand_snap = SourceSnapshot.from_payload(
        {
            "files": [{"path": "architecture.py", "content": ARCH_A, "sha256": "c" * 64}],
            "ast_features": ["A"],
            "token_shingles": ["t1"],
            "fingerprint": "f2",
        }
    )
    forced = DuplicatePolicyDecision(
        outcome="attach",
        reason="identical architecture graph",
        candidate=SimilarityCandidate(
            submission_id="prior-y",
            hotkey="hk2",
            code_hash="c" * 64,
            score=0.95,
            ast_similarity=0.95,
            token_similarity=0.95,
            file_similarity=0.9,
            snapshot=cand_snap,
            graph_similarity=1.0,
        ),
        report={"source_similarity": 0.95, "graph_similarity": 1.0, "outcome": "attach"},
    )
    monkeypatch.setattr(source_similarity, "classify_duplicate", lambda **kwargs: forced)

    class _CR:
        class fingerprints:
            family_hash = "fam"

        class components:
            entrypoint = "architecture.py"

        class semantic_signature:
            architecture_graph = {"nodes": ["x"], "edges": []}

    outcome = await worker._review_static_submission(  # noqa: SLF001
        submission_id="sub-copy",
        snapshot=snap,
        component_review=_CR(),  # type: ignore[arg-type]
        code_for_eval=ARCH_A,
        filename="project.zip",
        hotkey="miner-1",
        code_hash="f" * 64,
    )
    assert outcome.rejected is True
    assert outcome.reason and "llm_plagiarism" in outcome.reason
