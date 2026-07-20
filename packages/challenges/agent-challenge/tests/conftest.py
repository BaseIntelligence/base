from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, update

_TEST_DIR = Path(tempfile.mkdtemp(prefix="agent-challenge-tests-"))
_TEST_DB = _TEST_DIR / "challenge.sqlite3"

os.environ.setdefault("CHALLENGE_DATABASE_URL", f"sqlite+aiosqlite:///{_TEST_DB}")
os.environ.setdefault("CHALLENGE_SHARED_TOKEN", "test-token")
# Distinct from the internal bearer so least-privilege evidence encryption is
# exercised by default in offline route and module tests.
os.environ.setdefault("CHALLENGE_REVIEW_EVIDENCE_ENCRYPTION_KEY", "test-evidence-key")

from agent_challenge.app import app  # noqa: E402
from agent_challenge.db import database  # noqa: E402
from agent_challenge.models import (  # noqa: E402
    AdminReviewDecision,
    AgentSubmission,
    AnalysisRun,
    AnalyzerReport,
    EvalNonce,
    EvalResourceCounter,
    EvalRun,
    EvaluationAttempt,
    EvaluationJob,
    ExternalExecutionRef,
    LlmVerdict,
    OwnerActionAudit,
    PythonAstFeature,
    RateLimitReservation,
    ReplayAuditDispute,
    RequestNonce,
    ReviewAssignment,
    ReviewEvidenceObject,
    ReviewNonce,
    ReviewOperatorApproval,
    ReviewRulesSnapshot,
    ReviewSession,
    SimilarityMatch,
    SubmissionArtifact,
    SubmissionEnvVar,
    SubmissionFamily,
    SubmissionStatusEvent,
    TaskAttestation,
    TaskLogByteTotal,
    TaskLogEvent,
    TaskResult,
    TerminalBenchTrial,
)


@pytest.fixture(scope="session", autouse=True)
async def initialized_database():
    await database.init()
    yield
    await database.close()


@pytest.fixture(autouse=True)
async def clean_database(initialized_database):
    async with database.engine.begin() as connection:
        await connection.execute(delete(ReviewOperatorApproval))
        await connection.execute(delete(ReviewNonce))
        await connection.execute(delete(ReviewEvidenceObject))
        await connection.execute(delete(ReviewAssignment))
        await connection.execute(delete(ReviewRulesSnapshot))
        await connection.execute(delete(ReviewSession))
        await connection.execute(delete(OwnerActionAudit))
        await connection.execute(delete(AdminReviewDecision))
        await connection.execute(delete(RequestNonce))
        await connection.execute(delete(ReplayAuditDispute))
        await connection.execute(delete(RateLimitReservation))
        await connection.execute(delete(LlmVerdict))
        await connection.execute(delete(SimilarityMatch))
        await connection.execute(delete(PythonAstFeature))
        await connection.execute(delete(AnalysisRun))
        await connection.execute(delete(AnalyzerReport))
        await connection.execute(delete(TaskLogByteTotal))
        await connection.execute(delete(TaskLogEvent))
        await connection.execute(delete(TaskAttestation))
        await connection.execute(delete(TaskResult))
        await connection.execute(delete(ExternalExecutionRef))
        await connection.execute(delete(TerminalBenchTrial))
        await connection.execute(delete(EvaluationAttempt))
        await connection.execute(delete(EvaluationJob))
        await connection.execute(delete(EvalNonce))
        await connection.execute(delete(EvalRun))
        await connection.execute(delete(EvalResourceCounter))
        await connection.execute(delete(SubmissionEnvVar))
        await connection.execute(delete(SubmissionArtifact))
        await connection.execute(delete(SubmissionStatusEvent))
        await connection.execute(update(SubmissionFamily).values(latest_submission_id=None))
        await connection.execute(delete(AgentSubmission))
        await connection.execute(delete(SubmissionFamily))


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client


@pytest.fixture
def database_session():
    return database.session


@pytest.fixture
def internal_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer test-token",
        "X-Base-Challenge-Slug": "agent-challenge",
    }
