"""Durable, version-scoped authorization for full-attested evaluation paths."""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent_challenge.core.models import AgentSubmission, ReviewAssignment, ReviewSession


async def verified_review_assignment_for_submission(
    session: AsyncSession,
    submission: AgentSubmission,
) -> ReviewAssignment | None:
    """Return only the persisted verified allow for this exact submission.

    A session's mutable current assignment is deliberately irrelevant.  The
    durable authorizing assignment id must point at the same session and
    submission version, retain a receipted report, and contain the terminal
    verifier disposition.  No administrative status field is an input.

    Callers that **launch Eval CVM spend** must additionally re-verify the
    receipted envelope (see
    :func:`agent_challenge.evaluation.fresh_review_gate.admit_eval_cvm_launch_from_assignment`
    / ``create_eval_run``). This helper remains the durable locator of the
    authorizing assignment; production launch policies must not treat it as a
    cryptographic re-verify substitute.
    """

    review_session = await session.scalar(
        select(ReviewSession).where(ReviewSession.submission_id == submission.id).limit(1)
    )
    if review_session is None or review_session.authorizing_assignment_id is None:
        return None
    assignment = await session.scalar(
        select(ReviewAssignment)
        .where(ReviewAssignment.session_id == review_session.id)
        .where(ReviewAssignment.assignment_id == review_session.authorizing_assignment_id)
        .limit(1)
    )
    if (
        assignment is None
        or assignment.phase != "review_allowed"
        or assignment.review_report_envelope_json is None
        or assignment.review_digest is None
    ):
        return None
    try:
        outcome = json.loads(assignment.review_verification_outcome_json or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(outcome, dict):
        return None
    if outcome.get("status") != "verified_allow":
        return None
    if outcome.get("terminal") is not True or outcome.get("retryable") is not False:
        return None
    if outcome.get("nonce_consumed") is not True:
        return None
    return assignment


__all__ = ["verified_review_assignment_for_submission"]
