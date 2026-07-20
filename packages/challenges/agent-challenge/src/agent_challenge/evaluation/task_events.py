from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.models import TaskLogByteTotal, TaskLogEvent, TaskResult
from .benchmarks import BenchmarkTask

MAX_TASK_EVENT_BYTES = 64 * 1024
MAX_TASK_LOG_BYTES = 10 * 1024 * 1024
MAX_SUBMISSION_LOG_BYTES = 50 * 1024 * 1024
MAX_SEQUENCE_ALLOCATION_RETRIES = 5

# TaskLogByteTotal scopes: each mirrors exactly one filter used by the byte caps
# so a cap check reads one running-total row instead of re-summing prior rows.
LOG_BYTE_SCOPE_SUBMISSION = "submission"
LOG_BYTE_SCOPE_TASK_RESULT = "task_result"
LOG_BYTE_SCOPE_TASK = "task"
SAFE_TASK_PHASE_STATUSES = frozenset(
    {"assigned", "starting", "waiting", "running", "completed", "failed"}
)

TASK_LOG_CAP_EVENT_TYPE = "task_log_cap_reached"
SUBMISSION_LOG_CAP_EVENT_TYPE = "submission_log_cap_reached"

_NON_COUNTED_EVENT_TYPES = frozenset(
    {
        "task.progress",
        "task.status",
        "task.completed",
        "task.failed",
        "submission.status",
        "submission.completed",
        TASK_LOG_CAP_EVENT_TYPE,
        SUBMISSION_LOG_CAP_EVENT_TYPE,
    }
)
_API_KEY_RE = re.compile(r"\b(API_KEY=)[^\s]+")
_BEARER_RE = re.compile(r"\b(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE)
_SK_SECRET_RE = re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]*")

# Extended secret patterns used when redacting source code served publicly. These
# build on the task-event patterns above (``sk-``/``Bearer``/``API_KEY=``). Source
# redaction is BEST-EFFORT: it masks common, well-known secret shapes but cannot
# guarantee that every embedded credential is caught, so the output must not be
# treated as certified secret-free.
_AWS_ACCESS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_PEM_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
# ``scheme://user:password@host`` -> keep the scheme + host, mask the credentials.
# Any ``scheme://`` is handled generically; ``/`` is excluded from the credential
# span so a URL without credentials (``https://host/path``) never matches.
_URL_CREDENTIALS_RE = re.compile(r"(?P<scheme>[A-Za-z][A-Za-z0-9+.\-]*://)[^/:@\s]+:[^/@\s]+@")
# Bare provider tokens whose distinctive prefixes make them safe to always mask.
_GITHUB_TOKEN_RE = re.compile(r"\b(?:gh[opsur]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})")
_GITLAB_TOKEN_RE = re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}")
_SLACK_TOKEN_RE = re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")
_GOOGLE_API_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_\-]{35}")
# Stripe secret/restricted keys only -- publishable ``pk_`` keys are left intact.
_STRIPE_KEY_RE = re.compile(r"\b(?:sk_live_|sk_test_|rk_live_)[A-Za-z0-9]{10,}")
# A key-like identifier (e.g. ``OPENAI_API_KEY``, ``db_password``). The secret
# keyword must sit on an underscore/word boundary so innocuous identifiers such as
# ``AUTHOR`` (not ``AUTH``) or ``PASSWORDLESS`` (not ``PASSWORD``) never match.
_SECRET_KEY_WORD = (
    r"(?:API_?KEY|ACCESS_?KEY|SECRET_?KEY|SIGNING_?KEY|ENCRYPTION_?KEY|PRIVATE_?KEY|"
    r"CONNECTION_?STRING|AUTH_?TOKEN|CREDENTIALS?|PASSWORD|PASSWD|SECRET|TOKEN|"
    r"DSN|PWD|AUTH)"
)
_SECRET_KEY_NAME = r"(?:[A-Za-z0-9]+_)*" + _SECRET_KEY_WORD + r"(?:_[A-Za-z0-9]+)*"
# ``NAME = """literal"""`` (triple-quoted; mask the whole body, keep the delimiters).
_SECRET_ASSIGNMENT_TRIPLE_QUOTED_RE = re.compile(
    r"(?P<name>" + _SECRET_KEY_NAME + r")"
    r"(?P<pre>['\"]?\s*[:=]\s*)"
    r"(?P<q>'''|\"\"\")"
    r"(?P<value>.*?)"
    r"(?P=q)",
    re.IGNORECASE | re.DOTALL,
)
# ``NAME = "literal"`` / ``"NAME": "literal"`` (mask the quoted value, keep the
# name). A quoted literal assigned to a secret-named key is masked even when it is
# all-alpha; an empty string ("") is left untouched.
_SECRET_ASSIGNMENT_QUOTED_RE = re.compile(
    r"(?P<name>" + _SECRET_KEY_NAME + r")"
    r"(?P<pre>['\"]?\s*[:=]\s*)"
    r"(?P<q>['\"])"
    r"(?P<value>(?:\\.|[^'\"\\])+)"
    r"(?P=q)",
    re.IGNORECASE,
)
# ``NAME=literal`` with an unquoted ``.env``/config-style value. The value token is
# captured broadly (including any ``()``/``[]``/``.``) so ``_is_bare_secret_literal``
# can distinguish a bare literal from ordinary code before anything is masked.
_SECRET_ASSIGNMENT_UNQUOTED_RE = re.compile(
    r"(?P<name>" + _SECRET_KEY_NAME + r")"
    r"(?P<pre>\s*[:=]\s*)"
    r"(?P<value>[^\s#,;'\"][^\s#,;]*)",
    re.IGNORECASE,
)
# Attribute/method access such as ``config.api_key`` (identifier ``.`` identifier).
_ATTRIBUTE_ACCESS_RE = re.compile(r"[A-Za-z_]\w*\.[A-Za-z_]\w*")
_NON_SECRET_VALUE_LITERALS = frozenset({"None", "True", "False"})


def _is_bare_secret_literal(value: str) -> bool:
    """Return ``True`` when ``value`` is a bare secret literal rather than code.

    Guards the unquoted ``NAME=value`` redactor from corrupting ordinary source:
    calls/subscripts (``()``/``[]``), attribute access (``config.api_key``), Python
    constants (``None``/``True``/``False``), quoted strings, and empty values are
    treated as code references and left untouched.
    """

    if not value or value[0] in "'\"":
        return False
    if value in _NON_SECRET_VALUE_LITERALS:
        return False
    if any(char in "()[]" for char in value):
        return False
    return _ATTRIBUTE_ACCESS_RE.search(value) is None


def _redact_unquoted_assignment(match: re.Match[str]) -> str:
    if not _is_bare_secret_literal(match.group("value")):
        return match.group(0)
    return f"{match.group('name')}{match.group('pre')}[REDACTED]"


_SourceReplacement = str | Callable[[re.Match[str]], str]
_SOURCE_SECRET_SUBS: tuple[tuple[re.Pattern[str], _SourceReplacement], ...] = (
    (_PEM_PRIVATE_KEY_RE, "[REDACTED]"),
    (_URL_CREDENTIALS_RE, r"\g<scheme>[REDACTED]@"),
    (_GITHUB_TOKEN_RE, "[REDACTED]"),
    (_GITLAB_TOKEN_RE, "[REDACTED]"),
    (_SLACK_TOKEN_RE, "[REDACTED]"),
    (_GOOGLE_API_KEY_RE, "[REDACTED]"),
    (_STRIPE_KEY_RE, "[REDACTED]"),
    (_SECRET_ASSIGNMENT_TRIPLE_QUOTED_RE, r"\g<name>\g<pre>\g<q>[REDACTED]\g<q>"),
    (_SECRET_ASSIGNMENT_QUOTED_RE, r"\g<name>\g<pre>\g<q>[REDACTED]\g<q>"),
    (_SECRET_ASSIGNMENT_UNQUOTED_RE, _redact_unquoted_assignment),
    (_API_KEY_RE, r"\1[REDACTED]"),
    (_BEARER_RE, r"\1[REDACTED]"),
    (_SK_SECRET_RE, "sk-[REDACTED]"),
    (_AWS_ACCESS_KEY_RE, "[REDACTED]"),
)


async def record_task_event(
    session: AsyncSession,
    *,
    submission_id: int,
    event_type: str,
    message: str = "",
    job_id: int | None = None,
    task_result_id: int | None = None,
    task_id: str | None = None,
    stream: str | None = None,
    progress: float | None = None,
    status: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> list[TaskLogEvent]:
    stored_message, truncated = truncate_task_event_message(redact_task_event_message(message))
    counts_toward_caps = _counts_toward_log_caps(event_type)
    if counts_toward_caps:
        stored_message, cap_truncated = await _cap_message_for_remaining_budget(
            session,
            submission_id=submission_id,
            task_result_id=task_result_id,
            task_id=task_id,
            message=stored_message,
        )
        truncated = truncated or cap_truncated
        if not stored_message:
            return await _append_cap_markers(
                session,
                submission_id=submission_id,
                job_id=job_id,
                task_result_id=task_result_id,
                task_id=task_id,
            )

    event = await _append_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_result_id=task_result_id,
        task_id=task_id,
        event_type=event_type,
        stream=stream,
        message=stored_message,
        progress=progress,
        status=status,
        truncated=truncated,
        cap_reached=False,
        metadata=metadata,
    )
    events = [event]
    if counts_toward_caps:
        # Fold the just-inserted event into the persisted running totals BEFORE
        # the cap-marker check re-reads them, so accounting stays byte-exact with
        # the legacy behaviour where the post-insert scan included this row.
        await _record_counted_log_bytes(
            session,
            submission_id=submission_id,
            task_result_id=task_result_id,
            task_id=task_id,
            message_bytes=event.message_bytes,
        )
        events.extend(
            await _append_cap_markers(
                session,
                submission_id=submission_id,
                job_id=job_id,
                task_result_id=task_result_id,
                task_id=task_id,
            )
        )
    return events


async def record_task_phase_event(
    session: AsyncSession,
    *,
    submission_id: int,
    job_id: int | None,
    task: BenchmarkTask,
    phase: str,
    attempt: int | None = None,
) -> None:
    if phase not in SAFE_TASK_PHASE_STATUSES:
        raise ValueError(f"unsupported public task phase: {phase}")
    metadata: dict[str, object] = {
        "phase": phase,
        "benchmark": task.benchmark,
    }
    if attempt is not None:
        metadata["attempt"] = attempt
    await record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_id=task.task_id,
        event_type="task.status",
        message=f"task {task.task_id} {phase}",
        status=phase,
        metadata=metadata,
    )


async def record_task_result_events(
    session: AsyncSession,
    *,
    submission_id: int,
    job_id: int | None,
    result: TaskResult,
    progress: float | None = 1.0,
    metadata: Mapping[str, object] | None = None,
) -> None:
    result_metadata: dict[str, object] = {
        "returncode": result.returncode,
        "score": result.score,
        "duration_seconds": result.duration_seconds,
        **dict(metadata or {}),
    }
    await record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_result_id=result.id,
        task_id=result.task_id,
        event_type="task.progress",
        message=f"task {result.task_id} {result.status}",
        progress=progress,
        status=result.status,
        metadata=result_metadata,
    )
    if result.stdout:
        await record_task_event(
            session,
            submission_id=submission_id,
            job_id=job_id,
            task_result_id=result.id,
            task_id=result.task_id,
            event_type="task.log",
            stream="stdout",
            message=result.stdout,
            status=result.status,
        )
    if result.stderr:
        await record_task_event(
            session,
            submission_id=submission_id,
            job_id=job_id,
            task_result_id=result.id,
            task_id=result.task_id,
            event_type="task.log",
            stream="stderr",
            message=result.stderr,
            status=result.status,
        )
    terminal_event_type = "task.completed" if result.status == "completed" else "task.failed"
    await record_task_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_result_id=result.id,
        task_id=result.task_id,
        event_type=terminal_event_type,
        message=f"task {result.task_id} {result.status}",
        progress=1.0,
        status=result.status,
        metadata=result_metadata,
    )


def apply_miner_env_redaction(content: str, redaction_values: Mapping[str, str] | None) -> str:
    """Replace any locked miner-env secret value in ``content`` with a marker."""

    if not redaction_values:
        return content
    redacted = content
    for raw_value in sorted(set(redaction_values.values()), key=len, reverse=True):
        if raw_value:
            redacted = redacted.replace(raw_value, "[REDACTED_MINER_ENV]")
    return redacted


def _read_log_file(path: str | Path) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


async def record_separated_trial_logs(
    session: AsyncSession,
    *,
    submission_id: int,
    job_id: int | None,
    task_result_id: int | None,
    task_id: str | None,
    artifacts: Mapping[str, Any],
    status: str | None = None,
    redaction_values: Mapping[str, str] | None = None,
) -> None:
    """Emit one ``task.log`` event per separated harbor v2 trial channel.

    Maps agent_log_files->"agent", trial/exception->"harness",
    verifier stdout/stderr->"test_stdout"/"test_stderr". Miner-env secrets
    are redacted from file content before persistence.
    """
    single_sources: tuple[tuple[str, str], ...] = (
        ("trial_log_ref", "harness"),
        ("exception_ref", "harness"),
        ("test_stdout_ref", "test_stdout"),
        ("test_stderr_ref", "test_stderr"),
    )

    async def _emit(stream: str, content: str) -> None:
        redacted = apply_miner_env_redaction(content, redaction_values)
        if not redacted.strip():
            return
        await record_task_event(
            session,
            submission_id=submission_id,
            job_id=job_id,
            task_result_id=task_result_id,
            task_id=task_id,
            event_type="task.log",
            stream=stream,
            message=redacted,
            status=status,
        )

    for agent_file in artifacts.get("agent_log_files", []) or []:
        content = _read_log_file(agent_file)
        if content is not None:
            await _emit("agent", content)

    for ref_key, stream in single_sources:
        ref = artifacts.get(ref_key)
        if not ref:
            continue
        content = _read_log_file(ref)
        if content is not None:
            await _emit(stream, content)


async def next_task_event_sequence(session: AsyncSession, submission_id: int) -> int:
    current = await session.scalar(
        select(func.max(TaskLogEvent.sequence)).where(TaskLogEvent.submission_id == submission_id)
    )
    return int(current or 0) + 1


def redact_task_event_message(message: str) -> str:
    redacted = _API_KEY_RE.sub(r"\1[REDACTED]", message)
    redacted = _BEARER_RE.sub(r"\1[REDACTED]", redacted)
    return _SK_SECRET_RE.sub("sk-[REDACTED]", redacted)


def redact_secrets(text: str) -> tuple[str, bool]:
    """Redact embedded secrets from arbitrary text, returning ``(text, changed)``.

    Reuses the task-event secret patterns (``sk-``/``Bearer``/``API_KEY=``) and
    extends them with URL-embedded credentials, well-known provider tokens (GitHub,
    GitLab, Slack, Google, Stripe), AWS access keys, PEM private-key blocks, and
    key-like assignments before agent source is served over the public source
    endpoint. Redaction is BEST-EFFORT: it masks common secret shapes but cannot
    guarantee that every embedded credential is caught, so the output must not be
    treated as certified secret-free. ``changed`` is ``True`` only when a redaction
    actually altered the text.
    """

    redacted = text
    for pattern, replacement in _SOURCE_SECRET_SUBS:
        redacted = pattern.sub(replacement, redacted)
    return redacted, redacted != text


def truncate_task_event_message(message: str) -> tuple[str, bool]:
    return _truncate_utf8(message, MAX_TASK_EVENT_BYTES)


async def _cap_message_for_remaining_budget(
    session: AsyncSession,
    *,
    submission_id: int,
    task_result_id: int | None,
    task_id: str | None,
    message: str,
) -> tuple[str, bool]:
    task_used = await _stored_log_bytes(
        session,
        submission_id=submission_id,
        task_result_id=task_result_id,
        task_id=task_id,
        task_scope=True,
    )
    submission_used = await _stored_log_bytes(
        session,
        submission_id=submission_id,
        task_result_id=None,
        task_id=None,
        task_scope=False,
    )
    remaining = min(MAX_TASK_LOG_BYTES - task_used, MAX_SUBMISSION_LOG_BYTES - submission_used)
    if remaining <= 0:
        return "", bool(message)
    return _truncate_utf8(message, remaining)


async def _append_cap_markers(
    session: AsyncSession,
    *,
    submission_id: int,
    job_id: int | None,
    task_result_id: int | None,
    task_id: str | None,
) -> list[TaskLogEvent]:
    events: list[TaskLogEvent] = []
    task_used = await _stored_log_bytes(
        session,
        submission_id=submission_id,
        task_result_id=task_result_id,
        task_id=task_id,
        task_scope=True,
    )
    if task_used >= MAX_TASK_LOG_BYTES and not await _cap_marker_exists(
        session,
        submission_id=submission_id,
        event_type=TASK_LOG_CAP_EVENT_TYPE,
        task_result_id=task_result_id,
        task_id=task_id,
    ):
        events.append(
            await _append_cap_marker(
                session,
                submission_id=submission_id,
                job_id=job_id,
                task_result_id=task_result_id,
                task_id=task_id,
                event_type=TASK_LOG_CAP_EVENT_TYPE,
            )
        )

    submission_used = await _stored_log_bytes(
        session,
        submission_id=submission_id,
        task_result_id=None,
        task_id=None,
        task_scope=False,
    )
    if submission_used >= MAX_SUBMISSION_LOG_BYTES and not await _cap_marker_exists(
        session,
        submission_id=submission_id,
        event_type=SUBMISSION_LOG_CAP_EVENT_TYPE,
        task_result_id=None,
        task_id=None,
    ):
        events.append(
            await _append_cap_marker(
                session,
                submission_id=submission_id,
                job_id=job_id,
                task_result_id=task_result_id,
                task_id=task_id,
                event_type=SUBMISSION_LOG_CAP_EVENT_TYPE,
            )
        )
    return events


async def _append_cap_marker(
    session: AsyncSession,
    *,
    submission_id: int,
    job_id: int | None,
    task_result_id: int | None,
    task_id: str | None,
    event_type: str,
) -> TaskLogEvent:
    return await _append_event(
        session,
        submission_id=submission_id,
        job_id=job_id,
        task_result_id=task_result_id,
        task_id=task_id,
        event_type=event_type,
        stream=None,
        message="",
        progress=None,
        status=None,
        truncated=False,
        cap_reached=True,
        metadata=None,
    )


async def _append_event(
    session: AsyncSession,
    *,
    submission_id: int,
    job_id: int | None,
    task_result_id: int | None,
    task_id: str | None,
    event_type: str,
    stream: str | None,
    message: str,
    progress: float | None,
    status: str | None,
    truncated: bool,
    cap_reached: bool,
    metadata: Mapping[str, object] | None,
) -> TaskLogEvent:
    last_collision: IntegrityError | None = None
    for _ in range(MAX_SEQUENCE_ALLOCATION_RETRIES):
        try:
            async with session.begin_nested():
                event = TaskLogEvent(
                    submission_id=submission_id,
                    job_id=job_id,
                    task_result_id=task_result_id,
                    task_id=task_id,
                    sequence=await next_task_event_sequence(session, submission_id),
                    event_type=event_type,
                    stream=stream,
                    message=message,
                    message_bytes=len(message.encode("utf-8")),
                    progress=progress,
                    status=status,
                    truncated=truncated,
                    cap_reached=cap_reached,
                    metadata_json=json.dumps(
                        dict(metadata or {}),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
                session.add(event)
                await session.flush()
            return event
        except IntegrityError as exc:
            if not _is_sequence_collision(exc):
                raise
            last_collision = exc
    if last_collision is not None:
        raise last_collision
    raise RuntimeError("MAX_SEQUENCE_ALLOCATION_RETRIES must be greater than zero")


def _is_sequence_collision(exc: IntegrityError) -> bool:
    message = str(exc.orig)
    return (
        "uq_task_log_events_submission_sequence" in message
        or "task_log_events.submission_id, task_log_events.sequence" in message
    )


def _log_byte_scope(
    task_result_id: int | None,
    task_id: str | None,
    task_scope: bool,
) -> tuple[str, str] | None:
    """Map a byte-accounting query to its running-total ``(scope, scope_key)``.

    Mirrors the exact filter the legacy full-scan used: submission-wide when not
    task-scoped, otherwise the task_result_id filter (with the task_id filter as
    the fallback used by the live-ingest path, which has no task_result_id).
    """

    if not task_scope:
        return LOG_BYTE_SCOPE_SUBMISSION, ""
    if task_result_id is not None:
        return LOG_BYTE_SCOPE_TASK_RESULT, str(task_result_id)
    if task_id is not None:
        return LOG_BYTE_SCOPE_TASK, task_id
    return None


async def _stored_log_bytes(
    session: AsyncSession,
    *,
    submission_id: int,
    task_result_id: int | None,
    task_id: str | None,
    task_scope: bool,
) -> int:
    scope = _log_byte_scope(task_result_id, task_id, task_scope)
    if scope is None:
        return 0
    total = await session.scalar(
        select(TaskLogByteTotal.total_bytes).where(
            TaskLogByteTotal.submission_id == submission_id,
            TaskLogByteTotal.scope == scope[0],
            TaskLogByteTotal.scope_key == scope[1],
        )
    )
    return int(total or 0)


async def _record_counted_log_bytes(
    session: AsyncSession,
    *,
    submission_id: int,
    task_result_id: int | None,
    task_id: str | None,
    message_bytes: int,
) -> None:
    if message_bytes <= 0:
        return
    await _increment_log_byte_total(
        session, submission_id, LOG_BYTE_SCOPE_SUBMISSION, "", message_bytes
    )
    if task_result_id is not None:
        await _increment_log_byte_total(
            session,
            submission_id,
            LOG_BYTE_SCOPE_TASK_RESULT,
            str(task_result_id),
            message_bytes,
        )
    if task_id is not None:
        await _increment_log_byte_total(
            session, submission_id, LOG_BYTE_SCOPE_TASK, task_id, message_bytes
        )


async def _increment_log_byte_total(
    session: AsyncSession,
    submission_id: int,
    scope: str,
    scope_key: str,
    delta: int,
) -> None:
    """Atomically add ``delta`` to one running-total row (upsert).

    The in-place ``UPDATE ... SET total_bytes = total_bytes + delta`` keeps the
    increment correct under concurrent writers (no read-modify-write race), and
    the savepoint-guarded INSERT seeds a missing row. Works on SQLite + Postgres.
    """

    result = await session.execute(
        update(TaskLogByteTotal)
        .where(
            TaskLogByteTotal.submission_id == submission_id,
            TaskLogByteTotal.scope == scope,
            TaskLogByteTotal.scope_key == scope_key,
        )
        .values(total_bytes=TaskLogByteTotal.total_bytes + delta)
    )
    if result.rowcount:
        return
    try:
        async with session.begin_nested():
            await session.execute(
                insert(TaskLogByteTotal).values(
                    submission_id=submission_id,
                    scope=scope,
                    scope_key=scope_key,
                    total_bytes=delta,
                )
            )
    except IntegrityError:
        await session.execute(
            update(TaskLogByteTotal)
            .where(
                TaskLogByteTotal.submission_id == submission_id,
                TaskLogByteTotal.scope == scope,
                TaskLogByteTotal.scope_key == scope_key,
            )
            .values(total_bytes=TaskLogByteTotal.total_bytes + delta)
        )


async def _cap_marker_exists(
    session: AsyncSession,
    *,
    submission_id: int,
    event_type: str,
    task_result_id: int | None,
    task_id: str | None,
) -> bool:
    statement = select(TaskLogEvent.id).where(
        TaskLogEvent.submission_id == submission_id,
        TaskLogEvent.event_type == event_type,
    )
    if event_type == TASK_LOG_CAP_EVENT_TYPE:
        if task_result_id is not None:
            statement = statement.where(TaskLogEvent.task_result_id == task_result_id)
        elif task_id is not None:
            statement = statement.where(TaskLogEvent.task_id == task_id)
        else:
            return False
    return await session.scalar(statement.limit(1)) is not None


def _counts_toward_log_caps(event_type: str) -> bool:
    return event_type not in _NON_COUNTED_EVENT_TYPES


def _truncate_utf8(message: str, max_bytes: int) -> tuple[str, bool]:
    encoded = message.encode("utf-8")
    if len(encoded) <= max_bytes:
        return message, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True
