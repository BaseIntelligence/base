"""Strict, one-shot direct OpenRouter transport for the measured reviewer.

This module owns only the bounded pre-call plan and direct exchange. It does
not make policy decisions or create a review report, so later review stages can
bind the captured records without accepting an unbound model result.
"""

from __future__ import annotations

import json
import re
import ssl
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import httpx

from .canonical import canonical_json_v1, canonical_sha256, parse_json_object
from .policy import ModelPolicyOutput, ReviewPolicyError, parse_model_policy_output
from .schemas import (
    MAX_OPENROUTER_METADATA_BYTES,
    MAX_OPENROUTER_REQUEST_BYTES,
    MAX_OPENROUTER_RESPONSE_BYTES,
    OPENROUTER_HEADERS,
    OPENROUTER_ORIGIN,
    OPENROUTER_PATH,
    REVIEW_INFRASTRUCTURE_FAILURE_REASONS,
    REVIEW_MODEL,
    REVIEW_TRANSPORT_SCHEMA_VERSION,
    ReviewTransportSchemaError,
    is_pinned_review_model,
    review_policy_tools,
    validate_model_call_started,
    validate_observed_openrouter_transport,
    validate_planned_openrouter_request,
    validate_review_infrastructure_failure,
    validate_review_routing,
)

OPENROUTER_URL = f"{OPENROUTER_ORIGIN}{OPENROUTER_PATH}"
_REQUEST_TIMEOUT = httpx.Timeout(connect=10.0, read=240.0, write=10.0, pool=10.0)
_NETWORK_FAILURES = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadError, httpx.ReadTimeout)
_TLS_MESSAGE_RE = re.compile(
    r"(certificate|ssl|tls|handshake|hostname|name mismatch|verify failed|"
    r"self.?signed|certificate_verify_failed|SSLCertVerificationError|"
    r"CERTIFICATE_VERIFY_FAILED|WRONG_VERSION_NUMBER|handshake failure)",
    re.IGNORECASE,
)
_DNS_MESSAGE_RE = re.compile(
    r"(name or service not known|nodename nor servname|getaddrinfo|"
    r"temporary failure in name resolution|dns|name resolution|"
    r"no address associated with hostname|unknown host)",
    re.IGNORECASE,
)


def openrouter_timeout_from_settings(settings: object | None = None) -> httpx.Timeout:
    """Build the OpenRouter timeout object from challenge settings when present.

    Connect, TLS handshake, write, and pool stages share the tightest of the
    connect/tls settings; read uses the dedicated read budget. The total
    timeout is applied as an upper bound so a slow multi-stage exchange still
    terminates when ``review_https_total_timeout_seconds`` expires.
    """

    if settings is None:
        return _REQUEST_TIMEOUT
    connect = float(getattr(settings, "review_https_connect_timeout_seconds", 10.0))
    tls = float(getattr(settings, "review_https_tls_timeout_seconds", connect))
    read = float(getattr(settings, "review_https_read_timeout_seconds", 240.0))
    write = float(getattr(settings, "review_https_write_timeout_seconds", 10.0))
    total = float(getattr(settings, "review_https_total_timeout_seconds", 300.0))
    handshake = min(connect, tls)
    # httpx has no independent TLS-timeout knob; fold TLS into connect/pool so
    # the configured budget still aborts a hung handshake/connection pool wait.
    return httpx.Timeout(
        connect=handshake,
        read=min(read, total),
        write=min(write, total),
        pool=handshake,
    )


def openrouter_byte_limits_from_settings(settings: object | None = None) -> dict[str, int]:
    """Resolve request/response/metadata caps from live settings when present."""

    if settings is None:
        return {
            "request": MAX_OPENROUTER_REQUEST_BYTES,
            "response": MAX_OPENROUTER_RESPONSE_BYTES,
            "metadata": MAX_OPENROUTER_METADATA_BYTES,
        }
    return {
        "request": int(
            getattr(settings, "review_max_openrouter_request_bytes", MAX_OPENROUTER_REQUEST_BYTES)
        ),
        "response": int(
            getattr(settings, "review_max_openrouter_response_bytes", MAX_OPENROUTER_RESPONSE_BYTES)
        ),
        "metadata": int(
            getattr(settings, "review_max_openrouter_metadata_bytes", MAX_OPENROUTER_METADATA_BYTES)
        ),
    }


# Closed short classes for policy_output_malformed guest diag (no free text).
_POLICY_ERROR_CLASS_TOKENS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("verdict",), "verdict"),
    (("arguments",), "args"),
    (("tool call", "malformed"), "tool_shape"),
    (("function shape",), "tool_shape"),
    (("unassigned policy tool",), "tool_shape"),
    (("exactly one final tool",), "tool_count"),
    (("exactly one choice",), "tool_count"),
    (("prose is not a policy",), "tool_count"),
    (("duplicate-free json",), "shape"),
    (("response is not",), "shape"),
    (("binding is invalid",), "shape"),
    (("exceeds policy parser",), "shape"),
    # Assignment allowlist blew past bound (former 3N package-path expand residual).
    (("too many assigned evidence",), "allowed_cap"),
    (("assigned evidence paths",), "allowed_cap"),
)

_POLICY_DIAG_ALLOWLIST = frozenset(
    {"verdict", "args", "shape", "tool_shape", "tool_count", "allowed_cap", "other"}
)

# Closed diag tokens for quote_measurement_mismatch (field class only; no digests).
# From review_runtime ``_measurement_from_quote`` ValueError wording + soft residual.
_QUOTE_MEASUREMENT_DIAG_TOKENS: tuple[tuple[tuple[str, ...], str], ...] = (
    # Order matters: match the most specific field keywords first.
    (("quoted rtmr0",), "rtmr0"),
    (("quoted rtmr1",), "rtmr1"),
    (("quoted rtmr2",), "rtmr2"),
    (("quoted rtmr3",), "rtmr3"),
    (("rtmr0",), "rtmr0"),
    (("rtmr1",), "rtmr1"),
    (("rtmr2",), "rtmr2"),
    (("rtmr3",), "rtmr3"),
    (("quoted mrtd",), "mrtd"),
    (("mrtd",), "mrtd"),
    (("quoted compose",), "compose"),
    (("compose hash",), "compose"),
    (("quoted key provider",), "key_provider"),
    (("key provider",), "key_provider"),
    (("quoted os image",), "os"),
    (("os image",), "os"),
    # Event-log / RTMR3 replay residual that still classifies as mismatch soft set.
    (("event log", "rtmr3"), "event_log"),
    (("event_log",), "event_log"),
)

_QUOTE_MEASUREMENT_DIAG_ALLOWLIST = frozenset(
    {
        "compose",
        "key_provider",
        "os",
        "mrtd",
        "rtmr0",
        "rtmr1",
        "rtmr2",
        "rtmr3",
        "event_log",
        "other",
    }
)

# Closed diag tokens for report_envelope_invalid residual public_logs.
# Subclass message words / /report 4xx body codes only — never free-form or secrets.
# Feature residual sub24: opaque ValueError report_envelope_invalid after OS seal.
_REPORT_ENVELOPE_DIAG_TOKENS: tuple[tuple[tuple[str, ...], str], ...] = (
    # Order: most specific subclasses first.
    (("timeline",), "timeline"),
    (("started_at", "issued_at"), "timeline"),
    (("attestation_stale",), "timeline"),
    (("times are invalid",), "timeline"),
    (("evidence",), "evidence"),
    (("measurement",), "measurement"),
    (("allowlisted",), "measurement"),
    (("schema",), "schema"),
    (("report envelope invalid",), "schema"),
    (("review envelope",), "schema"),
    (("envelope digest",), "schema"),
    (("envelope domain",), "schema"),
    (("envelope core",), "schema"),
    (("unsupported review envelope",), "schema"),
    (("report data mismatches",), "schema"),
    (("http_status",), "http_status"),
    (("from /report status=",), "http_status"),
    (("report post status",), "http_status"),
)

_REPORT_ENVELOPE_DIAG_ALLOWLIST = frozenset(
    {
        "timeline",
        "evidence",
        "measurement",
        "schema",
        "http_status",
        "other",
    }
)

# Union of closed residual diag tokens accepted on OpenRouterTransportError.diag.
_TRANSPORT_DIAG_ALLOWLIST = (
    _POLICY_DIAG_ALLOWLIST | _QUOTE_MEASUREMENT_DIAG_ALLOWLIST | _REPORT_ENVELOPE_DIAG_ALLOWLIST
)


def short_policy_error_class(exc: BaseException) -> str:
    """Map ReviewPolicyError text to a closed short class (never free text).

    Guest public_logs may expose this string under ``diag`` when reason_code is
    ``policy_output_malformed``. Unknown residual → ``other``.
    """

    text = str(exc).lower()
    if not text:
        return "other"
    for tokens, label in _POLICY_ERROR_CLASS_TOKENS:
        if all(token in text for token in tokens):
            return label if label in _POLICY_DIAG_ALLOWLIST else "other"
    return "other"


def short_quote_measurement_diag_class(exc: BaseException) -> str:
    """Map quote measurement ValueError text to a closed field-class token.

    Live residual collapses every measured-register mismatch into
    ``quote_measurement_mismatch`` with no public field token. Captures only
    short allowlisted labels from message words
    (compose|key_provider|os|mrtd|rtmr0..3|event_log|other). Never re-emits
    digests, assignment values, or free-form text.
    """

    text = str(exc).lower()
    if not text:
        return "other"
    for tokens, label in _QUOTE_MEASUREMENT_DIAG_TOKENS:
        if all(token in text for token in tokens):
            return label if label in _QUOTE_MEASUREMENT_DIAG_ALLOWLIST else "other"
    return "other"


def short_report_envelope_diag_class(exc: BaseException) -> str:
    """Map report envelope residual text to a closed subclass diag token.

    Live residual (sub24 after OS identity) collapses /report 4xx and local
    envelope build ValueErrors into single ``report_envelope_invalid`` with no
    public subclass. Messages and /.diag only map to
    timeline|evidence|measurement|schema|http_status|other. Never re-emits
    HTTP bodies, digests, or free-form detail codes from /report.
    """

    # Prefer an explicit allowlisted .diag attribute (set by guest /report map).
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        raw = getattr(current, "diag", None)
        if isinstance(raw, str) and raw.strip():
            token = raw.strip().lower()
            if token in _REPORT_ENVELOPE_DIAG_ALLOWLIST:
                return token
        current = current.__cause__ or current.__context__

    text = str(exc).lower()
    if not text:
        return "other"
    for tokens, label in _REPORT_ENVELOPE_DIAG_TOKENS:
        if all(token in text for token in tokens):
            return label if label in _REPORT_ENVELOPE_DIAG_ALLOWLIST else "other"
    return "other"


class OpenRouterTransportError(ValueError):
    """Bounded direct-transport error, safe to map to infrastructure failure."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        planned_request_sha256: str | None = None,
        diag: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        # When set (post-announce), measured /failure stays plan-bound.
        self.planned_request_sha256 = planned_request_sha256
        # Optional short allowlisted residual class (policy error class or
        # quote-measurement field class). Never free-form messages or secrets.
        if diag is None:
            self.diag: str | None = None
        else:
            token = str(diag).strip().lower()
            self.diag = token if token in _TRANSPORT_DIAG_ALLOWLIST else "other"


# Tiny closed map for outer exceptions that never carry raw provider bodies.
# Transport-side TimeoutError stays openrouter_unavailable; quote timeouts use
# message-classified mapping so residual post-OpenRouter hangs are diagnosable.
# Live dstack quote replay raised QuoteVerificationError historically collapsed
# to report_generation_failed because that class is not a ValueError.
def _quote_error_types() -> tuple[type[BaseException], ...]:
    try:
        from agent_challenge.keyrelease.quote import (
            QuoteError,
            QuoteStructureError,
            QuoteVerificationError,
        )
    except Exception:  # pragma: no cover - import scaffolding only
        return ()
    return (QuoteVerificationError, QuoteStructureError, QuoteError)


_EXCEPTION_CLASS_REASON: dict[type[BaseException], str] = {
    TimeoutError: "openrouter_unavailable",
    ssl.SSLError: "tls_failed",
    ssl.SSLCertVerificationError: "tls_failed",
    ssl.CertificateError: "tls_failed",
}
for _quote_exc in _quote_error_types():
    # Prefer more specific message classification when present; otherwise map
    # any dstack/quote package error after OpenRouter to the event-log bucket.
    _EXCEPTION_CLASS_REASON.setdefault(_quote_exc, "quote_event_log_invalid")

# Safe substring classifiers for residual quote/report ValueError surfaces after
# OpenRouter. Matching is on lowercased exception text only; secrets and raw
# bodies are never retained on the /failure surface.
_MESSAGE_REASON_PATTERNS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("quote", "timeout"), "quote_timeout"),
    (("quote", "timed out"), "quote_timeout"),
    (("get_quote", "timeout"), "quote_timeout"),
    (("dstack", "timeout"), "quote_timeout"),
    (("quote event_log",), "quote_event_log_invalid"),
    (("quote event log",), "quote_event_log_invalid"),
    (("event log does not reproduce",), "quote_event_log_invalid"),
    (("event log is missing",), "quote_event_log_invalid"),
    (("event log has",), "quote_event_log_invalid"),
    (("event log must",), "quote_event_log_invalid"),
    (("event log entry",), "quote_event_log_invalid"),
    (("event log exceeds",), "quote_event_log_invalid"),
    (("boot identity",), "quote_event_log_invalid"),
    (("identity event",), "quote_event_log_invalid"),
    (("reserved identity",), "quote_event_log_invalid"),
    (("schema closed",), "quote_event_log_invalid"),
    (("digest", "does not match"), "quote_event_log_invalid"),
    (("digest is not valid",), "quote_event_log_invalid"),
    (("event_payload",), "quote_event_log_invalid"),
    (("key provider event",), "quote_event_log_invalid"),
    (("quoted compose hash",), "quote_measurement_mismatch"),
    (("quoted key provider",), "quote_measurement_mismatch"),
    (("quoted os image",), "quote_measurement_mismatch"),
    (("quoted mrtd",), "quote_measurement_mismatch"),
    (("quoted rtmr",), "quote_measurement_mismatch"),
    (("mismatches assignment",), "quote_measurement_mismatch"),
    (("measurement", "mismatch"), "quote_measurement_mismatch"),
    (("report timeline",), "report_timeline_invalid"),
    (("timeline does not match",), "report_timeline_invalid"),
    (("timeline is future",), "report_timeline_invalid"),
    (("started_at", "issued_at"), "report_timeline_invalid"),
    (("timestamps are not",), "report_timeline_invalid"),
    (("report times",), "report_timeline_invalid"),
    (("times are invalid",), "report_timeline_invalid"),
    (("report finished time",), "report_timeline_invalid"),
    (("report envelope",), "report_envelope_invalid"),
    (("report_envelope",), "report_envelope_invalid"),
    (("review envelope",), "report_envelope_invalid"),
    (("review core",), "report_envelope_invalid"),
    (("review decision",), "report_envelope_invalid"),
    (("openrouter observation",), "report_envelope_invalid"),
    (("returned model",), "report_envelope_invalid"),
    (("verifier result",), "report_envelope_invalid"),
    (("review verdict",), "report_envelope_invalid"),
    (("policy finding",), "report_envelope_invalid"),
    (("model verdict",), "report_envelope_invalid"),
    (("model policy",), "report_envelope_invalid"),
    (("model response",), "report_envelope_invalid"),
    (("model output",), "report_envelope_invalid"),
    (("evidence", "invalid"), "report_evidence_invalid"),
    (("evidence field",), "report_evidence_invalid"),
    (("evidence encoding",), "report_evidence_invalid"),
    (("evidence object is empty",), "report_evidence_invalid"),
    (("review evidence",), "report_evidence_invalid"),
    (("quote response is missing",), "quote_unavailable"),
    (("quote unavailable",), "quote_unavailable"),
    (("dstack", "unavailable"), "quote_unavailable"),
    (("get_quote",), "quote_unavailable"),
    (("quote signature",), "quote_unavailable"),
    (("dcap-qvl",), "quote_unavailable"),
    (("tdx quote",), "quote_unavailable"),
    (("stage quote",), "quote_unavailable"),
    (("stage measure",), "quote_event_log_invalid"),
    (("stage policy",), "report_envelope_invalid"),
    (("stage core",), "report_envelope_invalid"),
    (("stage envelope",), "report_envelope_invalid"),
)


def infrastructure_failure_reason(exc: BaseException) -> str:
    """Map a measured-runtime exception to a safe infrastructure failure code.

    Prefer ``OpenRouterTransportError.reason_code`` (already allowlisted). Also
    honor an explicit allowlisted ``.reason_code`` on residual ReportEnvelopeError
    instances from the guest /report mapper. A tiny class→reason table covers a
    few outer failures; residual quote/report ValueError message classes map to
    specific codes so live failures remain diagnosable without public_logs.
    Everything else collapses to ``report_generation_failed`` without retaining
    messages or bodies.
    """

    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OpenRouterTransportError):
            reason = str(current.reason_code)
            if reason in REVIEW_INFRASTRUCTURE_FAILURE_REASONS:
                return reason
            return "report_generation_failed"
        # Guest ReportEnvelopeError (and similar) may carry a closed reason_code
        # distinct from generic ValueError message classification.
        explicit = getattr(current, "reason_code", None)
        if isinstance(explicit, str) and explicit in REVIEW_INFRASTRUCTURE_FAILURE_REASONS:
            return explicit
        for cls, reason in _EXCEPTION_CLASS_REASON.items():
            if isinstance(current, cls):
                # Prefer more specific message classes for quote timeouts when a
                # bare TimeoutError was raised from the quote path.
                message_reason = _message_classified_reason(current)
                if message_reason is not None:
                    return message_reason
                return reason
        message_reason = _message_classified_reason(current)
        if message_reason is not None:
            return message_reason
        current = current.__cause__ or current.__context__
    return "report_generation_failed"


def _message_classified_reason(exc: BaseException) -> str | None:
    """Return an allowlisted reason from a closed message classifier, or None.

    Uses only lowercased ``str(exc)`` tokens. Never re-emits the original text.
    """

    text = str(exc).lower()
    if not text:
        return None
    for tokens, reason in _MESSAGE_REASON_PATTERNS:
        if all(token in text for token in tokens):
            if reason in REVIEW_INFRASTRUCTURE_FAILURE_REASONS:
                return reason
            return None
    return None


@dataclass(frozen=True)
class OpenRouterCapture:
    """Credential-free records and raw bytes retained by encrypted evidence."""

    planned: dict[str, Any]
    planned_bytes: bytes
    planned_sha256: str
    observed: dict[str, Any]
    observed_bytes: bytes
    request_body: bytes
    response_body: bytes
    metadata: bytes | None
    model_output: ModelPolicyOutput


def build_openrouter_request_body(
    *,
    messages: Sequence[Mapping[str, str]],
    routing: Mapping[str, Any],
) -> bytes:
    """Create the only accepted non-streaming direct request body."""

    normalized_messages: list[dict[str, str]] = []
    if not isinstance(messages, Sequence) or not messages:
        raise OpenRouterTransportError("report_generation_failed", "review messages are required")
    for message in messages:
        if not isinstance(message, Mapping) or set(message) != {"role", "content"}:
            raise OpenRouterTransportError(
                "report_generation_failed", "review message is malformed"
            )
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise OpenRouterTransportError(
                "report_generation_failed", "review message role is invalid"
            )
        if not isinstance(content, str):
            raise OpenRouterTransportError(
                "report_generation_failed", "review message content is invalid"
            )
        normalized_messages.append({"role": role, "content": content})
    try:
        routing_value = validate_review_routing(routing)
    except ReviewTransportSchemaError as exc:
        raise OpenRouterTransportError(
            "report_generation_failed", "review routing is invalid"
        ) from exc
    # Live pin for x-ai/grok-4.5:
    # - omit parallel_tool_calls (OpenRouter 404s endpoints under that parameter)
    # - tool_choice "auto" (forced named tool_choice is rejected while thinking is on)
    # The model is still instructed to call submit_verdict exactly once; the
    # verifier remains final authority on the tool payload.
    body = canonical_json_v1(
        {
            "messages": normalized_messages,
            "model": REVIEW_MODEL,
            "provider": routing_value,
            "stream": False,
            "tool_choice": "auto",
            "tools": review_policy_tools(),
        }
    )
    _validate_direct_request_body(
        body,
        routing_sha256=canonical_sha256(routing_value),
        max_request_bytes=MAX_OPENROUTER_REQUEST_BYTES,
    )
    return body


def build_planned_openrouter_request(
    *,
    body: bytes,
    routing_sha256: str,
    settings: object | None = None,
) -> tuple[dict[str, Any], bytes, str]:
    """Build the credential-redacted pre-network plan and its canonical digest."""

    limits = openrouter_byte_limits_from_settings(settings)
    _validate_direct_request_body(
        body,
        routing_sha256=routing_sha256,
        max_request_bytes=limits["request"],
    )
    planned = {
        "schema_version": REVIEW_TRANSPORT_SCHEMA_VERSION,
        "method": "POST",
        "origin": OPENROUTER_ORIGIN,
        "path": OPENROUTER_PATH,
        "headers": dict(OPENROUTER_HEADERS),
        "body_sha256": sha256(body).hexdigest(),
        "body_length": len(body),
        "model": REVIEW_MODEL,
        "routing_sha256": routing_sha256,
    }
    planned_bytes = validate_planned_openrouter_request(planned)
    return planned, planned_bytes, sha256(planned_bytes).hexdigest()


def build_model_call_started(
    *,
    assignment_id: str,
    planned_request_sha256: str,
    request_body_sha256: str,
    request_body_length: int,
) -> dict[str, Any]:
    """Build exact Model Call Started v1, with copied plan body facts."""

    marker = {
        "schema_version": REVIEW_TRANSPORT_SCHEMA_VERSION,
        "assignment_id": assignment_id,
        "planned_request_sha256": planned_request_sha256,
        "request_body_sha256": request_body_sha256,
        "request_body_length": request_body_length,
    }
    validate_model_call_started(marker)
    return marker


def build_review_infrastructure_failure(
    *,
    assignment_id: str,
    planned_request_sha256: str | None,
    reason_code: str,
) -> dict[str, Any]:
    """Build exact Review Infrastructure Failure v1 without raw error details."""

    failure = {
        "schema_version": REVIEW_TRANSPORT_SCHEMA_VERSION,
        "assignment_id": assignment_id,
        "planned_request_sha256": planned_request_sha256,
        "reason_code": reason_code,
    }
    validate_review_infrastructure_failure(failure)
    return failure


class DirectOpenRouterClient:
    """A single direct TLS request, with proxies and redirects disabled."""

    def __init__(
        self,
        *,
        assignment_id: str,
        api_key: str,
        announce: Callable[[dict[str, Any]], bool],
        transport: httpx.BaseTransport | None = None,
        settings: object | None = None,
    ) -> None:
        self._assignment_id = assignment_id
        self._api_key = api_key
        self._announce = announce
        self._called = False
        self._settings = settings
        self._limits = openrouter_byte_limits_from_settings(settings)
        self._client = httpx.Client(
            transport=transport,
            verify=True,
            trust_env=False,
            follow_redirects=False,
            timeout=openrouter_timeout_from_settings(settings),
        )

    def __repr__(self) -> str:
        return f"DirectOpenRouterClient(api_key=<redacted>, called={self._called})"

    def close(self) -> None:
        """Close the direct transport without exposing connection details."""

        self._client.close()

    def call(
        self,
        *,
        body: bytes,
        routing_sha256: str,
        allowed_evidence_paths: set[str] | frozenset[str],
    ) -> OpenRouterCapture:
        """Announce, then perform exactly one pinned non-streaming exchange."""

        if self._called:
            raise OpenRouterTransportError(
                "report_generation_failed", "model call already attempted"
            )
        if not isinstance(self._api_key, str) or not self._api_key:
            raise OpenRouterTransportError("missing_credential", "OpenRouter credential is missing")
        planned, planned_bytes, planned_digest = build_planned_openrouter_request(
            body=body,
            routing_sha256=routing_sha256,
            settings=self._settings,
        )
        marker = build_model_call_started(
            assignment_id=self._assignment_id,
            planned_request_sha256=planned_digest,
            request_body_sha256=planned["body_sha256"],
            request_body_length=planned["body_length"],
        )
        try:
            announced = self._announce(marker)
        except Exception as exc:
            raise OpenRouterTransportError(
                "report_generation_failed",
                "model call announcement failed",
            ) from exc
        if announced is not True:
            raise OpenRouterTransportError(
                "report_generation_failed",
                "model call announcement rejected",
                planned_request_sha256=planned_digest,
            )

        self._called = True

        def _post_announce_error(
            reason_code: str,
            message: str,
            *,
            diag: str | None = None,
        ) -> OpenRouterTransportError:
            return OpenRouterTransportError(
                reason_code,
                message,
                planned_request_sha256=planned_digest,
                diag=diag,
            )

        try:
            # Stream the body so the configured byte cap can abort mid-transfer
            # without first buffering an unbounded successful response.
            with self._client.stream(
                "POST",
                OPENROUTER_URL,
                content=body,
                headers={
                    **OPENROUTER_HEADERS,
                    "authorization": f"Bearer {self._api_key}",
                },
            ) as response:
                if response.is_redirect or response.history:
                    raise _post_announce_error(
                        "openrouter_unavailable", "OpenRouter redirect refused"
                    )
                final_port = response.url.port or (443 if response.url.scheme == "https" else None)
                if (
                    response.url.scheme != "https"
                    or response.url.host != "openrouter.ai"
                    or final_port != 443
                    or response.url.path != OPENROUTER_PATH
                    or response.url.query
                ):
                    raise _post_announce_error("tls_failed", "OpenRouter destination drifted")
                if response.status_code in {401, 403}:
                    raise _post_announce_error(
                        "openrouter_auth_failed", "OpenRouter authentication failed"
                    )
                if response.status_code == 429:
                    raise _post_announce_error("openrouter_rate_limited", "OpenRouter rate limited")
                if not 200 <= response.status_code < 300:
                    raise _post_announce_error(
                        "openrouter_unavailable", "OpenRouter returned an error"
                    )
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise _post_announce_error(
                        "compressed_response_forbidden",
                        "compressed response is forbidden",
                    )
                response_body = _read_response_body_bounded(
                    response, maximum=self._limits["response"]
                )
                _require_exact_returned_model(response_body)
                try:
                    model_output = parse_model_policy_output(
                        response_body,
                        allowed_evidence_paths=allowed_evidence_paths,
                    )
                except ReviewPolicyError as exc:
                    raise _post_announce_error(
                        "policy_output_malformed",
                        "OpenRouter policy output is malformed",
                        diag=short_policy_error_class(exc),
                    ) from exc
                metadata = _metadata_bytes(response)
                if metadata is not None and len(metadata) > self._limits["metadata"]:
                    raise _post_announce_error(
                        "metadata_bounds",
                        "metadata exceeds configured bound",
                    )
                observed = {
                    "schema_version": REVIEW_TRANSPORT_SCHEMA_VERSION,
                    "planned_request_sha256": planned_digest,
                    "final_origin": OPENROUTER_ORIGIN,
                    "final_path": OPENROUTER_PATH,
                    "tls_hostname": "openrouter.ai",
                    "tls_hostname_verified": True,
                    "redirected": False,
                    "proxied": False,
                    "response_status": response.status_code,
                    "response_content_encoding": "identity",
                    "response_body_sha256": sha256(response_body).hexdigest(),
                    "response_body_length": len(response_body),
                    "metadata_sha256": (
                        sha256(metadata).hexdigest() if metadata is not None else None
                    ),
                }
                observed_bytes = validate_observed_openrouter_transport(observed)
                # Live OR path: confirm planned/observed digests are coherent for
                # production admission (not library-only helpers). Fail closed if
                # capture digests cannot form a real OpenRouter observation bind.
                from .or_outcome_bind import (
                    ReviewOrOutcomeError,
                    build_openrouter_observation,
                    require_real_or_digests,
                )

                try:
                    # reconstruct response model from body for pin check
                    import json as _json

                    try:
                        body_obj = _json.loads(response_body.decode("utf-8"))
                        returned_model = str(
                            body_obj.get("model") if isinstance(body_obj, dict) else REVIEW_MODEL
                        )
                    except Exception:
                        returned_model = REVIEW_MODEL
                    if not is_pinned_review_model(returned_model):
                        returned_model = REVIEW_MODEL
                    try:
                        response_id = str(
                            body_obj.get("id") if isinstance(body_obj, dict) else "openrouter-live"
                        )
                    except Exception:
                        response_id = "openrouter-live"
                    if not response_id:
                        response_id = "openrouter-live"
                    or_obs = build_openrouter_observation(
                        planned=planned,
                        observed=observed,
                        request_body_sha256=str(planned["body_sha256"]),
                        request_body_length=int(planned["body_length"]),
                        response_id=response_id,
                        returned_model=returned_model,
                        response_status=int(response.status_code),
                        metadata_sha256=(
                            sha256(metadata).hexdigest() if metadata is not None else None
                        ),
                    )
                    require_real_or_digests(
                        planned=planned,
                        observed=observed,
                        openrouter_observation=or_obs,
                    )
                except ReviewOrOutcomeError as exc:
                    raise _post_announce_error(
                        "planned_digest_unbound",
                        f"openrouter digests unbound: {exc.code}",
                    ) from exc
                return OpenRouterCapture(
                    planned=planned,
                    planned_bytes=planned_bytes,
                    planned_sha256=planned_digest,
                    observed=observed,
                    observed_bytes=observed_bytes,
                    request_body=body,
                    response_body=response_body,
                    metadata=metadata,
                    model_output=model_output,
                )
        except OpenRouterTransportError as exc:
            if exc.planned_request_sha256 is None:
                exc.planned_request_sha256 = planned_digest
            raise
        except httpx.ProxyError as exc:
            raise _post_announce_error("tls_failed", "proxy use is forbidden") from exc
        except _NETWORK_FAILURES as exc:
            raise _post_announce_error(
                _network_reason(exc), "direct OpenRouter exchange failed"
            ) from exc
        except httpx.HTTPError as exc:
            raise _post_announce_error(
                "openrouter_unavailable", "direct OpenRouter exchange failed"
            ) from exc


def _validate_direct_request_body(
    body: bytes,
    *,
    routing_sha256: str,
    max_request_bytes: int = MAX_OPENROUTER_REQUEST_BYTES,
) -> None:
    if not isinstance(body, bytes) or not 1 <= len(body) <= max_request_bytes:
        raise OpenRouterTransportError(
            "report_generation_failed", "request body exceeds configured bound"
        )
    try:
        value = parse_json_object(body)
    except ValueError as exc:
        raise OpenRouterTransportError(
            "report_generation_failed", "request body is malformed"
        ) from exc
    if set(value) != {
        "messages",
        "model",
        "provider",
        "stream",
        "tool_choice",
        "tools",
    }:
        raise OpenRouterTransportError(
            "report_generation_failed", "request body fields are not exact"
        )
    if (
        value["model"] != REVIEW_MODEL
        or value["stream"] is not False
        or value["tool_choice"] != "auto"
        or value["tools"] != review_policy_tools()
    ):
        raise OpenRouterTransportError(
            "report_generation_failed", "request model or stream mode is invalid"
        )
    try:
        routing = validate_review_routing(value["provider"])
    except ReviewTransportSchemaError as exc:
        raise OpenRouterTransportError(
            "report_generation_failed", "request routing is invalid"
        ) from exc
    if canonical_sha256(routing) != routing_sha256:
        raise OpenRouterTransportError(
            "report_generation_failed", "request routing digest mismatches"
        )


def _network_reason(exc: BaseException) -> str:
    """Map transport failures: TLS cert/host/handshake → tls_failed, else dns/unavailable."""

    if _exception_chain_has_tls(exc):
        return "tls_failed"
    message = _exception_chain_message(exc)
    if _TLS_MESSAGE_RE.search(message):
        return "tls_failed"
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        if _DNS_MESSAGE_RE.search(message):
            return "dns_failed"
        # Conservative default for connect/timeout without TLS evidence is dns.
        return "dns_failed"
    return "openrouter_unavailable"


def _exception_chain_has_tls(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                ssl.SSLError,
                ssl.SSLCertVerificationError,
                ssl.CertificateError,
                ssl.SSLZeroReturnError,
                ssl.SSLWantReadError,
                ssl.SSLWantWriteError,
                ssl.SSLSyscallError,
                ssl.SSLEOFError,
            ),
        ):
            return True
        # Some environments stack CertificateError / SSLError under ConnectError.
        if type(current).__name__ in {
            "SSLCertVerificationError",
            "CertificateError",
            "SSLError",
            "SSLHandshakeError",
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


def _exception_chain_message(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current))
        parts.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    return " | ".join(parts)


def _read_response_body_bounded(response: httpx.Response, *, maximum: int) -> bytes:
    """Accumulate response chunks, aborting as soon as the byte cap is exceeded."""

    if maximum < 1:
        raise OpenRouterTransportError("response_malformed", "response exceeds configured bound")
    chunks: list[bytes] = []
    total = 0
    try:
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            total += len(chunk)
            if total > maximum:
                # Abort without retaining the unbounded tail.
                response.close()
                raise OpenRouterTransportError(
                    "response_malformed", "response exceeds configured bound"
                )
            chunks.append(chunk)
    except OpenRouterTransportError:
        raise
    except httpx.HTTPError as exc:
        raise OpenRouterTransportError(
            _network_reason(exc), "direct OpenRouter exchange failed"
        ) from exc
    body = b"".join(chunks)
    if not 1 <= len(body) <= maximum:
        raise OpenRouterTransportError("response_malformed", "response exceeds configured bound")
    return body


def _require_exact_returned_model(response_body: bytes) -> None:
    try:
        response = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenRouterTransportError(
            "openrouter_body_not_json", "OpenRouter response is not JSON"
        ) from exc
    if not isinstance(response, Mapping):
        raise OpenRouterTransportError(
            "openrouter_body_not_json", "OpenRouter response is not a JSON object"
        )
    if not is_pinned_review_model(response.get("model")):
        raise OpenRouterTransportError(
            "model_pin_mismatch", "OpenRouter returned model mismatches pin"
        )


def _metadata_bytes(response: httpx.Response) -> bytes | None:
    value = response.headers.get("x-openrouter-metadata")
    return value.encode("utf-8") if value else None


__all__ = [
    "DirectOpenRouterClient",
    "OPENROUTER_ORIGIN",
    "OPENROUTER_PATH",
    "OPENROUTER_URL",
    "OpenRouterCapture",
    "OpenRouterTransportError",
    "build_model_call_started",
    "build_openrouter_request_body",
    "build_planned_openrouter_request",
    "build_review_infrastructure_failure",
    "infrastructure_failure_reason",
    "openrouter_byte_limits_from_settings",
    "openrouter_timeout_from_settings",
    "short_policy_error_class",
    "short_quote_measurement_diag_class",
    "short_report_envelope_diag_class",
]
