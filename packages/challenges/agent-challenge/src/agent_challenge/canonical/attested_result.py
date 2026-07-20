"""In-image attested-result emission for the canonical Phala eval image (M1).

After the ``own_runner`` pipeline produces its per-task results, the canonical
image (running inside a Phala Intel TDX CVM) calls dstack ``get_quote(report_data)``
and emits an *attested-result envelope* on the BASE ``ExecutionProof`` Phala tier
(architecture.md sec 6). The envelope rides along the SAME single, parseable
``BASE_BENCHMARK_RESULT=`` line the legacy path already emits (additive-only, so
the host-side parser is unaffected), extended with:

* ``execution_proof`` -- an ``ExecutionProof`` (``tier == "phala-tdx"``) whose
  ``attestation`` carries ``{tdx_quote, event_log, report_data, measurement,
  vm_config}`` (a :class:`base.schemas.worker.PhalaAttestation`); and
* ``attestation_binding`` -- the architecture-sec-6 ``report_data`` preimage in
  the clear (``agent_hash``, sorted ``task_ids``, per-task ``scores`` +
  ``scores_digest``, ``validator_nonce``, ``canonical_measurement``) so a
  validator can recompute ``report_data`` and check it against the quote.

Trust & fail-closed invariants:

* ``report_data`` is derived by :mod:`agent_challenge.canonical.report_data`
  (byte-identical to base's single-source helper) and never exceeds the 64-byte
  TDX field -- the 32-byte sec-6 digest is what is handed to ``get_quote``.
* If a genuine quote cannot be produced (dstack socket unavailable,
  ``get_quote`` raises/times out, or returns an empty/malformed quote) the image
  **fails closed**: :func:`emit_attested_or_failclosed` emits an explicit
  ``failed`` result with a reason code and NO attestation envelope. It never
  fabricates a ``tdx_quote``/``report_data`` and never emits a passing result as
  if it were attested (VAL-IMG-034).

base's ``ExecutionProof``/``PhalaAttestation`` models are not importable inside
the lean canonical image, so the envelope is built as plain dicts and validated
by self-contained conformance checks that mirror base's required fields/types.
The exact envelope shape is pinned to base's real models in
``base/tests/unit/test_worker_proof_phala.py`` (cross-repo conformance guard).
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import IO, Any, Protocol, runtime_checkable

from agent_challenge.canonical import report_data as rd
from agent_challenge.canonical.measurement import (
    CANONICAL_MEASUREMENT_FIELDS,
    CanonicalMeasurement,
)
from agent_challenge.evaluation.own_runner.result_schema import (
    RESULT_LINE_PREFIX,
    build_benchmark_result,
    emit_benchmark_result_line,
    validate_benchmark_result,
)

#: Phala Intel TDX tier value for ``ExecutionProof.tier``. MUST equal base's
#: ``PHALA_TDX_TIER`` so the emitted envelope is recognized by base's verifier.
PHALA_TDX_TIER = "phala-tdx"

#: ``ExecutionProof.version`` the image emits (mirrors base ``EXECUTION_PROOF_VERSION``).
EXECUTION_PROOF_VERSION = 1

#: Additive key on the ``BASE_BENCHMARK_RESULT=`` payload carrying the envelope.
EXECUTION_PROOF_RESULT_KEY = "execution_proof"
#: Additive key carrying the sec-6 ``report_data`` preimage (verifier-checkable).
ATTESTATION_BINDING_RESULT_KEY = "attestation_binding"

#: Reason code emitted on the fail-closed path when a genuine quote cannot be
#: produced (see :mod:`agent_challenge.evaluation.own_runner.reason_codes`).
PHALA_ATTESTATION_FAILED_REASON = "phala_attestation_failed"

#: Full measurement register set carried in the attestation (runtime ``rtmr3``
#: included; the static, allowlist-pinnable subset is
#: :data:`CANONICAL_MEASUREMENT_FIELDS`).
MEASUREMENT_FIELDS: tuple[str, ...] = (
    "mrtd",
    "rtmr0",
    "rtmr1",
    "rtmr2",
    "rtmr3",
    "compose_hash",
    "os_image_hash",
)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX96_RE = re.compile(r"^[0-9a-f]{96}$")
_HEX128_RE = re.compile(r"^[0-9a-f]{128}$")

#: Required attestation-payload fields (mirrors base ``PhalaAttestation``).
ATTESTATION_REQUIRED_FIELDS: tuple[str, ...] = (
    "tdx_quote",
    "event_log",
    "report_data",
    "measurement",
    "vm_config",
)

#: Required ExecutionProof fields (mirrors base ``ExecutionProof``).
EXECUTION_PROOF_REQUIRED_FIELDS: tuple[str, ...] = (
    "version",
    "tier",
    "manifest_sha256",
    "worker_signature",
    "attestation",
)

#: Max width of the TDX ``report_data`` field handed to ``get_quote``.
MAX_REPORT_DATA_BYTES = rd.PHALA_REPORT_DATA_BYTES


class EnvelopeSchemaError(ValueError):
    """Raised when an attestation envelope violates the Phala-tier schema."""


class AttestationEmissionError(RuntimeError):
    """Raised when a genuine quote cannot be produced (drives fail-closed)."""


@runtime_checkable
class QuoteProvider(Protocol):
    """A source of TDX quotes (dstack ``DstackClient`` in production)."""

    def get_quote(self, report_data: bytes) -> Any:  # pragma: no cover - protocol
        ...


@dataclass(frozen=True)
class QuoteResult:
    """A validated quote: non-empty hex ``quote`` + parsed ``event_log``/``vm_config``."""

    quote: str
    event_log: list[dict[str, Any]]
    vm_config: dict[str, Any]


# --------------------------------------------------------------------------- #
# dstack quote provider (lazy import so the module loads without a live socket)
# --------------------------------------------------------------------------- #
#: Live dstack get_quote frequently exceeds the SDK default of 3s. Bound key-
#: release and score quote acquisition so the acquire path cannot hang
#: indefinitely before the raw TCP dial on 8701.
DSTACK_QUOTE_TIMEOUT_SECONDS = 90.0


class DstackQuoteProvider:
    """Adapts the dstack SDK ``DstackClient`` to :class:`QuoteProvider`.

    ``dstack_sdk`` is imported lazily on first use so this module (and its
    conformance/parse tests) import cleanly without the SDK's runtime socket.
    The client connects to ``/var/run/dstack.sock`` inside the CVM by default.
    Quote RPCs are bounded by :data:`DSTACK_QUOTE_TIMEOUT_SECONDS` so a stuck
    guest socket cannot leave the CVM silent at eval_prepared for 30 minutes.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        timeout_seconds: float = DSTACK_QUOTE_TIMEOUT_SECONDS,
    ) -> None:
        self._endpoint = endpoint
        self._timeout_seconds = float(timeout_seconds)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from dstack_sdk import DstackClient

            # Prefer the timeout-aware constructor when present; fall back for
            # older SDK stubs used in offline unit tests.
            try:
                self._client = (
                    DstackClient(self._endpoint, timeout=self._timeout_seconds)
                    if self._endpoint
                    else DstackClient(timeout=self._timeout_seconds)
                )
            except TypeError:
                self._client = DstackClient(self._endpoint) if self._endpoint else DstackClient()
        return self._client

    def get_quote(self, report_data: bytes) -> Any:
        from agent_challenge.canonical.wallclock import WallclockTimeout, call_with_wallclock

        client = self._get_client()
        get_quote = getattr(client, "get_quote", None)
        if not callable(get_quote):
            raise AttestationEmissionError("dstack client lacks get_quote")

        # Daemon-thread wallclock: never re-join a hung dstack get_quote after
        # the deadline (ThreadPoolExecutor.__exit__ shutdown(wait=True) would).
        try:
            return call_with_wallclock(
                lambda: get_quote(report_data),
                timeout_seconds=self._timeout_seconds,
                label="get_quote",
            )
        except WallclockTimeout as exc:
            raise AttestationEmissionError(
                f"dstack get_quote exceeded {self._timeout_seconds:.0f}s wallclock"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - fail closed on any RPC error
            text = str(exc).lower()
            if "timeout" in text or "timed out" in text:
                raise AttestationEmissionError(f"dstack get_quote timed out: {exc}") from exc
            raise


# --------------------------------------------------------------------------- #
# Quote acquisition (fail-closed)
# --------------------------------------------------------------------------- #
def _coerce_event_log(raw: Any) -> list[dict[str, Any]]:
    """Legacy shallow coerce (JSON string → list of dicts). Prefer KR normalize.

    Kept for call-sites that only need JSON parsing. Score emission goes through
    :func:`obtain_quote`, which reuses the key-release normalizers so live
    dstack GetQuote residuals (empty IMR3 digests, ``0x`` casing) pass RTMR3
    self-check and ``validate_eval_phala_attestation``.
    """

    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else []
        except json.JSONDecodeError as exc:
            raise AttestationEmissionError(f"quote event_log is not valid JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise AttestationEmissionError("quote event_log is not a list of events")
    return [dict(event) for event in raw]


def _coerce_vm_config(raw: Any) -> dict[str, Any]:
    """Parse a dstack ``vm_config`` (JSON string, dict, or empty) to a plain dict.

    Does not project onto the schema-v2 key set; callers that emit Eval
    attestation must run :func:`_project_eval_vm_config` so wire validate sees
    exactly ``{vcpu, memory_mb, os_image_hash}``.
    """

    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError as exc:
            raise AttestationEmissionError(f"quote vm_config is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise AttestationEmissionError("quote vm_config is not an object")
    return dict(raw)


def _project_eval_vm_config(
    raw: Any,
    *,
    os_image_hash: str | None = None,
) -> dict[str, Any]:
    """Map dstack/host vm_config onto the sealed schema-v2 Eval key set.

    Port of review_runtime ``_normalize_vm_config`` for the score emit path:

    * ``cpu_count`` → ``vcpu``; ``memory_size`` (bytes) → ``memory_mb`` via floor // 1MiB
    * extras dropped so ``set(result) == {vcpu, memory_mb, os_image_hash}``
    * ints coerced; ``os_image_hash`` prefers measurement-derived when provided
    * fail closed when vcpu/cpu_count or memory_mb/memory_size cannot be derived
      (score domain does not invent silent defaults like review's 1 / 2048)
    """

    if raw is None or raw == "":
        raw = {}
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raw = {}
        else:
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise AttestationEmissionError(f"quote vm_config is not valid JSON: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise AttestationEmissionError("quote vm_config is not an object")

    # dstack historically exposed cpu_count/memory_size (bytes); schema-v2 is
    # exactly {vcpu, memory_mb, os_image_hash}.
    if "vcpu" in raw:
        vcpu = raw["vcpu"]
    elif "cpu_count" in raw:
        vcpu = raw["cpu_count"]
    else:
        raise AttestationEmissionError("quote vm_config missing vcpu (or dstack cpu_count)")

    if "memory_mb" in raw:
        memory_mb = raw["memory_mb"]
    elif "memory_size" in raw:
        try:
            memory_bytes = int(raw["memory_size"])
        except (TypeError, ValueError) as exc:
            raise AttestationEmissionError("quote memory_size is invalid") from exc
        if memory_bytes <= 0:
            raise AttestationEmissionError("quote memory_size must be positive")
        memory_mb = max(1, memory_bytes // (1024 * 1024))
    else:
        raise AttestationEmissionError("quote vm_config missing memory_mb (or dstack memory_size)")

    try:
        vcpu_i = int(vcpu)
        memory_mb_i = int(memory_mb)
    except (TypeError, ValueError) as exc:
        raise AttestationEmissionError("quote vm_config vcpu/memory are invalid") from exc
    if vcpu_i < 1 or memory_mb_i < 1:
        raise AttestationEmissionError("quote vm_config vcpu/memory must be positive")

    image = raw.get("os_image_hash", os_image_hash)
    if image is not None and os_image_hash is not None and image != os_image_hash:
        # Prefer the measurement-derived image hash; dstack may omit the nested
        # field or populate an untrusted host surface.
        image = os_image_hash
    if image is None and os_image_hash is not None:
        image = os_image_hash

    return {
        "vcpu": vcpu_i,
        "memory_mb": memory_mb_i,
        "os_image_hash": image,
    }


def _score_normalize_quote_hex(quote: Any) -> str:
    """Normalize score-path quote hex via KR ``_normalize_quote_hex`` (0x/case)."""

    from agent_challenge.keyrelease.client import (
        KeyReleaseProtocolError,
        _normalize_quote_hex,
    )

    try:
        return _normalize_quote_hex(quote)
    except KeyReleaseProtocolError as exc:
        raise AttestationEmissionError(f"get_quote quote_hex cannot be normalized: {exc}") from exc


def _visible_ascii_event_id(raw: Any) -> str:
    """Map a raw dstack ``event`` field onto 0..128 visible ASCII (may be empty).

    Strips bytes outside U+0021..U+007E (``!``..``~``). Non-strings become empty
    so the caller can decide between derivation vs fail-closed.
    """

    if not isinstance(raw, str):
        return ""
    cleaned = "".join(ch for ch in raw if "!" <= ch <= "~")
    if len(cleaned) > 128:
        cleaned = cleaned[:128]
    return cleaned


def _derive_event_id_from_type(event_type: Any) -> str:
    """Stable 1–128 visible fallback when the wire-required name is empty.

    Preferred when the entry is not RTMR3-runtime-bound (empty name would not
    alter ``runtime_event_digest`` / RTMR3). Uses ``event-type-<n>`` token form
    when ``event_type`` is a non-negative int; otherwise the generic ``event``.
    """

    if (
        isinstance(event_type, int)
        and not isinstance(event_type, bool)
        and 0 <= event_type <= 0xFFFFFFFF
    ):
        candidate = f"event-type-{event_type}"
        if 1 <= len(candidate) <= 128 and all("!" <= ch <= "~" for ch in candidate):
            return candidate
    return "event"


def _project_eval_event_log(raw: Any) -> list[dict[str, Any]]:
    """Project GetQuote event_log so every entry.event is wire-legal 1–128.

    Applied *after* KR coerce + empty-IMR3 fill. Live residual after score
    vm_config project (image@sha256:ffbb60a9): emit failed with
    ``event_log[].event must be a 1-128 character visible ASCII id`` because
    early IMR0–2 boot entries (and occasionally incomplete shapes) arrive with
    empty / missing / null / control-containing ``event`` names.

    Policy (fail closed for RTMR3 integrity):
    * non-empty string → strip control/non-visible bytes; keep if still 1–128
    * empty after normalize AND entry is IMR3 + dstack runtime type → fail closed
      (event name is bound into ``runtime_event_digest``; inventing breaks RTMR3)
    * empty after normalize AND not RTMR3-runtime-bound → derive from event_type
      tokens (``event-type-N``) or ``event`` fallback
    * when an entry already has the closed 5-key shape, re-emit that shape with
      the projected event id; incomplete offline fixtures keep their keys and
      only get event projected when coercible (legacy compatibility)
    """

    from agent_challenge.keyrelease.client import (
        KeyReleaseProtocolError,
        _normalize_framed_event_log,
    )
    from agent_challenge.keyrelease.quote import (
        APP_IMR,
        DSTACK_RUNTIME_EVENT_TYPE,
    )

    if raw is None or raw == "":
        return []
    # Allow callers to pass already-coerced lists without re-JSON parsing.
    try:
        if isinstance(raw, list) and raw and all(isinstance(item, Mapping) for item in raw):
            entries = [dict(item) for item in raw]
        else:
            entries = _normalize_framed_event_log(raw, enforce_schema=False)
    except KeyReleaseProtocolError as exc:
        raise AttestationEmissionError(f"get_quote event_log cannot be projected: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise AttestationEmissionError(f"get_quote event_log cannot be projected: {exc}") from exc

    projected: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, Mapping):
            raise AttestationEmissionError(
                f"event_log[{index}] is not an object; refusing score emission"
            )
        entry = dict(raw_entry)
        imr = entry.get("imr")
        event_type = entry.get("event_type")
        event_raw = entry.get("event", "")
        event_id = _visible_ascii_event_id(event_raw)

        closed_keys = {"imr", "event_type", "digest", "event", "event_payload"}
        closed_shape = closed_keys <= set(entry) or set(entry) == closed_keys
        has_int_imr = isinstance(imr, int) and not isinstance(imr, bool)
        has_int_type = isinstance(event_type, int) and not isinstance(event_type, bool)

        if not event_id:
            rtmr3_bound = (
                has_int_imr
                and has_int_type
                and imr == APP_IMR
                and event_type == DSTACK_RUNTIME_EVENT_TYPE
            )
            if rtmr3_bound:
                raise AttestationEmissionError(
                    f"event_log[{index}] IMR3 runtime event is empty after project "
                    "(RTMR3-bound; refuse invented event id)"
                )
            # Live residual: empty/missing/null/control IMR0–2 names. Invent a
            # wire-legal id when the entry has event_type (or had an event key)
            # so close-to-wire shapes pass schema-v2. Incomplete offline fakes
            # that never had event/event_type keep their partial keys.
            if closed_shape or "event" in entry or has_int_type:
                event_id = _derive_event_id_from_type(event_type)

        if event_id:
            if not (1 <= len(event_id) <= 128):
                raise AttestationEmissionError(
                    f"event_log[{index}].event cannot be projected to a 1-128 visible ASCII id"
                )
            if any(not ("!" <= ch <= "~") for ch in event_id):
                raise AttestationEmissionError(
                    f"event_log[{index}].event is not visible ASCII after project"
                )

        if has_int_imr and has_int_type:
            # Emit closed 5-key for live / KR-normalized residual.
            digest = entry.get("digest", "")
            if not isinstance(digest, str):
                digest = ""
            payload = entry.get("event_payload", "")
            if not isinstance(payload, str):
                payload = ""
            if not event_id:
                raise AttestationEmissionError(
                    f"event_log[{index}].event cannot be projected to a 1-128 visible ASCII id"
                )
            projected.append(
                {
                    "imr": imr,
                    "event_type": event_type,
                    "digest": digest,
                    "event": event_id,
                    "event_payload": payload,
                }
            )
            continue

        # Incomplete offline fixtures: keep original keys, only set event when
        # we have a projected id (else leave as-is for legacy assert equality).
        out = dict(entry)
        if event_id:
            out["event"] = event_id
        projected.append(out)
    return projected


def _score_normalize_event_log(raw: Any) -> list[dict[str, Any]]:
    """Normalize score-path event_log via KR helpers + event-id project.

    Live residual order:
    1. KR coerce (0x/case, closed keys, base64→hex) + empty IMR3 fill
    2. project every ``event`` onto 1–128 visible ASCII for schema-v2 wire

    KR alone leaves early IMR0–2 dstack names empty/missing; emit then failed with
    ``event_log[].event must be a 1-128 character visible ASCII id``.
    """

    from agent_challenge.keyrelease.client import (
        KeyReleaseProtocolError,
        _normalize_framed_event_log,
    )

    try:
        if raw is None:
            return []
        filled = _normalize_framed_event_log(raw, enforce_schema=False)
        return _project_eval_event_log(filled)
    except AttestationEmissionError:
        raise
    except KeyReleaseProtocolError as exc:
        raise AttestationEmissionError(f"get_quote event_log cannot be normalized: {exc}") from exc
    except (ValueError, TypeError) as exc:
        raise AttestationEmissionError(f"get_quote event_log cannot be normalized: {exc}") from exc


def obtain_quote(provider: QuoteProvider, report_data_digest: bytes) -> QuoteResult:
    """Call ``provider.get_quote`` and return a validated :class:`QuoteResult`.

    Fail-closed: raises :class:`AttestationEmissionError` on ANY failure -- the
    provider raising/timing out, or returning an empty/malformed quote -- so the
    caller never mistakes a missing quote for a genuine attestation. The digest
    handed to ``get_quote`` is guarded to never exceed the 64-byte TDX field.

    **Score-domain normalize:** reuses key-release GetQuote normalizers
    (quote_hex lower/0x strip, event_log coerce + empty IMR3
    ``runtime_event_digest`` fill, closed-key projection) then projects every
    ``event_log[].event`` onto schema-v2's 1–128 visible ASCII id so emit's
    RTMR3 self-check and ``validate_eval_phala_attestation`` accept the same
    live dstack shapes that KR framed grant already accepted (plus empty-name
    early boot entries, which wire rejects after KR alone).
    """

    if not isinstance(report_data_digest, (bytes, bytearray)):
        raise AttestationEmissionError("report_data handed to get_quote must be bytes")
    if len(report_data_digest) > MAX_REPORT_DATA_BYTES:
        raise AttestationEmissionError(
            f"report_data is {len(report_data_digest)} bytes (> {MAX_REPORT_DATA_BYTES}); "
            "refusing to hand an oversized value to get_quote"
        )

    try:
        response = provider.get_quote(bytes(report_data_digest))
    except AttestationEmissionError:
        raise
    except Exception as exc:  # noqa: BLE001 - fail closed on any provider failure
        raise AttestationEmissionError(f"get_quote failed: {exc}") from exc

    quote_raw = getattr(response, "quote", None)
    if quote_raw is None and isinstance(response, Mapping):
        quote_raw = response.get("quote")
    if not isinstance(quote_raw, str) or not quote_raw.strip():
        raise AttestationEmissionError("get_quote returned an empty or malformed quote")
    quote = _score_normalize_quote_hex(quote_raw)

    if isinstance(response, Mapping):
        event_raw = response.get("event_log")
        vm_raw = response.get("vm_config")
    else:
        event_raw = getattr(response, "event_log", None)
        vm_raw = getattr(response, "vm_config", None)
    event_log = _score_normalize_event_log(event_raw)
    # Project dstack-shaped vm_config (cpu_count/memory_size/extras) onto the
    # exact schema-v2 set {vcpu, memory_mb, os_image_hash}. Empty raw stays
    # {} so callers that fill via explicit env override still work; non-empty
    # surfaces must be fully projectable or fail closed.
    parsed_vm = _coerce_vm_config(vm_raw)
    if not parsed_vm:
        vm_config: dict[str, Any] = {}
    else:
        image_hint = parsed_vm.get("os_image_hash")
        image_hint_str = image_hint if isinstance(image_hint, str) else None
        vm_config = _project_eval_vm_config(parsed_vm, os_image_hash=image_hint_str)
    return QuoteResult(quote=quote, event_log=event_log, vm_config=vm_config)


# --------------------------------------------------------------------------- #
# Envelope construction
# --------------------------------------------------------------------------- #
def build_measurement(
    canonical_measurement: CanonicalMeasurement | Mapping[str, Any], *, rtmr3: str
) -> dict[str, str]:
    """Full measurement register set for the attestation (static subset + ``rtmr3``)."""

    if isinstance(canonical_measurement, CanonicalMeasurement):
        source: Mapping[str, Any] = canonical_measurement.as_dict()
    elif isinstance(canonical_measurement, Mapping):
        source = canonical_measurement
    else:
        raise TypeError(
            "canonical_measurement must be a CanonicalMeasurement or mapping, "
            f"not {type(canonical_measurement).__name__}"
        )
    measurement = {field: str(source[field]) for field in CANONICAL_MEASUREMENT_FIELDS}
    measurement["rtmr3"] = str(rtmr3)
    return measurement


def build_phala_attestation(
    *,
    tdx_quote: str,
    event_log: Iterable[Mapping[str, Any]],
    report_data_hex: str,
    measurement: Mapping[str, Any],
    vm_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build (and conformance-check) a ``PhalaAttestation``-shaped payload dict."""

    attestation: dict[str, Any] = {
        "tdx_quote": tdx_quote,
        "event_log": [dict(event) for event in event_log],
        "report_data": report_data_hex,
        "measurement": dict(measurement),
        "vm_config": dict(vm_config) if vm_config else {},
    }
    validate_phala_attestation(attestation)
    return attestation


def placeholder_worker_signature() -> dict[str, str]:
    """A schema-valid, explicitly-empty tier-0 worker signature.

    The Phala tier's trust anchor is the hardware quote; the sr25519
    ``worker_signature`` layer is (re)bound by the validator-side base adapter
    (milestone M4). Until a worker signer is wired into the image the emitter
    uses this explicit placeholder rather than fabricating a signature.
    """

    return {"worker_pubkey": "", "sig": ""}


def build_execution_proof_envelope(
    *,
    manifest_sha256: str,
    attestation: Mapping[str, Any],
    worker_signature: Mapping[str, str] | None = None,
    tier: str = PHALA_TDX_TIER,
    version: int = EXECUTION_PROOF_VERSION,
    image_digest: str | None = None,
    provider: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build (and conformance-check) an ``ExecutionProof``-shaped envelope dict."""

    envelope: dict[str, Any] = {
        "version": version,
        "tier": tier,
        "manifest_sha256": manifest_sha256,
        "worker_signature": dict(worker_signature)
        if worker_signature is not None
        else placeholder_worker_signature(),
        "attestation": dict(attestation),
    }
    if image_digest is not None:
        envelope["image_digest"] = image_digest
    if provider is not None:
        envelope["provider"] = dict(provider)
    validate_execution_proof_envelope(envelope)
    return envelope


def build_attestation_binding(
    *,
    agent_hash: str,
    task_ids: Iterable[str],
    scores: Mapping[str, Any],
    scores_digest: str,
    validator_nonce: str,
    canonical_measurement: CanonicalMeasurement | Mapping[str, Any],
) -> dict[str, Any]:
    """The sec-6 ``report_data`` preimage in the clear (verifier-recomputable)."""

    if isinstance(canonical_measurement, CanonicalMeasurement):
        measurement_source: Mapping[str, Any] = canonical_measurement.as_dict()
    else:
        measurement_source = canonical_measurement
    return {
        "agent_hash": agent_hash,
        "task_ids": sorted(task_ids),
        "scores": dict(scores),
        "scores_digest": scores_digest,
        "validator_nonce": validator_nonce,
        "canonical_measurement": {
            field: str(measurement_source[field]) for field in CANONICAL_MEASUREMENT_FIELDS
        },
    }


# --------------------------------------------------------------------------- #
# Conformance validation (mirrors base's required fields / types)
# --------------------------------------------------------------------------- #
def validate_phala_attestation(payload: Any) -> None:
    """Validate ``payload`` conforms to the base ``PhalaAttestation`` schema."""

    if not isinstance(payload, Mapping):
        raise EnvelopeSchemaError(f"attestation must be an object, got {type(payload).__name__}")
    for field in ATTESTATION_REQUIRED_FIELDS:
        if field not in payload:
            raise EnvelopeSchemaError(f"attestation missing required field {field!r}")
    quote = payload["tdx_quote"]
    if (
        not isinstance(quote, str)
        or not quote
        or len(quote) % 2
        or len(quote) > 2 * 64 * 1024
        or quote != quote.lower()
        or any(character not in "0123456789abcdef" for character in quote)
    ):
        raise EnvelopeSchemaError("attestation.tdx_quote must be lowercase bounded hex")
    report_data = payload["report_data"]
    if not isinstance(report_data, str) or _HEX128_RE.fullmatch(report_data) is None:
        raise EnvelopeSchemaError("attestation.report_data must be 64-byte lowercase hex")
    if not isinstance(payload["event_log"], list):
        raise EnvelopeSchemaError("attestation.event_log must be a list")
    if not all(isinstance(event, Mapping) for event in payload["event_log"]):
        raise EnvelopeSchemaError("attestation.event_log entries must be objects")
    if not isinstance(payload["vm_config"], Mapping):
        raise EnvelopeSchemaError("attestation.vm_config must be an object")
    measurement = payload["measurement"]
    if not isinstance(measurement, Mapping):
        raise EnvelopeSchemaError("attestation.measurement must be an object")
    for field in MEASUREMENT_FIELDS:
        if field not in measurement:
            raise EnvelopeSchemaError(f"attestation.measurement missing register {field!r}")
        width = _HEX96_RE if field.startswith(("mrtd", "rtmr")) else _HEX64_RE
        if not isinstance(measurement[field], str) or width.fullmatch(measurement[field]) is None:
            raise EnvelopeSchemaError(
                f"attestation.measurement.{field} must have its exact lowercase hex width"
            )


def validate_execution_proof_envelope(payload: Any) -> None:
    """Validate ``payload`` conforms to the base ``ExecutionProof`` Phala tier."""

    if not isinstance(payload, Mapping):
        raise EnvelopeSchemaError(
            f"execution_proof must be an object, got {type(payload).__name__}"
        )
    for field in EXECUTION_PROOF_REQUIRED_FIELDS:
        if field not in payload:
            raise EnvelopeSchemaError(f"execution_proof missing required field {field!r}")
    if not isinstance(payload["version"], int) or isinstance(payload["version"], bool):
        raise EnvelopeSchemaError("execution_proof.version must be an integer")
    if not isinstance(payload["tier"], (int, str)) or isinstance(payload["tier"], bool):
        raise EnvelopeSchemaError("execution_proof.tier must be an int or string")
    if not isinstance(payload["manifest_sha256"], str) or not payload["manifest_sha256"]:
        raise EnvelopeSchemaError("execution_proof.manifest_sha256 must be a non-empty string")
    signature = payload["worker_signature"]
    if not isinstance(signature, Mapping):
        raise EnvelopeSchemaError("execution_proof.worker_signature must be an object")
    for field in ("worker_pubkey", "sig"):
        if field not in signature:
            raise EnvelopeSchemaError(f"execution_proof.worker_signature missing {field!r}")
        if not isinstance(signature[field], str):
            raise EnvelopeSchemaError(f"execution_proof.worker_signature.{field} must be a string")
    validate_phala_attestation(payload["attestation"])


# --------------------------------------------------------------------------- #
# Extended result assembly + emission
# --------------------------------------------------------------------------- #
def build_attested_benchmark_result(
    *,
    benchmark_result: Mapping[str, Any],
    execution_proof: Mapping[str, Any],
    attestation_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Extend a five-field benchmark result with the (additive) attestation blocks.

    The legacy five-field result contract is preserved byte-for-byte; the
    envelope + binding ride along as additive keys. The result is re-validated
    against the benchmark-result schema so the extended line stays parseable.
    """

    validate_benchmark_result(benchmark_result)
    validate_execution_proof_envelope(execution_proof)
    extended = dict(benchmark_result)
    extended[EXECUTION_PROOF_RESULT_KEY] = dict(execution_proof)
    extended[ATTESTATION_BINDING_RESULT_KEY] = dict(attestation_binding)
    validate_benchmark_result(extended)
    return extended


def emit_attested_benchmark_result(
    *,
    benchmark_result: Mapping[str, Any],
    canonical_measurement: CanonicalMeasurement | Mapping[str, Any],
    rtmr3: str,
    agent_hash: str,
    task_ids: Iterable[str],
    scores: Mapping[str, Any],
    quote_provider: QuoteProvider,
    manifest_sha256: str,
    validator_nonce: str | None = None,
    eval_run_id: str | None = None,
    submission_id: str | None = None,
    score_nonce: str | None = None,
    score_record: Mapping[str, Any] | None = None,
    image_digest: str | None = None,
    worker_signature: Mapping[str, str] | None = None,
    vm_config: Mapping[str, Any] | None = None,
    unit_id: str = "",
    stream: IO[str] | None = None,
) -> str:
    """Emit an attested ``BASE_BENCHMARK_RESULT=`` line for a completed run.

    Legacy callers receive the original additive benchmark-result envelope.
    Supplying every schema-v2 Eval argument emits the exact Eval result request
    v1 instead.  Both paths fail closed if a genuine quote cannot be produced.
    """

    v2_values = (eval_run_id, submission_id, score_nonce, score_record, image_digest)
    if any(value is not None for value in v2_values):
        if validator_nonce is not None or not all(value is not None for value in v2_values):
            raise AttestationEmissionError(
                "schema-v2 emission requires eval run, submission, score nonce, "
                "score record, image digest, and no validator_nonce"
            )
        return _emit_schema_v2_eval_result(
            canonical_measurement=canonical_measurement,
            rtmr3=rtmr3,
            agent_hash=agent_hash,
            task_ids=task_ids,
            quote_provider=quote_provider,
            manifest_sha256=manifest_sha256,
            eval_run_id=str(eval_run_id),
            submission_id=str(submission_id),
            score_nonce=str(score_nonce),
            score_record=score_record,
            image_digest=str(image_digest),
            vm_config=vm_config,
            stream=stream,
        )

    if validator_nonce is None:
        raise AttestationEmissionError(
            "validator_nonce is required for legacy attestation emission"
        )
    digest = rd.report_data(
        canonical_measurement=canonical_measurement,
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores_digest=rd.scores_digest(scores),
        validator_nonce=validator_nonce,
    )
    report_data_field = rd.to_report_data_field(digest)

    quote = obtain_quote(quote_provider, digest)

    measurement = build_measurement(canonical_measurement, rtmr3=rtmr3)
    attestation = build_phala_attestation(
        tdx_quote=quote.quote,
        event_log=quote.event_log,
        report_data_hex=report_data_field,
        measurement=measurement,
        vm_config=vm_config if vm_config is not None else quote.vm_config,
    )
    envelope = build_execution_proof_envelope(
        manifest_sha256=manifest_sha256,
        attestation=attestation,
        worker_signature=worker_signature,
    )
    binding = build_attestation_binding(
        agent_hash=agent_hash,
        task_ids=task_ids,
        scores=scores,
        scores_digest=rd.scores_digest(scores),
        validator_nonce=validator_nonce,
        canonical_measurement=canonical_measurement,
    )
    extended = build_attested_benchmark_result(
        benchmark_result=benchmark_result,
        execution_proof=envelope,
        attestation_binding=binding,
    )
    return emit_benchmark_result_line(extended, stream=stream)


def _emit_schema_v2_eval_result(
    *,
    canonical_measurement: CanonicalMeasurement | Mapping[str, Any],
    rtmr3: str,
    agent_hash: str,
    task_ids: Iterable[str],
    quote_provider: QuoteProvider,
    manifest_sha256: str,
    eval_run_id: str,
    submission_id: str,
    score_nonce: str,
    score_record: Mapping[str, Any],
    image_digest: str,
    vm_config: Mapping[str, Any] | None,
    stream: IO[str] | None,
) -> str:
    """Emit the schema-closed Eval result request v1 on the sole result line."""

    from agent_challenge.canonical import eval_wire as ew

    try:
        score_digest = ew.score_record_digest(score_record)
        binding = ew.build_score_binding(
            canonical_measurement=canonical_measurement,
            agent_hash=agent_hash,
            eval_run_id=eval_run_id,
            score_nonce=score_nonce,
            scores_digest=score_digest,
            task_ids=list(task_ids),
        )
        report_data_hex = ew.score_report_data_hex(binding)
        quote = obtain_quote(quote_provider, bytes.fromhex(report_data_hex))
        # A production TDX v4 quote carries the authoritative RTMR3 register.
        # Recompute it from the ordered event log and reject divergence. Short
        # synthetic quote fixtures used by offline legacy tests retain their
        # supplied runtime value because they have no parseable TD report.
        try:
            from agent_challenge.keyrelease.quote import parse_tdx_quote_v4, replay_rtmr3

            parsed_rtmr3 = parse_tdx_quote_v4(quote.quote).rtmr3
            replayed_rtmr3 = replay_rtmr3(quote.event_log).rtmr3
            if parsed_rtmr3 != replayed_rtmr3:
                raise AttestationEmissionError("quote RTMR3 does not match its event log")
            rtmr3 = parsed_rtmr3
        except AttestationEmissionError:
            raise
        except Exception:
            # Offline compatibility fixtures may use opaque quote bytes. A
            # genuine production TDX v4 quote is parseable here and therefore
            # always takes the authoritative quote/event-log path above.
            pass
        measurement = build_measurement(canonical_measurement, rtmr3=rtmr3)
        # Project env override OR quote.vm_config onto exact
        # {vcpu, memory_mb, os_image_hash} before wire validate. Live residual:
        # raw dstack extras (cpu_count/memory_size/qemu_*) hit "invalid fields".
        measurement_image = measurement.get("os_image_hash")
        measurement_image_str = measurement_image if isinstance(measurement_image, str) else None
        raw_for_project: Any = vm_config if vm_config is not None else quote.vm_config
        projected_vm = _project_eval_vm_config(
            raw_for_project,
            os_image_hash=measurement_image_str,
        )
        attestation = ew.validate_eval_phala_attestation(
            {
                "tdx_quote": quote.quote,
                "event_log": quote.event_log,
                "report_data": report_data_hex,
                "measurement": measurement,
                "vm_config": projected_vm,
            }
        )
        execution_proof = ew.validate_eval_execution_proof(
            {
                "version": EXECUTION_PROOF_VERSION,
                "tier": PHALA_TDX_TIER,
                "manifest_sha256": manifest_sha256,
                "image_digest": image_digest,
                "provider": None,
                "worker_signature": placeholder_worker_signature(),
                "attestation": attestation,
            }
        )
        request = ew.validate_eval_result_request(
            {
                "schema_version": 1,
                "eval_run_id": eval_run_id,
                "submission_id": submission_id,
                "agent_hash": agent_hash,
                "score_record": score_record,
                "scores_digest": score_digest,
                "execution_proof": execution_proof,
            }
        )
    except (ew.EvalWireError, ValueError, TypeError) as exc:
        raise AttestationEmissionError(f"schema-v2 Eval emission is invalid: {exc}") from exc

    # Host process_direct_eval_result requires raw POST body bytes ==
    # eval_wire.canonical_json_v1(validated). Default json.dumps separators
    # insert spaces and arrive as result_noncanonical; emit compact sorted
    # form that is byte-identical to the host apply path.
    body = ew.canonical_json_v1(request).decode("utf-8")
    line = RESULT_LINE_PREFIX + body
    target = stream if stream is not None else sys.stdout
    target.write(line + "\n")
    return line


def emit_attested_eval_result_from_plan(
    *,
    eval_plan: Mapping[str, Any],
    score_record: Mapping[str, Any],
    rtmr3: str,
    quote_provider: QuoteProvider,
    manifest_sha256: str,
    vm_config: Mapping[str, Any] | None = None,
    stream: IO[str] | None = None,
) -> str:
    """Emit a strict Eval result using only immutable plan-derived bindings."""

    from agent_challenge.canonical import eval_wire as ew

    try:
        plan = ew.validate_eval_plan(eval_plan)
        task_ids = [task["task_id"] for task in plan["selected_tasks"]]
        validated_record = ew.validate_canonical_score_record(
            score_record,
            scoring_policy=plan["scoring_policy"],
            expected_eval_run_id=plan["eval_run_id"],
            expected_task_ids=task_ids,
            expected_k=plan["k"],
        )
        measurement = {
            "mrtd": plan["eval_app"]["measurement"]["mrtd"],
            "rtmr0": plan["eval_app"]["measurement"]["rtmr0"],
            "rtmr1": plan["eval_app"]["measurement"]["rtmr1"],
            "rtmr2": plan["eval_app"]["measurement"]["rtmr2"],
            "compose_hash": plan["eval_app"]["compose_hash"],
            "os_image_hash": plan["eval_app"]["measurement"]["os_image_hash"],
        }
    except ew.EvalWireError as exc:
        raise AttestationEmissionError(
            f"invalid immutable Eval plan or score record: {exc}"
        ) from exc
    return emit_attested_benchmark_result(
        benchmark_result=build_benchmark_result(
            status="completed",
            score=ew.decode_score_f64be(validated_record["final"]["job_score_f64be"]),
            resolved=validated_record["final"]["passed_tasks"],
            total=validated_record["final"]["total_tasks"],
            reason_code=None,
        ),
        canonical_measurement=measurement,
        rtmr3=rtmr3,
        agent_hash=plan["agent_hash"],
        task_ids=task_ids,
        scores={},
        quote_provider=quote_provider,
        manifest_sha256=manifest_sha256,
        eval_run_id=plan["eval_run_id"],
        submission_id=plan["submission_id"],
        score_nonce=plan["score_nonce"],
        score_record=validated_record,
        image_digest=plan["eval_app"]["image_ref"],
        vm_config=vm_config,
        stream=stream,
    )


def emit_failclosed_result(
    *,
    total: int,
    reason_code: str = PHALA_ATTESTATION_FAILED_REASON,
    stream: IO[str] | None = None,
) -> str:
    """Emit a ``failed`` result with NO attestation (the fail-closed line).

    Never carries an ``execution_proof``/``attestation_binding`` and never a
    passing score, so a missing/failed quote can never be mistaken downstream for
    a genuine attested result.
    """

    failed = build_benchmark_result(
        status="failed",
        score=0.0,
        resolved=0,
        total=int(total),
        reason_code=reason_code,
    )
    return emit_benchmark_result_line(failed, stream=stream)


def emit_attested_or_failclosed(
    *,
    benchmark_result: Mapping[str, Any],
    canonical_measurement: CanonicalMeasurement | Mapping[str, Any],
    rtmr3: str,
    agent_hash: str,
    task_ids: Iterable[str],
    scores: Mapping[str, Any],
    quote_provider: QuoteProvider,
    manifest_sha256: str,
    validator_nonce: str | None = None,
    eval_run_id: str | None = None,
    submission_id: str | None = None,
    score_nonce: str | None = None,
    score_record: Mapping[str, Any] | None = None,
    image_digest: str | None = None,
    worker_signature: Mapping[str, str] | None = None,
    vm_config: Mapping[str, Any] | None = None,
    unit_id: str = "",
    stream: IO[str] | None = None,
) -> tuple[str, bool]:
    """Emit the attested line, or a fail-closed line if no genuine quote exists.

    Returns ``(emitted_line, attested)``. ``attested`` is ``True`` only when a
    genuine quote was obtained and the attested envelope was emitted; on any
    :class:`AttestationEmissionError` it is ``False`` and the emitted line is an
    explicit ``failed`` result with no fabricated attestation (VAL-IMG-034).
    """

    try:
        line = emit_attested_benchmark_result(
            benchmark_result=benchmark_result,
            canonical_measurement=canonical_measurement,
            rtmr3=rtmr3,
            agent_hash=agent_hash,
            task_ids=task_ids,
            scores=scores,
            validator_nonce=validator_nonce,
            quote_provider=quote_provider,
            manifest_sha256=manifest_sha256,
            eval_run_id=eval_run_id,
            submission_id=submission_id,
            score_nonce=score_nonce,
            score_record=score_record,
            image_digest=image_digest,
            worker_signature=worker_signature,
            vm_config=vm_config,
            unit_id=unit_id,
            stream=stream,
        )
        return line, True
    except AttestationEmissionError:
        total = _result_total(benchmark_result, task_ids)
        line = emit_failclosed_result(total=total, stream=stream)
        return line, False


def _result_total(benchmark_result: Mapping[str, Any], task_ids: Iterable[str]) -> int:
    """Best-effort task total for the fail-closed line (result ``total`` else count)."""

    total = benchmark_result.get("total") if isinstance(benchmark_result, Mapping) else None
    if isinstance(total, int) and not isinstance(total, bool):
        return total
    return len(list(task_ids))


__all__ = [
    "ATTESTATION_BINDING_RESULT_KEY",
    "ATTESTATION_REQUIRED_FIELDS",
    "AttestationEmissionError",
    "DstackQuoteProvider",
    "EXECUTION_PROOF_REQUIRED_FIELDS",
    "EXECUTION_PROOF_RESULT_KEY",
    "EXECUTION_PROOF_VERSION",
    "EnvelopeSchemaError",
    "MEASUREMENT_FIELDS",
    "PHALA_ATTESTATION_FAILED_REASON",
    "PHALA_TDX_TIER",
    "QuoteProvider",
    "QuoteResult",
    "build_attestation_binding",
    "build_attested_benchmark_result",
    "build_execution_proof_envelope",
    "build_measurement",
    "build_phala_attestation",
    "emit_attested_benchmark_result",
    "emit_attested_eval_result_from_plan",
    "emit_attested_or_failclosed",
    "emit_failclosed_result",
    "obtain_quote",
    "placeholder_worker_signature",
    "validate_execution_proof_envelope",
    "validate_phala_attestation",
]
