"""TDX quote parsing, event-log replay, and signature/TCB verification (M4).

The validator/master Phala-tier verifier (``verify_execution_proof``) must, before
accepting an attested result, confirm that its Intel TDX quote:

1. is cryptographically valid on genuine hardware with an acceptable TCB posture
   (delegated to a :class:`QuoteVerifier` -- ``dcap-qvl`` / Phala verify); and
2. carries the expected measurement registers + ``report_data`` (parsed here
   structurally from the hardware-signed TD report); and
3. has an RTMR3 that a replay of its event log reproduces, yielding the canonical
   ``compose_hash`` (so the compose is bound by content, not trusted by value).

Structural parsing (register/report_data offsets) and the dstack ``cc-eventlog``
RTMR replay follow the same hardware/dstack facts the agent-challenge in-CVM
key-release path uses, so a live dstack event log verifies here unmodified. The
crucial base-side addition is the **park vs reject** distinction: a *cryptographic*
failure raises :class:`QuoteVerificationError` (the verifier returns False), while
a *transient* dependency outage/timeout raises :class:`VerifierUnavailableError`
so the caller PARKS the result (never accepts, never fraud-rejects) -- VAL-VERIFY-014.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------- #
# TDX quote v4 (ECDSA) layout: 48-byte header + TD report (TDREPORT10)
# --------------------------------------------------------------------------- #
#: Quote header length (bytes) preceding the TD report body.
QUOTE_HEADER_LEN = 48
#: SHA-384 measurement register width (bytes).
REGISTER_LEN = 48
#: TDX ``report_data`` field width (bytes).
REPORT_DATA_LEN = 64

# Absolute byte offsets of the fields within a v4 quote (header + TD report body).
_MRTD_OFFSET = QUOTE_HEADER_LEN + 136
_RTMR0_OFFSET = QUOTE_HEADER_LEN + 328
_RTMR1_OFFSET = QUOTE_HEADER_LEN + 376
_RTMR2_OFFSET = QUOTE_HEADER_LEN + 424
_RTMR3_OFFSET = QUOTE_HEADER_LEN + 472
_REPORT_DATA_OFFSET = QUOTE_HEADER_LEN + 520

#: Minimum length (bytes) a quote must have to contain the full TD report.
MIN_QUOTE_LEN = _REPORT_DATA_OFFSET + REPORT_DATA_LEN

# --------------------------------------------------------------------------- #
# dstack event-log constants (cc-eventlog)
# --------------------------------------------------------------------------- #
#: dstack runtime event type (RTMR3 app events). Not a TCG-defined value.
DSTACK_RUNTIME_EVENT_TYPE = 0x08000001
#: RTMR3 event name carrying the normalized compose hash.
COMPOSE_HASH_EVENT = "compose-hash"
#: RTMR3 event name carrying the KMS key-provider identity.
KEY_PROVIDER_EVENT = "key-provider"
#: The RTMR index whose event log binds the app compose (RTMR3).
APP_IMR = 3


class QuoteError(Exception):
    """Base error for quote parsing / verification failures (fail closed)."""


class QuoteStructureError(QuoteError):
    """The quote bytes are malformed / too short to contain a TD report."""


class QuoteVerificationError(QuoteError):
    """The quote is cryptographically invalid, or its event log is inconsistent.

    A *cryptographic* verdict: the caller should REJECT the result (verify False).
    """


class VerifierUnavailableError(QuoteError):
    """The verification dependency is transiently unreachable / timed out.

    NOT a cryptographic verdict: the caller should PARK the result (retry later),
    never accept it and never permanently fraud-reject it (VAL-VERIFY-014).
    """


@dataclass(frozen=True)
class TdReport:
    """Measurement registers + ``report_data`` read from a TDX quote's TD report."""

    mrtd: str
    rtmr0: str
    rtmr1: str
    rtmr2: str
    rtmr3: str
    report_data: bytes


def _coerce_register(value: bytes | str, *, field_name: str) -> bytes:
    if isinstance(value, str):
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError(f"{field_name} is not valid hex: {exc}") from exc
    elif isinstance(value, bytes | bytearray):
        raw = bytes(value)
    else:
        raise TypeError(
            f"{field_name} must be hex str or bytes, not {type(value).__name__}"
        )
    if len(raw) != REGISTER_LEN:
        raise ValueError(f"{field_name} must be {REGISTER_LEN} bytes, got {len(raw)}")
    return raw


def parse_quote_hex(quote_hex: str) -> bytes:
    """Decode a hex quote string to bytes; fail closed on malformed hex."""

    if not isinstance(quote_hex, str) or not quote_hex:
        raise QuoteStructureError("quote must be a non-empty hex string")
    text = quote_hex.strip()
    if text.startswith(("0x", "0X")):
        text = text[2:]
    try:
        return bytes.fromhex(text)
    except ValueError as exc:
        raise QuoteStructureError(f"quote is not valid hex: {exc}") from exc


def parse_td_report(quote: bytes | str) -> TdReport:
    """Parse the measurement registers + ``report_data`` from a TDX quote.

    Reads MRTD/RTMR0-3 (SHA-384) and the 64-byte ``report_data`` at their fixed
    v4-quote offsets. Raises :class:`QuoteStructureError` if the quote is too
    short to contain a full TD report.
    """

    raw = parse_quote_hex(quote) if isinstance(quote, str) else bytes(quote)
    if len(raw) < MIN_QUOTE_LEN:
        raise QuoteStructureError(
            f"quote is {len(raw)} bytes, need at least {MIN_QUOTE_LEN} for a TD report"
        )

    def _reg(offset: int) -> str:
        return raw[offset : offset + REGISTER_LEN].hex()

    return TdReport(
        mrtd=_reg(_MRTD_OFFSET),
        rtmr0=_reg(_RTMR0_OFFSET),
        rtmr1=_reg(_RTMR1_OFFSET),
        rtmr2=_reg(_RTMR2_OFFSET),
        rtmr3=_reg(_RTMR3_OFFSET),
        report_data=raw[_REPORT_DATA_OFFSET : _REPORT_DATA_OFFSET + REPORT_DATA_LEN],
    )


def os_image_hash_from_registers(mrtd: str, rtmr1: str, rtmr2: str) -> str:
    """The dstack ``mr_image`` OS identity: ``sha256(MRTD ∥ RTMR1 ∥ RTMR2)``.

    Matches the canonical eval image's ``os_image_hash`` (dstack ``mr_image``) so
    a quote's registers reproduce the value a validator pins in the allowlist.
    """

    preimage = (
        _coerce_register(mrtd, field_name="mrtd")
        + _coerce_register(rtmr1, field_name="rtmr1")
        + _coerce_register(rtmr2, field_name="rtmr2")
    )
    return hashlib.sha256(preimage).hexdigest()


# --------------------------------------------------------------------------- #
# Event-log replay (RTMR3)
# --------------------------------------------------------------------------- #
def runtime_event_digest(event_name: str, payload: bytes) -> bytes:
    """SHA-384 digest of a dstack runtime (RTMR3) event (cc-eventlog format).

    ``SHA384( event_type(le32) ∥ b":" ∥ event_name ∥ b":" ∥ payload )`` with
    ``event_type == DSTACK_RUNTIME_EVENT_TYPE``. Binding the digest to the payload
    means a forged compose-hash payload cannot keep a matching RTMR3.
    """

    hasher = hashlib.sha384()
    hasher.update(DSTACK_RUNTIME_EVENT_TYPE.to_bytes(4, "little"))
    hasher.update(b":")
    hasher.update(event_name.encode("utf-8"))
    hasher.update(b":")
    hasher.update(payload)
    return hasher.digest()


def _rtmr_extend(mr: bytes, digest: bytes) -> bytes:
    return hashlib.sha384(mr + digest).digest()


@dataclass(frozen=True)
class Rtmr3Replay:
    """Result of replaying an event log into RTMR3."""

    rtmr3: str
    compose_hash: str | None
    key_provider: str | None


def _payload_bytes(entry: Mapping[str, Any]) -> bytes:
    payload = entry.get("event_payload", "")
    if isinstance(payload, bytes | bytearray):
        return bytes(payload)
    if not isinstance(payload, str):
        raise QuoteVerificationError("event_payload must be a hex string or bytes")
    if payload == "":
        return b""
    try:
        return bytes.fromhex(payload)
    except ValueError as exc:
        raise QuoteVerificationError(f"event_payload is not valid hex: {exc}") from exc


def replay_rtmr3(event_log: Iterable[Mapping[str, Any]]) -> Rtmr3Replay:
    """Replay the RTMR3 (``imr == 3``) events, binding each digest to its payload.

    For dstack runtime events the digest is recomputed from the event contents
    and must equal the logged digest (else the log is inconsistent and rejected),
    then folded ``RTMR = SHA384(RTMR || digest)``. The ``compose-hash`` and
    ``key-provider`` event payloads are surfaced for the allowlist check.
    """

    mr = bytes(REGISTER_LEN)
    compose_hash: str | None = None
    key_provider: str | None = None

    for entry in event_log:
        if not isinstance(entry, Mapping):
            raise QuoteVerificationError("event log entries must be objects")
        if entry.get("imr") != APP_IMR:
            continue
        event_name = entry.get("event", "")
        if not isinstance(event_name, str):
            raise QuoteVerificationError("event 'event' name must be a string")
        payload = _payload_bytes(entry)
        event_type = entry.get("event_type")

        if event_type == DSTACK_RUNTIME_EVENT_TYPE:
            digest = runtime_event_digest(event_name, payload)
            logged = entry.get("digest")
            if isinstance(logged, str) and logged:
                try:
                    logged_bytes = bytes.fromhex(logged)
                except ValueError as exc:
                    raise QuoteVerificationError(
                        f"event digest is not valid hex: {exc}"
                    ) from exc
                if logged_bytes != digest:
                    raise QuoteVerificationError(
                        f"event '{event_name}' digest does not match its payload"
                    )
        else:
            logged = entry.get("digest")
            if not isinstance(logged, str) or not logged:
                raise QuoteVerificationError(
                    "non-runtime RTMR3 event is missing a digest"
                )
            try:
                digest = bytes.fromhex(logged)
            except ValueError as exc:
                raise QuoteVerificationError(
                    f"event digest is not valid hex: {exc}"
                ) from exc
            if len(digest) != REGISTER_LEN:
                raise QuoteVerificationError("RTMR3 event digest must be 48 bytes")

        mr = _rtmr_extend(mr, digest)
        if event_name == COMPOSE_HASH_EVENT:
            compose_hash = payload.hex()
        elif event_name == KEY_PROVIDER_EVENT:
            key_provider = payload.hex()

    return Rtmr3Replay(
        rtmr3=mr.hex(), compose_hash=compose_hash, key_provider=key_provider
    )


# --------------------------------------------------------------------------- #
# Cryptographic verification (signature + TCB)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuoteVerdict:
    """A verified quote's TCB posture (its bytes are Intel-signed / authentic)."""

    tcb_status: str
    advisory_ids: tuple[str, ...] = ()


@runtime_checkable
class QuoteVerifier(Protocol):
    """Verifies a TDX quote's DCAP signature + certificate chain.

    Returns a :class:`QuoteVerdict` (carrying the TCB status) when the quote is
    cryptographically valid, raises :class:`QuoteVerificationError` when it is not
    (invalid signature / broken cert chain / malformed), and raises
    :class:`VerifierUnavailableError` when the verification dependency itself is
    transiently unreachable (so the caller parks rather than rejects).
    """

    def verify(self, quote_hex: str) -> QuoteVerdict:  # pragma: no cover - protocol
        ...


@dataclass
class DcapQvlVerifier:
    """Trustless quote verification via the ``dcap-qvl`` CLI (Intel PCS).

    ``dcap-qvl`` verifies the quote against Intel collateral and reports the TCB
    status; this adapter shells out and parses that verdict. ``runner`` is
    injectable for testing. The accept / reject / park mapping is deliberate
    (VAL-VERIFY-014):

    * a **non-zero exit** is a cryptographic verdict -- the tool judged the quote
      invalid / its TCB unacceptable -- so it PERMANENTLY rejects
      (:class:`QuoteVerificationError`); and
    * a **timeout / missing binary / subprocess error** is a transient outage, so
      it PARKS (:class:`VerifierUnavailableError`, retryable); and
    * an **exit-0-but-unparseable / non-object / missing-TCB-status** output is a
      *tooling* regression (dcap-qvl accepted the quote's cryptography but changed
      its stdout format), NOT a fraud verdict -- so it PARKS
      (:class:`VerifierUnavailableError`) rather than permanently fraud-rejecting a
      legitimate result. It is never accepted (no verdict is returned).
    """

    binary: str = "dcap-qvl"
    timeout: float = 30.0
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        if self.runner is not None:
            return self.runner(args)
        return subprocess.run(  # pragma: no cover - real CLI invoked live (M6)
            args, capture_output=True, text=True, timeout=self.timeout
        )

    def verify(self, quote_hex: str) -> QuoteVerdict:
        args = [self.binary, "verify", "--hex", quote_hex]
        try:
            proc = self._run(args)
        except FileNotFoundError as exc:
            raise VerifierUnavailableError(f"dcap-qvl not available: {exc}") from exc
        except subprocess.TimeoutExpired as exc:
            raise VerifierUnavailableError(f"dcap-qvl timed out: {exc}") from exc
        except subprocess.SubprocessError as exc:
            raise VerifierUnavailableError(
                f"dcap-qvl invocation failed: {exc}"
            ) from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise QuoteVerificationError(f"dcap-qvl rejected the quote: {detail}")

        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise VerifierUnavailableError(
                f"dcap-qvl exited 0 but its output is not JSON "
                f"(tooling regression -- park, do not reject): {exc}"
            ) from exc
        if not isinstance(report, Mapping):
            raise VerifierUnavailableError(
                "dcap-qvl exited 0 but its output was not a JSON object "
                "(tooling regression -- park, do not reject)"
            )

        status = (
            report.get("status") or report.get("tcbStatus") or report.get("tcb_status")
        )
        if not isinstance(status, str) or not status:
            raise VerifierUnavailableError(
                "dcap-qvl exited 0 but its output is missing a TCB status "
                "(tooling regression -- park, do not reject)"
            )
        advisories = report.get("advisory_ids") or report.get("advisoryIDs") or []
        if not isinstance(advisories, Sequence) or isinstance(advisories, str | bytes):
            advisories = []
        return QuoteVerdict(
            tcb_status=status, advisory_ids=tuple(str(a) for a in advisories)
        )


@dataclass(frozen=True)
class StaticQuoteVerifier:
    """A :class:`QuoteVerifier` with a fixed verdict (tests / offline harness).

    ``tcb_status`` is the posture reported for any quote; ``valid=False`` rejects
    every quote as if its signature did not verify (:class:`QuoteVerificationError`);
    ``unavailable=True`` models a transient verifier outage
    (:class:`VerifierUnavailableError`, park).
    """

    tcb_status: str = "UpToDate"
    valid: bool = True
    unavailable: bool = False
    advisory_ids: tuple[str, ...] = field(default_factory=tuple)

    def verify(self, quote_hex: str) -> QuoteVerdict:
        if self.unavailable:
            raise VerifierUnavailableError("quote verifier is unavailable")
        if not self.valid:
            raise QuoteVerificationError("quote signature verification failed")
        return QuoteVerdict(tcb_status=self.tcb_status, advisory_ids=self.advisory_ids)


# --------------------------------------------------------------------------- #
# Quote / event-log assembly (tooling + test-vector generation)
# --------------------------------------------------------------------------- #
def build_tdx_quote(
    *,
    mrtd: bytes | str,
    rtmr0: bytes | str,
    rtmr1: bytes | str,
    rtmr2: bytes | str,
    rtmr3: bytes | str,
    report_data: bytes | str,
    header: bytes = b"",
    tail: bytes = b"",
) -> str:
    """Assemble a minimal v4-layout TDX quote (hex) with the given fields.

    The inverse of :func:`parse_td_report`: places each register + ``report_data``
    at its fixed offset. Used to generate deterministic test vectors / fixtures
    (the real quote is produced by dstack ``get_quote`` on a live CVM).
    """

    buf = bytearray(MIN_QUOTE_LEN)
    if header:
        buf[: min(len(header), QUOTE_HEADER_LEN)] = header[:QUOTE_HEADER_LEN]
    buf[_MRTD_OFFSET : _MRTD_OFFSET + REGISTER_LEN] = _coerce_register(
        mrtd, field_name="mrtd"
    )
    buf[_RTMR0_OFFSET : _RTMR0_OFFSET + REGISTER_LEN] = _coerce_register(
        rtmr0, field_name="rtmr0"
    )
    buf[_RTMR1_OFFSET : _RTMR1_OFFSET + REGISTER_LEN] = _coerce_register(
        rtmr1, field_name="rtmr1"
    )
    buf[_RTMR2_OFFSET : _RTMR2_OFFSET + REGISTER_LEN] = _coerce_register(
        rtmr2, field_name="rtmr2"
    )
    buf[_RTMR3_OFFSET : _RTMR3_OFFSET + REGISTER_LEN] = _coerce_register(
        rtmr3, field_name="rtmr3"
    )
    if isinstance(report_data, str):
        rd = bytes.fromhex(report_data)
    else:
        rd = bytes(report_data)
    rd = rd[:REPORT_DATA_LEN].ljust(REPORT_DATA_LEN, b"\x00")
    buf[_REPORT_DATA_OFFSET : _REPORT_DATA_OFFSET + REPORT_DATA_LEN] = rd
    return (bytes(buf) + tail).hex()


def build_rtmr3_event_log(
    events: Sequence[tuple[str, bytes]],
) -> tuple[list[dict[str, Any]], str]:
    """Build a consistent RTMR3 event log + its replayed RTMR3 hex.

    ``events`` is an ordered sequence of ``(event_name, payload_bytes)``; each is
    emitted as a dstack runtime event with the correctly derived digest, and the
    folded RTMR3 is returned so a caller can pin it into a matching quote.
    """

    log: list[dict[str, Any]] = []
    mr = bytes(REGISTER_LEN)
    for name, payload in events:
        digest = runtime_event_digest(name, payload)
        log.append(
            {
                "imr": APP_IMR,
                "event_type": DSTACK_RUNTIME_EVENT_TYPE,
                "digest": digest.hex(),
                "event": name,
                "event_payload": payload.hex(),
            }
        )
        mr = _rtmr_extend(mr, digest)
    return log, mr.hex()


__all__ = [
    "APP_IMR",
    "COMPOSE_HASH_EVENT",
    "DSTACK_RUNTIME_EVENT_TYPE",
    "KEY_PROVIDER_EVENT",
    "MIN_QUOTE_LEN",
    "REGISTER_LEN",
    "REPORT_DATA_LEN",
    "DcapQvlVerifier",
    "QuoteError",
    "QuoteStructureError",
    "QuoteVerdict",
    "QuoteVerificationError",
    "QuoteVerifier",
    "Rtmr3Replay",
    "StaticQuoteVerifier",
    "TdReport",
    "VerifierUnavailableError",
    "build_rtmr3_event_log",
    "build_tdx_quote",
    "os_image_hash_from_registers",
    "parse_quote_hex",
    "parse_td_report",
    "replay_rtmr3",
    "runtime_event_digest",
]
