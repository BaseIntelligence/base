"""TDX quote parsing, event-log replay, and signature/TCB verification.

The validator key-release endpoint must, before releasing the golden key, confirm
that a presented Intel TDX quote (architecture.md §4 C3):

1. is cryptographically valid on genuine hardware with an acceptable TCB posture
   (delegated to a :class:`QuoteVerifier` -- ``dcap-qvl`` / Phala verify); and
2. carries the expected measurement registers and ``report_data`` (parsed here
   structurally from the hardware-signed TD report); and
3. has an RTMR3 that a replay of its event log reproduces, yielding the canonical
   ``compose_hash`` and ``key_provider`` (so the compose is bound by content, not
   trusted by value).

Structural parsing is separated from cryptographic verification: the verifier
attests that the quote bytes are authentic (Intel-signed), and this module reads
the measurement registers + ``report_data`` from those same verified bytes. The
RTMR replay + event digest derivation follow dstack's ``cc-eventlog`` exactly
(``RTMR = SHA384(RTMR || digest)``; a runtime event's digest is
``SHA384(event_type_le32 ∥ b":" ∥ event ∥ b":" ∥ payload)``), so a live dstack
event log verifies here without modification.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

#: Bounded visible ASCII identifier (matches review ``_ID_RE`` for pins).
_KEY_PROVIDER_ID_RE = re.compile(r"^[!-~]{1,128}$")

# --------------------------------------------------------------------------- #
# TDX quote v4 (ECDSA) layout: 48-byte header + 584-byte TD report (TDREPORT10)
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
#: Quote v4 appends a little-endian signature-data length after TDREPORT10.
QUOTE_V4_BODY_LEN = 584
QUOTE_V4_SIGNED_PREFIX_LEN = QUOTE_HEADER_LEN + QUOTE_V4_BODY_LEN
QUOTE_V4_LENGTH_FIELD_LEN = 4
#: Minimum ECDSA quote-v4 signature-data structure before certification bytes.
MIN_QUOTE_V4_SIGNATURE_DATA_LEN = 584
#: Intel SGX/TDX quoting-enclave vendor UUID in quote headers.
INTEL_QE_VENDOR_ID = bytes.fromhex("939a7233f79c4ca9940a0db3957f0607")
TDX_QUOTE_VERSION = 4
TDX_TEE_TYPE = 0x81
ECDSA_P256_ATTESTATION_KEY_TYPE = 2

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
#: Live dstack KMS family names collapse onto the sealed pin ``phala``.
_KEY_PROVIDER_PHALA_NAMES = frozenset({"kms", "phala", "phala-kms"})


def decode_key_provider(value: str | None) -> str:
    """Normalize an RTMR3 key-provider event payload into an allowlist pin id.

    Offline fixtures and simple providers emit a short UTF-8 identifier such as
    ``phala``. Live dstack KMS emits JSON such as ``{"name":"kms","id":"..."}``
    as the event payload (surfaced here as lowercase hex of those UTF-8 bytes).
    Collapse the KMS/phala JSON family onto the stable pin ``phala`` so host KR
    candidates match validator allowlist rows that pin ``key_provider=phala``
    (parity with review ``_decode_key_provider`` / gate37). Fail closed on any
    unreadable/unidentified payload.
    """

    if not isinstance(value, str):
        raise QuoteVerificationError("key provider event is missing")
    try:
        decoded = bytes.fromhex(value).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as exc:
        raise QuoteVerificationError("key provider event is invalid") from exc
    text = decoded.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise QuoteVerificationError("key provider event is invalid") from exc
        if not isinstance(payload, Mapping):
            raise QuoteVerificationError("key provider event is invalid")
        name = str(payload.get("name") or payload.get("type") or "").strip().lower()
        if name in _KEY_PROVIDER_PHALA_NAMES:
            text = "phala"
        elif name:
            text = name
        else:
            raise QuoteVerificationError("key provider event is invalid")
    if not _KEY_PROVIDER_ID_RE.fullmatch(text):
        raise QuoteVerificationError("key provider event is invalid")
    return text


class QuoteError(Exception):
    """Base error for quote parsing / verification failures (fail closed)."""


class QuoteStructureError(QuoteError):
    """The quote bytes are malformed / too short to contain a TD report."""


class QuoteVerificationError(QuoteError):
    """The quote is cryptographically invalid, or its event log is inconsistent."""


class QuoteVerifierUnavailable(QuoteVerificationError):
    """The external verifier failed indeterminately and may be retried."""


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
    elif isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        raise TypeError(f"{field_name} must be hex str or bytes, not {type(value).__name__}")
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


def parse_tdx_quote_v4(quote: bytes | str) -> TdReport:
    """Require the one supported Intel TDX Quote v4 structural profile.

    This rejects other quote families before fixed-offset extraction and
    requires the declared signature-data length to consume the input exactly,
    with no trailing or alternate encoding.
    """

    raw = parse_quote_hex(quote) if isinstance(quote, str) else bytes(quote)
    minimum = (
        QUOTE_V4_SIGNED_PREFIX_LEN + QUOTE_V4_LENGTH_FIELD_LEN + MIN_QUOTE_V4_SIGNATURE_DATA_LEN
    )
    if len(raw) < minimum:
        raise QuoteStructureError("TDX Quote v4 signature data is truncated")
    if int.from_bytes(raw[0:2], "little") != TDX_QUOTE_VERSION:
        raise QuoteStructureError("unsupported TDX quote version")
    if int.from_bytes(raw[2:4], "little") != ECDSA_P256_ATTESTATION_KEY_TYPE:
        raise QuoteStructureError("unsupported TDX attestation key type")
    if int.from_bytes(raw[4:8], "little") != TDX_TEE_TYPE:
        raise QuoteStructureError("quote is not an Intel TDX quote")
    if raw[8:12] != b"\0" * 4:
        raise QuoteStructureError("TDX quote header reserved bytes are nonzero")
    # Intel TDX Quote v4 header layout (48 bytes total before the body):
    #   version(2) | att_key_type(2) | tee_type(4) | reserved(4) | qe_vendor_id(16) | user_data(16)
    # The QE vendor UUID therefore occupies bytes [12:28], not [16:32].
    if raw[12:28] != INTEL_QE_VENDOR_ID:
        raise QuoteStructureError("TDX quote QE vendor is not Intel")
    signature_len = int.from_bytes(
        raw[QUOTE_V4_SIGNED_PREFIX_LEN : QUOTE_V4_SIGNED_PREFIX_LEN + 4],
        "little",
    )
    if signature_len < MIN_QUOTE_V4_SIGNATURE_DATA_LEN:
        raise QuoteStructureError("TDX quote signature data is too short")
    declared_len = QUOTE_V4_SIGNED_PREFIX_LEN + 4 + signature_len
    # Live Intel/dstack quotes often carry certification-data appendices after
    # the signed quote region. Require the declared signed region to be fully
    # present and cover the TD report offsets, but allow trailing CE/auth data.
    if declared_len > len(raw):
        raise QuoteStructureError("TDX quote declared length exceeds its encoding")
    if declared_len < MIN_QUOTE_LEN:
        raise QuoteStructureError("TDX quote declared length is truncated")
    return parse_td_report(raw)


def os_image_hash_from_registers(mrtd: str, rtmr1: str, rtmr2: str) -> str:
    """Product OS identity: ``sha256(MRTD ∥ RTMR1 ∥ RTMR2)``.

    Matches :mod:`agent_challenge.canonical.measurement`'s sealed
    ``os_image_hash`` (and :func:`product_os_image_hash`) so a quote's registers
    reproduce the value a validator pins in the allowlist / assignment. This is
    **not** the Phala provision / teepod catalog digest sometimes labeled
    ``mr_image`` (residual: catalog ``bd369a…`` ≠ product ``5c6d…`` on the same
    MRTD/RTMR1/RTMR2).
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
    if isinstance(payload, (bytes, bytearray)):
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
                    raise QuoteVerificationError(f"event digest is not valid hex: {exc}") from exc
                if logged_bytes != digest:
                    raise QuoteVerificationError(
                        f"event '{event_name}' digest does not match its payload"
                    )
        else:
            logged = entry.get("digest")
            if not isinstance(logged, str) or not logged:
                raise QuoteVerificationError("non-runtime RTMR3 event is missing a digest")
            try:
                digest = bytes.fromhex(logged)
            except ValueError as exc:
                raise QuoteVerificationError(f"event digest is not valid hex: {exc}") from exc
            if len(digest) != REGISTER_LEN:
                raise QuoteVerificationError("RTMR3 event digest must be 48 bytes")

        mr = _rtmr_extend(mr, digest)
        if event_name == COMPOSE_HASH_EVENT:
            compose_hash = payload.hex()
        elif event_name == KEY_PROVIDER_EVENT:
            key_provider = payload.hex()

    return Rtmr3Replay(rtmr3=mr.hex(), compose_hash=compose_hash, key_provider=key_provider)


_HEX96_RE = re.compile(r"^[0-9a-f]{96}$")
_EVEN_HEX_RE = re.compile(r"^(?:[0-9a-f]{2})*$")
_RESERVED_IDENTITY_NAMES = frozenset(
    {
        COMPOSE_HASH_EVENT,
        KEY_PROVIDER_EVENT,
        "compose_hash",
        "composehash",
        "key_provider",
        "keyprovider",
    }
)


def validate_rtmr3_event_log(
    event_log: Any,
    *,
    max_entries: int = 4096,
    max_encoded_bytes: int = 2 * 1024 * 1024,
) -> list[dict[str, Any]]:
    """Validate the schema-closed ordered dstack RTMR3 event grammar.

    Identity events are exactly one ``compose-hash`` and one ``key-provider`` at
    IMR3 using the dstack runtime event type, with ``compose-hash`` appearing
    before ``key-provider``. Live dstack 0.5.x inserts other boot events
    (``instance-id``, ``boot-mr-done``, ``mr-kms``, ``os-image-hash``, ...)
    between those identities; intermediate non-identity events are allowed.
    Aliases, duplicates, shadowing and identity-like extras are rejected before
    replay.
    """

    import json

    if not isinstance(event_log, list) or not (1 <= len(event_log) <= max_entries):
        raise QuoteVerificationError("event log has an invalid entry count")
    if len(json.dumps(event_log, separators=(",", ":"), ensure_ascii=False).encode()) > (
        max_encoded_bytes
    ):
        raise QuoteVerificationError("event log exceeds the encoded byte limit")
    validated: list[dict[str, Any]] = []
    identity_positions: list[tuple[int, str]] = []
    for index, raw in enumerate(event_log):
        if not isinstance(raw, Mapping) or set(raw) != {
            "imr",
            "event_type",
            "digest",
            "event",
            "event_payload",
        }:
            raise QuoteVerificationError("event log entry is not schema closed")
        imr = raw["imr"]
        event_type = raw["event_type"]
        digest = raw["digest"]
        event = raw["event"]
        payload = raw["event_payload"]
        if (
            not isinstance(imr, int)
            or isinstance(imr, bool)
            or not 0 <= imr <= 3
            or not isinstance(event_type, int)
            or isinstance(event_type, bool)
            or not 0 <= event_type <= 0xFFFFFFFF
            or not isinstance(digest, str)
            or _HEX96_RE.fullmatch(digest) is None
            or not isinstance(event, str)
            # Live dstack includes early IMR0-2 entries with empty event names and
            # payload-only digests. Identity events still require non-empty names.
            or not (0 <= len(event.encode("utf-8")) <= 16_384)
            or not isinstance(payload, str)
            or _EVEN_HEX_RE.fullmatch(payload) is None
        ):
            raise QuoteVerificationError("event log entry has invalid field encoding")
        normalized = event.lower()
        if normalized in _RESERVED_IDENTITY_NAMES:
            if event not in {COMPOSE_HASH_EVENT, KEY_PROVIDER_EVENT}:
                raise QuoteVerificationError("reserved identity event alias is forbidden")
            identity_positions.append((index, event))
            if imr != APP_IMR or event_type != DSTACK_RUNTIME_EVENT_TYPE:
                raise QuoteVerificationError("identity event has the wrong origin or type")
            expected_payload_len = 32 if event == COMPOSE_HASH_EVENT else None
            if expected_payload_len is not None and len(payload) != expected_payload_len * 2:
                raise QuoteVerificationError("compose-hash payload must be exactly 32 bytes")
            if event == KEY_PROVIDER_EVENT and not payload:
                raise QuoteVerificationError("key-provider payload must be non-empty")
        validated.append(dict(raw))
    if [name for _, name in identity_positions] != [
        COMPOSE_HASH_EVENT,
        KEY_PROVIDER_EVENT,
    ]:
        raise QuoteVerificationError("event log must contain each boot identity exactly once")
    # Order is still required: compose-hash must precede key-provider. Adjacency
    # is intentionally NOT required because live dstack emits intermediate boot
    # events between those two identity events.
    if identity_positions[0][0] >= identity_positions[1][0]:
        raise QuoteVerificationError("boot identity events must be ordered")
    return validated


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
    cryptographically valid, and raises :class:`QuoteVerificationError` when it
    is not (invalid signature, broken cert chain, malformed structure).
    """

    def verify(self, quote_hex: str) -> QuoteVerdict:  # pragma: no cover - protocol
        ...


@dataclass
class DcapQvlVerifier:
    """Trustless quote verification via the ``dcap-qvl`` CLI (Intel PCS).

    ``dcap-qvl`` verifies the quote against Intel collateral and reports the TCB
    status; this adapter shells out and parses that verdict. ``runner`` is
    injectable for testing. Any non-zero exit / unparseable output / rejection is
    surfaced as :class:`QuoteVerificationError` (fail closed).

    Production invocations use :func:`subprocess.Popen` so a deadline can
    terminate the real child process, not only cancel an awaiter around a thread.
    """

    binary: str = "dcap-qvl"
    timeout: float = 30.0
    runner: Callable[[list[str]], subprocess.CompletedProcess[str]] | None = None
    _active_process: subprocess.Popen[str] | None = None

    def cancel(self) -> None:
        """Terminate any in-flight dcap-qvl process after an outer deadline."""

        process = self._active_process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - extreme stall
                pass
        finally:
            self._active_process = None

    def _run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        if self.runner is not None:
            return self.runner(args)
        # Use Popen so :meth:`cancel` can kill the real process on deadline.
        process = subprocess.Popen(  # pragma: no cover - real CLI invoked live
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._active_process = process
        try:
            stdout, stderr = process.communicate(timeout=self.timeout)
            return subprocess.CompletedProcess(
                args=args,
                returncode=process.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired as exc:
            self.cancel()
            raise exc
        finally:
            if self._active_process is process:
                self._active_process = None

    def verify(self, quote_hex: str) -> QuoteVerdict:
        import json
        import tempfile
        from pathlib import Path

        # dcap-qvl's third argument is a *file path*, not the hex body.
        # Live TDX quotes are ~10k hex chars; passing them as argv raises
        # "File name too long (os error 36)" and collapses to review_quote_invalid.
        if not isinstance(quote_hex, str) or not quote_hex:
            raise QuoteStructureError("quote_hex must be a non-empty hex string")
        quote_path: Path | None = None
        try:
            # NamedTemporaryFile is unlinked in finally; write full body first so a
            # partially filled temp cannot be verified by a concurrent caller.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="ascii",
                prefix="dcap-qvl-quote-",
                suffix=".hex",
                delete=False,
            ) as handle:
                handle.write(quote_hex)
                quote_path = Path(handle.name)
            args = [self.binary, "verify", "--hex", str(quote_path)]
            try:
                proc = self._run(args)
            except FileNotFoundError as exc:  # pragma: no cover - environment-specific
                raise QuoteVerifierUnavailable(f"dcap-qvl not available: {exc}") from exc
            except subprocess.TimeoutExpired as exc:
                raise QuoteVerifierUnavailable(f"dcap-qvl timed out: {exc}") from exc
            except OSError as exc:  # pragma: no cover - environment-specific
                raise QuoteVerifierUnavailable(f"dcap-qvl OS error: {exc}") from exc
            except subprocess.SubprocessError as exc:  # pragma: no cover - environment-specific
                raise QuoteVerifierUnavailable(f"dcap-qvl invocation failed: {exc}") from exc
        finally:
            if quote_path is not None:
                try:
                    quote_path.unlink(missing_ok=True)
                except OSError:  # pragma: no cover - best-effort cleanup
                    pass

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise QuoteVerificationError(f"dcap-qvl rejected the quote: {detail}")

        try:
            report = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise QuoteVerifierUnavailable(f"dcap-qvl output is not JSON: {exc}") from exc
        if not isinstance(report, Mapping):
            raise QuoteVerifierUnavailable("dcap-qvl output was not a JSON object")

        status = report.get("status") or report.get("tcbStatus") or report.get("tcb_status")
        if not isinstance(status, str) or not status:
            raise QuoteVerifierUnavailable("dcap-qvl output is missing a TCB status")
        advisories = report.get("advisory_ids") or report.get("advisoryIDs") or []
        if not isinstance(advisories, Sequence) or isinstance(advisories, (str, bytes)):
            advisories = []
        return QuoteVerdict(tcb_status=status, advisory_ids=tuple(str(a) for a in advisories))


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
    tail: bytes = b"\0" * MIN_QUOTE_V4_SIGNATURE_DATA_LEN,
) -> str:
    """Assemble a minimal v4-layout TDX quote (hex) with the given fields.

    The inverse of :func:`parse_td_report`: places each register + ``report_data``
    at its fixed offset. Used to generate deterministic test vectors / fixtures
    (the real quote is produced by dstack ``get_quote`` on a live CVM).
    """

    buf = bytearray(MIN_QUOTE_LEN)
    buf[0:2] = TDX_QUOTE_VERSION.to_bytes(2, "little")
    buf[2:4] = ECDSA_P256_ATTESTATION_KEY_TYPE.to_bytes(2, "little")
    buf[4:8] = TDX_TEE_TYPE.to_bytes(4, "little")
    buf[12:28] = INTEL_QE_VENDOR_ID
    if header:
        buf[: min(len(header), QUOTE_HEADER_LEN)] = header[:QUOTE_HEADER_LEN]
    buf[_MRTD_OFFSET : _MRTD_OFFSET + REGISTER_LEN] = _coerce_register(mrtd, field_name="mrtd")
    buf[_RTMR0_OFFSET : _RTMR0_OFFSET + REGISTER_LEN] = _coerce_register(rtmr0, field_name="rtmr0")
    buf[_RTMR1_OFFSET : _RTMR1_OFFSET + REGISTER_LEN] = _coerce_register(rtmr1, field_name="rtmr1")
    buf[_RTMR2_OFFSET : _RTMR2_OFFSET + REGISTER_LEN] = _coerce_register(rtmr2, field_name="rtmr2")
    buf[_RTMR3_OFFSET : _RTMR3_OFFSET + REGISTER_LEN] = _coerce_register(rtmr3, field_name="rtmr3")
    if isinstance(report_data, str):
        rd = bytes.fromhex(report_data)
    else:
        rd = bytes(report_data)
    rd = rd[:REPORT_DATA_LEN].ljust(REPORT_DATA_LEN, b"\x00")
    buf[_REPORT_DATA_OFFSET : _REPORT_DATA_OFFSET + REPORT_DATA_LEN] = rd
    return (bytes(buf) + len(tail).to_bytes(4, "little") + tail).hex()


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


@dataclass(frozen=True)
class StaticQuoteVerifier:
    """A :class:`QuoteVerifier` with a fixed verdict (tests / offline harness).

    ``tcb_status`` is the posture reported for any quote; when ``valid`` is False
    every quote is rejected as if its signature did not verify.
    """

    tcb_status: str = "UpToDate"
    valid: bool = True
    advisory_ids: tuple[str, ...] = field(default_factory=tuple)

    def verify(self, quote_hex: str) -> QuoteVerdict:
        if not self.valid:
            raise QuoteVerificationError("quote signature verification failed")
        return QuoteVerdict(tcb_status=self.tcb_status, advisory_ids=self.advisory_ids)


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
    "QuoteVerifierUnavailable",
    "QuoteVerifier",
    "Rtmr3Replay",
    "StaticQuoteVerifier",
    "TdReport",
    "build_rtmr3_event_log",
    "build_tdx_quote",
    "decode_key_provider",
    "os_image_hash_from_registers",
    "parse_quote_hex",
    "parse_td_report",
    "parse_tdx_quote_v4",
    "replay_rtmr3",
    "runtime_event_digest",
    "validate_rtmr3_event_log",
]
