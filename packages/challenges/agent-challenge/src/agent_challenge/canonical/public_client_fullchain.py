"""Public-only client fullchain export for operator client-trust install.

After guest ``GetTlsKey`` materializes ``/run/secrets/ra_tls/client.crt``
(leaf + intermediates), non-dev Phala blocks SSH/SCP and secret-free bootstrap
markers omit PEM, so operators cannot harvest the public chain needed to set
``KEY_RELEASE_RA_TLS_CA_FILE`` / host client-trust distinct from the server CA.

This module exposes **public-only** material via two surfaces (no invent PEMs):

1. Flushed log lines labeled ``ra_tls_public_fullchain`` that host scrapers already
   reach with ``phala logs`` (intentionally contain ``BEGIN CERTIFICATE`` blocks).
2. A well-known non-secret path under ``/var/log/agent-challenge/`` for any host
   inspect tooling that can read guest files without full SSH.

Hard rules:
* NEVER includes private key material (``BEGIN PRIVATE KEY`` / PKCS8 / etc.).
* Private key stays only on the mTLS paths used for the raw dial.
* Fail closed when the public chain is missing or empty.
* Never invents roots or issuer PEMs.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: Host log / serial scrapers look for this prefix (intentional public PEM).
PUBLIC_FULLCHAIN_MARKER = "ra_tls_public_fullchain"

#: Default path (public-only PEM). Not under /run/secrets and never holds keys.
DEFAULT_PUBLIC_EXPORT_PATH = Path("/var/log/agent-challenge/client-fullchain.pem")

#: Override well-known export path (tests / residual ops).
PUBLIC_EXPORT_PATH_ENV = "CHALLENGE_PHALA_PUBLIC_CLIENT_FULLCHAIN_PATH"

_CERT_BLOCK_RE = re.compile(
    r"-----BEGIN CERTIFICATE-----\r?\n"
    r"[A-Za-z0-9+/=\r\n]+"
    r"-----END CERTIFICATE-----",
    re.MULTILINE,
)

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----",
    re.IGNORECASE,
)

_PRIVATE_HINTS = (
    "BEGIN PRIVATE KEY",
    "BEGIN RSA PRIVATE KEY",
    "BEGIN EC PRIVATE KEY",
    "BEGIN ENCRYPTED PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
)


@dataclass(frozen=True)
class PublicFullchainExport:
    """Result of a successful public-only fullchain export (safe to log)."""

    pem: str
    cert_count: int
    pem_len: int
    sha256_hex: str
    leaf_spki_sha256: str | None
    export_path: str | None
    path_written: bool


class PublicFullchainExportError(RuntimeError):
    """Fail-closed error for missing/invalid public fullchain export."""


def assert_public_only(surface: str, *, label: str = "export_surface") -> None:
    """Raise when any private-key material appears in an export surface.

    The check is intentionally wide (header substrings + regex) so unit tests
    and production log emission share one fail-closed gate.
    """

    text = surface or ""
    upper = text.upper()
    for hint in _PRIVATE_HINTS:
        if hint in upper:
            raise PublicFullchainExportError(
                f"private_key_in_export: {label} contains private key material"
            )
    if _PRIVATE_KEY_RE.search(text):
        raise PublicFullchainExportError(
            f"private_key_in_export: {label} matched private key PEM header"
        )


def extract_public_fullchain_pem(cert_pem: str) -> str:
    """Return leaf+intermediates CERTIFICATE blocks only (no invent, no keys).

    Raises:
        PublicFullchainExportError: when chain is empty/missing or result is not
        public-only.
    """

    if cert_pem is None or not str(cert_pem).strip():
        raise PublicFullchainExportError("public_chain_missing: client certificate chain is empty")

    raw = str(cert_pem)
    # Fail closed if operator accidentally passed a combined cert+key blob.
    assert_public_only(raw, label="input_cert_pem")

    blocks = _CERT_BLOCK_RE.findall(raw)
    if not blocks:
        raise PublicFullchainExportError(
            "public_chain_missing: no BEGIN CERTIFICATE blocks in client chain"
        )

    pieces: list[str] = []
    for block in blocks:
        body = block.strip()
        if not body.endswith("\n"):
            body = body + "\n"
        pieces.append(body)
    pem = "".join(pieces)
    if not pem.endswith("\n"):
        pem = pem + "\n"

    assert_public_only(pem, label="extracted_fullchain")
    if "BEGIN CERTIFICATE" not in pem or "END CERTIFICATE" not in pem:
        raise PublicFullchainExportError(
            "public_chain_missing: extracted fullchain lacks PEM markers"
        )
    return pem


def _leaf_spki_sha256(pem: str) -> str | None:
    """Best-effort leaf SPKI digest; Non-fatal when cryptography is unavailable."""

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import serialization
    except ImportError:  # pragma: no cover - cryptography is a product dep
        return None
    try:
        # load_pem_x509_certificate accepts the first cert in a multi-PEM string
        # only when given a single block; take first block explicitly.
        first = _CERT_BLOCK_RE.search(pem)
        if first is None:
            return None
        cert = x509.load_pem_x509_certificate(first.group(0).encode("utf-8"))
        spki = cert.public_key().public_bytes(
            serialization.Encoding.DER,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        return hashlib.sha256(spki).hexdigest()
    except (ValueError, TypeError):
        return None


def resolve_public_export_path(explicit: Path | str | None = None) -> Path:
    """Resolve well-known public export path (env override wins over default)."""

    if explicit is not None:
        return Path(explicit)
    import os

    raw = (os.environ.get(PUBLIC_EXPORT_PATH_ENV) or "").strip()
    if raw:
        return Path(raw)
    return DEFAULT_PUBLIC_EXPORT_PATH


def write_public_fullchain_file(pem: str, path: Path) -> bool:
    """Write public-only PEM to ``path``. Returns True on success, False on I/O fail.

    Never writes private keys. Caller must pass already-validated public PEM.
    """

    assert_public_only(pem, label="write_path_input")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pem, encoding="utf-8")
        # Public material only; world-readable so host inspect tools can scrape.
        try:
            path.chmod(0o644)
        except OSError:
            pass
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def emit_public_fullchain_log(
    pem: str,
    *,
    cert_count: int,
    pem_len: int,
    sha256_hex: str,
    leaf_spki_sha256: str | None,
    export_path: str | None,
    path_written: bool,
) -> str:
    """Emit flushed public-fullchain log surface; return combined text for tests.

    Stage markers carry only redacted lengths/digests; PEM body is intentional
    public CERTIFICATE blocks. Private keys are rejected before emit.
    """

    assert_public_only(pem, label="log_emit_pem")
    meta_parts = [
        f"{PUBLIC_FULLCHAIN_MARKER} stage=export_ready",
        f"cert_count={cert_count}",
        f"pem_len={pem_len}",
        f"sha256={sha256_hex}",
    ]
    if leaf_spki_sha256:
        meta_parts.append(f"leaf_spki_sha256={leaf_spki_sha256}")
    if export_path:
        meta_parts.append(f"path={export_path}")
    meta_parts.append(f"path_written={'yes' if path_written else 'no'}")
    meta_line = " ".join(meta_parts)

    end_line = f"{PUBLIC_FULLCHAIN_MARKER} stage=export_end sha256={sha256_hex}"
    body = pem if pem.endswith("\n") else pem + "\n"
    combined = f"{meta_line}\n{body}{end_line}\n"
    assert_public_only(combined, label="log_emit_combined")

    print(meta_line, flush=True)
    # Multi-line PEM intentionally printed so phala logs harvestors can scrape.
    sys.stdout.write(body)
    sys.stdout.flush()
    print(end_line, flush=True)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - best-effort flush only
        pass
    return combined


def export_public_client_fullchain(
    *,
    cert_path: Path | str,
    export_path: Path | str | None = None,
    emit_log: bool = True,
    write_file: bool = True,
) -> PublicFullchainExport:
    """Read guest client.crt and export public-only fullchain (fail-closed).

    Args:
        cert_path: Path to the leaf+intermediates client certificate file.
        export_path: Optional override for the well-known public path.
        emit_log: When True, print the public PEM on the log surface.
        write_file: When True, attempt the well-known path write.

    Returns:
        :class:`PublicFullchainExport` with digests and path status.

    Raises:
        PublicFullchainExportError: missing chain, private key present, I/O for
        cert read fails, or export surface validation fails.
    """

    path = Path(cert_path)
    if not path.is_file():
        raise PublicFullchainExportError(
            f"public_chain_missing: client certificate file not found: {path}"
        )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PublicFullchainExportError(
            f"public_chain_missing: cannot read client certificate: {exc}"
        ) from exc

    pem = extract_public_fullchain_pem(raw)
    cert_count = len(_CERT_BLOCK_RE.findall(pem))
    pem_len = len(pem.encode("utf-8"))
    sha256_hex = hashlib.sha256(pem.encode("utf-8")).hexdigest()
    leaf_spki = _leaf_spki_sha256(pem)

    target = resolve_public_export_path(export_path)
    path_written = False
    if write_file:
        path_written = write_public_fullchain_file(pem, target)

    export = PublicFullchainExport(
        pem=pem,
        cert_count=cert_count,
        pem_len=pem_len,
        sha256_hex=sha256_hex,
        leaf_spki_sha256=leaf_spki,
        export_path=str(target),
        path_written=path_written,
    )

    if emit_log:
        combined = emit_public_fullchain_log(
            pem,
            cert_count=cert_count,
            pem_len=pem_len,
            sha256_hex=sha256_hex,
            leaf_spki_sha256=leaf_spki,
            export_path=str(target),
            path_written=path_written,
        )
        assert_public_only(combined, label="post_emit_surface")
        # Dedicated safeguard: private key material must never reach the surface.
        if "PRIVATE KEY" in combined.upper():
            raise PublicFullchainExportError(
                "private_key_in_export: log surface leaked private key material"
            )

    return export


def export_surface_contains_private_key(surface: str) -> bool:
    """Predicate helper for tests: True when surface has private key material."""

    try:
        assert_public_only(surface, label="surface_probe")
    except PublicFullchainExportError:
        return True
    return "PRIVATE KEY" in (surface or "").upper()
