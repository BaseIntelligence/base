"""Secret scanning for the canonical eval image and build artifacts.

Detects credential-shaped values and golden-test plaintext markers so the
canonical image (and any generated artifact) can be proven free of baked-in
secrets. Matches are reported as ``(member, pattern)`` pairs and never carry the
matched secret value, so scan results are always safe to log.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

# The golden-plaintext marker is assembled from fragments so this scanner's own
# source (which ships inside the image) never self-matches; only real golden
# files carry the contiguous marker.
_GOLDEN_MARKER = "harbor-independence/" + "oracle-golden"

# High-confidence patterns: each is specific enough not to fire on a stock
# python base image, yet catches the secret classes VAL-IMG-005 forbids.
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "phala_api_key": re.compile(r"phak_[A-Za-z0-9]{16,}"),
    "openai_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9]{32,}"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    # A REAL PEM private key: the BEGIN header followed by a newline, optional
    # legacy encryption headers (Proc-Type/DEK-Info, as in "traditional" OpenSSL
    # encrypted keys), a base64 body, and the matching END footer. Requiring the
    # body (not a bare header) avoids false positives on library source/binaries
    # that embed the header string as a constant (e.g. cryptography's ssh.py
    # ``_SK_START``); allowing the header block keeps legacy encrypted keys caught.
    "pem_private_key": re.compile(
        r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----\r?\n"
        r"(?:(?:Proc-Type|DEK-Info):[^\r\n]*\r?\n)*"
        r"(?:\r?\n)?"
        r"[A-Za-z0-9+/=\r\n]+"
        r"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"
    ),
    "golden_oracle_plaintext": re.compile(re.escape(_GOLDEN_MARKER)),
}

# Skip members larger than this when scanning (secrets are small; huge binary
# blobs only slow the scan and never legitimately hold a pasted credential).
MAX_MEMBER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class SecretHit:
    """A single secret-pattern match. Deliberately omits the matched value."""

    member: str
    pattern: str


def scan_bytes(data: bytes, *, member: str) -> list[SecretHit]:
    """Return the secret patterns present in ``data`` (value never included)."""

    text = data.decode("latin-1", errors="ignore")
    hits: list[SecretHit] = []
    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            hits.append(SecretHit(member=member, pattern=name))
    return hits


def scan_text(text: str, *, member: str = "<text>") -> list[SecretHit]:
    return scan_bytes(text.encode("utf-8"), member=member)


def scan_path(root: Path | str) -> list[SecretHit]:
    """Scan a file or directory tree for secret patterns."""

    root = Path(root)
    hits: list[SecretHit] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > MAX_MEMBER_BYTES:
                continue
            data = path.read_bytes()
        except OSError:  # pragma: no cover - defensive
            continue
        hits.extend(scan_bytes(data, member=str(path)))
    return hits


def scan_tar(tar_path: Path | str) -> list[SecretHit]:
    """Scan every regular file inside a tar archive (e.g. ``docker export``)."""

    hits: list[SecretHit] = []
    with tarfile.open(tar_path, "r:*") as archive:
        for member in archive:
            if not member.isfile() or member.size > MAX_MEMBER_BYTES:
                continue
            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - defensive
                continue
            with handle:
                hits.extend(scan_bytes(handle.read(), member=member.name))
    return hits
