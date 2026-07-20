"""Helpers for submission family naming and labels."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_MAX_DISPLAY_NAME_LENGTH = 128
_ALLOWED_PUNCTUATION = {" ", "_", "-", ".", ":"}


def normalize_submission_name(value: str) -> str:
    """Return the normalized comparison key for a submission display name."""

    normalized = unicodedata.normalize("NFKC", value)
    display_name = _WHITESPACE_RE.sub(" ", normalized.strip())
    if not display_name:
        raise ValueError("submission name must not be blank")
    if len(display_name) > _MAX_DISPLAY_NAME_LENGTH:
        raise ValueError("submission name must be 128 characters or fewer")
    if any(not _is_allowed_name_character(character) for character in display_name):
        raise ValueError(
            "submission name may contain only letters, numbers, spaces, underscores, hyphens, "
            "periods, and colons"
        )
    return display_name.casefold()


def _is_allowed_name_character(character: str) -> bool:
    return character in _ALLOWED_PUNCTUATION or unicodedata.category(character)[0] in {"L", "N"}


def version_label(version_number: int) -> str:
    """Return the public label for a 1-indexed submission version."""

    if version_number < 1:
        raise ValueError("version number must be positive")
    return f"v{version_number}"
