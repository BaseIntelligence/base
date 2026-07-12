"""Small, deterministic SemVer compatibility checks for SDK wire contracts.

The challenge SDK intentionally does not depend on a third-party version
solver.  Release configuration only needs the conservative range forms
documented by the API contract: exact versions, wildcards, caret/tilde ranges,
and whitespace/comma-separated comparison clauses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_VERSION_RE = re.compile(r"^(0|[1-9]\d*)(?:\.(0|[1-9]\d*))?(?:\.(0|[1-9]\d*))?$")


@dataclass(frozen=True, order=True)
class SemVer:
    """A normalized semantic version, with omitted components treated as zero."""

    major: int
    minor: int = 0
    patch: int = 0

    @classmethod
    def parse(cls, value: str) -> SemVer:
        match = _VERSION_RE.fullmatch(value.strip())
        if match is None:
            raise ValueError(f"invalid semantic version: {value!r}")
        return cls(*(int(part or 0) for part in match.groups()))

    def short(self, *, components: int = 3) -> str:
        values = (self.major, self.minor, self.patch)
        return ".".join(str(item) for item in values[:components])


def _parse_bound(value: str) -> SemVer:
    return SemVer.parse(value.removeprefix("v"))


def _caret_bounds(version: SemVer) -> tuple[SemVer, SemVer]:
    if version.major > 0:
        return version, SemVer(version.major + 1)
    if version.minor > 0:
        return version, SemVer(version.major, version.minor + 1)
    return version, SemVer(version.major, version.minor, version.patch + 1)


def _tilde_bounds(version: SemVer, raw: str) -> tuple[SemVer, SemVer]:
    # "~1" and "~1.2" preserve the precision supplied by the operator.
    components = len(raw.split("."))
    upper = (
        SemVer(version.major + 1)
        if components == 1
        else SemVer(version.major, version.minor + 1)
    )
    return version, upper


def _clause_matches(version: SemVer, clause: str) -> bool:
    clause = clause.strip()
    if not clause or clause in {"*", "x", "X"}:
        return True
    if clause.startswith("^"):
        lower = _parse_bound(clause[1:])
        start, end = _caret_bounds(lower)
        return start <= version < end
    if clause.startswith("~"):
        raw = clause[1:]
        lower = _parse_bound(raw)
        start, end = _tilde_bounds(lower, raw)
        return start <= version < end
    for operator in (">=", "<=", ">", "<", "="):
        if clause.startswith(operator):
            bound = _parse_bound(clause[len(operator) :])
            return {
                ">=": version >= bound,
                "<=": version <= bound,
                ">": version > bound,
                "<": version < bound,
                "=": version == bound,
            }[operator]
    if "x" in clause.lower() or "*" in clause:
        pieces = clause.replace("*", "x").split(".")
        actual = (version.major, version.minor, version.patch)
        return all(
            part.lower() in {"x", ""} or int(part) == actual[index]
            for index, part in enumerate(pieces)
        )
    return version == _parse_bound(clause)


def is_compatible(version: str, compatibility_range: str) -> bool:
    """Return whether ``version`` satisfies every clause in a supported range."""

    try:
        parsed = SemVer.parse(version)
        clauses = [
            clause
            for clause in re.split(r"[,\s]+", compatibility_range.strip())
            if clause
        ]
        return bool(clauses) and all(
            _clause_matches(parsed, clause) for clause in clauses
        )
    except (TypeError, ValueError):
        return False


def require_compatible(
    *,
    version: str,
    compatibility_range: str,
    label: str,
) -> None:
    """Raise a field-specific error for an invalid or unsatisfied range."""

    if not compatibility_range.strip():
        raise ValueError(f"{label} compatibility range must not be empty")
    if not is_compatible(version, compatibility_range):
        raise ValueError(
            f"Incompatible {label}: expected {compatibility_range!r}, "
            f"actual {version!r}"
        )


__all__ = ["SemVer", "is_compatible", "require_compatible"]
