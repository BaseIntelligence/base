from __future__ import annotations

from dataclasses import dataclass

from .sandbox import inspect_code


@dataclass(frozen=True)
class CheatFinding:
    kind: str
    severity: float
    details: str


@dataclass(frozen=True)
class AntiCheatResult:
    multiplier: float
    diversity_bonus: float
    findings: list[CheatFinding]


def ast_similarity(left: str, right: str, *, allowed_import_roots: set[str] | None = None) -> float:
    try:
        left_fingerprint = inspect_code(
            left, require_contract=False, allowed_import_roots=allowed_import_roots
        ).ast_fingerprint
        right_fingerprint = inspect_code(
            right, require_contract=False, allowed_import_roots=allowed_import_roots
        ).ast_fingerprint
    except Exception:
        return 0.0
    if not left_fingerprint or not right_fingerprint:
        return 0.0
    return len(left_fingerprint & right_fingerprint) / len(left_fingerprint | right_fingerprint)


def jaccard_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - (len(left & right) / len(left | right))


def evaluate_anti_cheat(
    code: str,
    previous_codes: list[str],
    *,
    allowed_import_roots: set[str] | None = None,
) -> AntiCheatResult:
    report = inspect_code(code, require_contract=False, allowed_import_roots=allowed_import_roots)
    findings: list[CheatFinding] = []
    max_similarity = 0.0
    distances: list[float] = []
    for previous in previous_codes:
        similarity = ast_similarity(code, previous, allowed_import_roots=allowed_import_roots)
        max_similarity = max(max_similarity, similarity)
        try:
            previous_report = inspect_code(
                previous, require_contract=False, allowed_import_roots=allowed_import_roots
            )
        except Exception:
            continue
        distances.append(jaccard_distance(report.ast_fingerprint, previous_report.ast_fingerprint))
    if max_similarity >= 0.96:
        findings.append(CheatFinding("ast_similarity", 0.8, "near-duplicate submission"))
    avg_distance = sum(distances) / len(distances) if distances else 1.0
    diversity_bonus = min(0.05, max(0.0, avg_distance * 0.05))
    multiplier = max(0.0, 1.0 - sum(f.severity for f in findings))
    return AntiCheatResult(multiplier, diversity_bonus, findings)
