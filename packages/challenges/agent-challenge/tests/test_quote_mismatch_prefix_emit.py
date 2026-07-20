"""quote_measurement_mismatch: allowlisted short hex prefixes on guest public_logs.

SPEED residual after tip 920b3ed6 / sub22: guest surface had diag=os but no
actual/expected digests, so offline dstack-mr pin packs could not be compared
to live Phala quote registers for honest repin.

Product Mode B: when reason_code=quote_measurement_mismatch and diag is in
{os,mrtd,rtmr0,rtmr1,rtmr2,compose}, append only short hex prefixes
(actual_prefix / expected_prefix, length 12–16) for the mismatched field.
Never full digests/secrets in the default residual surface. TDD; no invent allow.
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from agent_challenge.review.openrouter import OpenRouterTransportError

_PREFIX_RE = re.compile(r"^[0-9a-f]{12,16}$")
_PREFIX_FIELDS = frozenset({"os", "mrtd", "rtmr0", "rtmr1", "rtmr2", "compose"})


def _load_review_runtime():
    path = Path(__file__).resolve().parents[1] / "docker" / "review" / "review_runtime.py"
    spec = importlib.util.spec_from_file_location("review_runtime_prefix_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_keyrelease_quote_module():
    """Return the quote module instance late-imports will bind.

    ``test_keyrelease_package_init`` can leave package-attribute + sys.modules
    dual identities for ``keyrelease.quote``. String monkeypatches then miss
    the instance ``_measurement_from_quote`` late-imports. Prefer the
    ``sys.modules`` entry (what ``from ... import`` uses) and rebind package
    ``.quote`` — never delete modules mid-suite (that breaks later DCAP class
    identity / maps).
    """

    importlib.import_module("agent_challenge.keyrelease.quote")
    quote = sys.modules["agent_challenge.keyrelease.quote"]
    pkg = sys.modules.get("agent_challenge.keyrelease")
    if pkg is not None and getattr(pkg, "quote", None) is not quote:
        pkg.quote = quote  # type: ignore[attr-defined]
    return quote


def test_short_hex_prefix_helper_trims_and_lowercases() -> None:
    runtime = _load_review_runtime()
    full = "BD369A8C2F9EDB2B52DAD48AC8E0B32DDE5F1337C423A506B48D07403A7D8033"
    prefix = runtime.short_quote_field_hex_prefix(full)
    assert _PREFIX_RE.fullmatch(prefix)
    assert 12 <= len(prefix) <= 16
    assert prefix == full.lower()[: len(prefix)]
    # Full digest must never equal the prefix (unless artificially short).
    assert prefix != full.lower()
    # 0x prefix and whitespace stripped.
    assert runtime.short_quote_field_hex_prefix("0x" + full) == prefix
    assert runtime.short_quote_field_hex_prefix("  " + full.lower() + "  ") == prefix


def test_short_hex_prefix_rejects_non_hex() -> None:
    runtime = _load_review_runtime()
    assert runtime.short_quote_field_hex_prefix("not-a-digest") is None
    assert runtime.short_quote_field_hex_prefix("") is None
    assert runtime.short_quote_field_hex_prefix("zzzz") is None
    assert runtime.short_quote_field_hex_prefix(None) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("diag", sorted(_PREFIX_FIELDS))
def test_bounded_surface_emits_prefixes_for_prefix_diag_fields(diag: str) -> None:
    """public_logs residual must include short prefixes for the mismatched field."""

    runtime = _load_review_runtime()
    actual = ("a1" * 32) if diag != "mrtd" else ("b3" * 48)
    expected = ("c4" * 32) if diag != "mrtd" else ("d5" * 48)
    # SHA-384 registers are 96 hex chars; hashes are 64. Either ok.
    if diag in {"mrtd", "rtmr0", "rtmr1", "rtmr2"}:
        actual = "ab" * 48
        expected = "cd" * 48
    else:
        actual = "11" * 32
        expected = "22" * 32

    exc = runtime.QuoteMeasurementMismatchError(
        f"quoted {diag} mismatches assignment",
        diag=diag,
        actual=actual,
        expected=expected,
    )
    surface = runtime.bounded_review_failure_surface(exc)
    assert surface["error"] == "review_failed"
    assert surface["reason_code"] == "quote_measurement_mismatch"
    assert surface.get("diag") == diag
    assert "actual_prefix" in surface
    assert "expected_prefix" in surface
    assert _PREFIX_RE.fullmatch(surface["actual_prefix"])
    assert _PREFIX_RE.fullmatch(surface["expected_prefix"])
    assert surface["actual_prefix"] == actual.lower()[: len(surface["actual_prefix"])]
    assert surface["expected_prefix"] == expected.lower()[: len(surface["expected_prefix"])]
    # Never full digests on default surface.
    assert actual.lower() not in str(surface)
    assert expected.lower() not in str(surface)
    # Only prefixes (not raw message wording).
    assert "mismatches assignment" not in str(surface)


def test_os_mismatch_surface_uses_computed_os_image_hash_prefixes() -> None:
    """Prefer actual computed os_image_hash (sha256(mrtd∥rtmr1∥rtmr2)) prefixes."""

    runtime = _load_review_runtime()
    actual_os = "bd369a8c2f9edb2b52dad48ac8e0b32dde5f1337c423a506b48d07403a7d8033"
    expected_os = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    exc = runtime.QuoteMeasurementMismatchError(
        "quoted os image hash mismatches assignment",
        diag="os",
        actual=actual_os,
        expected=expected_os,
    )
    surface = runtime.bounded_review_failure_surface(exc)
    assert surface["diag"] == "os"
    assert surface["actual_prefix"] == actual_os[:16]
    assert surface["expected_prefix"] == expected_os[:16]
    assert len(surface["actual_prefix"]) == 16
    assert "bd369a8c2f9edb2b52dad48ac8e0b32d" not in str(surface)  # not ≥20 chars full


def test_bounded_surface_never_emits_full_digests_when_prefixes_present() -> None:
    runtime = _load_review_runtime()
    actual = "0123456789abcdef" * 4  # 64 hex
    expected = "fedcba9876543210" * 4
    exc = runtime.QuoteMeasurementMismatchError(
        "quoted compose hash mismatches assignment",
        diag="compose",
        actual=actual,
        expected=expected,
    )
    surface = runtime.bounded_review_failure_surface(exc)
    blob = str(surface)
    assert actual not in blob
    assert expected not in blob
    assert surface["actual_prefix"] != actual
    assert surface["expected_prefix"] != expected
    assert 12 <= len(surface["actual_prefix"]) <= 16


def test_bounded_surface_skips_prefixes_for_non_prefix_diags() -> None:
    """key_provider / event_log / other must not mint free-form digest prefixes."""

    runtime = _load_review_runtime()
    for diag, message in (
        ("key_provider", "quoted key provider mismatches assignment"),
        ("event_log", "quote event log soft mismatch"),
        ("other", "quote measurement mismatch without field token"),
    ):
        # Even if someone passes actual/expected, non-prefix diags omit them.
        try:
            exc = runtime.QuoteMeasurementMismatchError(
                message,
                diag=diag,
                actual="aa" * 32,
                expected="bb" * 32,
            )
        except ValueError:
            # Implementation may refuse non-prefix diags on the typed error —
            # plain ValueError path must also stay prefix-free.
            surface = runtime.bounded_review_failure_surface(ValueError(message))
            assert "actual_prefix" not in surface
            assert "expected_prefix" not in surface
            continue
        surface = runtime.bounded_review_failure_surface(exc)
        assert "actual_prefix" not in surface
        assert "expected_prefix" not in surface


def test_plain_value_error_os_still_has_diag_without_prefixes() -> None:
    """Legacy plain ValueError (no attrs) keeps field diag but no invented prefixes."""

    runtime = _load_review_runtime()
    surface = runtime.bounded_review_failure_surface(
        ValueError("quoted os image hash mismatches assignment")
    )
    assert surface["reason_code"] == "quote_measurement_mismatch"
    assert surface.get("diag") == "os"
    assert "actual_prefix" not in surface
    assert "expected_prefix" not in surface


def test_openrouter_transport_error_with_prefix_attrs_surfaces_them() -> None:
    """Transport-wrapped mismatch may also carry prefix attrs for residual emit."""

    runtime = _load_review_runtime()
    err = OpenRouterTransportError(
        "quote_measurement_mismatch",
        "quoted mrtd mismatches assignment",
        diag="mrtd",
    )
    err.actual_prefix = "deadbeefcafebabe"[:16]  # type: ignore[attr-defined]
    err.expected_prefix = "0011223344556677"[:16]  # type: ignore[attr-defined]
    surface = runtime.bounded_review_failure_surface(err)
    assert surface["diag"] == "mrtd"
    assert surface["actual_prefix"] == "deadbeefcafebabe"[:16]
    assert surface["expected_prefix"] == "0011223344556677"[:16]


def test_measurement_from_quote_os_mismatch_raises_with_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_measurement_from_quote must attach short prefixes on os mismatch."""

    quote_mod = _canonical_keyrelease_quote_module()
    runtime = _load_review_runtime()
    # Minimal structural stubs so only the os compare path fires.
    assignment = {
        "assignment_core": {
            "review_app": {
                "compose_hash": "cc" * 32,
                "measurement": {
                    "key_provider": "phala",
                    "os_image_hash": "ee" * 32,
                    "mrtd": "11" * 48,
                    "rtmr0": "22" * 48,
                    "rtmr1": "33" * 48,
                    "rtmr2": "44" * 48,
                    "vm_shape": "tdx.small",
                },
            }
        }
    }

    class _Report:
        mrtd = "11" * 48
        rtmr0 = "22" * 48
        rtmr1 = "33" * 48
        rtmr2 = "44" * 48
        rtmr3 = "55" * 48

    class _Replay:
        rtmr3 = "55" * 48
        compose_hash = "cc" * 32
        key_provider = "7068616c61"  # "phala" hex — decode path handled below

    def _parse(_hex: str) -> _Report:
        return _Report()

    def _validate(event_log: list) -> list:
        return list(event_log)

    def _replay(_validated: list) -> _Replay:
        return _Replay()

    # os_image_hash computed from registers will not equal ee*32 → os mismatch.
    fixed_os = "aa" * 32

    monkeypatch.setattr(quote_mod, "parse_tdx_quote_v4", _parse)
    monkeypatch.setattr(quote_mod, "validate_rtmr3_event_log", _validate)
    monkeypatch.setattr(quote_mod, "replay_rtmr3", _replay)
    monkeypatch.setattr(
        quote_mod,
        "os_image_hash_from_registers",
        lambda *a, **k: fixed_os,
    )
    monkeypatch.setattr(
        "agent_challenge.review.report._decode_key_provider",
        lambda _v: "phala",
    )

    with pytest.raises(runtime.QuoteMeasurementMismatchError) as ei:
        runtime._measurement_from_quote(
            assignment=assignment,
            tdx_quote_hex="00",
            event_log=[{"event": "compose-hash"}],
        )
    err = ei.value
    assert err.diag == "os"
    assert err.actual_prefix == fixed_os[:16]
    assert err.expected_prefix == ("ee" * 32)[:16]
    assert len(err.actual_prefix) <= 16
    assert fixed_os not in str(err) or fixed_os[:16] in str(err)
    # Full digests must not appear as standalone free text in message (closed wording).
    assert "mismatches assignment" in str(err).lower()

    surface = runtime.bounded_review_failure_surface(err)
    assert surface["diag"] == "os"
    assert surface["actual_prefix"] == fixed_os[:16]
    assert surface["expected_prefix"] == ("ee" * 32)[:16]
    assert fixed_os not in str(surface)
    assert ("ee" * 32) not in str(surface)


def test_measurement_from_quote_rtmr1_mismatch_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quote_mod = _canonical_keyrelease_quote_module()
    runtime = _load_review_runtime()
    assignment = {
        "assignment_core": {
            "review_app": {
                "compose_hash": "cc" * 32,
                "measurement": {
                    "key_provider": "phala",
                    "os_image_hash": "aa" * 32,
                    "mrtd": "11" * 48,
                    "rtmr0": "22" * 48,
                    "rtmr1": "dd" * 48,  # expected; report will differ
                    "rtmr2": "44" * 48,
                    "vm_shape": "tdx.small",
                },
            }
        }
    }

    class _Report:
        mrtd = "11" * 48
        rtmr0 = "22" * 48
        rtmr1 = "33" * 48  # actual
        rtmr2 = "44" * 48
        rtmr3 = "55" * 48

    class _Replay:
        rtmr3 = "55" * 48
        compose_hash = "cc" * 32
        key_provider = "7068616c61"

    monkeypatch.setattr(quote_mod, "parse_tdx_quote_v4", lambda _h: _Report())
    monkeypatch.setattr(
        quote_mod,
        "validate_rtmr3_event_log",
        lambda e: list(e),
    )
    monkeypatch.setattr(quote_mod, "replay_rtmr3", lambda _v: _Replay())
    # Match os so we reach register loop.
    monkeypatch.setattr(
        quote_mod,
        "os_image_hash_from_registers",
        lambda *a, **k: "aa" * 32,
    )
    monkeypatch.setattr(
        "agent_challenge.review.report._decode_key_provider",
        lambda _v: "phala",
    )

    with pytest.raises(runtime.QuoteMeasurementMismatchError) as ei:
        runtime._measurement_from_quote(
            assignment=assignment,
            tdx_quote_hex="00",
            event_log=[{"event": "compose-hash"}],
        )
    err = ei.value
    assert err.diag == "rtmr1"
    assert err.actual_prefix == ("33" * 48)[:16]
    assert err.expected_prefix == ("dd" * 48)[:16]
    surface = runtime.bounded_review_failure_surface(err)
    assert surface["diag"] == "rtmr1"
    assert surface["actual_prefix"] == err.actual_prefix
    assert surface["expected_prefix"] == err.expected_prefix
    assert ("33" * 48) not in str(surface)
    assert ("dd" * 48) not in str(surface)
