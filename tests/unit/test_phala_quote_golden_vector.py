"""Shared cross-repo golden TDX quote/measurement/RTMR3 anti-drift vector (base).

base ``src/base/worker/phala_quote.py`` re-implements agent-challenge
``src/agent_challenge/keyrelease/quote.py`` almost verbatim (TDX register
byte-offsets, ``os_image_hash = sha256(MRTD||RTMR1||RTMR2)``, dstack RTMR3
event-log replay) because base cannot import the lean in-CVM module. That
duplication is drift-prone: a one-sided offset/hash tweak could let a real dstack
quote verify in one repo and silently fail in the other.

This test pins a FIXED quote + event log (``tests/unit/phala_quote_golden_vector.json``)
to the exact registers / os_image_hash / RTMR3 / compose-hash a correct parser
must reproduce, and asserts base's parser reproduces them. The SAME fixture bytes
and the SAME :data:`GOLDEN_VECTOR_SHA256` are asserted in agent-challenge
(``tests/test_quote_golden_vector.py``); because the pinned expected values come
from the frozen fixture (not recomputed with the same offsets under test), a
one-sided offset/hash change diverges from these values and fails here or there.
Do NOT edit one repo's copy without the other (AGENTS.md
'TDX-quote-parse / measurement / RTMR3 anti-drift').
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from base.worker.phala_quote import (
    _MRTD_OFFSET,
    REGISTER_LEN,
    build_rtmr3_event_log,
    os_image_hash_from_registers,
    parse_td_report,
    replay_rtmr3,
)

# Same literal in BOTH repos: the SHA-256 of the byte-identical golden fixture.
# If either repo's fixture is edited, that side's pin fails; a reviewer can grep
# this constant across both repos to confirm they still match.
GOLDEN_VECTOR_SHA256 = (
    "053979e8445c147798ec6f9165f9849a38ff5797e4dccd2ee38aa329b7f673bf"
)

_VECTOR_PATH = Path(__file__).parent / "phala_quote_golden_vector.json"


def _vector() -> dict:
    return json.loads(_VECTOR_PATH.read_text(encoding="utf-8"))


def test_golden_fixture_is_byte_identical_across_repos() -> None:
    digest = hashlib.sha256(_VECTOR_PATH.read_bytes()).hexdigest()
    assert digest == GOLDEN_VECTOR_SHA256


def test_golden_quote_parses_to_expected_registers() -> None:
    vector = _vector()
    expected = vector["expected"]
    report = parse_td_report(vector["quote_hex"])
    assert report.mrtd == expected["mrtd"]
    assert report.rtmr0 == expected["rtmr0"]
    assert report.rtmr1 == expected["rtmr1"]
    assert report.rtmr2 == expected["rtmr2"]
    assert report.rtmr3 == expected["rtmr3"]
    assert report.report_data.hex() == expected["report_data_hex"]


def test_golden_os_image_hash_matches() -> None:
    vector = _vector()
    expected = vector["expected"]
    assert (
        os_image_hash_from_registers(
            expected["mrtd"], expected["rtmr1"], expected["rtmr2"]
        )
        == expected["os_image_hash"]
    )


def test_golden_event_log_replays_to_expected_rtmr3() -> None:
    vector = _vector()
    expected = vector["expected"]
    replay = replay_rtmr3(vector["event_log"])
    assert replay.rtmr3 == expected["rtmr3"]
    assert replay.compose_hash == expected["compose_hash"]
    assert replay.key_provider == expected["key_provider"]
    # The event-log replay reproduces the RTMR3 the fixed quote carries.
    assert replay.rtmr3 == parse_td_report(vector["quote_hex"]).rtmr3


def test_golden_vector_offset_sensitivity_discriminator() -> None:
    # Non-vacuity: the pinned MRTD is not read from a constant. Reading one byte
    # off the register offset (a simulated one-sided off-by-one) yields a value
    # that differs from the golden -- exactly the divergence the pin above catches.
    vector = _vector()
    raw = bytes.fromhex(vector["quote_hex"])
    correct = raw[_MRTD_OFFSET : _MRTD_OFFSET + REGISTER_LEN].hex()
    tweaked = raw[_MRTD_OFFSET + 1 : _MRTD_OFFSET + 1 + REGISTER_LEN].hex()
    assert correct == vector["expected"]["mrtd"]
    assert tweaked != vector["expected"]["mrtd"]


def test_golden_vector_replay_sensitivity_discriminator() -> None:
    # Non-vacuity: a different compose payload replays to a different RTMR3, so the
    # pinned RTMR3 genuinely binds the event-log digest/extend formula.
    vector = _vector()
    _log, other_rtmr3 = build_rtmr3_event_log(
        [("compose-hash", bytes.fromhex("00" * 32))]
    )
    assert other_rtmr3 != vector["expected"]["rtmr3"]
