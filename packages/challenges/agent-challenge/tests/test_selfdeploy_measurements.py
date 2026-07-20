"""Measurement reproduction + allowlist verdict for the miner self-deploy CLI.

Covers VAL-DEPLOY-003 (measurement publish/reproduce is deterministic),
VAL-DEPLOY-004 (miner and validator agree on the canonical measurement set), and
VAL-DEPLOY-012 (the CLI reports a measurement and a correct allowlist verdict).

``dstack-mr`` needs the multi-hundred-MB dstack OS image files (only feasible
live), so these offline tests drive the wrapper via a faithful stub binary that
derives deterministic registers from its inputs (matching the M1 measurement-tool
tests); real-tool equivalence is a live (M6) concern.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from agent_challenge.selfdeploy import measurements as measure

_STUB = """#!{python}
import hashlib, json, sys
args = sys.argv[1:]
opts = {{}}
i = 0
while i < len(args):
    tok = args[i]
    if tok == "-json":
        opts["json"] = True; i += 1
    elif tok.startswith("-"):
        opts[tok[1:]] = args[i + 1] if i + 1 < len(args) else ""; i += 2
    else:
        i += 1
meta = open(opts.get("metadata", ""), "rb").read()
seed = meta + opts.get("cpu", "").encode() + opts.get("memory", "").encode()
reg = lambda t: hashlib.sha384(t + seed).hexdigest()
img = lambda t: hashlib.sha256(t + seed).hexdigest()
print(json.dumps({{"mrtd": reg(b"mrtd"), "rtmr0": reg(b"rtmr0"), "rtmr1": reg(b"rtmr1"),
                   "rtmr2": reg(b"rtmr2"), "mr_image": img(b"img")}}))
"""


@pytest.fixture
def stub_mr(tmp_path) -> str:
    path = tmp_path / "dstack-mr-stub"
    path.write_text(_STUB.format(python=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)
    return str(path)


@pytest.fixture
def metadata(tmp_path) -> Path:
    meta = tmp_path / "metadata.json"
    meta.write_text(json.dumps({"bios": "ovmf.fd", "kernel": "bzImage"}))
    return meta


@pytest.fixture
def compose() -> dict:
    return {
        "manifest_version": 2,
        "name": "agent-challenge-canonical",
        "runner": "docker-compose",
        "docker_compose_file": "services:\n  orchestrator:\n    image: repo@sha256:"
        + ("a" * 64)
        + "\n",
        "allowed_envs": ["BASE_GATEWAY_TOKEN"],
    }


CANONICAL_FIELDS = ("mrtd", "rtmr0", "rtmr1", "rtmr2", "compose_hash", "os_image_hash")


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-003: measurement publish/reproduce is deterministic
# --------------------------------------------------------------------------- #
def test_reproduce_is_byte_identical_across_runs(stub_mr, metadata, compose):
    first = measure.reproduce_measurement(
        metadata_path=metadata, cpu=1, memory="2G", compose=compose, dstack_mr_bin=stub_mr
    )
    second = measure.reproduce_measurement(
        metadata_path=metadata, cpu=1, memory="2G", compose=compose, dstack_mr_bin=stub_mr
    )
    assert first.to_json() == second.to_json()  # byte-identical, empty diff


def test_reproduce_covers_every_field_non_empty(stub_mr, metadata, compose):
    record = measure.reproduce_measurement(
        metadata_path=metadata, cpu=1, memory="2G", compose=compose, dstack_mr_bin=stub_mr
    ).as_dict()
    for field in CANONICAL_FIELDS:
        assert record.get(field), field


def test_reproduce_via_cli_is_deterministic(tmp_path, stub_mr, metadata, compose, capsys):
    from agent_challenge.selfdeploy import cli

    compose_path = tmp_path / "app-compose.json"
    compose_path.write_text(json.dumps(compose))
    argv = [
        "measurements",
        "--metadata",
        str(metadata),
        "--cpu",
        "1",
        "--memory",
        "2G",
        "--compose",
        str(compose_path),
        "--dstack-mr",
        stub_mr,
    ]
    assert cli.main(argv) == 0
    first = capsys.readouterr().out.strip()
    assert cli.main(argv) == 0
    second = capsys.readouterr().out.strip()
    assert first == second
    assert set(json.loads(first)) == set(CANONICAL_FIELDS)


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-004: miner and validator agree on the canonical measurement set
# --------------------------------------------------------------------------- #
def test_miner_record_equals_validator_allowlist_entry(stub_mr, metadata, compose):
    # Both sides recompute with the same tooling/inputs → identical record.
    miner = measure.reproduce_measurement(
        metadata_path=metadata, cpu=1, memory="2G", compose=compose, dstack_mr_bin=stub_mr
    )
    validator = measure.reproduce_measurement(
        metadata_path=metadata, cpu=1, memory="2G", compose=compose, dstack_mr_bin=stub_mr
    )
    # The validator pins the canonical subset (+ its key_provider) into an allowlist.
    entry = {**validator.as_dict(), "key_provider": "kms"}
    assert measure.measurements_agree(miner.as_dict(), entry)


def test_miner_disagrees_when_validator_entry_differs(stub_mr, metadata, compose):
    miner = measure.reproduce_measurement(
        metadata_path=metadata, cpu=1, memory="2G", compose=compose, dstack_mr_bin=stub_mr
    ).as_dict()
    tampered = {**miner, "compose_hash": "f" * 64}
    assert not measure.measurements_agree(miner, tampered)


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-012: CLI reports a measurement and a correct allowlist verdict
# --------------------------------------------------------------------------- #
def _record(stub_mr, metadata, compose) -> dict:
    return measure.reproduce_measurement(
        metadata_path=metadata, cpu=1, memory="2G", compose=compose, dstack_mr_bin=stub_mr
    ).as_dict()


def test_matching_measurement_reports_in_list(stub_mr, metadata, compose):
    record = _record(stub_mr, metadata, compose)
    allowlist = [{**record, "key_provider": "kms"}]
    verdict = measure.allowlist_verdict(record, allowlist)
    assert verdict.in_allowlist is True
    assert verdict.as_dict()["verdict"] == "IN-LIST"
    # The full canonical set is reported.
    assert set(verdict.measurement) == set(CANONICAL_FIELDS)


def test_single_field_tampered_measurement_reports_not_in_list(stub_mr, metadata, compose):
    record = _record(stub_mr, metadata, compose)
    allowlist = [{**record, "key_provider": "kms"}]
    tampered = {**record, "rtmr0": "0" * 96}
    verdict = measure.allowlist_verdict(tampered, allowlist)
    assert verdict.in_allowlist is False
    assert verdict.as_dict()["verdict"] == "NOT-IN-LIST"


def test_empty_allowlist_fails_closed(stub_mr, metadata, compose):
    record = _record(stub_mr, metadata, compose)
    assert measure.allowlist_verdict(record, []).in_allowlist is False


def test_verdict_cli_reports_measurement_and_verdict(tmp_path, stub_mr, metadata, compose, capsys):
    from agent_challenge.selfdeploy import cli

    record = _record(stub_mr, metadata, compose)
    allow_path = tmp_path / "allowlist.json"
    allow_path.write_text(json.dumps([{**record, "key_provider": "kms"}]))
    meas_path = tmp_path / "measurement.json"
    meas_path.write_text(json.dumps(record))

    code = cli.main(["verdict", "--measurement", str(meas_path), "--allowlist", str(allow_path)])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "IN-LIST"
    assert set(payload["measurement"]) == set(CANONICAL_FIELDS)

    # A one-field-tampered measurement → NOT-IN-LIST and a non-zero exit.
    meas_path.write_text(json.dumps({**record, "mrtd": "1" * 96}))
    code = cli.main(["verdict", "--measurement", str(meas_path), "--allowlist", str(allow_path)])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "NOT-IN-LIST"
