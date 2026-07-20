"""Behavioral tests for the canonical measurement tooling (M1).

Fulfils VAL-IMG-006..009:
  * VAL-IMG-006 normalized compose-hash is deterministic and normalization-invariant
  * VAL-IMG-007 compose-hash changes deterministically on any material change (drift)
  * VAL-IMG-008 dstack-mr wrapper computes stable, well-formed MRTD/RTMR0-2
  * VAL-IMG-009 the canonical measurement record is emitted in a stable, pinnable form

The compose-hash assertions are pure/offline. The dstack-mr assertions drive the
wrapper against an injected stub ``dstack-mr`` binary (the real tool needs the
multi-hundred-MB dstack OS image files, which are only available on a live CVM;
the real-tool equivalence is checked live at M6 via VAL-IMG-011). The stub is a
faithful stand-in: it mirrors the real ``dstack-mr -cpu -memory -json -metadata``
interface and JSON output shape, and derives its registers deterministically from
its inputs so both determinism and input-sensitivity are exercised.
"""

from __future__ import annotations

import copy
import json
import os
import re
import stat
import sys
from pathlib import Path

import pytest

from agent_challenge.canonical import measurement as m

HEX96_RE = re.compile(r"^[0-9a-f]{96}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- #
# Fixtures: a representative app-compose and a stub dstack-mr binary
# --------------------------------------------------------------------------- #


def _base_compose() -> dict:
    return {
        "manifest_version": 2,
        "name": "agent-challenge-canonical",
        "runner": "docker-compose",
        "docker_compose_file": (
            "services:\n"
            "  orchestrator:\n"
            "    image: ghcr.io/base/agent-challenge@sha256:" + ("a" * 64) + "\n"
            "    ports:\n"
            '      - "8700:8700"\n'
        ),
        "kms_enabled": True,
        "gateway_enabled": False,
        "public_logs": True,
        "public_sysinfo": True,
        "local_key_provider_enabled": False,
        "allowed_envs": ["BASE_LLM_GATEWAY_URL", "BASE_GATEWAY_TOKEN"],
        "no_instance_id": False,
        "features": {"kms": True, "gateway": False, "logs": True},
    }


_STUB_TEMPLATE = """#!{python}
import hashlib
import json
import sys

MODE = {mode!r}

args = sys.argv[1:]
opts = {{}}
i = 0
while i < len(args):
    tok = args[i]
    if tok == "-json":
        opts["json"] = True
        i += 1
    elif tok.startswith("-"):
        opts[tok[1:]] = args[i + 1] if i + 1 < len(args) else ""
        i += 2
    else:
        i += 1

metadata_path = opts.get("metadata", "")
try:
    meta = open(metadata_path, "rb").read()
except OSError:
    sys.stderr.write("stub dstack-mr: cannot read metadata\\n")
    sys.exit(1)

seed = meta + opts.get("cpu", "").encode() + opts.get("memory", "").encode()


def reg(tag):
    return hashlib.sha384(tag + seed).hexdigest()


def img(tag):
    return hashlib.sha256(tag + seed).hexdigest()


if MODE == "fail":
    sys.stderr.write("stub dstack-mr: forced failure\\n")
    sys.exit(3)

if MODE == "bad_hex":
    out = {{"mrtd": "nothexvalue", "rtmr0": reg(b"rtmr0"), "rtmr1": reg(b"rtmr1"),
            "rtmr2": reg(b"rtmr2"), "mr_aggregated": img(b"agg"), "mr_image": img(b"img")}}
elif MODE == "bad_width":
    out = {{"mrtd": "abcdef", "rtmr0": reg(b"rtmr0"), "rtmr1": reg(b"rtmr1"),
            "rtmr2": reg(b"rtmr2"), "mr_aggregated": img(b"agg"), "mr_image": img(b"img")}}
else:
    out = {{"mrtd": reg(b"mrtd"), "rtmr0": reg(b"rtmr0"), "rtmr1": reg(b"rtmr1"),
            "rtmr2": reg(b"rtmr2"), "mr_aggregated": img(b"agg"), "mr_image": img(b"img")}}

print(json.dumps(out, indent=2))
"""


def _write_stub(path: Path, *, mode: str = "ok") -> Path:
    path.write_text(_STUB_TEMPLATE.format(python=sys.executable, mode=mode))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IRUSR)
    return path


@pytest.fixture
def stub_mr(tmp_path) -> str:
    return str(_write_stub(tmp_path / "dstack-mr-stub", mode="ok"))


@pytest.fixture
def metadata_file(tmp_path) -> Path:
    meta = tmp_path / "metadata.json"
    meta.write_text(
        json.dumps(
            {
                "bios": "ovmf.fd",
                "kernel": "bzImage",
                "cmdline": "console=ttyS0 root=/dev/vda",
                "initrd": "initramfs.cpio.gz",
            }
        )
    )
    return meta


# --------------------------------------------------------------------------- #
# VAL-IMG-006: deterministic + normalization-invariant compose-hash
# --------------------------------------------------------------------------- #


def test_compose_hash_is_deterministic_and_sha256_shaped():
    compose = _base_compose()
    first = m.compose_hash(compose)
    second = m.compose_hash(copy.deepcopy(compose))
    assert first == second
    assert HEX64_RE.match(first), first


def test_compose_hash_invariant_to_top_level_key_order():
    compose = _base_compose()
    reordered = {k: compose[k] for k in reversed(list(compose))}
    assert list(reordered) != list(compose)
    assert m.compose_hash(reordered) == m.compose_hash(compose)


def test_compose_hash_invariant_to_nested_key_order():
    compose = _base_compose()
    variant = copy.deepcopy(compose)
    variant["features"] = {k: variant["features"][k] for k in reversed(list(variant["features"]))}
    assert m.compose_hash(variant) == m.compose_hash(compose)


def test_compose_hash_invariant_to_insignificant_whitespace():
    compose = _base_compose()
    compact = json.dumps(compose, separators=(",", ":"))
    pretty = json.dumps(compose, indent=4, sort_keys=False)
    baseline = m.compose_hash(compose)
    assert m.compose_hash(compact) == baseline
    assert m.compose_hash(pretty) == baseline


def test_compose_hash_accepts_dict_and_equivalent_json_string():
    compose = _base_compose()
    as_string = json.dumps(compose)
    assert m.compose_hash(as_string) == m.compose_hash(compose)


def test_normalize_app_compose_is_sorted_and_compact():
    compose = _base_compose()
    normalized = m.normalize_app_compose(compose)
    # Sorted keys + compact separators => round-trips and is byte-identical to a
    # sorted compact dump.
    assert normalized == json.dumps(compose, sort_keys=True, separators=(",", ":"))
    # keys appear in sorted order at the top level
    top_keys = [k for k in compose]
    assert list(json.loads(normalized)) == sorted(top_keys)
    assert json.loads(normalized) == compose


def test_normalize_app_compose_rejects_unsupported_type():
    with pytest.raises(TypeError):
        m.normalize_app_compose(12345)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# VAL-IMG-007: material change => different deterministic hash; revert restores
# --------------------------------------------------------------------------- #


def _mutations():
    def change_image_digest(c):
        c["docker_compose_file"] = c["docker_compose_file"].replace("a" * 64, "b" * 64)

    def change_ports(c):
        c["docker_compose_file"] = c["docker_compose_file"].replace("8700:8700", "8701:8701")

    def change_env_keys(c):
        c["allowed_envs"] = ["BASE_LLM_GATEWAY_URL"]

    def change_service_definition(c):
        c["docker_compose_file"] += "  sidecar:\n    image: busybox@sha256:" + ("c" * 64) + "\n"

    def change_top_level_flag(c):
        c["gateway_enabled"] = True

    return {
        "image_digest": change_image_digest,
        "ports": change_ports,
        "env_keys": change_env_keys,
        "service_definition": change_service_definition,
        "top_level_flag": change_top_level_flag,
    }


def test_material_change_yields_distinct_reproducible_hash_and_revert_restores():
    baseline_compose = _base_compose()
    baseline = m.compose_hash(baseline_compose)

    seen = {baseline}
    for name, mutate in _mutations().items():
        mutated = copy.deepcopy(baseline_compose)
        mutate(mutated)
        changed = m.compose_hash(mutated)
        assert changed != baseline, f"{name} did not change the hash"
        assert changed not in seen, f"{name} collided with another hash"
        seen.add(changed)
        # deterministic: recomputing the same mutation reproduces the hash
        again = copy.deepcopy(baseline_compose)
        mutate(again)
        assert m.compose_hash(again) == changed
        # reverting reproduces the baseline exactly
        assert m.compose_hash(_base_compose()) == baseline


# --------------------------------------------------------------------------- #
# VAL-IMG-008: dstack-mr wrapper — stable, well-formed MRTD/RTMR0-2
# --------------------------------------------------------------------------- #


def test_compute_image_measurement_is_deterministic_and_well_formed(stub_mr, metadata_file):
    first = m.compute_image_measurement(metadata_file, cpu=4, memory="4G", dstack_mr_bin=stub_mr)
    second = m.compute_image_measurement(metadata_file, cpu=4, memory="4G", dstack_mr_bin=stub_mr)
    for reg_value in (first.mrtd, first.rtmr0, first.rtmr1, first.rtmr2):
        assert HEX96_RE.match(reg_value), reg_value
    assert (first.mrtd, first.rtmr0, first.rtmr1, first.rtmr2) == (
        second.mrtd,
        second.rtmr0,
        second.rtmr1,
        second.rtmr2,
    )
    assert HEX64_RE.match(first.os_image_hash), first.os_image_hash
    # Product formula, not raw tool mr_image unless they coincide.
    assert first.os_image_hash == m.product_os_image_hash(
        mrtd=first.mrtd, rtmr1=first.rtmr1, rtmr2=first.rtmr2
    )


def test_compute_image_measurement_is_input_sensitive(stub_mr, tmp_path, metadata_file):
    baseline = m.compute_image_measurement(metadata_file, cpu=4, memory="4G", dstack_mr_bin=stub_mr)

    # different cpu -> different registers (inputs really flow to the tool)
    diff_cpu = m.compute_image_measurement(metadata_file, cpu=8, memory="4G", dstack_mr_bin=stub_mr)
    assert diff_cpu.mrtd != baseline.mrtd

    # different image metadata -> different registers (drift on image change)
    other_meta = tmp_path / "metadata2.json"
    other_meta.write_text(metadata_file.read_text().replace("bzImage", "bzImage-v2"))
    diff_image = m.compute_image_measurement(other_meta, cpu=4, memory="4G", dstack_mr_bin=stub_mr)
    assert diff_image.mrtd != baseline.mrtd
    assert diff_image.os_image_hash != baseline.os_image_hash


def test_compute_image_measurement_passes_cpu_and_memory(monkeypatch, metadata_file):
    captured = {}

    class _Proc:
        returncode = 0
        stdout = json.dumps(
            {
                "mrtd": "a" * 96,
                "rtmr0": "b" * 96,
                "rtmr1": "c" * 96,
                "rtmr2": "d" * 96,
                "mr_image": "e" * 64,
            }
        )
        stderr = ""

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return _Proc()

    monkeypatch.setattr(m.subprocess, "run", fake_run)
    m.compute_image_measurement(metadata_file, cpu=2, memory="2G", dstack_mr_bin="dstack-mr")
    cmd = captured["cmd"]
    assert cmd[0] == "dstack-mr"
    assert "-cpu" in cmd and cmd[cmd.index("-cpu") + 1] == "2"
    assert "-memory" in cmd and cmd[cmd.index("-memory") + 1] == "2G"
    assert "-json" in cmd
    assert "-metadata" in cmd and cmd[cmd.index("-metadata") + 1] == str(metadata_file)


def test_compute_image_measurement_raises_on_tool_failure(tmp_path, metadata_file):
    failing = str(_write_stub(tmp_path / "dstack-mr-fail", mode="fail"))
    with pytest.raises(RuntimeError, match="dstack-mr"):
        m.compute_image_measurement(metadata_file, cpu=4, memory="4G", dstack_mr_bin=failing)


def test_compute_image_measurement_rejects_non_hex_register(tmp_path, metadata_file):
    bad = str(_write_stub(tmp_path / "dstack-mr-badhex", mode="bad_hex"))
    with pytest.raises(ValueError, match="mrtd"):
        m.compute_image_measurement(metadata_file, cpu=4, memory="4G", dstack_mr_bin=bad)


def test_compute_image_measurement_rejects_wrong_width_register(tmp_path, metadata_file):
    bad = str(_write_stub(tmp_path / "dstack-mr-badwidth", mode="bad_width"))
    with pytest.raises(ValueError, match="mrtd"):
        m.compute_image_measurement(metadata_file, cpu=4, memory="4G", dstack_mr_bin=bad)


def test_memory_int_is_formatted_as_gigabytes(monkeypatch, metadata_file):
    captured = {}

    class _Proc:
        returncode = 0
        stdout = json.dumps(
            {
                "mrtd": "a" * 96,
                "rtmr0": "b" * 96,
                "rtmr1": "c" * 96,
                "rtmr2": "d" * 96,
                "mr_image": "e" * 64,
            }
        )
        stderr = ""

    monkeypatch.setattr(
        m.subprocess, "run", lambda cmd, *a, **k: captured.update(cmd=cmd) or _Proc()
    )
    m.compute_image_measurement(metadata_file, cpu=1, memory=8, dstack_mr_bin="dstack-mr")
    cmd = captured["cmd"]
    assert cmd[cmd.index("-memory") + 1] == "8G"


def test_dstack_mr_binary_resolution_prefers_explicit_then_env(monkeypatch):
    monkeypatch.delenv("DSTACK_MR_BIN", raising=False)
    assert m.dstack_mr_binary() == m.DEFAULT_DSTACK_MR_BIN
    assert m.dstack_mr_binary("/opt/dstack-mr") == "/opt/dstack-mr"
    monkeypatch.setenv("DSTACK_MR_BIN", "/env/dstack-mr")
    assert m.dstack_mr_binary() == "/env/dstack-mr"
    assert m.dstack_mr_binary("/opt/dstack-mr") == "/opt/dstack-mr"


# --------------------------------------------------------------------------- #
# VAL-IMG-009: stable, pinnable canonical measurement record
# --------------------------------------------------------------------------- #


def test_canonical_measurement_has_exactly_the_six_pinnable_fields(stub_mr, metadata_file):
    record = m.build_canonical_measurement(
        metadata_path=metadata_file,
        cpu=4,
        memory="4G",
        compose=_base_compose(),
        dstack_mr_bin=stub_mr,
    )
    data = record.as_dict()
    assert set(data) == {"mrtd", "rtmr0", "rtmr1", "rtmr2", "compose_hash", "os_image_hash"}
    # rtmr3 is a runtime register and must NOT be part of the pinnable record.
    assert "rtmr3" not in data


def test_canonical_measurement_fields_are_correctly_shaped(stub_mr, metadata_file):
    record = m.build_canonical_measurement(
        metadata_path=metadata_file,
        cpu=4,
        memory="4G",
        compose=_base_compose(),
        dstack_mr_bin=stub_mr,
    )
    for field in ("mrtd", "rtmr0", "rtmr1", "rtmr2"):
        assert HEX96_RE.match(getattr(record, field)), field
    assert HEX64_RE.match(record.compose_hash)
    assert HEX64_RE.match(record.os_image_hash)
    assert record.compose_hash == m.compose_hash(_base_compose())


def test_canonical_measurement_serialization_is_byte_stable_across_reemission(
    stub_mr, metadata_file
):
    def build():
        return m.build_canonical_measurement(
            metadata_path=metadata_file,
            cpu=4,
            memory="4G",
            compose=_base_compose(),
            dstack_mr_bin=stub_mr,
        )

    first = build().to_json()
    second = build().to_json()
    assert first == second


def test_canonical_measurement_json_is_sorted_and_pinnable(stub_mr, metadata_file):
    record = m.build_canonical_measurement(
        metadata_path=metadata_file,
        cpu=4,
        memory="4G",
        compose=_base_compose(),
        dstack_mr_bin=stub_mr,
    )
    serialized = record.to_json()
    parsed = json.loads(serialized)
    # A validator copies this verbatim into an allowlist entry: sorted keys +
    # compact separators make it byte-stable.
    assert serialized == json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    assert parsed == record.as_dict()


def test_canonical_measurement_drift_on_changed_image_or_compose(stub_mr, tmp_path, metadata_file):
    baseline = m.build_canonical_measurement(
        metadata_path=metadata_file,
        cpu=4,
        memory="4G",
        compose=_base_compose(),
        dstack_mr_bin=stub_mr,
    )

    changed_compose = copy.deepcopy(_base_compose())
    changed_compose["docker_compose_file"] = changed_compose["docker_compose_file"].replace(
        "a" * 64, "b" * 64
    )
    drift_compose = m.build_canonical_measurement(
        metadata_path=metadata_file,
        cpu=4,
        memory="4G",
        compose=changed_compose,
        dstack_mr_bin=stub_mr,
    )
    assert drift_compose.compose_hash != baseline.compose_hash
    assert drift_compose.to_json() != baseline.to_json()

    other_meta = tmp_path / "metadata3.json"
    other_meta.write_text(metadata_file.read_text().replace("ovmf.fd", "ovmf-next.fd"))
    drift_image = m.build_canonical_measurement(
        metadata_path=other_meta,
        cpu=4,
        memory="4G",
        compose=_base_compose(),
        dstack_mr_bin=stub_mr,
    )
    assert drift_image.mrtd != baseline.mrtd
    assert drift_image.os_image_hash != baseline.os_image_hash


def test_build_canonical_measurement_accepts_env_binary(monkeypatch, stub_mr, metadata_file):
    monkeypatch.setenv("DSTACK_MR_BIN", stub_mr)
    record = m.build_canonical_measurement(
        metadata_path=metadata_file,
        cpu=4,
        memory="4G",
        compose=_base_compose(),
    )
    assert HEX96_RE.match(record.mrtd)


def test_dstack_mr_available_reflects_binary_presence(stub_mr):
    assert m.dstack_mr_available(stub_mr) is True
    assert m.dstack_mr_available(os.path.join(os.sep, "nonexistent", "dstack-mr")) is False


def test_dstack_mr_available_uses_path_lookup_for_bare_name(monkeypatch):
    monkeypatch.delenv("DSTACK_MR_BIN", raising=False)
    monkeypatch.setattr(m.shutil, "which", lambda name: "/usr/bin/" + name)
    assert m.dstack_mr_available() is True
    monkeypatch.setattr(m.shutil, "which", lambda name: None)
    assert m.dstack_mr_available() is False


def test_format_memory_rejects_bool():
    with pytest.raises(TypeError):
        m._format_memory(True)


def test_compute_image_measurement_rejects_non_json_output(monkeypatch, metadata_file):
    class _Proc:
        returncode = 0
        stdout = "not json at all"
        stderr = ""

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(RuntimeError, match="non-JSON"):
        m.compute_image_measurement(metadata_file, cpu=4, memory="4G", dstack_mr_bin="dstack-mr")


def test_compute_image_measurement_derives_product_os_when_mr_image_absent(
    monkeypatch, metadata_file
):
    """Product seals sha256(registers); catalog mr_image is optional."""

    class _Proc:
        returncode = 0
        stdout = json.dumps(
            {"mrtd": "a" * 96, "rtmr0": "b" * 96, "rtmr1": "c" * 96, "rtmr2": "d" * 96}
        )
        stderr = ""

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Proc())
    image = m.compute_image_measurement(
        metadata_file, cpu=4, memory="4G", dstack_mr_bin="dstack-mr"
    )
    expected = m.product_os_image_hash(mrtd="a" * 96, rtmr1="c" * 96, rtmr2="d" * 96)
    assert image.os_image_hash == expected
    assert image.dstack_mr_image is None


def test_compute_image_measurement_rejects_non_string_register(monkeypatch, metadata_file):
    class _Proc:
        returncode = 0
        stdout = json.dumps(
            {
                "mrtd": 123,
                "rtmr0": "b" * 96,
                "rtmr1": "c" * 96,
                "rtmr2": "d" * 96,
                "mr_image": "e" * 64,
            }
        )
        stderr = ""

    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(ValueError, match="mrtd"):
        m.compute_image_measurement(metadata_file, cpu=4, memory="4G", dstack_mr_bin="dstack-mr")
