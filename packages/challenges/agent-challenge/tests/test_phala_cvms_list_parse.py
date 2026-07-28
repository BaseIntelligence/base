"""Fail-loud CVM list parsing — never under-report spend as count 0.

Safety guard: an unrecognized GET /cvms (or CLI) payload must raise, not
silently become an empty list. Known-good paginated and bare-list shapes
must parse. Teardown confirmation fails closed when the count is
indeterminate.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from agent_challenge.selfdeploy.phala import (
    DEFAULT_PHALA_API_VERSION,
    DEFAULT_PHALA_USER_AGENT,
    CvmListParseError,
    PhalaApiError,
    PhalaCloudClient,
    parse_cvms_list_response,
    resolve_cvm_id_from_list,
)
from agent_challenge.selfdeploy.plan import PHALA_API_KEY_ENV

POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "staging"
    / "cvm_teardown_policy.py"
)


def _load_policy():
    spec = importlib.util.spec_from_file_location("cvm_teardown_policy", POLICY_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _CapturingOpener:
    def __init__(self, payloads: list[Any] | Any) -> None:
        self.requests: list[Request] = []
        if isinstance(payloads, list) and payloads and not isinstance(
            payloads[0], dict
        ):
            # list of sequential response bodies
            self._queue = list(payloads)
        elif isinstance(payloads, list) and all(
            isinstance(p, (dict, list)) for p in payloads
        ):
            # could be one list-body OR queue of bodies — treat multi as queue
            # when first element looks like a full response object with items/total
            if (
                len(payloads) > 1
                and isinstance(payloads[0], dict)
                and ("items" in payloads[0] or "total" in payloads[0])
            ):
                self._queue = list(payloads)
            elif len(payloads) == 1:
                self._queue = list(payloads)
            else:
                self._queue = list(payloads)
        else:
            self._queue = [payloads]

    def __call__(self, request: Request, timeout: float = 0.0):  # noqa: ARG002
        self.requests.append(request)
        if not self._queue:
            body: Any = {"items": [], "total": 0, "page": 1, "page_size": 50, "pages": 0}
        else:
            body = self._queue.pop(0)

        class _Resp:
            def __init__(self, raw: bytes) -> None:
                self._body = raw

            def read(self, n: int = -1) -> bytes:  # noqa: ARG002
                return self._body

        return _Resp(json.dumps(body).encode())


# --------------------------------------------------------------------------- #
# parse_cvms_list_response — known good
# --------------------------------------------------------------------------- #


class TestParseCvmsListKnownGood:
    def test_paginated_cli_shape_with_total(self) -> None:
        # Given: CLI /cvms/paginated envelope (2026-06-23)
        payload = {
            "success": True,
            "page": 1,
            "pageSize": 50,
            "total": 1,
            "totalPages": 1,
            "items": [
                {
                    "id": "cvm_abc",
                    "app_id": "be7f13772257facda88080a25ef2ac0d1ab9dfe5",
                    "name": "agent-challenge-canonical",
                    "status": "running",
                }
            ],
        }
        # When
        snap = parse_cvms_list_response(payload)
        # Then: count comes from total, not silent empty
        assert snap.total == 1
        assert list(snap.ids) == ["cvm_abc"]
        assert len(snap.items) == 1
        assert snap.items[0]["app_id"].startswith("be7f")

    def test_paginated_api_snake_case(self) -> None:
        payload = {
            "items": [{"id": "cvm_x", "vm_uuid": "u-1"}],
            "total": 1,
            "page": 1,
            "page_size": 30,
            "pages": 1,
        }
        snap = parse_cvms_list_response(payload)
        assert snap.total == 1
        assert list(snap.ids) == ["cvm_x"]

    def test_bare_list_of_dicts(self) -> None:
        payload = [{"id": "cvm_1"}, {"id": 42, "name": "n"}]
        snap = parse_cvms_list_response(payload)
        assert snap.total == 2
        assert list(snap.ids) == ["cvm_1", "42"]

    def test_empty_paginated_is_zero_not_error(self) -> None:
        snap = parse_cvms_list_response(
            {"items": [], "total": 0, "page": 1, "page_size": 50, "pages": 0}
        )
        assert snap.total == 0
        assert list(snap.ids) == []
        assert list(snap.items) == []

    def test_data_key_list(self) -> None:
        snap = parse_cvms_list_response({"data": [{"id": "cvm_d"}]})
        assert snap.total == 1
        assert list(snap.ids) == ["cvm_d"]

    def test_cvms_key_list(self) -> None:
        snap = parse_cvms_list_response({"cvms": [{"cvm_id": "cvm_c"}]})
        assert snap.total == 1
        assert list(snap.ids) == ["cvm_c"]

    def test_total_preferred_over_page_len_when_consistent(self) -> None:
        # Single page fully loaded: total matches len(items)
        snap = parse_cvms_list_response(
            {"items": [{"id": "a"}, {"id": "b"}], "total": 2}
        )
        assert snap.total == 2


# --------------------------------------------------------------------------- #
# parse_cvms_list_response — fail loud (never 0 on confusion)
# --------------------------------------------------------------------------- #


class TestParseCvmsListFailLoud:
    def test_unknown_object_shape_raises(self) -> None:
        # Given: object with neither items/data/cvms nor a list body
        payload = {"success": True, "result": {"vms": [{"id": "hidden"}]}, "ok": 1}
        # When / Then: raise — must NOT become count 0
        with pytest.raises((CvmListParseError, PhalaApiError, ValueError)) as ei:
            parse_cvms_list_response(payload)
        msg = str(ei.value).lower()
        assert "unrecognized" in msg or "unknown" in msg or "shape" in msg

    def test_null_payload_raises(self) -> None:
        with pytest.raises((CvmListParseError, PhalaApiError, ValueError)):
            parse_cvms_list_response(None)

    def test_string_payload_raises(self) -> None:
        with pytest.raises((CvmListParseError, PhalaApiError, ValueError)):
            parse_cvms_list_response("not-json-object")

    def test_items_not_a_list_raises(self) -> None:
        with pytest.raises((CvmListParseError, PhalaApiError, ValueError)):
            parse_cvms_list_response({"items": {"id": "cvm_x"}, "total": 1})

    def test_total_not_int_raises(self) -> None:
        with pytest.raises((CvmListParseError, PhalaApiError, ValueError)):
            parse_cvms_list_response({"items": [], "total": "zero"})

    def test_total_disagrees_with_single_page_items_raises(self) -> None:
        # Safety: total=0 with non-empty items (or vice versa on full page) is
        # indeterminate — fail closed rather than pick the wrong number.
        with pytest.raises((CvmListParseError, PhalaApiError, ValueError)):
            parse_cvms_list_response(
                {
                    "items": [{"id": "cvm_live"}],
                    "total": 0,
                    "page": 1,
                    "page_size": 50,
                    "pages": 0,
                }
            )

    def test_resolve_cvm_id_propagates_unknown_shape(self) -> None:
        with pytest.raises((CvmListParseError, PhalaApiError, ValueError)):
            resolve_cvm_id_from_list(
                {"weird": True},
                app_id="be7f13772257facda88080a25ef2ac0d1ab9dfe5",
            )


# --------------------------------------------------------------------------- #
# Client pins + list_cvms uses paginated route
# --------------------------------------------------------------------------- #


class TestPhalaClientListCvms:
    def test_api_version_and_user_agent_match_cli_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_test")
        assert DEFAULT_PHALA_API_VERSION == "2026-06-23"
        assert DEFAULT_PHALA_USER_AGENT == "phala-cloud-cli/1.1.19"
        opener = _CapturingOpener(
            {"items": [], "total": 0, "page": 1, "page_size": 50, "pages": 0}
        )
        client = PhalaCloudClient(api_key="phak_test", opener=opener)
        client.list_cvms()
        headers = {k.lower(): v for k, v in opener.requests[0].header_items()}
        assert headers.get("x-phala-version") == "2026-06-23"
        assert headers.get("user-agent") == "phala-cloud-cli/1.1.19"
        assert headers.get("x-api-key") == "phak_test"
        url = opener.requests[0].full_url
        assert "/cvms/paginated" in url

    def test_list_cvms_parses_known_good(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_test")
        opener = _CapturingOpener(
            {
                "items": [
                    {
                        "id": "cvm_live",
                        "app_id": "be7f13772257facda88080a25ef2ac0d1ab9dfe5",
                    }
                ],
                "total": 1,
                "page": 1,
                "page_size": 50,
                "pages": 1,
            }
        )
        client = PhalaCloudClient(api_key="phak_test", opener=opener)
        snap = client.list_cvms()
        assert snap.total == 1
        assert list(snap.ids) == ["cvm_live"]

    def test_list_cvms_unknown_shape_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_test")
        opener = _CapturingOpener({"status": "ok", "payload": []})
        client = PhalaCloudClient(api_key="phak_test", opener=opener)
        with pytest.raises((CvmListParseError, PhalaApiError, ValueError)):
            client.list_cvms()


# --------------------------------------------------------------------------- #
# Teardown policy: indeterminate listing fails closed
# --------------------------------------------------------------------------- #


class TestTeardownFailsClosedOnIndeterminate:
    def test_cli_account_json_unknown_shape_exits_nonzero(self, tmp_path: Path) -> None:
        track = tmp_path / "owned.txt"
        track.write_text("cvm_mine\n", encoding="utf-8")
        bad = json.dumps({"status": "ok", "vms": [{"id": "cvm_mine"}]})
        proc = subprocess.run(
            [
                sys.executable,
                str(POLICY_PATH),
                "--owned-file",
                str(track),
                "--account-ids-json",
                bad,
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0, proc.stdout + proc.stderr
        blob = (proc.stderr + proc.stdout).lower()
        assert "unrecognized" in blob or "unknown" in blob or "shape" in blob

    def test_cli_account_json_paginated_parses(self, tmp_path: Path) -> None:
        track = tmp_path / "owned.txt"
        track.write_text("cvm_mine\n", encoding="utf-8")
        good = json.dumps(
            {
                "items": [
                    {"id": "cvm_mine", "name": "staging"},
                    {"id": "cvm_foreign", "name": "prod"},
                ],
                "total": 2,
            }
        )
        proc = subprocess.run(
            [
                sys.executable,
                str(POLICY_PATH),
                "--owned-file",
                str(track),
                "--account-ids-json",
                good,
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        plan = json.loads(proc.stdout)
        assert plan["will_delete"] == ["cvm_mine"]
        assert "cvm_foreign" in plan["will_not_delete_foreign"]

    def test_policy_parse_account_payload_raises(self) -> None:
        policy = _load_policy()
        with pytest.raises((SystemExit, ValueError, TypeError)):
            policy.parse_account_cvms_payload({"nope": True})
