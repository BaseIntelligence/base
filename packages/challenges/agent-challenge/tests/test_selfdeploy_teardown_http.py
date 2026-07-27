"""HTTP teardown for Phala CVMs (no external ``phala`` binary).

Covers:
  * DELETE /cvms/{id} allowlisted on PhalaCloudClient
  * 204 success, 404 idempotent success, other status → PhalaApiError
  * default_phala_teardown never shells out to a ``phala`` binary
  * optional --cvm-id resolved via GET /cvms + unique app_id match
  * ambiguous multi-match refused
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy.phala import (
    PhalaApiError,
    PhalaCloudClient,
    resolve_cvm_id_from_list,
)


class _FakeResponse:
    def __init__(self, body: bytes = b"", *, status: int = 204) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body


def test_delete_cvm_issues_delete_to_cvms_id_path() -> None:
    seen: list[Request] = []

    def opener(request: Request, timeout: float = 0) -> _FakeResponse:  # noqa: ARG001
        seen.append(request)
        return _FakeResponse(b"", status=204)

    client = PhalaCloudClient(api_key="k" * 32, opener=opener)
    client.delete_cvm("cvm-abc-1")
    assert len(seen) == 1
    assert seen[0].get_method() == "DELETE"
    assert seen[0].full_url.endswith("/cvms/cvm-abc-1")


def test_delete_cvm_treats_204_as_success() -> None:
    client = PhalaCloudClient(
        api_key="k" * 32,
        opener=lambda request, timeout=0: _FakeResponse(b"", status=204),  # noqa: ARG005
    )
    client.delete_cvm("42")  # must not raise


def test_delete_cvm_treats_404_as_idempotent_success() -> None:
    def opener(request: Request, timeout: float = 0) -> _FakeResponse:  # noqa: ARG001
        raise HTTPError(
            url=request.full_url,
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b""),
        )

    client = PhalaCloudClient(api_key="k" * 32, opener=opener)
    client.delete_cvm("already-gone")  # must not raise


@pytest.mark.parametrize("code", [400, 401, 403, 500])
def test_delete_cvm_raises_on_non_success_status(code: int) -> None:
    def opener(request: Request, timeout: float = 0) -> _FakeResponse:  # noqa: ARG001
        raise HTTPError(
            url=request.full_url,
            code=code,
            msg="err",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b"{}"),
        )

    client = PhalaCloudClient(api_key="k" * 32, opener=opener)
    with pytest.raises(PhalaApiError, match=f"HTTP {code}"):
        client.delete_cvm("cvm-1")


def test_delete_cvm_refuses_non_allowlisted_path_shape() -> None:
    client = PhalaCloudClient(
        api_key="k" * 32,
        opener=lambda *_a, **_k: _FakeResponse(),  # pragma: no cover
    )
    with pytest.raises(PhalaApiError, match="unsupported|invalid"):
        client.delete_cvm("../escape")
    with pytest.raises(PhalaApiError, match="unsupported|invalid"):
        client.delete_cvm("id/with/slash")


def test_delete_is_not_available_via_post_allowlist() -> None:
    client = PhalaCloudClient(
        api_key="k" * 32,
        opener=lambda *_a, **_k: _FakeResponse(b"{}"),  # pragma: no cover
    )
    with pytest.raises(PhalaApiError, match="unsupported"):
        client.post("/cvms/cvm-1", {})


def test_resolve_cvm_id_require_unique_refuses_ambiguous_match() -> None:
    listing = {
        "items": [
            {"id": 1, "app_id": "same-app"},
            {"id": 2, "app_id": "same-app"},
        ]
    }
    with pytest.raises(PhalaApiError, match="multiple|ambiguous"):
        resolve_cvm_id_from_list(listing, app_id="same-app", require_unique=True)


def test_resolve_cvm_id_require_unique_returns_single_match() -> None:
    listing = {
        "items": [
            {"id": 9, "app_id": "other"},
            {"id": 77, "app_id": "target-app"},
        ]
    }
    assert resolve_cvm_id_from_list(listing, app_id="target-app", require_unique=True) == "77"


def test_default_phala_teardown_uses_http_delete_not_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        MagicMock(side_effect=AssertionError("no subprocess")),
    )
    deleted: list[str] = []

    class _Client:
        def delete_cvm(self, cvm_id: str) -> None:
            deleted.append(cvm_id)

    result = cli.default_phala_teardown("cvm-live-1", client=_Client())  # type: ignore[arg-type]
    assert result["ok"] is True
    assert result["returncode"] == 0
    assert deleted == ["cvm-live-1"]
    assert "phala" not in json.dumps(result).lower() or result.get("error") is None


def test_default_phala_teardown_maps_api_error() -> None:
    class _Client:
        def delete_cvm(self, cvm_id: str) -> None:
            raise PhalaApiError("Phala delete returned HTTP 500")

    result = cli.default_phala_teardown("cvm-x", client=_Client())  # type: ignore[arg-type]
    assert result["ok"] is False
    assert result["returncode"] != 0
    assert "500" in str(result.get("error") or "")


def test_cli_teardown_resolves_cvm_id_from_app_id(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = {"items": [{"id": "resolved-9", "app_id": "app-hex-1"}]}
    deleted: list[str] = []

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def get(self, path: str) -> dict[str, Any]:
            assert path == "/cvms"
            return listing

        def delete_cvm(self, cvm_id: str) -> None:
            deleted.append(cvm_id)

    monkeypatch.setattr(cli, "PhalaCloudClient", _Client)
    monkeypatch.setenv("PHALA_CLOUD_API_KEY", "k" * 32)
    capture: list[Any] = []
    monkeypatch.setattr(cli, "_print", lambda payload: capture.append(payload))

    parser = cli.build_parser()
    args = parser.parse_args(["review", "teardown", "--app-id", "app-hex-1"])
    code = cli._ordered_review_command(args)
    assert code == 0
    assert deleted == ["resolved-9"]
    assert capture and capture[0].get("ok") is True
    assert capture[0].get("torn_down") == "resolved-9"


def test_cli_teardown_refuses_ambiguous_app_id(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = {
        "items": [
            {"id": "a", "app_id": "dup"},
            {"id": "b", "app_id": "dup"},
        ]
    }

    class _Client:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def get(self, path: str) -> dict[str, Any]:
            return listing

        def delete_cvm(self, cvm_id: str) -> None:  # pragma: no cover
            raise AssertionError(f"must not delete on ambiguity: {cvm_id}")

    monkeypatch.setattr(cli, "PhalaCloudClient", _Client)
    monkeypatch.setenv("PHALA_CLOUD_API_KEY", "k" * 32)
    args = SimpleNamespace(review_command="teardown", cvm_id=None, app_id="dup", phala_api=None)
    capture: list = []
    monkeypatch.setattr(cli, "_print", lambda payload: capture.append(payload))
    code = cli._ordered_review_command(args)
    assert code != 0
    assert capture and "multiple" in str(capture[0].get("diagnostics", {}).get("error", "")).lower()


def test_cli_teardown_requires_cvm_id_or_app_id() -> None:
    parser = cli.build_parser()
    # --cvm-id no longer required at parse time; runtime refuses empty identity.
    args = parser.parse_args(["teardown"])
    assert getattr(args, "cvm_id", None) in (None, "")
    # Runtime path
    import sys
    from io import StringIO

    buf = StringIO()
    old = sys.stderr
    try:
        sys.stderr = buf
        code = cli.main(["teardown"], teardowner=lambda *_a, **_k: {"ok": True, "returncode": 0})
    finally:
        sys.stderr = old
    # Without cvm-id/app-id should fail closed (non-zero) rather than call teardowner with None
    assert code != 0
