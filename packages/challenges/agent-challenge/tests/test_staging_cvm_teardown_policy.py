"""Owned-only CVM teardown policy for staging (foreign ids never selected)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "staging"
    / "cvm_teardown_policy.py"
)


def _load_policy():
    spec = importlib.util.spec_from_file_location(
        "cvm_teardown_policy", POLICY_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def policy():
    return _load_policy()


class TestSelectTeardownIds:
    def test_empty_owned_selects_nothing_even_with_account_ids(self, policy) -> None:
        # Given: no owned ids, account has foreign CVMs
        account = ["cvm_prod_eval_11", "cvm_other"]
        # When: default selection
        to_delete, rejected = policy.select_teardown_ids(
            owned_ids=[],
            account_ids=account,
            account_sweep=False,
        )
        # Then: nothing deleted; foreign never selected
        assert to_delete == []
        assert "cvm_prod_eval_11" not in to_delete

    def test_only_owned_ids_selected(self, policy) -> None:
        owned = ["cvm_staging_a", "cvm_staging_b"]
        account = ["cvm_staging_a", "cvm_prod_eval_11", "cvm_staging_b", "cvm_x"]
        to_delete, _ = policy.select_teardown_ids(
            owned_ids=owned,
            account_ids=account,
            account_sweep=False,
        )
        assert to_delete == owned
        assert "cvm_prod_eval_11" not in to_delete
        assert "cvm_x" not in to_delete

    def test_account_sweep_flag_does_not_expand_delete_set(self, policy) -> None:
        # Given: opt-in account_sweep (loud path in shell) still cannot expand
        owned = ["cvm_mine"]
        account = ["cvm_mine", "cvm_foreign_prod"]
        to_delete, _ = policy.select_teardown_ids(
            owned_ids=owned,
            account_ids=account,
            account_sweep=True,
        )
        assert to_delete == ["cvm_mine"]
        assert "cvm_foreign_prod" not in to_delete

    def test_dedup_preserves_order(self, policy) -> None:
        to_delete, _ = policy.select_teardown_ids(
            owned_ids=["cvm_a", "cvm_b", "cvm_a", "  cvm_c  ", ""],
        )
        assert to_delete == ["cvm_a", "cvm_b", "cvm_c"]


class TestAssertIdOwned:
    def test_foreign_id_refused(self, policy) -> None:
        with pytest.raises(SystemExit, match="foreign CVM"):
            policy.assert_id_owned("cvm_foreign", ["cvm_mine"])

    def test_owned_id_allowed(self, policy) -> None:
        policy.assert_id_owned("cvm_mine", ["cvm_mine", "cvm_other"])

    def test_empty_id_refused(self, policy) -> None:
        with pytest.raises(SystemExit, match="empty"):
            policy.assert_id_owned("  ", ["cvm_mine"])


class TestPlanAndCli:
    def test_plan_reports_foreign_not_deleted(self, policy, tmp_path: Path) -> None:
        track = tmp_path / "cvms.txt"
        track.write_text("cvm_owned_1\ncvm_owned_2\n", encoding="utf-8")
        plan = policy.plan_teardown(
            owned_paths=[track],
            account_ids=["cvm_owned_1", "cvm_foreign_prod", "cvm_owned_2"],
            account_sweep=False,
        )
        assert plan["will_delete"] == ["cvm_owned_1", "cvm_owned_2"]
        assert "cvm_foreign_prod" in plan["will_not_delete_foreign"]
        assert "cvm_foreign_prod" not in plan["will_delete"]

    def test_cli_dry_run_never_lists_foreign(
        self, tmp_path: Path
    ) -> None:
        track = tmp_path / "owned.txt"
        track.write_text("cvm_run_only\n", encoding="utf-8")
        account = json.dumps(
            {"ids": ["cvm_run_only", "cvm_prod_submission_11", "cvm_noise"]}
        )
        proc = subprocess.run(
            [
                sys.executable,
                str(POLICY_PATH),
                "--owned-file",
                str(track),
                "--account-ids-json",
                account,
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        plan = json.loads(proc.stdout)
        assert plan["will_delete"] == ["cvm_run_only"]
        assert "cvm_prod_submission_11" not in plan["will_delete"]
        assert "cvm_prod_submission_11" in plan["will_not_delete_foreign"]

    def test_cli_check_id_rejects_foreign(self, tmp_path: Path) -> None:
        track = tmp_path / "owned.txt"
        track.write_text("cvm_mine\n", encoding="utf-8")
        proc = subprocess.run(
            [
                sys.executable,
                str(POLICY_PATH),
                "--owned-file",
                str(track),
                "--check-id",
                "cvm_foreign",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "foreign" in (proc.stderr + proc.stdout).lower()

    def test_load_owned_merges_multiple_files(self, policy, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("cvm_1\n# comment\ncvm_2\n", encoding="utf-8")
        b.write_text("cvm_2\ncvm_3\n", encoding="utf-8")
        assert policy.load_owned_ids(a, b) == ["cvm_1", "cvm_2", "cvm_3"]
