"""Per-validator on-chain weight submitter tests.

Covers durable ledger idempotence, exact master-vector submission, multi-validator
wallet ownership, gated default-off no-op, rejection/retry/ambiguous outcomes,
restart safety, new-vector supersession, unusable-vector rejection before
wallet/chain side effects, identity binding, and master/validator separation.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from base.challenge_sdk.roles import Role, activate_role
from base.schemas.weights import MasterWeightsResponse
from base.validator.submission_ledger import (
    SubmissionStatus,
    ValidatorSubmissionLedger,
)
from base.validator.weight_submitter import (
    ValidatorSubmitOutcome,
    ValidatorWeightSubmitter,
)
from base.validator.weights_client import validate_master_weights_payload


@pytest.fixture(autouse=True)
def _activate_validator_role() -> Iterator[None]:
    with activate_role(Role.VALIDATOR):
        yield


# Fixed reference time. Payloads anchor their computed_at/expires_at to REF and the
# submitter clock is pinned to REF, so freshness/expiry are deterministic. REF is in
# the future so MasterWeightsResponse's construction-time not-expired validator passes.
REF = datetime(2030, 1, 1, 12, 0, tzinfo=UTC)


def _fresh_payload(
    *,
    netuid: int = 100,
    computed_offset_seconds: int = 0,
    uids: list[int] | None = None,
    weights: list[float] | None = None,
    vector_id: str | None = "vector-1",
    vector_digest: str | None = "a" * 64,
    chain_endpoint: str = "wss://chain.example:9944",
) -> MasterWeightsResponse:
    """A valid, fresh master vector whose computed_at is ``offset`` secs before REF."""

    final_uids = uids if uids is not None else [0, 1]
    final_weights = weights if weights is not None else [0.5, 0.5]
    chain_domain = json.dumps(
        {"netuid": netuid, "uids": final_uids, "weights": final_weights},
        sort_keys=True,
        separators=(",", ":"),
    )
    return MasterWeightsResponse(
        protocol_version="1.0",
        vector_id=vector_id,
        vector_digest=vector_digest,
        netuid=netuid,
        chain_endpoint=chain_endpoint,
        uids=final_uids,
        weights=final_weights,
        chain_domain_bytes=chain_domain,
        computed_at=REF - timedelta(seconds=computed_offset_seconds),
        expires_at=REF + timedelta(seconds=1260),
        source_challenges=[],
        metagraph_updated_at=REF,
    )


class _FetchClient:
    """Stand-in ``WeightsClient`` returning a (mutable) canned master vector."""

    def __init__(self, payload: MasterWeightsResponse) -> None:
        self._payload = payload
        self.calls = 0

    async def fetch_latest(self) -> MasterWeightsResponse:
        self.calls += 1
        return self._payload

    def set_payload(self, payload: MasterWeightsResponse) -> None:
        self._payload = payload


class _RecordingSetter:
    """A ``WeightSetter`` stand-in bound to THIS validator's own hotkey."""

    def __init__(
        self,
        hotkey: str,
        *,
        result: Any = None,
        raises: BaseException | None = None,
        results: list[Any] | None = None,
    ) -> None:
        self.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address=hotkey))
        self.calls: list[tuple[list[int], list[float]]] = []
        self._result = (
            result
            if result is not None
            else SimpleNamespace(success=True, message="ok")
        )
        self._raises = raises
        self._results = list(results) if results is not None else None

    def set_weights(self, uids: list[int], weights: list[float]) -> Any:
        self.calls.append((list(uids), list(weights)))
        if self._raises is not None:
            raise self._raises
        if self._results is not None:
            if not self._results:
                return SimpleNamespace(success=True, message="ok")
            return self._results.pop(0)
        return self._result


def _submitter(
    *,
    client: Any,
    setter: Any = None,
    factory: Any = None,
    submit_enabled: bool = True,
    netuid: int = 100,
    state_dir: Path | None = None,
    expected_hotkey: str | None = None,
    max_attempts: int = 5,
    backoff_base_seconds: float = 0.0,
    require_provenance: bool = False,
    observation_reporter: Any = None,
) -> ValidatorWeightSubmitter:
    def default_factory() -> Any:
        return setter

    return ValidatorWeightSubmitter(
        submit_enabled=submit_enabled,
        netuid=netuid,
        weights_client=client,
        weight_setter_factory=factory or default_factory,
        clock=lambda: REF,
        state_dir=state_dir,
        expected_hotkey=expected_hotkey,
        max_attempts=max_attempts,
        backoff_base_seconds=backoff_base_seconds,
        backoff_max_seconds=0.0,
        require_provenance=require_provenance,
        observation_reporter=observation_reporter,
    )


# --- VAL-CODE-VWGT / VAL-SDK-065 / VAL-WEIGHT-057 ------------------------------


async def test_fetches_master_vector_and_submits_with_own_keypair(
    tmp_path: Path,
) -> None:
    payload = _fresh_payload(uids=[0, 3, 7], weights=[0.2, 0.3, 0.5])
    client = _FetchClient(payload)
    setter = _RecordingSetter("validator-a-hotkey")
    built = {"n": 0}

    def factory() -> _RecordingSetter:
        built["n"] += 1
        return setter

    submitter = _submitter(
        client=client,
        factory=factory,
        state_dir=tmp_path,
        expected_hotkey="validator-a-hotkey",
    )

    outcome = await submitter.run_once()

    assert outcome is ValidatorSubmitOutcome.SUBMITTED
    assert client.calls == 1
    assert setter.calls == [([0, 3, 7], [0.2, 0.3, 0.5])]
    assert setter.wallet.hotkey.ss58_address == "validator-a-hotkey"
    assert built["n"] == 1
    records = submitter.ledger.all_records()
    assert len(records) == 1
    assert records[0].status == SubmissionStatus.ACCEPTED.value
    assert records[0].vector_id == "vector-1"


async def test_two_independent_validators_each_submit_with_own_hotkey(
    tmp_path: Path,
) -> None:
    payload = _fresh_payload(uids=[0, 1], weights=[0.4, 0.6])
    setter_a = _RecordingSetter("hotkey-A")
    setter_b = _RecordingSetter("hotkey-B")
    sub_a = _submitter(
        client=_FetchClient(payload),
        setter=setter_a,
        state_dir=tmp_path / "a",
        expected_hotkey="hotkey-A",
    )
    sub_b = _submitter(
        client=_FetchClient(payload),
        setter=setter_b,
        state_dir=tmp_path / "b",
        expected_hotkey="hotkey-B",
    )

    assert await sub_a.run_once() is ValidatorSubmitOutcome.SUBMITTED
    assert await sub_b.run_once() is ValidatorSubmitOutcome.SUBMITTED

    assert setter_a.calls == [([0, 1], [0.4, 0.6])]
    assert setter_b.calls == [([0, 1], [0.4, 0.6])]
    assert setter_a.wallet.hotkey.ss58_address == "hotkey-A"
    assert setter_b.wallet.hotkey.ss58_address == "hotkey-B"

    assert await sub_a.run_once() is ValidatorSubmitOutcome.ALREADY_SUBMITTED
    assert len(setter_a.calls) == 1
    assert len(setter_b.calls) == 1


async def test_idempotent_rerun_is_noop_until_master_publishes_new_vector(
    tmp_path: Path,
) -> None:
    client = _FetchClient(
        _fresh_payload(
            computed_offset_seconds=30,
            weights=[0.5, 0.5],
            vector_id="v1",
            vector_digest="b" * 64,
        )
    )
    setter = _RecordingSetter("hotkey-A")
    submitter = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
    )

    assert await submitter.run_once() is ValidatorSubmitOutcome.SUBMITTED
    assert await submitter.run_once() is ValidatorSubmitOutcome.ALREADY_SUBMITTED
    assert await submitter.run_once() is ValidatorSubmitOutcome.ALREADY_SUBMITTED
    assert len(setter.calls) == 1

    client.set_payload(
        _fresh_payload(
            computed_offset_seconds=10,
            weights=[0.3, 0.7],
            vector_id="v2",
            vector_digest="c" * 64,
        )
    )
    assert await submitter.run_once() is ValidatorSubmitOutcome.SUBMITTED
    assert setter.calls == [([0, 1], [0.5, 0.5]), ([0, 1], [0.3, 0.7])]


async def test_gate_off_is_a_full_noop_no_fetch_no_setter_build(tmp_path: Path) -> None:
    client = _FetchClient(_fresh_payload())

    def exploding_factory() -> Any:
        raise AssertionError("gate off must NOT build a WeightSetter / live Subtensor")

    submitter = _submitter(
        client=client,
        factory=exploding_factory,
        submit_enabled=False,
        state_dir=tmp_path,
    )

    outcome = await submitter.run_once()

    assert outcome is ValidatorSubmitOutcome.DISABLED
    assert submitter.submit_enabled is False
    assert client.calls == 0
    assert submitter.last_submitted_key is None
    assert submitter.ledger.all_records() == []


async def test_rejected_commit_is_retried_and_not_marked_submitted(
    tmp_path: Path,
) -> None:
    client = _FetchClient(_fresh_payload(computed_offset_seconds=30))
    setter = _RecordingSetter(
        "hotkey-A",
        raises=RuntimeError("subtensor rejected weight submission: TooFast"),
    )
    submitter = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
        backoff_base_seconds=0.0,
    )

    assert await submitter.run_once() is ValidatorSubmitOutcome.REJECTED
    record = submitter.ledger.all_records()[0]
    assert record.status == SubmissionStatus.REJECTED.value
    assert len(setter.calls) == 1
    assert await submitter.run_once() is ValidatorSubmitOutcome.REJECTED
    assert len(setter.calls) == 2


async def test_rejected_result_object_also_retries(tmp_path: Path) -> None:
    client = _FetchClient(_fresh_payload(computed_offset_seconds=30))
    setter = _RecordingSetter(
        "hotkey-A", result=SimpleNamespace(success=False, message="too fast")
    )
    submitter = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
        backoff_base_seconds=0.0,
    )

    assert await submitter.run_once() is ValidatorSubmitOutcome.REJECTED
    assert submitter.ledger.all_records()[0].status == SubmissionStatus.REJECTED.value
    assert len(setter.calls) == 1


async def test_ambiguous_timeout_is_unknown_not_accepted(tmp_path: Path) -> None:
    client = _FetchClient(_fresh_payload())
    setter = _RecordingSetter(
        "hotkey-A", raises=RuntimeError("timeout waiting for inclusion")
    )
    submitter = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
        backoff_base_seconds=0.0,
    )
    assert await submitter.run_once() is ValidatorSubmitOutcome.UNKNOWN
    assert submitter.ledger.all_records()[0].status == SubmissionStatus.UNKNOWN.value


async def test_fetch_failure_is_no_vector_and_no_submit(tmp_path: Path) -> None:
    class _BoomClient:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch_latest(self) -> MasterWeightsResponse:
            self.calls += 1
            raise RuntimeError("master unreachable")

    setter = _RecordingSetter("hotkey-A")
    submitter = _submitter(
        client=_BoomClient(),
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
    )

    assert await submitter.run_once() is ValidatorSubmitOutcome.NO_VECTOR
    assert setter.calls == []


async def test_invalid_master_vector_is_skipped(tmp_path: Path) -> None:
    client = _FetchClient(_fresh_payload(netuid=999))
    setter = _RecordingSetter("hotkey-A")
    submitter = _submitter(
        client=client,
        setter=setter,
        netuid=100,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
    )

    assert await submitter.run_once() is ValidatorSubmitOutcome.NO_VECTOR
    assert setter.calls == []


def test_validate_rejects_unusable_vectors() -> None:
    cases: list[tuple[dict[str, Any], str, bool]] = [
        (
            {
                "uids": [],
                "weights": [],
                "chain_domain_bytes": json.dumps(
                    {"netuid": 100, "uids": [], "weights": []},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
            "empty",
            False,
        ),
        ({"weights": [0.5]}, "lengths", False),
        (
            {
                "uids": [1, 0],
                "chain_domain_bytes": json.dumps(
                    {"netuid": 100, "uids": [1, 0], "weights": [0.5, 0.5]},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
            "sorted",
            False,
        ),
        (
            {
                "uids": [1, 1],
                "chain_domain_bytes": json.dumps(
                    {"netuid": 100, "uids": [1, 1], "weights": [0.5, 0.5]},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
            "duplicate",
            False,
        ),
        ({"weights": [0.5, -0.1]}, "negative", False),
        ({"weights": [0.9, 0.9]}, "exceeds 1", False),
        ({"vector_id": None}, "vector_id", True),
        ({"vector_digest": None}, "vector_digest", True),
    ]
    for patch, token, require in cases:
        data = _fresh_payload().model_dump()
        data.update(patch)
        payload = MasterWeightsResponse.model_validate(data)
        failure = validate_master_weights_payload(
            payload,
            netuid=100,
            weights_freshness_seconds=720,
            now=REF,
            require_provenance=require,
        )
        assert failure is not None, token
        assert token.split()[0] in failure or token in failure

    # Non-finite values cannot pass pydantic construction; construct minimal bypass.
    bad = _fresh_payload()
    object.__setattr__(bad, "weights", [0.5, float("nan")])
    failure = validate_master_weights_payload(
        bad, netuid=100, weights_freshness_seconds=720, now=REF
    )
    assert failure is not None and "non-finite" in failure


async def test_unusable_vector_never_builds_wallet(tmp_path: Path) -> None:
    client = _FetchClient(_fresh_payload(netuid=999))

    def exploding_factory() -> Any:
        raise AssertionError("unusable vector must not construct WeightSetter")

    submitter = _submitter(
        client=client,
        factory=exploding_factory,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
    )
    assert await submitter.run_once() is ValidatorSubmitOutcome.NO_VECTOR


async def test_identity_mismatch_rejects_before_set_weights(tmp_path: Path) -> None:
    client = _FetchClient(_fresh_payload())
    setter = _RecordingSetter("wallet-hotkey-X")
    submitter = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="registered-hotkey-Y",
    )
    outcome = await submitter.run_once()
    assert outcome is ValidatorSubmitOutcome.IDENTITY_MISMATCH
    assert setter.calls == []
    assert (
        submitter.ledger.all_records()[0].status
        == SubmissionStatus.PRE_SEND_FAILED.value
    )


async def test_durable_ledger_survives_restart(tmp_path: Path) -> None:
    payload = _fresh_payload(vector_id="persist-me", vector_digest="d" * 64)
    client = _FetchClient(payload)
    setter = _RecordingSetter("hotkey-A")
    first = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
    )
    assert await first.run_once() is ValidatorSubmitOutcome.SUBMITTED
    assert len(setter.calls) == 1

    relaunched = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
    )
    assert await relaunched.run_once() is ValidatorSubmitOutcome.ALREADY_SUBMITTED
    assert len(setter.calls) == 1
    assert Path(tmp_path / "submission_ledger.json").exists()


async def test_new_vector_supersedes_active_retry(tmp_path: Path) -> None:
    old = _fresh_payload(vector_id="old", vector_digest="e" * 64, weights=[0.5, 0.5])
    client = _FetchClient(old)
    setter = _RecordingSetter(
        "hotkey-A",
        raises=RuntimeError("subtensor rejected weight submission: TooFast"),
    )
    submitter = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
        backoff_base_seconds=0.0,
    )
    assert await submitter.run_once() is ValidatorSubmitOutcome.REJECTED
    assert submitter.ledger.all_records()[0].status == SubmissionStatus.REJECTED.value

    setter._raises = None
    client.set_payload(
        _fresh_payload(
            vector_id="new",
            vector_digest="f" * 64,
            weights=[0.2, 0.8],
            computed_offset_seconds=5,
        )
    )
    assert await submitter.run_once() is ValidatorSubmitOutcome.SUBMITTED
    records = {r.vector_id: r for r in submitter.ledger.all_records()}
    assert records["old"].status == SubmissionStatus.SUPERSEDED.value
    assert records["new"].status == SubmissionStatus.ACCEPTED.value
    assert setter.calls[-1] == ([0, 1], [0.2, 0.8])


async def test_retry_exhausted_after_max_attempts(tmp_path: Path) -> None:
    client = _FetchClient(_fresh_payload())
    setter = _RecordingSetter(
        "hotkey-A", raises=RuntimeError("rejected permanently for test")
    )
    submitter = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
        max_attempts=2,
        backoff_base_seconds=0.0,
    )
    assert await submitter.run_once() is ValidatorSubmitOutcome.REJECTED
    assert await submitter.run_once() is ValidatorSubmitOutcome.REJECTED
    assert await submitter.run_once() is ValidatorSubmitOutcome.RETRY_EXHAUSTED
    assert (
        submitter.ledger.all_records()[0].status
        == SubmissionStatus.RETRY_EXHAUSTED.value
    )


async def test_observation_reporter_called_on_success(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    async def reporter(payload: dict[str, Any]) -> None:
        seen.append(payload)

    client = _FetchClient(_fresh_payload())
    setter = _RecordingSetter("hotkey-A")
    submitter = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="hotkey-A",
        observation_reporter=reporter,
    )
    assert await submitter.run_once() is ValidatorSubmitOutcome.SUBMITTED
    assert seen
    assert seen[0]["outcome"] == "accepted"
    assert seen[0]["vector_id"] == "vector-1"
    assert submitter.ledger.all_records()[0].observed_to_master is True


async def test_logs_do_not_include_wallet_secrets(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    client = _FetchClient(_fresh_payload())
    setter = _RecordingSetter("public-hotkey-only")
    submitter = _submitter(
        client=client,
        setter=setter,
        state_dir=tmp_path,
        expected_hotkey="public-hotkey-only",
    )
    with caplog.at_level("INFO"):
        await submitter.run_once()
    joined = "\n".join(record.getMessage() for record in caplog.records)
    assert "public-hotkey-only" in joined
    assert "private" not in joined.lower()
    assert "mnemonic" not in joined.lower()
    assert "seed" not in joined.lower()


def test_validator_submit_path_has_no_aggregation_dependency() -> None:
    import base.validator.weight_submitter as mod

    tree = ast.parse(inspect.getsource(mod))
    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)

    assert not any(m.startswith("base.master") for m in imported_modules), (
        imported_modules
    )
    assert "base.validator.weights_client" in imported_modules


def test_ledger_atomic_and_keyed(tmp_path: Path) -> None:
    ledger = ValidatorSubmissionLedger(tmp_path / "ledger.json")
    from base.validator.submission_ledger import SubmissionRecord

    rec = SubmissionRecord(
        validator_hotkey="hk",
        vector_id="v",
        vector_digest="d",
        netuid=100,
        chain_endpoint="ep",
        uids=[0],
        weights=[1.0],
    )
    ledger.upsert(rec)
    reloaded = ValidatorSubmissionLedger(tmp_path / "ledger.json")
    got = reloaded.get(
        validator_hotkey="hk",
        vector_id="v",
        vector_digest="d",
        netuid=100,
        chain_endpoint="ep",
    )
    assert got is not None
    assert got.status == SubmissionStatus.PENDING.value
