"""Architecture-lab serving layer: repository read methods + ``/v1`` routes + LLM auto-report.

Seeds the lab tables directly (the producer-side writers are covered in
``test_prism_lab_producer.py``) and asserts the read aggregation/ordering/404s, the curve
downsampling + compute derivation. Architecture LLM auto-report routes were removed with the
gateway; residual report generator tests stay skip-only below.
"""

from __future__ import annotations

import json
from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient

from prism_challenge.db import Database
from prism_challenge.repository import PrismRepository

NOW = "2026-06-30T12:00:00+00:00"
EARLIER = "2026-06-20T08:00:00+00:00"
EPOCH_SECONDS = 3600


# --------------------------------------------------------------------------------------------------
# Seeding helpers (work against any aiosqlite connection from ``database.connect()``).
# --------------------------------------------------------------------------------------------------
async def _insert_family(
    conn,
    *,
    architecture_id: str,
    family_hash: str,
    owner_hotkey: str,
    canonical_submission_id: str,
    q_arch_best: float,
    display_name: str | None = "Arch",
    owner_submission_id: str | None = None,
    created_at: str = EARLIER,
    updated_at: str = NOW,
) -> None:
    await conn.execute(
        "INSERT INTO architecture_families("
        "id, family_hash, arch_fingerprint, behavior_fingerprint, owner_hotkey, "
        "owner_submission_id, canonical_submission_id, q_arch_best, display_name, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            architecture_id,
            family_hash,
            f"{family_hash}-af",
            f"{family_hash}-bf",
            owner_hotkey,
            owner_submission_id or canonical_submission_id,
            canonical_submission_id,
            q_arch_best,
            display_name,
            created_at,
            updated_at,
        ),
    )


async def _insert_submission(
    conn,
    *,
    submission_id: str,
    hotkey: str,
    epoch_id: int,
    arch_hash: str | None,
    name: str | None = None,
    status: str = "completed",
    created_at: str = NOW,
) -> None:
    await conn.execute(
        "INSERT INTO submissions("
        "id, hotkey, epoch_id, filename, code, code_hash, arch_hash, name, metadata, status, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            submission_id,
            hotkey,
            epoch_id,
            "project.zip",
            "code",
            f"hash-{submission_id}",
            arch_hash,
            name,
            "{}",
            status,
            created_at,
            created_at,
        ),
    )


async def _insert_score(conn, *, submission_id: str, final_score: float, metrics: dict) -> None:
    await conn.execute(
        "INSERT INTO scores("
        "submission_id, q_arch, q_recipe, anti_cheat_multiplier, diversity_bonus, penalty, "
        "final_score, metrics, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (submission_id, final_score, 0.0, 1.0, 0.0, 0.0, final_score, json.dumps(metrics), NOW),
    )


async def _insert_variant(
    conn,
    *,
    variant_id: str,
    architecture_id: str,
    training_hash: str,
    owner_hotkey: str,
    submission_id: str,
    q_recipe: float,
    is_current_best: bool,
    created_at: str = NOW,
) -> None:
    await conn.execute(
        "INSERT INTO training_variants("
        "id, architecture_id, training_hash, owner_hotkey, submission_id, q_recipe, "
        "metric_mean, metric_std, is_current_best, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            variant_id,
            architecture_id,
            training_hash,
            owner_hotkey,
            submission_id,
            q_recipe,
            q_recipe,
            0.0,
            int(is_current_best),
            created_at,
            created_at,
        ),
    )


async def _insert_curve(
    conn,
    *,
    submission_id: str,
    online_loss: list[float],
    covered_bytes_cumulative: list[float],
    step0_loss: float | None,
    baseline_nats: float | None,
    compute: dict,
    train_series: dict | None = None,
) -> None:
    await conn.execute(
        "INSERT INTO submission_curves("
        "submission_id, online_loss, covered_bytes_cumulative, step0_loss, baseline_nats, "
        "compute, train_series, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            submission_id,
            json.dumps(online_loss),
            json.dumps(covered_bytes_cumulative),
            step0_loss,
            baseline_nats,
            json.dumps(compute),
            json.dumps(train_series) if train_series is not None else None,
            NOW,
        ),
    )


async def _make_repo(tmp_path: Path) -> PrismRepository:
    database = Database(tmp_path / "lab.sqlite3")
    await database.init()
    return PrismRepository(database, EPOCH_SECONDS)


# --------------------------------------------------------------------------------------------------
# Repository method tests.
# --------------------------------------------------------------------------------------------------
async def test_list_architectures_aggregates_and_ranks(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        # Family A: 2 submissions, 2 variants, best score 0.9 (epoch 100).
        await _insert_family(
            conn,
            architecture_id="af-A",
            family_hash="hashA",
            owner_hotkey="hkA",
            canonical_submission_id="subA2",
            q_arch_best=0.9,
            display_name="Alpha",
        )
        await _insert_submission(
            conn, submission_id="subA1", hotkey="hkA", epoch_id=100, arch_hash="hashA"
        )
        await _insert_submission(
            conn, submission_id="subA2", hotkey="hkA2", epoch_id=100, arch_hash="hashA"
        )
        await _insert_variant(
            conn,
            variant_id="tvA1",
            architecture_id="af-A",
            training_hash="tA1",
            owner_hotkey="hkA",
            submission_id="subA1",
            q_recipe=0.5,
            is_current_best=False,
        )
        await _insert_variant(
            conn,
            variant_id="tvA2",
            architecture_id="af-A",
            training_hash="tA2",
            owner_hotkey="hkA2",
            submission_id="subA2",
            q_recipe=0.9,
            is_current_best=True,
        )
        # Family B: 1 submission, 1 variant, best score 0.5 (epoch 100).
        await _insert_family(
            conn,
            architecture_id="af-B",
            family_hash="hashB",
            owner_hotkey="hkB",
            canonical_submission_id="subB1",
            q_arch_best=0.5,
            display_name=None,
        )
        await _insert_submission(
            conn, submission_id="subB1", hotkey="hkB", epoch_id=100, arch_hash="hashB"
        )
        await _insert_variant(
            conn,
            variant_id="tvB1",
            architecture_id="af-B",
            training_hash="tB1",
            owner_hotkey="hkB",
            submission_id="subB1",
            q_recipe=0.5,
            is_current_best=True,
        )

    resolved_epoch, rows = await repo.list_architectures(100)
    assert resolved_epoch == 100
    assert [r["architecture_id"] for r in rows] == ["af-A", "af-B"]
    top = rows[0]
    assert top["arch_hash"] == "hashA"
    assert top["name"] == "Alpha"
    assert top["best_final_score"] == 0.9
    assert top["best_submission_id"] == "subA2"
    assert top["variant_count"] == 2
    assert top["submission_count"] == 2
    assert rows[1]["name"] is None
    assert rows[1]["variant_count"] == 1
    assert rows[1]["submission_count"] == 1


async def test_list_architectures_epoch_filter_and_none_fallback(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        await _insert_family(
            conn,
            architecture_id="af-A",
            family_hash="hashA",
            owner_hotkey="hkA",
            canonical_submission_id="subA",
            q_arch_best=0.9,
        )
        await _insert_submission(
            conn, submission_id="subA", hotkey="hkA", epoch_id=100, arch_hash="hashA"
        )
        # Family C only has a submission in the LATER epoch 105.
        await _insert_family(
            conn,
            architecture_id="af-C",
            family_hash="hashC",
            owner_hotkey="hkC",
            canonical_submission_id="subC",
            q_arch_best=0.7,
        )
        await _insert_submission(
            conn, submission_id="subC", hotkey="hkC", epoch_id=105, arch_hash="hashC"
        )

    # Explicit epoch scopes by submission presence.
    _, only_a = await repo.list_architectures(100)
    assert [r["architecture_id"] for r in only_a] == ["af-A"]
    _, only_c = await repo.list_architectures(105)
    assert [r["architecture_id"] for r in only_c] == ["af-C"]
    # None resolves to the most-recent non-empty epoch (105 -> family C).
    resolved, fallback_rows = await repo.list_architectures(None)
    assert resolved == 105
    assert [r["architecture_id"] for r in fallback_rows] == ["af-C"]


async def test_get_architecture_detail_and_missing(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        await _insert_family(
            conn,
            architecture_id="af-A",
            family_hash="hashA",
            owner_hotkey="hkA",
            canonical_submission_id="subA",
            q_arch_best=0.42,
            display_name="Alpha",
            created_at=EARLIER,
            updated_at=NOW,
        )
        await _insert_submission(
            conn, submission_id="subA", hotkey="hkA", epoch_id=100, arch_hash="hashA"
        )
        await _insert_variant(
            conn,
            variant_id="tvA",
            architecture_id="af-A",
            training_hash="tA",
            owner_hotkey="hkA",
            submission_id="subA",
            q_recipe=0.42,
            is_current_best=True,
        )

    detail = await repo.get_architecture("af-A")
    assert detail is not None
    assert detail["arch_hash"] == "hashA"
    assert detail["name"] == "Alpha"
    assert detail["best_submission_id"] == "subA"
    assert detail["best_final_score"] == 0.42
    assert detail["variant_count"] == 1
    assert detail["submission_count"] == 1
    assert detail["first_seen_at"] == EARLIER
    assert detail["updated_at"] == NOW
    assert await repo.get_architecture("missing") is None


async def test_list_training_variants_orders_best_first(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        await _insert_family(
            conn,
            architecture_id="af-A",
            family_hash="hashA",
            owner_hotkey="hkA",
            canonical_submission_id="sub3",
            q_arch_best=0.9,
        )
        await _insert_variant(
            conn,
            variant_id="tv1",
            architecture_id="af-A",
            training_hash="t1",
            owner_hotkey="hk1",
            submission_id="sub1",
            q_recipe=0.5,
            is_current_best=False,
        )
        await _insert_variant(
            conn,
            variant_id="tv2",
            architecture_id="af-A",
            training_hash="t2",
            owner_hotkey="hk2",
            submission_id="sub2",
            q_recipe=0.2,
            is_current_best=False,
        )
        await _insert_variant(
            conn,
            variant_id="tv3",
            architecture_id="af-A",
            training_hash="t3",
            owner_hotkey="hk3",
            submission_id="sub3",
            q_recipe=0.9,
            is_current_best=True,
        )

    variants = await repo.list_training_variants("af-A")
    assert [v["variant_id"] for v in variants] == ["tv3", "tv1", "tv2"]
    assert variants[0]["is_current_best"] == 1
    assert variants[0]["final_score"] == 0.9
    assert variants[0]["owner_hotkey"] == "hk3"
    assert await repo.list_training_variants("missing") == []


async def test_get_submission_curve_and_missing(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    async with repo.database.connect() as conn:
        await _insert_submission(
            conn, submission_id="sub1", hotkey="hk", epoch_id=100, arch_hash="hashA"
        )
        await _insert_score(
            conn,
            submission_id="sub1",
            final_score=0.8,
            metrics={"prequential_bpb": 0.95, "bits_per_byte": 0.95, "tokens_consumed": 5000.0},
        )
        await _insert_curve(
            conn,
            submission_id="sub1",
            online_loss=[2.5, 2.0, 1.5],
            covered_bytes_cumulative=[100.0, 200.0, 300.0],
            step0_loss=2.5,
            baseline_nats=5.6,
            compute={"gpu_count": 1, "model_params": 1000},
        )

    curve = await repo.get_submission_curve("sub1")
    assert curve is not None
    assert curve["online_loss"] == [2.5, 2.0, 1.5]
    assert curve["covered_bytes_cumulative"] == [100.0, 200.0, 300.0]
    assert curve["step0_loss"] == 2.5
    assert curve["baseline_nats"] == 5.6
    assert curve["compute"] == {"gpu_count": 1, "model_params": 1000}
    assert curve["prequential_bpb"] == 0.95
    assert curve["bits_per_byte"] == 0.95
    assert curve["tokens_consumed"] == 5000.0
    assert await repo.get_submission_curve("missing") is None


async def test_architecture_report_cache_round_trip(tmp_path: Path) -> None:
    repo = await _make_repo(tmp_path)
    assert await repo.get_architecture_report("af-A") is None
    await repo.store_architecture_report(
        architecture_id="af-A",
        content="## Summary",
        model="test-model",
        source_submission_id="subA",
        generated_at=NOW,
    )
    cached = await repo.get_architecture_report("af-A")
    assert cached is not None
    assert cached["content"] == "## Summary"
    assert cached["model"] == "test-model"
    assert cached["source_submission_id"] == "subA"
    assert cached["generated_at"] == NOW


# --------------------------------------------------------------------------------------------------
# Route tests (use the shared ``client`` fixture; seed via the repository connection).
# --------------------------------------------------------------------------------------------------
def _seed(client: TestClient, coro_factory) -> None:
    repository = client.app.state.repository

    async def run() -> None:
        async with repository.database.connect() as conn:
            await coro_factory(conn)

    anyio.run(run)


def test_route_list_architectures(client: TestClient) -> None:
    async def seed(conn):
        await _insert_family(
            conn,
            architecture_id="af-A",
            family_hash="hashA",
            owner_hotkey="hkA",
            canonical_submission_id="subA",
            q_arch_best=0.9,
            display_name="Alpha",
        )
        await _insert_submission(
            conn, submission_id="subA", hotkey="hkA", epoch_id=7, arch_hash="hashA"
        )
        await _insert_variant(
            conn,
            variant_id="tvA",
            architecture_id="af-A",
            training_hash="tA",
            owner_hotkey="hkA",
            submission_id="subA",
            q_recipe=0.9,
            is_current_best=True,
        )
        await _insert_family(
            conn,
            architecture_id="af-B",
            family_hash="hashB",
            owner_hotkey="hkB",
            canonical_submission_id="subB",
            q_arch_best=0.4,
            display_name=None,
        )
        await _insert_submission(
            conn, submission_id="subB", hotkey="hkB", epoch_id=7, arch_hash="hashB"
        )

    _seed(client, seed)
    body = client.get("/v1/architectures", params={"epoch_id": 7}).json()
    assert body["epoch_id"] == 7
    assert [a["rank"] for a in body["architectures"]] == [1, 2]
    first = body["architectures"][0]
    assert first["architecture_id"] == "af-A"
    assert first["arch_hash"] == "hashA"
    assert first["name"] == "Alpha"
    assert first["best_final_score"] == 0.9
    assert first["variant_count"] == 1
    assert first["submission_count"] == 1
    # Nullable name is returned as null, never omitted.
    assert body["architectures"][1]["name"] is None
    assert "updated_at" in first


def test_route_get_architecture_and_404(client: TestClient) -> None:
    async def seed(conn):
        await _insert_family(
            conn,
            architecture_id="af-A",
            family_hash="hashA",
            owner_hotkey="hkA",
            canonical_submission_id="subA",
            q_arch_best=0.9,
            display_name="Alpha",
        )
        await _insert_submission(
            conn, submission_id="subA", hotkey="hkA", epoch_id=7, arch_hash="hashA"
        )

    _seed(client, seed)
    ok = client.get("/v1/architectures/af-A")
    assert ok.status_code == 200, ok.text
    detail = ok.json()
    assert detail["architecture_id"] == "af-A"
    assert detail["best_submission_id"] == "subA"
    assert "first_seen_at" in detail and "updated_at" in detail
    assert client.get("/v1/architectures/missing").status_code == 404


def test_route_variants_empty_ok_and_404(client: TestClient) -> None:
    async def seed(conn):
        await _insert_family(
            conn,
            architecture_id="af-A",
            family_hash="hashA",
            owner_hotkey="hkA",
            canonical_submission_id="subA",
            q_arch_best=0.9,
        )

    _seed(client, seed)
    ok = client.get("/v1/architectures/af-A/variants")
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body == {"architecture_id": "af-A", "variants": []}
    missing = client.get("/v1/architectures/missing/variants")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "architecture not found"


def test_route_curve_downsamples_and_preserves_endpoints(client: TestClient) -> None:
    loss = [float(i) for i in range(1200)]
    cumulative = [float(i * 10) for i in range(1200)]

    async def seed(conn):
        await _insert_score(
            conn,
            submission_id="sub1",
            final_score=0.8,
            metrics={"prequential_bpb": 0.9, "bits_per_byte": 0.9, "tokens_consumed": 500.0},
        )
        await _insert_curve(
            conn,
            submission_id="sub1",
            online_loss=loss,
            covered_bytes_cumulative=cumulative,
            step0_loss=0.0,
            baseline_nats=5.0,
            compute={
                "gpu_count": 2,
                "device": "cuda:0",
                "model_params": 1000,
                "wall_clock_seconds": 720.0,
                "peak_vram_bytes": 123,
            },
        )

    _seed(client, seed)
    body = client.get("/v1/submissions/sub1/curve").json()
    series = body["loss_curve"]
    assert series["downsampled"] is True
    assert series["points"] == 500
    assert len(series["online_loss"]) == 500
    assert len(series["covered_bytes_cumulative"]) == 500
    # First and last samples are preserved for both axes.
    assert series["online_loss"][0] == 0.0
    assert series["online_loss"][-1] == 1199.0
    assert series["covered_bytes_cumulative"][0] == 0.0
    assert series["covered_bytes_cumulative"][-1] == 11990.0
    assert body["bpb"] == {"prequential_bpb": 0.9, "bits_per_byte": 0.9}
    compute = body["compute"]
    # estimated_flops = 6 * model_params * tokens_consumed; gpu_hours = gpu_count * wall / 3600.
    assert compute["estimated_flops"] == 6.0 * 1000 * 500
    assert compute["gpu_hours"] == 2 * 720.0 / 3600.0
    assert compute["tokens_consumed"] == 500
    assert compute["device"] == "cuda:0"
    assert compute["peak_vram_bytes"] == 123
    # Fields absent from the stored profile are returned as null, never omitted.
    assert compute["gpu_tier"] is None
    assert compute["peak_rss_bytes"] is None


def test_route_curve_small_series_not_downsampled(client: TestClient) -> None:
    async def seed(conn):
        await _insert_score(conn, submission_id="sub1", final_score=0.8, metrics={})
        await _insert_curve(
            conn,
            submission_id="sub1",
            online_loss=[2.5, 2.0, 1.5],
            covered_bytes_cumulative=[10.0, 20.0, 30.0],
            step0_loss=2.5,
            baseline_nats=None,
            compute={},
        )

    _seed(client, seed)
    body = client.get("/v1/submissions/sub1/curve").json()
    series = body["loss_curve"]
    assert series["downsampled"] is False
    assert series["points"] == 3
    assert series["online_loss"] == [2.5, 2.0, 1.5]
    # No inputs -> derived compute scalars are null.
    assert body["compute"]["estimated_flops"] is None
    assert body["compute"]["gpu_hours"] is None
    assert body["compute"]["tokens_consumed"] is None


def test_route_curve_uses_stored_estimates_when_present(client: TestClient) -> None:
    async def seed(conn):
        await _insert_score(
            conn, submission_id="sub1", final_score=0.8, metrics={"tokens_consumed": 500.0}
        )
        await _insert_curve(
            conn,
            submission_id="sub1",
            online_loss=[1.0],
            covered_bytes_cumulative=[1.0],
            step0_loss=1.0,
            baseline_nats=1.0,
            compute={
                "gpu_count": 4,
                "model_params": 1000,
                "wall_clock_seconds": 36.0,
                "estimated_flops": 999.0,
                "gpu_hours": 1.5,
            },
        )

    _seed(client, seed)
    compute = client.get("/v1/submissions/sub1/curve").json()["compute"]
    # Pre-computed values in the stored profile are used as-is (not recomputed).
    assert compute["estimated_flops"] == 999.0
    assert compute["gpu_hours"] == 1.5


def test_route_curve_404_when_absent(client: TestClient) -> None:
    assert client.get("/v1/submissions/nope/curve").status_code == 404


def test_route_curve_null_train_series_when_absent(client: TestClient) -> None:
    """Legacy curve rows without train_series still return loss_curve; train_series is null."""

    async def seed(conn):
        await _insert_score(conn, submission_id="sub1", final_score=0.8, metrics={})
        await _insert_curve(
            conn,
            submission_id="sub1",
            online_loss=[2.0, 1.5],
            covered_bytes_cumulative=[10.0, 20.0],
            step0_loss=2.0,
            baseline_nats=None,
            compute={},
            train_series=None,
        )

    _seed(client, seed)
    body = client.get("/v1/submissions/sub1/curve").json()
    assert body["train_series"] is None
    assert body["loss_curve"]["points"] == 2


def _challenge_series(*, n: int = 3, extra_secret: bool = False) -> dict:
    points = []
    for i in range(n):
        p: dict = {
            "i": i,
            "tokens_seen": (i + 1) * 10,
            "covered_bytes": float((i + 1) * 40),
            "train_ce_nats": 2.5 - i * 0.1,
            "running_bpb": 3.6 - i * 0.1,
            "wall_s": float(i) * 0.5,
            "grad_norm": 1.0 + i * 0.1,
            "clip_event": i % 2 == 0,
            "nan_inf": False,
        }
        if extra_secret:
            p["api_token"] = "secret-token-must-not-leak"
            p["wallet_mnemonic"] = "abandon abandon abandon"
        points.append(p)
    series: dict = {
        "schema": "prism_train_series.v1",
        "submission_id": "sub1",
        "run_id": "run1",
        "authority": "challenge",
        "x_axis": "batch_index",
        "token_budget": 1000,
        "points": points,
        "aggregates": {
            "n_points": n,
            "mean_step_ms": 120.0,
            "p99_step_ms": 200.0,
            "grad_spike_rate": 0.0,
            "nan_inf_batches": 0,
            "clip_events": sum(1 for p in points if p["clip_event"]),
        },
        "miner_reported_ignored": True,
    }
    if extra_secret:
        series["internal_proof_secret"] = "do-not-leak"
        series["provider_api_key"] = "sk-test-should-not-appear"
    return series


def test_route_curve_exposes_train_series_grad_and_clip(client: TestClient) -> None:
    """VAL-TELE-007: GET /curve returns prism_train_series.v1 with grad_norm + clip for charts."""

    async def seed(conn):
        await _insert_score(
            conn,
            submission_id="sub1",
            final_score=0.8,
            metrics={"prequential_bpb": 0.91, "bits_per_byte": 0.91},
        )
        await _insert_curve(
            conn,
            submission_id="sub1",
            online_loss=[2.5, 2.4, 2.3],
            covered_bytes_cumulative=[40.0, 80.0, 120.0],
            step0_loss=2.5,
            baseline_nats=5.0,
            compute={"gpu_count": 1},
            train_series=_challenge_series(n=3),
        )

    _seed(client, seed)
    body = client.get("/v1/submissions/sub1/curve").json()
    ts = body["train_series"]
    assert ts is not None
    assert ts["schema"] == "prism_train_series.v1"
    assert ts["authority"] == "challenge"
    assert ts["miner_reported_ignored"] is True
    assert ts["downsampled"] is False
    assert ts["points_total"] == 3
    assert len(ts["points"]) == 3
    assert ts["points"][0]["train_ce_nats"] == 2.5
    assert ts["points"][0]["running_bpb"] == 3.6
    assert ts["points"][0]["grad_norm"] == 1.0
    assert ts["points"][0]["clip_event"] is True
    assert ts["points"][0]["tokens_seen"] == 10
    assert ts["points"][0]["wall_s"] == 0.0
    assert ts["aggregates"]["clip_events"] == 2
    # Time-flow axes present for UI: loss/bpb + grad_norm vs tokens/wall.
    for p in ts["points"]:
        assert "tokens_seen" in p
        assert "wall_s" in p
        assert "grad_norm" in p
        assert "train_ce_nats" in p or "running_bpb" in p


def test_route_curve_downsamples_train_series_preserve_ends(client: TestClient) -> None:
    """Downsample-safe series: cap → 500 points, first/last preserved (chart-ready)."""

    async def seed(conn):
        await _insert_score(conn, submission_id="sub1", final_score=0.8, metrics={})
        await _insert_curve(
            conn,
            submission_id="sub1",
            online_loss=[1.0],
            covered_bytes_cumulative=[1.0],
            step0_loss=1.0,
            baseline_nats=1.0,
            compute={},
            train_series=_challenge_series(n=1200),
        )

    _seed(client, seed)
    ts = client.get("/v1/submissions/sub1/curve").json()["train_series"]
    assert ts is not None
    assert ts["downsampled"] is True
    assert ts["points_total"] == 1200
    assert len(ts["points"]) == 500
    assert ts["points"][0]["i"] == 0
    assert ts["points"][-1]["i"] == 1199
    assert ts["points"][0]["grad_norm"] == 1.0
    assert ts["points"][-1]["grad_norm"] == pytest.approx(1.0 + 1199 * 0.1)


def test_route_curve_ignores_miner_authority_series(client: TestClient) -> None:
    """Non-challenge authority never surfaces on the public curve (VAL-TELE-006 + API)."""

    miner_series = _challenge_series(n=2)
    miner_series["authority"] = "miner"
    miner_series["miner_reported_ignored"] = False

    async def seed(conn):
        await _insert_score(conn, submission_id="sub1", final_score=0.8, metrics={})
        await _insert_curve(
            conn,
            submission_id="sub1",
            online_loss=[1.0],
            covered_bytes_cumulative=[1.0],
            step0_loss=1.0,
            baseline_nats=1.0,
            compute={},
            train_series=miner_series,
        )

    _seed(client, seed)
    body = client.get("/v1/submissions/sub1/curve").json()
    assert body["train_series"] is None


def test_route_curve_train_series_does_not_leak_secrets(client: TestClient) -> None:
    """VAL-TELE-007 non-leak: unknown / secret-looking keys are stripped from the response."""

    async def seed(conn):
        await _insert_score(conn, submission_id="sub1", final_score=0.8, metrics={})
        await _insert_curve(
            conn,
            submission_id="sub1",
            online_loss=[1.0, 0.9],
            covered_bytes_cumulative=[10.0, 20.0],
            step0_loss=1.0,
            baseline_nats=1.0,
            compute={},
            train_series=_challenge_series(n=2, extra_secret=True),
        )

    _seed(client, seed)
    raw = client.get("/v1/submissions/sub1/curve").text
    body = json.loads(raw)
    ts = body["train_series"]
    assert ts is not None
    # Top-level and point secrets must not appear in the wire body.
    assert "secret-token-must-not-leak" not in raw
    assert "abandon abandon abandon" not in raw
    assert "sk-test-should-not-appear" not in raw
    assert "do-not-leak" not in raw
    assert "api_token" not in raw
    assert "internal_proof_secret" not in raw
    assert "provider_api_key" not in raw
    assert "wallet_mnemonic" not in raw
    for p in ts["points"]:
        assert set(p.keys()) <= {
            "i",
            "tokens_seen",
            "covered_bytes",
            "train_ce_nats",
            "running_bpb",
            "wall_s",
            "grad_norm",
            "clip_event",
            "param_norm",
            "lr",
            "nan_inf",
        }


def test_architecture_report_routes_removed(client: TestClient) -> None:
    """Architecture auto-report / LLM gateway report paths were removed.

    Listing and detail routes remain; the legacy report surface is gone (404).
    Residual generator module tests live as skipped stubs below.
    """

    async def seed(conn):
        await _insert_family(
            conn,
            architecture_id="af-A",
            family_hash="hashA",
            owner_hotkey="hkA",
            canonical_submission_id="subA",
            q_arch_best=0.9,
        )

    _seed(client, seed)
    response = client.get("/v1/architectures/af-A/report")
    assert response.status_code == 404


pytestmark_report = pytest.mark.skip(
    reason=(
        "Architecture LLM report generation removed with gateway; "
        "see test_architecture_report_routes_removed"
    )
)


@pytestmark_report
def test_route_report_cached_ready() -> None:
    return None


@pytestmark_report
def test_route_report_stale_cache_is_not_served() -> None:
    return None


@pytestmark_report
def test_route_report_generates_in_background() -> None:
    return None


@pytestmark_report
def test_route_report_generation_error_then_unavailable() -> None:
    return None


@pytestmark_report
def test_report_generation_available_reflects_credentials() -> None:
    return None


@pytestmark_report
def test_build_report_prompt_grounds_only_in_facts() -> None:
    return None


@pytestmark_report
def test_generate_report_content_uses_resolved_client() -> None:
    return None


@pytestmark_report
def test_generate_report_content_rejects_empty_completion() -> None:
    return None
