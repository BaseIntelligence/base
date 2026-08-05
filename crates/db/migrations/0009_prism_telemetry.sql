-- PRISM per-step training telemetry (recipe >= 1.1.0).
--
-- Miner `training.py` reports through the harness-provided `prism_telemetry`
-- shim (`report(loss=, step=, grad_norm=, layer_stats=)`); the harness ships
-- the captured series in `METRICS_JSON.telemetry.loss_series` and the master
-- persists one row per report here. `prism_submission.metrics_json` keeps the
-- whole metrics blob (authoritative row-level copy); this table is the
-- granular, queryable series behind the site telemetry/loss-curve surfaces.
-- Rows are replaced wholesale when a submission is re-scored and cascade-deleted
-- with the submission, mirroring the stage-event journal semantics.

CREATE TABLE prism_telemetry (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id    TEXT NOT NULL REFERENCES prism_submission(id) ON DELETE CASCADE,
    step             BIGINT NOT NULL,             -- miner-reported optimizer step
    loss             DOUBLE PRECISION NOT NULL,   -- miner-reported train loss
    grad_norm        DOUBLE PRECISION,            -- global gradient norm when reported
    layer_stats      JSONB,                       -- per-layer gradient/activation stats (bounded in-pod)
    reported_at_secs DOUBLE PRECISION,            -- seconds since train start (pod clock)
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prism_telemetry_step_nonneg CHECK (step >= 0),
    -- Postgres treats float8 NaN as equal to NaN, so NOT IN rejects NaN/±Inf.
    CONSTRAINT prism_telemetry_loss_finite
        CHECK (loss NOT IN ('NaN'::float8, 'Infinity'::float8, '-Infinity'::float8))
);

CREATE INDEX ix_prism_telemetry_subm ON prism_telemetry (submission_id, step);

-- Append-mostly analytics table: the app role may insert and replace rows
-- (retry path deletes the submission's series before re-inserting).
GRANT SELECT, INSERT, DELETE ON TABLE prism_telemetry TO base_app;
