-- PRISM submissions + per-stage event trail (orchestrator state machine).
--
-- `prism_submission` is the single source of truth behind the public status
-- API (`GET /v1/submissions/{id}`). It is mutable on purpose: the same row
-- carries the submission through its lifecycle
--   queued -> provisioning -> running -> llm_review -> similarity
--   -> scoring -> terminated | failed
-- so the API reads one row per submission instead of replaying a journal.
-- The journal lives in `prism_stage_event` (append-only): every stage entry,
-- exit, retry and verdict lands there with a timestamp, which is what the
-- timeline view renders and what makes an incident replayable months later.
--
-- Sources are stored inline (128 KiB cap each, enforced in Rust before
-- insert): similarity review needs the full corpus of historical code, and
-- replaying a queued job after a restart needs the same bytes the miner sent.
--
-- Score columns mirror `raw_weight_snapshot` shape (`kind` + `score` +
-- `absence_reason`) so the leaf emitter is a pure projection of the row and
-- the API shows exactly what the chain will remember.
--
-- `miner_hotkey` follows the same lowercase 64-hex CHECK as
-- `attestation.miner_hotkey` so rows join byte-for-byte.

CREATE TABLE prism_submission (
    id              TEXT PRIMARY KEY,            -- sha256 hex of the contract bytes
    miner_hotkey    TEXT NOT NULL,
    epoch           BIGINT NOT NULL,             -- chain epoch at acceptance
    netuid          INTEGER NOT NULL,
    status          TEXT NOT NULL,               -- queued|provisioning|running|llm_review|similarity|scoring|terminated|failed
    label           TEXT,
    architecture_py TEXT NOT NULL,
    training_py     TEXT NOT NULL,
    pod_id          TEXT,
    pod_provider    TEXT,
    image_digest    TEXT,
    receipt_json    JSONB,
    metrics_json    JSONB,
    bpb             DOUBLE PRECISION,
    review_json     JSONB,
    similarity_json JSONB,
    kind            TEXT,                        -- score|no_score (set at scoring)
    score           BIGINT,
    absence_reason  SMALLINT,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    error_detail    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prism_submission_status_check CHECK (status IN (
        'queued','provisioning','running','llm_review','similarity','scoring','terminated','failed')),
    CONSTRAINT prism_submission_kind_check CHECK (kind IS NULL OR kind IN ('score','no_score')),
    CONSTRAINT prism_submission_epoch_nonneg CHECK (epoch >= 0),
    CONSTRAINT prism_submission_netuid_range CHECK (netuid >= 0 AND netuid <= 65535),
    CONSTRAINT prism_submission_miner_hotkey_hex CHECK (miner_hotkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT prism_submission_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 1000000)),
    CONSTRAINT prism_submission_absence_range CHECK (absence_reason IS NULL OR (absence_reason >= 0 AND absence_reason <= 7)),
    CONSTRAINT prism_submission_source_len
        CHECK (octet_length(architecture_py) <= 131072 AND octet_length(training_py) <= 131072)
);

CREATE TABLE prism_stage_event (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id   TEXT NOT NULL REFERENCES prism_submission(id) ON DELETE CASCADE,
    stage           TEXT NOT NULL,
    detail          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prism_stage_event_stage_len CHECK (char_length(stage) <= 64)
);

CREATE INDEX ix_prism_submission_status ON prism_submission (status);
CREATE INDEX ix_prism_submission_miner ON prism_submission (miner_hotkey, epoch);
CREATE INDEX ix_prism_submission_created ON prism_submission (created_at DESC);
CREATE INDEX ix_prism_stage_event_subm ON prism_stage_event (submission_id, created_at);

-- Mutable state table vs append-only journal: the app role can update the
-- state row but never the journal.
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE prism_submission TO base_app;
GRANT SELECT, INSERT ON TABLE prism_stage_event TO base_app;
