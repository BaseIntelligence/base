-- Bounty challenge: video bug reports, stage journal, epoch scores.
--
-- `bounty_bug` is the mutable state row behind miner/admin APIs. Video bytes
-- live on the `bounty-artifacts` volume (`video_path`); only sha256 + size
-- are stored here. `bounty_stage_event` is append-only (INSERT for base_app).
-- `bounty_epoch_score` holds approved points + leaf projection per epoch.

CREATE TABLE bounty_bug (
    id                  TEXT PRIMARY KEY,
    miner_hotkey        TEXT NOT NULL,
    miner_coldkey       TEXT,
    app_id              TEXT NOT NULL,
    title               TEXT NOT NULL,
    description         TEXT NOT NULL,
    steps               TEXT,
    status              TEXT NOT NULL,
    agentic_verdict     JSONB,
    nearest_id          TEXT,
    video_sha256        TEXT,
    video_bytes         BIGINT,
    video_path          TEXT,
    reject_reason       TEXT,
    epoch               BIGINT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT bounty_bug_status_check CHECK (status IN (
        'uploaded','processing','agentic_review','pending_admin',
        'approved','rejected','failed')),
    CONSTRAINT bounty_bug_epoch_nonneg CHECK (epoch >= 0),
    CONSTRAINT bounty_bug_miner_hotkey_hex CHECK (miner_hotkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT bounty_bug_miner_coldkey_hex
        CHECK (miner_coldkey IS NULL OR miner_coldkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT bounty_bug_app_id_len CHECK (char_length(app_id) <= 128),
    CONSTRAINT bounty_bug_title_len CHECK (char_length(title) <= 256),
    CONSTRAINT bounty_bug_description_len CHECK (octet_length(description) <= 65536),
    CONSTRAINT bounty_bug_steps_len
        CHECK (steps IS NULL OR octet_length(steps) <= 65536),
    CONSTRAINT bounty_bug_video_sha_hex
        CHECK (video_sha256 IS NULL OR video_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT bounty_bug_video_bytes_nonneg
        CHECK (video_bytes IS NULL OR video_bytes >= 0),
    CONSTRAINT bounty_bug_video_path_len
        CHECK (video_path IS NULL OR char_length(video_path) <= 512),
    CONSTRAINT bounty_bug_reject_reason_len
        CHECK (reject_reason IS NULL OR char_length(reject_reason) <= 500)
);

CREATE TABLE bounty_stage_event (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bug_id          TEXT NOT NULL REFERENCES bounty_bug(id) ON DELETE CASCADE,
    stage           TEXT NOT NULL,
    detail          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT bounty_stage_event_stage_len CHECK (char_length(stage) <= 64)
);

CREATE TABLE bounty_epoch_score (
    epoch               BIGINT NOT NULL,
    miner_hotkey        TEXT NOT NULL,
    points              INTEGER NOT NULL DEFAULT 0,
    kind                TEXT,
    score               BIGINT,
    absence_reason      SMALLINT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (epoch, miner_hotkey),
    CONSTRAINT bounty_epoch_score_epoch_nonneg CHECK (epoch >= 0),
    CONSTRAINT bounty_epoch_score_points_nonneg CHECK (points >= 0),
    CONSTRAINT bounty_epoch_score_miner_hotkey_hex CHECK (miner_hotkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT bounty_epoch_score_kind_check CHECK (kind IS NULL OR kind IN ('score','no_score')),
    CONSTRAINT bounty_epoch_score_score_range
        CHECK (score IS NULL OR (score >= 0 AND score <= 1000000)),
    CONSTRAINT bounty_epoch_score_absence_range
        CHECK (absence_reason IS NULL OR (absence_reason >= 0 AND absence_reason <= 7))
);

CREATE INDEX ix_bounty_bug_status ON bounty_bug (status);
CREATE INDEX ix_bounty_bug_miner_epoch ON bounty_bug (miner_hotkey, epoch);
CREATE INDEX ix_bounty_bug_created ON bounty_bug (created_at DESC);
CREATE INDEX ix_bounty_bug_corpus ON bounty_bug (created_at DESC)
    WHERE status IN ('approved','pending_admin','rejected');
CREATE INDEX ix_bounty_bug_coldkey ON bounty_bug (miner_coldkey)
    WHERE miner_coldkey IS NOT NULL;
CREATE INDEX ix_bounty_bug_claim ON bounty_bug (created_at)
    WHERE status = 'uploaded';
CREATE INDEX ix_bounty_stage_event_bug ON bounty_stage_event (bug_id, created_at);
CREATE INDEX ix_bounty_epoch_score_epoch ON bounty_epoch_score (epoch);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE bounty_bug TO base_app;
GRANT SELECT, INSERT ON TABLE bounty_stage_event TO base_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE bounty_epoch_score TO base_app;
