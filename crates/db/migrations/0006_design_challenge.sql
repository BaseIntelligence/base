-- Design challenge tables: harness submissions, 12h rounds, sandboxed runs,
-- sanitized artifacts, human pairwise annotations, Elo ratings, quotas.
--
-- Mutable state rows (harness/round/run/rating/quota) vs append-only
-- `design_stage_event` journal (INSERT-only for base_app).
-- `raw_html` is audit-only and must never be served by the viewer API.

CREATE TABLE design_harness (
    id                      TEXT PRIMARY KEY,            -- sha256 hex of hotkey+sources
    miner_hotkey            TEXT NOT NULL,
    agent_py                TEXT NOT NULL,
    pyproject_toml          TEXT NOT NULL,
    extra_files             JSONB NOT NULL DEFAULT '{}'::jsonb,
    active                  BOOLEAN NOT NULL DEFAULT TRUE,
    eliminated_until_round  BIGINT NOT NULL DEFAULT 0,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT design_harness_miner_hotkey_hex CHECK (miner_hotkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT design_harness_agent_len CHECK (octet_length(agent_py) <= 1048576),
    CONSTRAINT design_harness_pyproject_len CHECK (octet_length(pyproject_toml) <= 65536),
    CONSTRAINT design_harness_elim_nonneg CHECK (eliminated_until_round >= 0)
);

CREATE TABLE design_round (
    round_id            BIGINT PRIMARY KEY,
    epoch               BIGINT NOT NULL,
    netuid              INTEGER NOT NULL,
    opens_at            TIMESTAMPTZ NOT NULL,
    closes_at           TIMESTAMPTZ NOT NULL,
    prompt_set_digest   TEXT NOT NULL,
    status              TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT design_round_status_check CHECK (status IN (
        'open','closed','scoring','emitted')),
    CONSTRAINT design_round_epoch_nonneg CHECK (epoch >= 0),
    CONSTRAINT design_round_netuid_range CHECK (netuid >= 0 AND netuid <= 65535),
    CONSTRAINT design_round_digest_hex CHECK (prompt_set_digest ~ '^[0-9a-f]{64}$')
);

CREATE TABLE design_run (
    id                  TEXT PRIMARY KEY,
    round_id            BIGINT NOT NULL REFERENCES design_round(round_id),
    harness_id          TEXT NOT NULL REFERENCES design_harness(id),
    prompt_id           TEXT NOT NULL,
    status              TEXT NOT NULL,
    artifact_digest     TEXT,
    sanitize_report     JSONB,
    error_detail        TEXT,
    kind                TEXT,
    score               BIGINT,
    absence_reason      SMALLINT,
    retry_count         INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT design_run_status_check CHECK (status IN (
        'queued','installing','running','sanitizing','awaiting_annotation','scored','failed')),
    CONSTRAINT design_run_kind_check CHECK (kind IS NULL OR kind IN ('score','no_score')),
    CONSTRAINT design_run_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 1000000)),
    CONSTRAINT design_run_absence_range CHECK (absence_reason IS NULL OR (absence_reason >= 0 AND absence_reason <= 7))
);

CREATE TABLE design_artifact (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              TEXT NOT NULL REFERENCES design_run(id) ON DELETE CASCADE,
    path                TEXT NOT NULL,
    sanitized_html      TEXT NOT NULL,
    raw_html            TEXT NOT NULL,
    raw_sha256          TEXT NOT NULL,
    bytes               INTEGER NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT design_artifact_path_len CHECK (char_length(path) <= 256),
    CONSTRAINT design_artifact_sha_hex CHECK (raw_sha256 ~ '^[0-9a-f]{64}$'),
    CONSTRAINT design_artifact_bytes_nonneg CHECK (bytes >= 0),
    CONSTRAINT design_artifact_unique_path UNIQUE (run_id, path)
);

CREATE TABLE design_pair (
    id                  TEXT PRIMARY KEY,
    round_id            BIGINT NOT NULL REFERENCES design_round(round_id),
    prompt_id           TEXT NOT NULL,
    run_a_id            TEXT NOT NULL REFERENCES design_run(id),
    run_b_id            TEXT NOT NULL REFERENCES design_run(id),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT design_pair_distinct CHECK (run_a_id <> run_b_id)
);

CREATE TABLE design_annotation (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pair_id             TEXT NOT NULL REFERENCES design_pair(id) ON DELETE CASCADE,
    annotator_id        TEXT NOT NULL,
    winner_run_id       TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT design_annotation_annotator_len CHECK (char_length(annotator_id) <= 128),
    CONSTRAINT design_annotation_unique UNIQUE (pair_id, annotator_id)
);

CREATE TABLE design_rating (
    round_id            BIGINT NOT NULL REFERENCES design_round(round_id),
    miner_hotkey        TEXT NOT NULL,
    rating              INTEGER NOT NULL,
    wins                INTEGER NOT NULL DEFAULT 0,
    losses              INTEGER NOT NULL DEFAULT 0,
    score               BIGINT,
    absence_reason      SMALLINT,
    kind                TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (round_id, miner_hotkey),
    CONSTRAINT design_rating_miner_hotkey_hex CHECK (miner_hotkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT design_rating_kind_check CHECK (kind IS NULL OR kind IN ('score','no_score')),
    CONSTRAINT design_rating_score_range CHECK (score IS NULL OR (score >= 0 AND score <= 1000000))
);

CREATE TABLE design_quota (
    miner_hotkey        TEXT NOT NULL,
    day                 DATE NOT NULL,
    runs_used           INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (miner_hotkey, day),
    CONSTRAINT design_quota_miner_hotkey_hex CHECK (miner_hotkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT design_quota_runs_nonneg CHECK (runs_used >= 0)
);

CREATE TABLE design_stage_event (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id              TEXT NOT NULL REFERENCES design_run(id) ON DELETE CASCADE,
    stage               TEXT NOT NULL,
    detail              JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT design_stage_event_stage_len CHECK (char_length(stage) <= 64)
);

CREATE INDEX ix_design_harness_miner ON design_harness (miner_hotkey);
CREATE INDEX ix_design_harness_active ON design_harness (active) WHERE active;
CREATE INDEX ix_design_run_status ON design_run (status);
CREATE INDEX ix_design_run_round ON design_run (round_id, status);
CREATE INDEX ix_design_run_harness ON design_run (harness_id);
CREATE INDEX ix_design_pair_round ON design_pair (round_id, prompt_id);
CREATE INDEX ix_design_annotation_pair ON design_annotation (pair_id);
CREATE INDEX ix_design_stage_event_run ON design_stage_event (run_id, created_at);
CREATE INDEX ix_design_rating_round ON design_rating (round_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE design_harness TO base_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE design_round TO base_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE design_run TO base_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE design_artifact TO base_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE design_pair TO base_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE design_annotation TO base_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE design_rating TO base_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE design_quota TO base_app;
GRANT SELECT, INSERT ON TABLE design_stage_event TO base_app;
