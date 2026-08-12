-- PRISM v3 eval storage: composite runs, Zone A metrics, anchor registry,
-- pre-registration hash-commits, and Zone B participant metric reports
-- (docs/spikes/prism-v3/research/09-miner-metrics-leaderboards.md §7).
--
-- `prism_eval_run` is the per-submission composite record: at most one row
-- per submission (UNIQUE), replaced wholesale on re-score — the same
-- semantics as `prism_telemetry` (0009). Children cascade from it. The full
-- `CompositeOutcome` serde blob lives in `outcome_json` (authoritative
-- copy); `prism_eval_group` is its queryable projection behind the API
-- eval panel. `pod_manifest` / `harness_files_sha256` / `netns` are the
-- METRICS_JSON v2 provenance captured at measure time.
--
-- Zone A (`prism_eval_metric`, `prism_mirror_pair`) holds organizer-measured
-- `org.*` keys only; miner-emitted `org.*` is rejected at Zone B ingest.
-- Zone B (`prism_metric_report`) is participant-reported, hash-chained
-- (`prev_hash` links to the previous report's `report_hash`; the first
-- report chains to the submission id), and verdicted
-- (`ok` | `flagged` | `quarantined`). Zone B is never read by the scoring
-- path; quarantined rows are kept as anti-cheat evidence (no retraction).

CREATE TABLE prism_eval_run (
    run_id               TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    submission_id        TEXT NOT NULL REFERENCES prism_submission(id) ON DELETE CASCADE,
    anchor_version       INTEGER NOT NULL,
    prereg_hash          TEXT NOT NULL,          -- sha256 hex over the canonical anchor JSON
    scoring_mode         TEXT NOT NULL,          -- shadow | composite (PRISM_SCORING_MODE)
    pod_manifest         JSONB,                  -- nvidia-smi -q snapshot, versions, netns facts
    harness_files_sha256 TEXT,                   -- sha256 hex of the uploaded harness file set
    netns                BOOLEAN,                -- miner subprocess ran in an empty netns
    eval_tier            TEXT,                   -- battery tier label when reported
    outcome_json         JSONB NOT NULL,         -- CompositeOutcome (serde tagged scored|ineligible)
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prism_eval_run_submission_unique UNIQUE (submission_id),
    CONSTRAINT prism_eval_run_mode_check CHECK (scoring_mode IN ('shadow', 'composite')),
    CONSTRAINT prism_eval_run_anchor_nonneg CHECK (anchor_version >= 0)
);

CREATE TABLE prism_eval_group (
    run_id         TEXT NOT NULL REFERENCES prism_eval_run(run_id) ON DELETE CASCADE,
    grp            TEXT NOT NULL,                -- g1..g8 ("group" is SQL-reserved)
    g              DOUBLE PRECISION NOT NULL,    -- point estimate after mirror-gap penalty
    ci_lo          DOUBLE PRECISION,             -- clustered-bootstrap 2.5% (NULL = no bootstrap)
    ci_hi          DOUBLE PRECISION,             -- clustered-bootstrap 97.5%
    mirror_penalty DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, grp),
    CONSTRAINT prism_eval_group_grp_check
        CHECK (grp IN ('g1','g2','g3','g4','g5','g6','g7','g8'))
);

CREATE TABLE prism_eval_metric (
    run_id   TEXT NOT NULL REFERENCES prism_eval_run(run_id) ON DELETE CASCADE,
    key      TEXT NOT NULL,                      -- org.<group>.<name> (Zone A raw)
    value    DOUBLE PRECISION NOT NULL,          -- aggregate on the metric's natural scale
    clusters JSONB,                              -- cluster id -> per-cluster value (bootstrap units)
    PRIMARY KEY (run_id, key),
    CONSTRAINT prism_eval_metric_key_org CHECK (key LIKE 'org.%')
);

CREATE TABLE prism_mirror_pair (
    run_id          TEXT NOT NULL REFERENCES prism_eval_run(run_id) ON DELETE CASCADE,
    grp             TEXT NOT NULL,               -- group the penalty applies to (g2 | g4)
    metric          TEXT NOT NULL,               -- org.* key whose anchors normalize both sides
    public_value    DOUBLE PRECISION NOT NULL,
    mirror_value    DOUBLE PRECISION NOT NULL,
    public_clusters JSONB,
    mirror_clusters JSONB,
    PRIMARY KEY (run_id, grp, metric),
    CONSTRAINT prism_mirror_pair_grp_check CHECK (grp IN ('g1','g2','g3','g4','g5','g6','g7','g8'))
);

CREATE TABLE prism_anchor_set (
    version      INTEGER PRIMARY KEY,
    json         JSONB NOT NULL,                 -- canonical anchor set (embedded contract)
    prereg_hash  TEXT NOT NULL,
    status       TEXT NOT NULL,                  -- placeholder | active
    activated_at TIMESTAMPTZ,                    -- set when governance activates the set
    CONSTRAINT prism_anchor_set_status_check CHECK (status IN ('placeholder', 'active')),
    CONSTRAINT prism_anchor_set_version_nonneg CHECK (version >= 0)
);

CREATE TABLE prism_prereg (
    version      INTEGER NOT NULL,
    hash         TEXT NOT NULL,                  -- sha256 hex over the canonical anchor JSON
    committed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    notes        TEXT,
    PRIMARY KEY (version, hash),
    CONSTRAINT prism_prereg_version_nonneg CHECK (version >= 0)
);

CREATE TABLE prism_metric_report (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id   TEXT NOT NULL REFERENCES prism_submission(id) ON DELETE CASCADE,
    seq             BIGINT NOT NULL,             -- per-submission chain sequence (0-based)
    schema_version  TEXT NOT NULL,               -- envelope schema (pinned to recipe_version)
    prev_hash       TEXT NOT NULL,               -- previous report_hash; submission_id when seq = 0
    report_hash     TEXT NOT NULL,               -- sha256 hex over the canonical payload
    payload         JSONB NOT NULL,              -- canonical Zone B payload (immutable)
    verdict         TEXT NOT NULL,               -- ok | flagged | quarantined
    verdict_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prism_metric_report_seq_unique UNIQUE (submission_id, seq),
    CONSTRAINT prism_metric_report_hash_unique UNIQUE (submission_id, report_hash),
    CONSTRAINT prism_metric_report_verdict_check
        CHECK (verdict IN ('ok', 'flagged', 'quarantined')),
    CONSTRAINT prism_metric_report_seq_nonneg CHECK (seq >= 0)
);

CREATE INDEX ix_prism_eval_metric_key ON prism_eval_metric (key);
CREATE INDEX ix_prism_metric_report_subm ON prism_metric_report (submission_id, seq);

-- Eval rows are written once at finalize and replaced wholesale on re-score
-- (the app role deletes only the run row; children cascade, mirroring the
-- `prism_stage_event` grant pattern). `prism_metric_report` is append-only
-- audit: flagged/quarantined rows stay visible (penalty-taxonomy
-- credibility); retries continue the chain at max(seq)+1.
-- `prism_anchor_set` is the one mutable registry row (status flip on
-- activation); `prism_prereg` hash-commits are append-only.
GRANT SELECT, INSERT, DELETE ON TABLE prism_eval_run TO base_app;
GRANT SELECT, INSERT ON TABLE prism_eval_group TO base_app;
GRANT SELECT, INSERT ON TABLE prism_eval_metric TO base_app;
GRANT SELECT, INSERT ON TABLE prism_mirror_pair TO base_app;
GRANT SELECT, INSERT, UPDATE ON TABLE prism_anchor_set TO base_app;
GRANT SELECT, INSERT ON TABLE prism_prereg TO base_app;
GRANT SELECT, INSERT ON TABLE prism_metric_report TO base_app;
