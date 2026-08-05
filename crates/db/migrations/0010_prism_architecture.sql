-- PRISM architecture registry + training-only submissions + top-model
-- publication record.
--
-- `prism_architecture` holds PUBLISHED architectures only: an architecture
-- becomes referenceable after its owning submission survived every gate
-- (pre-LLM copy gate, LLM review, agentic anti-cheat) and reached
-- `terminated` with a real measured score. Rejected/cheated architectures
-- never publish, so challengers can trust registry entries.
--
-- Submission kinds after this migration:
--   * architecture submission — miner ships architecture.py + training.py
--     (unchanged contract); on successful termination the arch registers as
--     arch_<hex16> and the submission row back-links `arch_id`.
--   * training-only submission — miner ships training.py + `arch_id` of an
--     already-published architecture; architecture.py is pulled from the
--     registry at intake and denormalized onto the row (replay fidelity).
--
-- Gating keys: one architecture submission per hotkey under challenge key
-- `prism` (0008); training-only entries gate per (hotkey, arch_id) under
-- the composite challenge key `prism:train:<arch_id>` (32 chars max — the
-- submission_gating challenge CHECK bound), same retry classes, same
-- metagraph-watcher resets (rows match on the plain hotkey column).
--
-- Also extends prism_submission.status with terminal 'rejected' (pre-LLM
-- copy gate on architecture.py, created_at ordered — mirrors the design_run
-- extension in 0008).
--
-- `prism_topmodel_publication` journals every top-model publish to the
-- public GitHub repo (BaseIntelligence/prism, top-model/): one row per new
-- global-best bpb at publish time; commit_sha is the GitHub commit that
-- landed the file set (NULL when the publish was a dry-run/no-op).

ALTER TABLE prism_submission DROP CONSTRAINT prism_submission_status_check;
ALTER TABLE prism_submission ADD CONSTRAINT prism_submission_status_check CHECK (status IN (
    'queued','provisioning','running','llm_review','similarity','scoring','terminated','failed','rejected'));

CREATE TABLE prism_architecture (
    arch_id           TEXT PRIMARY KEY,          -- arch_<first 16 hex chars of sha256(architecture_py)>
    owner_hotkey      TEXT NOT NULL,
    arch_digest       TEXT NOT NULL,             -- full sha256 hex of architecture_py
    architecture_py   TEXT NOT NULL,
    source_submission TEXT NOT NULL,             -- originating prism_submission.id
    best_bpb          DOUBLE PRECISION,          -- best measured bpb on this arch (any trainer)
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prism_architecture_id_shape CHECK (arch_id ~ '^arch_[0-9a-f]{16}$'),
    CONSTRAINT prism_architecture_digest_unique UNIQUE (arch_digest),
    CONSTRAINT prism_architecture_owner_hex CHECK (owner_hotkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT prism_architecture_source_len CHECK (octet_length(architecture_py) <= 131072)
);

CREATE INDEX ix_prism_architecture_owner ON prism_architecture (owner_hotkey);
CREATE INDEX ix_prism_architecture_created ON prism_architecture (created_at DESC);

ALTER TABLE prism_submission
    ADD COLUMN arch_id TEXT REFERENCES prism_architecture(arch_id);
CREATE INDEX ix_prism_submission_arch ON prism_submission (arch_id);

CREATE TABLE prism_topmodel_publication (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id   TEXT NOT NULL REFERENCES prism_submission(id) ON DELETE CASCADE,
    arch_id         TEXT REFERENCES prism_architecture(arch_id),
    owner_hotkey    TEXT NOT NULL,
    bpb             DOUBLE PRECISION NOT NULL,
    repo_path       TEXT NOT NULL,               -- e.g. top-model/arch_ab12cd34ef56/
    commit_sha      TEXT,
    published_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prism_topmodel_bpb_pos CHECK (bpb > 0),
    CONSTRAINT prism_topmodel_commit_len CHECK (commit_sha IS NULL OR char_length(commit_sha) <= 64)
);

CREATE INDEX ix_prism_topmodel_bpb ON prism_topmodel_publication (bpb ASC, published_at DESC);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE prism_architecture TO base_app;
GRANT SELECT, INSERT ON TABLE prism_topmodel_publication TO base_app;
