-- Submission gating shared by challenge intakes (design + prism):
-- metagraph membership, one accepted submission per (challenge, hotkey),
-- auto-retry budget accounting, and watcher-driven eligibility resets.
--
-- States: open → registered (intake accepted) → blocked (retry budget spent
-- on infra-class errors) | rejected (cheat / pre-LLM copy gate). The
-- metagraph watcher returns rows to open when the hotkey leaves the
-- metagraph (uid deregistered or hotkey replaced).
--
-- Also extends design_run with the terminal 'rejected' stage (pre-LLM copy
-- gate: byte/AST copy of an earlier harness, created_at ordered).

CREATE TABLE submission_gating (
    challenge           TEXT NOT NULL,
    hotkey              TEXT NOT NULL,
    uid                 INTEGER,
    state               TEXT NOT NULL DEFAULT 'open',
    attempt_count       INTEGER NOT NULL DEFAULT 0,
    last_error_class    TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (challenge, hotkey),
    CONSTRAINT submission_gating_challenge_len CHECK (char_length(challenge) BETWEEN 1 AND 32),
    CONSTRAINT submission_gating_hotkey_len CHECK (char_length(hotkey) BETWEEN 1 AND 128),
    CONSTRAINT submission_gating_state_check CHECK (state IN ('open','registered','blocked','rejected')),
    CONSTRAINT submission_gating_uid_range CHECK (uid IS NULL OR (uid >= 0 AND uid <= 65535)),
    CONSTRAINT submission_gating_attempt_nonneg CHECK (attempt_count >= 0)
);

CREATE INDEX ix_submission_gating_state ON submission_gating (challenge, state);

ALTER TABLE design_run DROP CONSTRAINT design_run_status_check;
ALTER TABLE design_run ADD CONSTRAINT design_run_status_check CHECK (status IN (
    'queued','installing','running','sanitizing',
    'agentic_review','awaiting_admin','awaiting_annotation',
    'scored','failed','rejected'));

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE submission_gating TO base_app;
