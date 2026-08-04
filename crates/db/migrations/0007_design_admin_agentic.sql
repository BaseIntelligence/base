-- Design admin winners + agentic anti-cheat stages.
-- Extends design_run lifecycle; stores agentic verdict JSON; round awards.

ALTER TABLE design_run DROP CONSTRAINT design_run_status_check;
ALTER TABLE design_run ADD CONSTRAINT design_run_status_check CHECK (status IN (
    'queued','installing','running','sanitizing',
    'agentic_review','awaiting_admin','awaiting_annotation',
    'scored','failed'));

ALTER TABLE design_run
    ADD COLUMN IF NOT EXISTS agentic_verdict JSONB;

CREATE TABLE design_round_award (
    round_id            BIGINT PRIMARY KEY REFERENCES design_round(round_id),
    winner_harness_ids  JSONB NOT NULL,
    awarded_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    admin_token_hash    TEXT,
    CONSTRAINT design_round_award_winners_len CHECK (
        jsonb_typeof(winner_harness_ids) = 'array'
        AND jsonb_array_length(winner_harness_ids) BETWEEN 1 AND 2
    )
);

CREATE INDEX ix_design_run_awaiting_admin ON design_run (round_id)
    WHERE status = 'awaiting_admin';

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE design_round_award TO base_app;
