-- PRISM epoch-close leaf emission outbox (epoch-semantics fix).
--
-- Before this migration the leaf emitter selected rows by ACCEPTANCE epoch
-- (`prism_submission.epoch = current_epoch`): independent scorers finalized in
-- the same epoch locked each other out (gateway leaves are append-only,
-- first-write-wins), and a submission whose eval crossed an epoch boundary
-- (prod train up to 6h >> 72-min epoch) never scored in any leaf set.
--
-- Emission is now epoch-close batched: one emitter tick per chain epoch
-- assigns every scored-not-yet-emitted row (`kind IS NOT NULL AND
-- emitted_epoch IS NULL`) to that epoch (`emitted_epoch`), submits one
-- D24-complete leaf set, and advances the per-netuid cursor. Exactly-once per
-- scoring run: assignment is sticky before submit and the cursor advances only
-- after the full set landed, so a crash mid-submit replays the assigned set
-- on the next tick (identical leaf values converge under first-write-wins).

ALTER TABLE prism_submission
    ADD COLUMN emitted_epoch BIGINT;        -- leaf epoch this scoring was emitted in (NULL = pending)

CREATE INDEX ix_prism_submission_emit_pending
    ON prism_submission (netuid)
    WHERE kind IS NOT NULL AND emitted_epoch IS NULL;

CREATE TABLE prism_emit_cursor (
    netuid          INTEGER PRIMARY KEY,
    epoch           BIGINT NOT NULL,        -- highest chain epoch whose leaf set fully landed
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT prism_emit_cursor_epoch_nonneg CHECK (epoch >= 0)
);

GRANT SELECT, INSERT, UPDATE ON TABLE prism_emit_cursor TO base_app;
