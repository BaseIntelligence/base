-- Persist miner coldkey (SubtensorModule.Owner) at intake so similarity /
-- copy corpora can exclude same-coldkey prior art after a miner iterates via
-- a new hotkey (1-max gating forces hotkey rotation under one coldkey).
-- Nullable for legacy rows; new intakes fill it from the metagraph cache.

ALTER TABLE design_harness
    ADD COLUMN miner_coldkey TEXT;

ALTER TABLE design_harness
    ADD CONSTRAINT design_harness_miner_coldkey_hex
    CHECK (miner_coldkey IS NULL OR miner_coldkey ~ '^[0-9a-f]{64}$');

CREATE INDEX ix_design_harness_coldkey
    ON design_harness (miner_coldkey)
    WHERE miner_coldkey IS NOT NULL;

ALTER TABLE prism_submission
    ADD COLUMN miner_coldkey TEXT;

ALTER TABLE prism_submission
    ADD CONSTRAINT prism_submission_miner_coldkey_hex
    CHECK (miner_coldkey IS NULL OR miner_coldkey ~ '^[0-9a-f]{64}$');

CREATE INDEX ix_prism_submission_coldkey
    ON prism_submission (miner_coldkey)
    WHERE miner_coldkey IS NOT NULL;
