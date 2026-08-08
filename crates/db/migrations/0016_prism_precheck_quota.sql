-- Prism miner similarity precheck quota: 3 attempts per coldkey per UTC day.
--
-- `POST /v1/submissions/precheck` runs the same pre-LLM copy gate as intake
-- without creating a submission or renting a pod. Quota is keyed by coldkey
-- (hotkey fallback when Owner is unknown) so rotating hotkeys cannot reset
-- the daily budget.

CREATE TABLE prism_precheck_quota (
    miner_coldkey   TEXT NOT NULL,
    day             DATE NOT NULL,
    checks_used     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (miner_coldkey, day),
    CONSTRAINT prism_precheck_quota_key_hex CHECK (miner_coldkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT prism_precheck_quota_checks_nonneg CHECK (checks_used >= 0)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE prism_precheck_quota TO base_app;
