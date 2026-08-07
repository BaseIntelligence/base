-- Split the design daily run quota by run origin.
--
-- `design_quota.runs_used` counted *every* sandbox run against a single
-- 10-run/hotkey/day ceiling, and the organizer's own round scheduler charged
-- the same bucket as the miner's `POST /v1/harness`. A full UTC day dispatches
-- `ROUNDS_PER_DAY (10) × PROMPTS_PER_ROUND (3)` = 30 runs to every registered
-- harness, so an honest, fully participating miner exhausted the day's quota
-- after ~3.3 rounds, sat out the remaining rounds, and could not even submit
-- (intake 409s when scheduling fails).
--
-- `manual_runs_used` isolates the anti-spam ceiling that actually belongs to
-- miner-initiated submissions; organizer-scheduled work is `runs_used -
-- manual_runs_used` and is bounded by a separate cap derived from the live
-- round schedule. Existing rows backfill to 0 manual runs, which only ever
-- widens a live miner's submission budget.

ALTER TABLE design_quota
    ADD COLUMN manual_runs_used INTEGER NOT NULL DEFAULT 0;

ALTER TABLE design_quota
    ADD CONSTRAINT design_quota_manual_runs_nonneg CHECK (manual_runs_used >= 0);
