-- Design product rules: admin/auto reject reason + epoch clock for unscored timeout.
-- `attempt_epoch` is stamped at run insert (chain epoch).
-- `awaiting_admin_epoch` is stamped when a clean run enters awaiting_admin.
-- Auto-reject when current_epoch - stamped_epoch >= 5 (see UNSCORED_EPOCH_LIMIT).

ALTER TABLE design_run
    ADD COLUMN IF NOT EXISTS reject_reason TEXT;

ALTER TABLE design_run
    ADD COLUMN IF NOT EXISTS attempt_epoch BIGINT;

ALTER TABLE design_run
    ADD COLUMN IF NOT EXISTS awaiting_admin_epoch BIGINT;

ALTER TABLE design_run
    ADD CONSTRAINT design_run_attempt_epoch_nonneg
    CHECK (attempt_epoch IS NULL OR attempt_epoch >= 0);

ALTER TABLE design_run
    ADD CONSTRAINT design_run_awaiting_admin_epoch_nonneg
    CHECK (awaiting_admin_epoch IS NULL OR awaiting_admin_epoch >= 0);

CREATE INDEX IF NOT EXISTS ix_design_run_awaiting_admin_epoch
    ON design_run (awaiting_admin_epoch)
    WHERE status = 'awaiting_admin' AND awaiting_admin_epoch IS NOT NULL;
