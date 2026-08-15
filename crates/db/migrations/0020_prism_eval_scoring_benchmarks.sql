-- PRISM_SCORING_MODE default is `benchmarks` (v4 G2 lattice). Finalize writes
-- that name into prism_eval_run.scoring_mode; the 0019 check only allowed
-- shadow|composite, so every v4 finalize failed:
--   new row for relation "prism_eval_run" violates check constraint
--   "prism_eval_run_mode_check"
-- Idempotent: prod may already have been patched by hand.

ALTER TABLE prism_eval_run DROP CONSTRAINT IF EXISTS prism_eval_run_mode_check;
ALTER TABLE prism_eval_run
    ADD CONSTRAINT prism_eval_run_mode_check
    CHECK (scoring_mode IN ('shadow', 'composite', 'benchmarks'));
