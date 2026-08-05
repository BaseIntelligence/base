-- Widen submission_gating.challenge to 64 chars.
--
-- Training-only prism entries gate on `prism:train:<arch_id>` where arch_id is
-- `arch_` + 16 lowercase hex (21 chars) — a 33-char key that violated the
-- original BETWEEN 1 AND 32 check, so the intake insert failed with a 500
-- *after* the submission row was already queued. 64 keeps the column bounded
-- while covering composed challenge keys.

ALTER TABLE submission_gating DROP CONSTRAINT submission_gating_challenge_len;
ALTER TABLE submission_gating ADD CONSTRAINT submission_gating_challenge_len
    CHECK (char_length(challenge) BETWEEN 1 AND 64);
