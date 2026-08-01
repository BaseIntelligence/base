-- Seal provenance for `GET /v1/weights/latest` (`computed_at` + `revision`).
--
-- 0001 modelled one immutable row per epoch, but the served response carries a
-- seal revision that increases when an epoch is re-sealed. The app role has no
-- UPDATE on epoch_bundle (append-only by design), so a re-seal must land as a
-- new row and the epoch-unique constraint has to widen to (epoch, revision).
-- `created_at` stays the seal instant and is served as `computed_at`.

ALTER TABLE epoch_bundle
    ADD COLUMN revision INTEGER NOT NULL DEFAULT 1;

ALTER TABLE epoch_bundle
    ADD CONSTRAINT epoch_bundle_revision_positive CHECK (revision > 0);

ALTER TABLE epoch_bundle
    DROP CONSTRAINT epoch_bundle_epoch_unique;

-- Still one row per (epoch, revision): two sealers racing on the same epoch
-- compute the same next revision and exactly one insert survives.
ALTER TABLE epoch_bundle
    ADD CONSTRAINT epoch_bundle_epoch_revision_unique UNIQUE (epoch, revision);

CREATE INDEX ix_epoch_bundle_epoch_revision ON epoch_bundle (epoch, revision DESC);
