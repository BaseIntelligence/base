-- Durable attestation outcomes plus the measured receipt key.
--
-- 0001 modelled one immutable attestation row per (epoch, miner, validator),
-- but a miner legitimately re-submits within an epoch: a PCS outage parks the
-- first attempt and a later retry verifies. Overwriting the parked row would
-- erase the evidence of that outage, so a retry lands as a new attempt and the
-- unique constraint widens to include it, the shape 0002 used for epoch_bundle.
--
-- `receipt_pk` is the sr25519 public key published as BASE_RECEIPT_PUBLIC_KEY
-- inside the measured app-compose. It is only trustworthy when the compose
-- preimage was checked against the RTMR3 compose hash, which only happens on
-- the verified path, so a non-verified row may never carry one.

ALTER TABLE attestation
    ADD COLUMN receipt_pk BYTEA;

ALTER TABLE attestation
    ADD CONSTRAINT attestation_receipt_pk_len
        CHECK (receipt_pk IS NULL OR octet_length(receipt_pk) = 32);

ALTER TABLE attestation
    ADD CONSTRAINT attestation_receipt_pk_verified_only
        CHECK (receipt_pk IS NULL OR outcome = 'verified');

ALTER TABLE attestation
    ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1;

ALTER TABLE attestation
    ADD CONSTRAINT attestation_attempt_positive CHECK (attempt > 0);

ALTER TABLE attestation
    DROP CONSTRAINT attestation_epoch_miner_validator_unique;

-- Two validators recording the same attempt number for the same miner still
-- get one row each; two writers racing on one attempt leave exactly one.
ALTER TABLE attestation
    ADD CONSTRAINT attestation_epoch_miner_validator_attempt_unique
        UNIQUE (epoch, miner_hotkey, validator_hotkey, attempt);
