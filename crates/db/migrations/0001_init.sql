-- db initial schema (task 12).
-- challenge_backends is operational routing only — never stores signing keys (D18).
-- raw_weight_snapshot, epoch_bundle, peer_root_statement are append-only for the app role.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

CREATE TABLE miners (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hotkey          TEXT NOT NULL,
    uid             INTEGER,
    status          TEXT NOT NULL DEFAULT 'active',
    last_seen_at    TIMESTAMPTZ,
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT miners_hotkey_unique UNIQUE (hotkey),
    CONSTRAINT miners_status_nonempty CHECK (char_length(status) > 0)
);

CREATE TABLE challenge_backends (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    challenge_id    TEXT NOT NULL,
    base_url        TEXT NOT NULL,
    weight          INTEGER NOT NULL DEFAULT 1,
    healthy         BOOLEAN NOT NULL DEFAULT TRUE,
    fail_count      INTEGER NOT NULL DEFAULT 0,
    last_failure_at TIMESTAMPTZ,
    ejected_until   TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT challenge_backends_challenge_url_unique UNIQUE (challenge_id, base_url),
    CONSTRAINT challenge_backends_weight_positive CHECK (weight > 0),
    CONSTRAINT challenge_backends_fail_count_nonneg CHECK (fail_count >= 0),
    CONSTRAINT challenge_backends_challenge_id_nonempty CHECK (char_length(challenge_id) > 0),
    CONSTRAINT challenge_backends_base_url_nonempty CHECK (char_length(base_url) > 0)
    -- D18: no signing_key / private_key / secret columns by design.
);

CREATE TABLE raw_weight_snapshot (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    challenge_id    TEXT NOT NULL,
    epoch           BIGINT NOT NULL,
    miner_hotkey    TEXT NOT NULL,
    kind            TEXT NOT NULL,
    score           BIGINT,
    absence_reason  TEXT,
    payload         BYTEA NOT NULL,
    payload_digest  BYTEA NOT NULL,
    signature       BYTEA NOT NULL,
    nonce           BYTEA NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT raw_weight_snapshot_challenge_epoch_miner_unique
        UNIQUE (challenge_id, epoch, miner_hotkey),
    CONSTRAINT raw_weight_snapshot_kind_check
        CHECK (kind IN ('score', 'no_score')),
    CONSTRAINT raw_weight_snapshot_score_shape_check
        CHECK (
            (kind = 'score' AND score IS NOT NULL AND absence_reason IS NULL)
            OR (kind = 'no_score' AND score IS NULL AND absence_reason IS NOT NULL)
        ),
    CONSTRAINT raw_weight_snapshot_digest_len CHECK (octet_length(payload_digest) = 32),
    CONSTRAINT raw_weight_snapshot_nonce_len CHECK (octet_length(nonce) = 32),
    CONSTRAINT raw_weight_snapshot_sig_nonempty CHECK (octet_length(signature) > 0),
    CONSTRAINT raw_weight_snapshot_payload_nonempty CHECK (octet_length(payload) > 0)
);

CREATE TABLE epoch_bundle (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    epoch                BIGINT NOT NULL,
    protocol_version     INTEGER NOT NULL,
    block_number         BIGINT NOT NULL,
    block_hash           BYTEA NOT NULL,
    metagraph_root       BYTEA NOT NULL,
    merkle_root          BYTEA NOT NULL,
    measurements_digest  BYTEA NOT NULL,
    vector_hash          BYTEA NOT NULL,
    payload              BYTEA NOT NULL,
    signature            BYTEA NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT epoch_bundle_epoch_unique UNIQUE (epoch),
    CONSTRAINT epoch_bundle_block_hash_len CHECK (octet_length(block_hash) = 32),
    CONSTRAINT epoch_bundle_metagraph_root_len CHECK (octet_length(metagraph_root) = 32),
    CONSTRAINT epoch_bundle_merkle_root_len CHECK (octet_length(merkle_root) = 32),
    CONSTRAINT epoch_bundle_measurements_digest_len CHECK (octet_length(measurements_digest) = 32),
    CONSTRAINT epoch_bundle_vector_hash_len CHECK (octet_length(vector_hash) = 32),
    CONSTRAINT epoch_bundle_payload_nonempty CHECK (octet_length(payload) > 0),
    CONSTRAINT epoch_bundle_sig_nonempty CHECK (octet_length(signature) > 0)
);

CREATE TABLE peer_root_statement (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    epoch           BIGINT NOT NULL,
    peer_hotkey     TEXT NOT NULL,
    merkle_root     BYTEA NOT NULL,
    payload         BYTEA NOT NULL,
    signature       BYTEA NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT peer_root_statement_epoch_peer_unique UNIQUE (epoch, peer_hotkey),
    CONSTRAINT peer_root_statement_merkle_root_len CHECK (octet_length(merkle_root) = 32),
    CONSTRAINT peer_root_statement_payload_nonempty CHECK (octet_length(payload) > 0),
    CONSTRAINT peer_root_statement_sig_nonempty CHECK (octet_length(signature) > 0),
    CONSTRAINT peer_root_statement_peer_hotkey_nonempty CHECK (char_length(peer_hotkey) > 0)
);

CREATE TABLE attestation (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    epoch             BIGINT NOT NULL,
    miner_hotkey      TEXT NOT NULL,
    validator_hotkey  TEXT NOT NULL,
    nonce             BYTEA NOT NULL,
    outcome           TEXT NOT NULL,
    quote             BYTEA,
    reason            TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT attestation_epoch_miner_validator_unique
        UNIQUE (epoch, miner_hotkey, validator_hotkey),
    CONSTRAINT attestation_outcome_check
        CHECK (outcome IN ('verified', 'park', 'reject')),
    CONSTRAINT attestation_nonce_len CHECK (octet_length(nonce) = 32),
    CONSTRAINT attestation_miner_hotkey_nonempty CHECK (char_length(miner_hotkey) > 0),
    CONSTRAINT attestation_validator_hotkey_nonempty CHECK (char_length(validator_hotkey) > 0)
);

CREATE TABLE attestation_nonce (
    nonce             BYTEA PRIMARY KEY,
    epoch             BIGINT NOT NULL,
    miner_hotkey      TEXT NOT NULL,
    validator_hotkey  TEXT NOT NULL,
    issued_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at        TIMESTAMPTZ NOT NULL,
    consumed_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT attestation_nonce_len CHECK (octet_length(nonce) = 32),
    CONSTRAINT attestation_nonce_expiry_after_issue CHECK (expires_at > issued_at),
    CONSTRAINT attestation_nonce_miner_hotkey_nonempty CHECK (char_length(miner_hotkey) > 0),
    CONSTRAINT attestation_nonce_validator_hotkey_nonempty CHECK (char_length(validator_hotkey) > 0)
);

CREATE TABLE dissent (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    epoch                 BIGINT NOT NULL,
    validator_hotkey      TEXT NOT NULL,
    protocol_version      INTEGER NOT NULL,
    bundle_root           BYTEA NOT NULL,
    expected_vector_hash  BYTEA NOT NULL,
    actual_vector_hash    BYTEA NOT NULL,
    reason_code           TEXT NOT NULL,
    payload               BYTEA NOT NULL,
    signature             BYTEA NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT dissent_epoch_validator_unique UNIQUE (epoch, validator_hotkey),
    CONSTRAINT dissent_bundle_root_len CHECK (octet_length(bundle_root) = 32),
    CONSTRAINT dissent_expected_vector_hash_len CHECK (octet_length(expected_vector_hash) = 32),
    CONSTRAINT dissent_actual_vector_hash_len CHECK (octet_length(actual_vector_hash) = 32),
    CONSTRAINT dissent_reason_code_nonempty CHECK (char_length(reason_code) > 0),
    CONSTRAINT dissent_payload_nonempty CHECK (octet_length(payload) > 0),
    CONSTRAINT dissent_sig_nonempty CHECK (octet_length(signature) > 0),
    CONSTRAINT dissent_validator_hotkey_nonempty CHECK (char_length(validator_hotkey) > 0)
);

CREATE TABLE promotion (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    environment     TEXT NOT NULL,
    from_digest     TEXT NOT NULL,
    to_digest       TEXT NOT NULL,
    status          TEXT NOT NULL,
    backup_path     TEXT,
    notes           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT promotion_environment_check CHECK (environment IN ('staging', 'prod')),
    CONSTRAINT promotion_status_check
        CHECK (status IN ('pending', 'succeeded', 'rolled_back', 'failed')),
    CONSTRAINT promotion_from_digest_nonempty CHECK (char_length(from_digest) > 0),
    CONSTRAINT promotion_to_digest_nonempty CHECK (char_length(to_digest) > 0)
);

-- ---------------------------------------------------------------------------
-- Indexes (query paths used by gateway / validator)
-- ---------------------------------------------------------------------------

CREATE INDEX ix_challenge_backends_challenge_id ON challenge_backends (challenge_id);
CREATE INDEX ix_challenge_backends_healthy ON challenge_backends (challenge_id, healthy);

CREATE INDEX ix_raw_weight_snapshot_challenge_epoch ON raw_weight_snapshot (challenge_id, epoch);
CREATE INDEX ix_raw_weight_snapshot_epoch ON raw_weight_snapshot (epoch);
CREATE INDEX ix_raw_weight_snapshot_created_at ON raw_weight_snapshot (created_at);

CREATE INDEX ix_epoch_bundle_merkle_root ON epoch_bundle (merkle_root);
CREATE INDEX ix_epoch_bundle_created_at ON epoch_bundle (created_at);

CREATE INDEX ix_peer_root_statement_epoch ON peer_root_statement (epoch);
CREATE INDEX ix_peer_root_statement_merkle_root ON peer_root_statement (merkle_root);

CREATE INDEX ix_attestation_epoch_miner ON attestation (epoch, miner_hotkey);
CREATE INDEX ix_attestation_outcome ON attestation (outcome);

CREATE INDEX ix_attestation_nonce_expires_at ON attestation_nonce (expires_at);
CREATE INDEX ix_attestation_nonce_epoch_miner ON attestation_nonce (epoch, miner_hotkey);

CREATE INDEX ix_dissent_epoch ON dissent (epoch);
CREATE INDEX ix_dissent_reason_code ON dissent (reason_code);

CREATE INDEX ix_promotion_environment_status ON promotion (environment, status);
CREATE INDEX ix_promotion_created_at ON promotion (created_at);

-- ---------------------------------------------------------------------------
-- Roles: migration owner vs application (no UPDATE on append-only tables)
-- ---------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'base_app') THEN
        CREATE ROLE base_app LOGIN PASSWORD 'base_app';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA public TO base_app;

-- Mutable tables: full DML for the app role.
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE
    miners,
    challenge_backends,
    attestation,
    attestation_nonce,
    dissent,
    promotion
TO base_app;

-- Append-only tables: SELECT + INSERT only (no UPDATE, no DELETE).
GRANT SELECT, INSERT ON TABLE
    raw_weight_snapshot,
    epoch_bundle,
    peer_root_statement
TO base_app;

-- Sequences / identity defaults (uuid via pgcrypto; no serials today).
-- Future serials would need: GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO base_app;
