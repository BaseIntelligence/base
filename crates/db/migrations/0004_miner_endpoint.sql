-- Miner-announced public CVM base URLs, signed by the miner hotkey.
--
-- AGENT_CHALLENGE.md §9.3 step 5 obliges a miner to register the base URL the
-- challenge service dispatches Harbor packs to. There was nowhere to put it:
-- `challenge_backends` is operator-owned routing keyed by challenge_id and has
-- no hotkey column, so it cannot answer "which URL belongs to this miner".
--
-- The row is keyed by epoch as well as hotkey because a miner redeploys its CVM
-- between epochs and the challenge service must not dispatch this epoch's packs
-- at last month's address. `miner_endpoints()` reads the highest epoch at or
-- above a caller-chosen floor, so an abandoned announcement ages out on its own
-- instead of needing a sweeper.
--
-- Re-announcing inside one epoch is an UPDATE, not a conflict: a miner that
-- moves its CVM mid-epoch has to be able to say so, and a retried POST after a
-- dropped response must not 409. That is why this table is mutable rather than
-- append-only — the signature column keeps the last announcement attributable.
--
-- `miner_hotkey` is lowercase unprefixed 64-hex, byte-for-byte the format
-- `attestation.miner_hotkey` uses (`hex::encode` on both the write and the read
-- side). The CHECK is what stops an SS58 or 0x-prefixed writer from creating a
-- second, silently unjoinable spelling of the same miner.
--
-- `base_url` is re-validated in Rust (miner-endpoint::validate_base_url) before
-- it ever reaches this table; the CHECK here is only a last-resort shape guard,
-- NOT the SSRF control. Anything reading this column for an outbound request
-- must re-check the resolved address at connect time.

CREATE TABLE miner_endpoint (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    epoch         BIGINT NOT NULL,
    netuid        INTEGER NOT NULL,
    miner_hotkey  TEXT NOT NULL,
    base_url      TEXT NOT NULL,
    signature     BYTEA NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT miner_endpoint_epoch_netuid_miner_unique
        UNIQUE (epoch, netuid, miner_hotkey),
    CONSTRAINT miner_endpoint_epoch_nonneg CHECK (epoch >= 0),
    CONSTRAINT miner_endpoint_netuid_range CHECK (netuid >= 0 AND netuid <= 65535),
    CONSTRAINT miner_endpoint_miner_hotkey_hex CHECK (miner_hotkey ~ '^[0-9a-f]{64}$'),
    CONSTRAINT miner_endpoint_base_url_scheme
        CHECK (base_url ~ '^https?://[^[:space:]/?#@]+(:[0-9]{1,5})?/?$'),
    CONSTRAINT miner_endpoint_base_url_len CHECK (char_length(base_url) <= 2048),
    CONSTRAINT miner_endpoint_sig_len CHECK (octet_length(signature) = 64)
);

-- The only production read is "every endpoint on this netuid at or after epoch
-- N", ordered per hotkey.
CREATE INDEX ix_miner_endpoint_netuid_epoch ON miner_endpoint (netuid, epoch);
CREATE INDEX ix_miner_endpoint_miner_hotkey ON miner_endpoint (miner_hotkey);

-- Mutable table (see the idempotent re-announce note above), so the app role
-- needs UPDATE. It is deliberately not added to APPEND_ONLY_TABLES.
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE miner_endpoint TO base_app;
