//! Typed persistence for the gateway's append-only tables.
//!
//! `raw_weight_snapshot` and `epoch_bundle` are `SELECT`/`INSERT` only for the
//! application role, so every helper here is an insert or a read — never an
//! update. All uniqueness and shape invariants are enforced by the schema
//! (`0001_init.sql`, `0002_epoch_bundle_revision.sql`); the Rust side only
//! feeds them and interprets the conflicts they raise.

use sqlx::PgPool;
use uuid::Uuid;

use crate::DbError;

/// Row to append to `raw_weight_snapshot`.
///
/// `kind` / `score` / `absence_reason` must satisfy
/// `raw_weight_snapshot_score_shape_check`: `'score'` carries a score and no
/// reason, `'no_score'` carries a reason and no score.
#[derive(Debug, Clone)]
pub struct NewRawWeight<'a> {
    /// Row id chosen by the caller so the HTTP ack and the row agree.
    pub id: Uuid,
    /// Challenge id (UTF-8).
    pub challenge_id: &'a str,
    /// Epoch index.
    pub epoch: i64,
    /// Miner hotkey hex (64 chars, lowercase).
    pub miner_hotkey: &'a str,
    /// `'score'` or `'no_score'`.
    pub kind: &'a str,
    /// Score, present iff `kind == "score"`.
    pub score: Option<i64>,
    /// Absence reason, present iff `kind == "no_score"`.
    pub absence_reason: Option<&'a str>,
    /// SCALE-encoded `RawWeightBodyV1`.
    pub payload: &'a [u8],
    /// SHA-256 of `payload` (exactly 32 bytes).
    pub payload_digest: &'a [u8],
    /// Challenge sr25519 signature (64 bytes).
    pub signature: &'a [u8],
    /// Per-submission nonce (exactly 32 bytes).
    pub nonce: &'a [u8],
}

/// A stored `raw_weight_snapshot` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawWeightRecord {
    /// Row id.
    pub id: Uuid,
    /// Challenge id.
    pub challenge_id: String,
    /// Epoch index.
    pub epoch: i64,
    /// Miner hotkey hex.
    pub miner_hotkey: String,
    /// `'score'` or `'no_score'`.
    pub kind: String,
    /// Score when `kind == "score"`.
    pub score: Option<i64>,
    /// Absence reason when `kind == "no_score"`.
    pub absence_reason: Option<String>,
    /// SCALE-encoded `RawWeightBodyV1`.
    pub payload: Vec<u8>,
    /// SHA-256 of `payload`.
    pub payload_digest: Vec<u8>,
    /// Challenge sr25519 signature.
    pub signature: Vec<u8>,
}

/// Append one raw-weight leaf.
///
/// Returns `Ok(None)` when `(challenge_id, epoch, miner_hotkey)` is already
/// stored — `raw_weight_snapshot_challenge_epoch_miner_unique` is what makes a
/// retried submission a conflict instead of a duplicate.
///
/// # Errors
///
/// Propagates sqlx query errors, including the schema `CHECK` violations that
/// reject a malformed row.
pub async fn insert_raw_weight(
    pool: &PgPool,
    row: &NewRawWeight<'_>,
) -> Result<Option<Uuid>, DbError> {
    let id = sqlx::query_scalar!(
        r#"
        INSERT INTO raw_weight_snapshot
            (id, challenge_id, epoch, miner_hotkey, kind, score, absence_reason,
             payload, payload_digest, signature, nonce)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
        ON CONFLICT (challenge_id, epoch, miner_hotkey) DO NOTHING
        RETURNING id
        "#,
        row.id,
        row.challenge_id,
        row.epoch,
        row.miner_hotkey,
        row.kind,
        row.score,
        row.absence_reason,
        row.payload,
        row.payload_digest,
        row.signature,
        row.nonce,
    )
    .fetch_optional(pool)
    .await?;
    Ok(id)
}

/// Read one raw-weight leaf by its unique key.
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn get_raw_weight(
    pool: &PgPool,
    challenge_id: &str,
    epoch: i64,
    miner_hotkey: &str,
) -> Result<Option<RawWeightRecord>, DbError> {
    let rec = sqlx::query_as!(
        RawWeightRecord,
        r#"
        SELECT id, challenge_id, epoch, miner_hotkey, kind, score, absence_reason,
               payload, payload_digest, signature
        FROM raw_weight_snapshot
        WHERE challenge_id = $1 AND epoch = $2 AND miner_hotkey = $3
        "#,
        challenge_id,
        epoch,
        miner_hotkey,
    )
    .fetch_optional(pool)
    .await?;
    Ok(rec)
}

/// All raw-weight leaves for `epoch`, in a stable order.
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn list_raw_weights_for_epoch(
    pool: &PgPool,
    epoch: i64,
) -> Result<Vec<RawWeightRecord>, DbError> {
    let rows = sqlx::query_as!(
        RawWeightRecord,
        r#"
        SELECT id, challenge_id, epoch, miner_hotkey, kind, score, absence_reason,
               payload, payload_digest, signature
        FROM raw_weight_snapshot
        WHERE epoch = $1
        ORDER BY challenge_id, miner_hotkey
        "#,
        epoch,
    )
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Count rows in `raw_weight_snapshot`.
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn count_raw_weights(pool: &PgPool) -> Result<i64, DbError> {
    let n: i64 =
        sqlx::query_scalar!(r#"SELECT COUNT(*)::bigint AS "count!" FROM raw_weight_snapshot"#)
            .fetch_one(pool)
            .await?;
    Ok(n)
}

/// Sealed bundle to append to `epoch_bundle`.
///
/// Every `*_root` / `*_hash` / `*_digest` field must be exactly 32 bytes
/// (`epoch_bundle_*_len` checks) and both byte blobs must be non-empty.
#[derive(Debug, Clone)]
pub struct NewEpochBundle<'a> {
    /// Epoch index.
    pub epoch: i64,
    /// Bundle `protocol_version`.
    pub protocol_version: i32,
    /// Inclusive epoch end block.
    pub block_number: i64,
    /// Hash of the end block (32 bytes).
    pub block_hash: &'a [u8],
    /// Metagraph root (32 bytes).
    pub metagraph_root: &'a [u8],
    /// Merkle root over leaves (32 bytes).
    pub merkle_root: &'a [u8],
    /// Measurements trust-root digest (32 bytes).
    pub measurements_digest: &'a [u8],
    /// `sha256(scale(final_vector))` (32 bytes).
    pub vector_hash: &'a [u8],
    /// SCALE-encoded `EpochBundleV1`.
    pub payload: &'a [u8],
    /// Gateway envelope signature.
    pub signature: &'a [u8],
}

/// A stored `epoch_bundle` row with its seal provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochBundleRecord {
    /// Epoch index.
    pub epoch: i64,
    /// Seal revision (1 for the first seal of the epoch).
    pub revision: i32,
    /// `created_at` in microseconds since the Unix epoch — the seal instant.
    pub sealed_at_micros: i64,
    /// SCALE-encoded `EpochBundleV1`.
    pub payload: Vec<u8>,
}

/// Append a sealed bundle as the next revision of its epoch.
///
/// The revision is derived in the same statement as the insert, so
/// `epoch_bundle_epoch_revision_unique` rejects a racing sealer that computed
/// the same next revision.
///
/// # Errors
///
/// Propagates sqlx query errors, including the 32-byte `CHECK` violations.
pub async fn insert_epoch_bundle(
    pool: &PgPool,
    bundle: &NewEpochBundle<'_>,
) -> Result<EpochBundleRecord, DbError> {
    let rec = sqlx::query!(
        r#"
        INSERT INTO epoch_bundle
            (epoch, revision, protocol_version, block_number, block_hash,
             metagraph_root, merkle_root, measurements_digest, vector_hash,
             payload, signature)
        SELECT $1,
               COALESCE((SELECT MAX(revision) FROM epoch_bundle WHERE epoch = $1), 0) + 1,
               $2, $3, $4, $5, $6, $7, $8, $9, $10
        RETURNING revision,
                  (EXTRACT(EPOCH FROM created_at) * 1000000)::bigint AS "sealed_at_micros!"
        "#,
        bundle.epoch,
        bundle.protocol_version,
        bundle.block_number,
        bundle.block_hash,
        bundle.metagraph_root,
        bundle.merkle_root,
        bundle.measurements_digest,
        bundle.vector_hash,
        bundle.payload,
        bundle.signature,
    )
    .fetch_one(pool)
    .await?;
    Ok(EpochBundleRecord {
        epoch: bundle.epoch,
        revision: rec.revision,
        sealed_at_micros: rec.sealed_at_micros,
        payload: bundle.payload.to_vec(),
    })
}

/// Newest sealed revision of `epoch`.
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn get_epoch_bundle(
    pool: &PgPool,
    epoch: i64,
) -> Result<Option<EpochBundleRecord>, DbError> {
    let rec = sqlx::query_as!(
        EpochBundleRecord,
        r#"
        SELECT epoch, revision, payload,
               (EXTRACT(EPOCH FROM created_at) * 1000000)::bigint AS "sealed_at_micros!"
        FROM epoch_bundle
        WHERE epoch = $1
        ORDER BY revision DESC
        LIMIT 1
        "#,
        epoch,
    )
    .fetch_optional(pool)
    .await?;
    Ok(rec)
}

/// Newest sealed bundle carrying `merkle_root`.
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn get_epoch_bundle_by_root(
    pool: &PgPool,
    merkle_root: &[u8],
) -> Result<Option<EpochBundleRecord>, DbError> {
    let rec = sqlx::query_as!(
        EpochBundleRecord,
        r#"
        SELECT epoch, revision, payload,
               (EXTRACT(EPOCH FROM created_at) * 1000000)::bigint AS "sealed_at_micros!"
        FROM epoch_bundle
        WHERE merkle_root = $1
        ORDER BY epoch DESC, revision DESC
        LIMIT 1
        "#,
        merkle_root,
    )
    .fetch_optional(pool)
    .await?;
    Ok(rec)
}

/// Highest sealed epoch, if any bundle exists.
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn latest_bundle_epoch(pool: &PgPool) -> Result<Option<i64>, DbError> {
    let epoch: Option<i64> = sqlx::query_scalar!(r#"SELECT MAX(epoch) FROM epoch_bundle"#)
        .fetch_one(pool)
        .await?;
    Ok(epoch)
}

/// Length of an sr25519 public key stored in `attestation.receipt_pk`.
pub const RECEIPT_PK_LEN: usize = 32;

/// One attestation outcome to append to `attestation`.
///
/// `outcome` must be `verified` | `park` | `reject`
/// (`attestation_outcome_check`) and `receipt_pk`, when present, must be
/// [`RECEIPT_PK_LEN`] bytes on a `verified` row only.
#[derive(Debug, Clone)]
pub struct NewAttestation<'a> {
    /// Epoch the quote bound.
    pub epoch: i64,
    /// Attested miner hotkey (hex).
    pub miner_hotkey: &'a str,
    /// Validator that ran the verification (hex).
    pub validator_hotkey: &'a str,
    /// Single-use nonce from the D10 binding (32 bytes).
    pub nonce: &'a [u8],
    /// `verified` | `park` | `reject`.
    pub outcome: &'a str,
    /// Raw quote bytes kept as evidence.
    pub quote: Option<&'a [u8]>,
    /// Machine reason when not verified.
    pub reason: Option<&'a str>,
    /// Receipt key read from the measured compose (verified rows only).
    pub receipt_pk: Option<&'a [u8]>,
}

/// Latest attestation outcome for a miner at an epoch, across validators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationRecord {
    /// `verified` | `park` | `reject`.
    pub outcome: String,
    /// Receipt key, only ever set on a `verified` outcome.
    pub receipt_pk: Option<[u8; RECEIPT_PK_LEN]>,
}

/// Append an attestation outcome as the next attempt for its
/// `(epoch, miner, validator)` slot.
///
/// The attempt is derived in the same statement as the insert, so
/// `attestation_epoch_miner_validator_attempt_unique` rejects a racing writer
/// that computed the same next attempt. Returns the attempt that landed.
///
/// # Errors
///
/// Propagates sqlx query errors, including the schema `CHECK` violations.
pub async fn insert_attestation(pool: &PgPool, row: &NewAttestation<'_>) -> Result<i32, DbError> {
    let attempt: i32 = sqlx::query_scalar!(
        r#"
        INSERT INTO attestation
            (epoch, miner_hotkey, validator_hotkey, attempt, nonce, outcome,
             quote, reason, receipt_pk)
        SELECT $1, $2, $3,
               COALESCE((SELECT MAX(attempt) FROM attestation
                         WHERE epoch = $1 AND miner_hotkey = $2
                           AND validator_hotkey = $3), 0) + 1,
               $4, $5, $6, $7, $8
        RETURNING attempt
        "#,
        row.epoch,
        row.miner_hotkey,
        row.validator_hotkey,
        row.nonce,
        row.outcome,
        row.quote,
        row.reason,
        row.receipt_pk,
    )
    .fetch_one(pool)
    .await?;
    Ok(attempt)
}

/// Latest attestation outcome for a miner at an epoch, across validators.
/// Returns the receipt key only for a `verified` outcome.
///
/// Tie-break across validators and retries: any `verified` row wins over a
/// `park` / `reject`, then the highest attempt, then the newest row. One
/// validator verifying a miner is enough to bind its receipt key, and the
/// binding is measurement-derived, so it cannot differ between validators that
/// both verified.
///
/// # Errors
///
/// Propagates sqlx query errors.
pub async fn attestation_for_miner(
    pool: &PgPool,
    epoch: i64,
    miner_hotkey: &str,
) -> Result<Option<AttestationRecord>, DbError> {
    let row = sqlx::query!(
        r#"
        SELECT outcome, receipt_pk
        FROM attestation
        WHERE epoch = $1 AND miner_hotkey = $2
        ORDER BY (outcome = 'verified') DESC, attempt DESC, created_at DESC
        LIMIT 1
        "#,
        epoch,
        miner_hotkey,
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| {
        let receipt_pk = r
            .receipt_pk
            .filter(|_| r.outcome == "verified")
            .and_then(|b| <[u8; RECEIPT_PK_LEN]>::try_from(b.as_slice()).ok());
        AttestationRecord {
            outcome: r.outcome,
            receipt_pk,
        }
    }))
}
