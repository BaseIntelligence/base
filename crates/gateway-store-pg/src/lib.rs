//! Postgres-backed implementations of the gateway's store traits.
//!
//! The gateway's in-memory stores stay for unit tests; production runs on these
//! so submitted raw weights and sealed bundles survive a restart. Rows land in
//! `raw_weight_snapshot` and `epoch_bundle`, and every uniqueness rule the HTTP
//! layer depends on is the schema's, not this crate's.

#![forbid(unsafe_code)]

use std::future::Future;
use std::sync::Arc;

use bundle::EpochBundleV1;
use db::{DbError, EpochBundleRecord, NewEpochBundle, NewRawWeight, PgPool, RawWeightRecord};
use gateway::{
    BundleStore, RawWeightRow, RawWeightStore, SharedBundleStore, SharedWeightStore, StoreError,
};
use parity_scale_codec::Encode;
use sha2::{Digest, Sha256};
use uuid::Uuid;
use weights_api::SealRecord;

/// Build the Postgres weight + bundle stores against `database_url`.
///
/// The pool is opened eagerly, so an unreachable database fails here instead of
/// degrading to a gateway that acknowledges submissions it never stored.
///
/// # Errors
///
/// When the dedicated database runtime, its thread, or the pool cannot be created.
pub fn stores(database_url: &str) -> Result<(SharedWeightStore, SharedBundleStore), String> {
    let exec = Arc::new(Executor::connect(database_url)?);
    let weights: SharedWeightStore = Arc::new(PgRawWeightStore {
        exec: Arc::clone(&exec),
    });
    let bundles: SharedBundleStore = Arc::new(PgBundleStore { exec });
    Ok((weights, bundles))
}

/// Drives async queries for the synchronous store traits.
///
/// The traits are called from inside axum handlers, and tokio forbids blocking
/// a runtime thread on that same runtime, so the queries run on a runtime of
/// their own and the caller only blocks on a one-shot channel. The pool is
/// opened on that runtime too: a tokio socket is only usable from the reactor
/// it was registered with.
struct Executor {
    pool: PgPool,
    jobs: std::sync::mpsc::Sender<Job>,
}

/// A query already bound to the channel its caller waits on.
type Job = std::pin::Pin<Box<dyn Future<Output = ()> + Send>>;

impl Executor {
    fn connect(database_url: &str) -> Result<Self, String> {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("gateway-db")
            .enable_all()
            .build()
            .map_err(|e| format!("gateway db runtime: {e}"))?;
        let (jobs, rx) = std::sync::mpsc::channel::<Job>();
        // The runtime is owned by this thread and dropped here once the last
        // store is gone: dropping a runtime inside an async context panics.
        std::thread::Builder::new()
            .name("gateway-db-exec".to_owned())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    rt.spawn(job);
                }
            })
            .map_err(|e| format!("gateway db thread: {e}"))?;
        let url = database_url.to_owned();
        let pool = submit(&jobs, async move { db::connect(&url).await })?;
        Ok(Self { pool, jobs })
    }

    fn run<F, T>(&self, query: impl FnOnce(PgPool) -> F) -> Result<T, String>
    where
        F: Future<Output = Result<T, DbError>> + Send + 'static,
        T: Send + 'static,
    {
        submit(&self.jobs, query(self.pool.clone()))
    }
}

/// Run `fut` on the database runtime and block until it answers.
fn submit<F, T>(jobs: &std::sync::mpsc::Sender<Job>, fut: F) -> Result<T, String>
where
    F: Future<Output = Result<T, DbError>> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    jobs.send(Box::pin(async move {
        let _ = tx.send(fut.await);
    }))
    .map_err(|_| "gateway db executor stopped".to_owned())?;
    match rx.recv() {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("gateway db task dropped".to_owned()),
    }
}

/// `raw_weight_snapshot`-backed [`RawWeightStore`].
struct PgRawWeightStore {
    exec: Arc<Executor>,
}

/// `epoch_bundle`-backed [`BundleStore`].
struct PgBundleStore {
    exec: Arc<Executor>,
}

fn epoch_i64(epoch: u64) -> Result<i64, String> {
    i64::try_from(epoch).map_err(|_| format!("epoch {epoch} exceeds BIGINT"))
}

fn record_to_row(rec: RawWeightRecord) -> Result<RawWeightRow, String> {
    let digest: [u8; 32] = rec
        .payload_digest
        .as_slice()
        .try_into()
        .map_err(|_| "stored payload_digest is not 32 bytes".to_owned())?;
    Ok(RawWeightRow {
        id: rec.id,
        challenge_id: rec.challenge_id,
        epoch: u64::try_from(rec.epoch).map_err(|_| "stored epoch is negative".to_owned())?,
        miner_hotkey: rec.miner_hotkey,
        kind: rec.kind,
        score: rec
            .score
            .map(u64::try_from)
            .transpose()
            .map_err(|_| "stored score is negative".to_owned())?,
        absence_reason: rec.absence_reason,
        payload: rec.payload,
        payload_digest: digest,
        challenge_sig: rec.signature,
    })
}

impl RawWeightStore for PgRawWeightStore {
    fn insert(&self, row: RawWeightRow) -> Result<RawWeightRow, StoreError> {
        let inserted = self.try_insert(&row).map_err(StoreError::Backend)?;
        if inserted {
            return Ok(row);
        }
        // The unique key already holds a row: the 409 body must echo the
        // stored original, not the rejected resubmission.
        match self.get(&row.challenge_id, row.epoch, &row.miner_hotkey) {
            Some(original) => Err(StoreError::Conflict {
                original: Box::new(original),
            }),
            None => Err(StoreError::Backend(
                "conflicting raw weight vanished before read-back".to_owned(),
            )),
        }
    }

    fn get(&self, challenge_id: &str, epoch: u64, miner_hotkey: &str) -> Option<RawWeightRow> {
        let epoch = log_err("raw_weight_get", epoch_i64(epoch))?;
        let challenge_id = challenge_id.to_owned();
        let miner_hotkey = miner_hotkey.to_owned();
        let found = self.exec.run(move |pool| async move {
            db::get_raw_weight(&pool, &challenge_id, epoch, &miner_hotkey).await
        });
        let rec = log_err("raw_weight_get", found)??;
        log_err("raw_weight_get", record_to_row(rec))
    }

    fn len(&self) -> usize {
        let count = self
            .exec
            .run(move |pool| async move { db::count_raw_weights(&pool).await });
        log_err("raw_weight_len", count)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0)
    }

    fn list_for_epoch(&self, epoch: u64) -> Vec<RawWeightRow> {
        let Some(epoch) = log_err("raw_weight_list", epoch_i64(epoch)) else {
            return Vec::new();
        };
        let rows = self
            .exec
            .run(move |pool| async move { db::list_raw_weights_for_epoch(&pool, epoch).await });
        let Some(rows) = log_err("raw_weight_list", rows) else {
            return Vec::new();
        };
        rows.into_iter()
            .filter_map(|rec| log_err("raw_weight_list", record_to_row(rec)))
            .collect()
    }
}

impl PgRawWeightStore {
    /// `Ok(true)` when the row was appended, `Ok(false)` on unique conflict.
    fn try_insert(&self, row: &RawWeightRow) -> Result<bool, String> {
        let epoch = epoch_i64(row.epoch)?;
        let score = row
            .score
            .map(i64::try_from)
            .transpose()
            .map_err(|_| "score exceeds BIGINT".to_owned())?;
        // sr25519 signatures are `R ‖ s`; `R` is the signer's per-submission
        // nonce commitment, which is exactly the 32 bytes the schema wants.
        let nonce: [u8; 32] = row
            .challenge_sig
            .get(..32)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| "challenge signature shorter than 32 bytes".to_owned())?;
        let owned = OwnedRawWeight {
            id: row.id,
            challenge_id: row.challenge_id.clone(),
            epoch,
            miner_hotkey: row.miner_hotkey.clone(),
            kind: row.kind.clone(),
            score,
            absence_reason: row.absence_reason.clone(),
            payload: row.payload.clone(),
            payload_digest: row.payload_digest,
            signature: row.challenge_sig.clone(),
            nonce,
        };
        let id = self
            .exec
            .run(move |pool| async move { db::insert_raw_weight(&pool, &owned.as_new()).await })?;
        Ok(id.is_some())
    }
}

/// Owned mirror of [`NewRawWeight`] so the insert future is `'static`.
struct OwnedRawWeight {
    id: Uuid,
    challenge_id: String,
    epoch: i64,
    miner_hotkey: String,
    kind: String,
    score: Option<i64>,
    absence_reason: Option<String>,
    payload: Vec<u8>,
    payload_digest: [u8; 32],
    signature: Vec<u8>,
    nonce: [u8; 32],
}

impl OwnedRawWeight {
    fn as_new(&self) -> NewRawWeight<'_> {
        NewRawWeight {
            id: self.id,
            challenge_id: &self.challenge_id,
            epoch: self.epoch,
            miner_hotkey: &self.miner_hotkey,
            kind: &self.kind,
            score: self.score,
            absence_reason: self.absence_reason.as_deref(),
            payload: &self.payload,
            payload_digest: &self.payload_digest,
            signature: &self.signature,
            nonce: &self.nonce,
        }
    }
}

impl BundleStore for PgBundleStore {
    fn put_if_absent(&self, epoch: u64, bytes: Vec<u8>) -> Vec<u8> {
        if let Some(existing) = self.get_by_epoch(epoch) {
            return existing;
        }
        match self.seal(epoch, &bytes) {
            Ok(()) => bytes,
            Err(e) => {
                // The trait cannot report this; the seal is only durable when
                // the insert lands, so surface it as loudly as possible here.
                tracing::error!(event = "bundle_persist_failed", epoch, error = %e);
                bytes
            }
        }
    }

    fn get_by_epoch(&self, epoch: u64) -> Option<Vec<u8>> {
        self.record_for_epoch(epoch).map(|rec| rec.payload)
    }

    fn get_by_root(&self, root: &[u8; 32]) -> Option<Vec<u8>> {
        let root = root.to_vec();
        let found = self
            .exec
            .run(move |pool| async move { db::get_epoch_bundle_by_root(&pool, &root).await });
        let rec = log_err("bundle_get_by_root", found)??;
        Some(rec.payload)
    }

    fn latest_epoch(&self) -> Option<u64> {
        let found = self
            .exec
            .run(move |pool| async move { db::latest_bundle_epoch(&pool).await });
        let epoch = log_err("bundle_latest_epoch", found)??;
        u64::try_from(epoch).ok()
    }

    fn seal_record(&self, epoch: u64) -> Option<SealRecord> {
        let rec = self.record_for_epoch(epoch)?;
        Some(SealRecord {
            sealed_at_micros: u64::try_from(rec.sealed_at_micros).unwrap_or(0),
            revision: u32::try_from(rec.revision).unwrap_or(1),
        })
    }
}

impl PgBundleStore {
    fn record_for_epoch(&self, epoch: u64) -> Option<EpochBundleRecord> {
        let epoch = log_err("bundle_get", epoch_i64(epoch))?;
        let found = self
            .exec
            .run(move |pool| async move { db::get_epoch_bundle(&pool, epoch).await });
        log_err("bundle_get", found)?
    }

    /// Append `bytes` as the next revision of `epoch`.
    fn seal(&self, epoch: u64, bytes: &[u8]) -> Result<(), String> {
        let epoch = epoch_i64(epoch)?;
        let decoded =
            EpochBundleV1::decode_bytes(bytes).map_err(|e| format!("bundle decode: {e}"))?;
        let body = decoded.body;
        // `epoch_bundle.vector_hash` mirrors `validator::vector_sha256`: the
        // digest of the SCALE-encoded final vector the bundle commits to.
        let vector_hash: [u8; 32] = Sha256::digest(body.final_vector.encode()).into();
        let owned = OwnedBundle {
            epoch,
            protocol_version: i32::from(body.protocol_version),
            block_number: i64::try_from(body.block_b)
                .map_err(|_| "block_b exceeds BIGINT".to_owned())?,
            block_hash: body.block_hash,
            metagraph_root: body.metagraph_root,
            merkle_root: body.merkle_root,
            measurements_digest: body.measurements_digest,
            vector_hash,
            payload: bytes.to_vec(),
            signature: decoded.gateway_sig.to_vec(),
        };
        self.exec.run(move |pool| async move {
            db::insert_epoch_bundle(&pool, &owned.as_new())
                .await
                .map(|rec: EpochBundleRecord| {
                    tracing::info!(
                        event = "bundle_persisted",
                        epoch = rec.epoch,
                        revision = rec.revision
                    );
                })
        })
    }
}

/// Owned mirror of [`NewEpochBundle`] so the insert future is `'static`.
struct OwnedBundle {
    epoch: i64,
    protocol_version: i32,
    block_number: i64,
    block_hash: [u8; 32],
    metagraph_root: [u8; 32],
    merkle_root: [u8; 32],
    measurements_digest: [u8; 32],
    vector_hash: [u8; 32],
    payload: Vec<u8>,
    signature: Vec<u8>,
}

impl OwnedBundle {
    fn as_new(&self) -> NewEpochBundle<'_> {
        NewEpochBundle {
            epoch: self.epoch,
            protocol_version: self.protocol_version,
            block_number: self.block_number,
            block_hash: &self.block_hash,
            metagraph_root: &self.metagraph_root,
            merkle_root: &self.merkle_root,
            measurements_digest: &self.measurements_digest,
            vector_hash: &self.vector_hash,
            payload: &self.payload,
            signature: &self.signature,
        }
    }
}

/// Log and swallow a failure the store trait has no way to return.
fn log_err<T>(event: &'static str, result: Result<T, String>) -> Option<T> {
    match result {
        Ok(value) => Some(value),
        Err(e) => {
            tracing::error!(event, error = %e, "gateway postgres store failure");
            None
        }
    }
}
