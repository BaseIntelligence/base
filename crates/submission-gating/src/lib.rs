//! Cross-challenge submission gating: metagraph membership at intake, one
//! accepted submission per `(challenge, hotkey)`, auto-retry budget accounting,
//! and a metagraph watcher that reopens eligibility when a hotkey leaves the
//! metagraph (uid deregistered or hotkey replaced).
//!
//! States per `(challenge, hotkey)`:
//!
//! ```text
//! open ──intake accept──▶ registered ──terminal retry-exhausted──▶ blocked
//!                        └─cheat / copy gate─────────────────────▶ rejected
//! blocked|rejected ──watcher: hotkey gone from metagraph──▶ open
//! ```
//!
//! Non-`open` rows make intake fail with an explicit 409; only the watcher
//! (or an out-of-band operator) returns a row to `open`.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use async_trait::async_trait;
use chain::ChainClient;
use thiserror::Error;

/// Gating lifecycle for one `(challenge, hotkey)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatingState {
    /// Eligible to submit.
    Open,
    /// Submission accepted; further intakes conflict (1-max).
    Registered,
    /// Retry budget exhausted on infra-class errors (terminal).
    Blocked,
    /// Cheat / copy-gate terminal reject (no retry).
    Rejected,
}

impl GatingState {
    /// DB string.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Registered => "registered",
            Self::Blocked => "blocked",
            Self::Rejected => "rejected",
        }
    }

    /// Parse DB string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "open" => Some(Self::Open),
            "registered" => Some(Self::Registered),
            "blocked" => Some(Self::Blocked),
            "rejected" => Some(Self::Rejected),
            _ => None,
        }
    }
}

/// One `submission_gating` row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatingRow {
    /// Challenge id (`design` / `prism`).
    pub challenge: String,
    /// Lowercase hex miner hotkey.
    pub hotkey: String,
    /// UID recorded at registration (advisory).
    pub uid: Option<u32>,
    /// Lifecycle state.
    pub state: GatingState,
    /// Auto-retry attempts consumed (infra classes).
    pub attempt_count: u32,
    /// Last classified error (`install` / `ast_infra` / `llm_infra` / `miner`).
    pub last_error_class: Option<String>,
    /// Created unix ms (0 when unknown).
    pub created_at_ms: u64,
    /// Updated unix ms (0 when unknown).
    pub updated_at_ms: u64,
}

/// Store failures.
#[derive(Debug, Error)]
pub enum GatingError {
    /// Backend (SQL / poison).
    #[error("gating backend: {0}")]
    Backend(String),
}

/// Persistence contract (memory + Postgres impls).
#[async_trait]
pub trait GatingStore: Send + Sync + std::fmt::Debug {
    /// Load one row.
    async fn get(&self, challenge: &str, hotkey: &str) -> Result<Option<GatingRow>, GatingError>;
    /// Intake accept: transition to `registered`, recording the metagraph uid.
    async fn mark_registered(
        &self,
        challenge: &str,
        hotkey: &str,
        uid: Option<u32>,
    ) -> Result<(), GatingError>;
    /// Terminal transition (`blocked` / `rejected`) with the last error class.
    async fn set_terminal(
        &self,
        challenge: &str,
        hotkey: &str,
        state: GatingState,
        error_class: Option<&str>,
    ) -> Result<(), GatingError>;
    /// Record one auto-retry attempt; returns the new attempt count.
    async fn bump_attempt(
        &self,
        challenge: &str,
        hotkey: &str,
        error_class: &str,
    ) -> Result<u32, GatingError>;
    /// Watcher: return to `open` and clear the retry budget.
    async fn reset_open(&self, challenge: &str, hotkey: &str) -> Result<(), GatingError>;
    /// All rows not in `open` state (watcher reconciliation input).
    async fn list_non_open(&self, challenge: &str) -> Result<Vec<GatingRow>, GatingError>;
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// In-memory store (tests / no-DB local mode).
#[derive(Debug, Default)]
pub struct MemoryGatingStore {
    rows: Mutex<HashMap<(String, String), GatingRow>>,
}

impl MemoryGatingStore {
    /// Empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl GatingStore for MemoryGatingStore {
    async fn get(&self, challenge: &str, hotkey: &str) -> Result<Option<GatingRow>, GatingError> {
        Ok(self
            .rows
            .lock()
            .map_err(|_| GatingError::Backend("poison".into()))?
            .get(&(challenge.to_owned(), hotkey.to_owned()))
            .cloned())
    }

    async fn mark_registered(
        &self,
        challenge: &str,
        hotkey: &str,
        uid: Option<u32>,
    ) -> Result<(), GatingError> {
        let mut m = self
            .rows
            .lock()
            .map_err(|_| GatingError::Backend("poison".into()))?;
        let key = (challenge.to_owned(), hotkey.to_owned());
        let now = now_ms();
        let row = m.entry(key).or_insert_with(|| GatingRow {
            challenge: challenge.to_owned(),
            hotkey: hotkey.to_owned(),
            uid: None,
            state: GatingState::Open,
            attempt_count: 0,
            last_error_class: None,
            created_at_ms: now,
            updated_at_ms: now,
        });
        row.state = GatingState::Registered;
        row.uid = uid;
        row.updated_at_ms = now;
        Ok(())
    }

    async fn set_terminal(
        &self,
        challenge: &str,
        hotkey: &str,
        state: GatingState,
        error_class: Option<&str>,
    ) -> Result<(), GatingError> {
        debug_assert!(matches!(
            state,
            GatingState::Blocked | GatingState::Rejected
        ));
        let mut m = self
            .rows
            .lock()
            .map_err(|_| GatingError::Backend("poison".into()))?;
        if let Some(row) = m.get_mut(&(challenge.to_owned(), hotkey.to_owned())) {
            row.state = state;
            if let Some(c) = error_class {
                row.last_error_class = Some(c.to_owned());
            }
            row.updated_at_ms = now_ms();
        }
        Ok(())
    }

    async fn bump_attempt(
        &self,
        challenge: &str,
        hotkey: &str,
        error_class: &str,
    ) -> Result<u32, GatingError> {
        let mut m = self
            .rows
            .lock()
            .map_err(|_| GatingError::Backend("poison".into()))?;
        let row = m
            .get_mut(&(challenge.to_owned(), hotkey.to_owned()))
            .ok_or_else(|| GatingError::Backend("gating row missing".into()))?;
        row.attempt_count = row.attempt_count.saturating_add(1);
        row.last_error_class = Some(error_class.to_owned());
        row.updated_at_ms = now_ms();
        Ok(row.attempt_count)
    }

    async fn reset_open(&self, challenge: &str, hotkey: &str) -> Result<(), GatingError> {
        let mut m = self
            .rows
            .lock()
            .map_err(|_| GatingError::Backend("poison".into()))?;
        if let Some(row) = m.get_mut(&(challenge.to_owned(), hotkey.to_owned())) {
            row.state = GatingState::Open;
            row.attempt_count = 0;
            row.last_error_class = None;
            row.updated_at_ms = now_ms();
        }
        Ok(())
    }

    async fn list_non_open(&self, challenge: &str) -> Result<Vec<GatingRow>, GatingError> {
        Ok(self
            .rows
            .lock()
            .map_err(|_| GatingError::Backend("poison".into()))?
            .values()
            .filter(|r| challenge_matches(&r.challenge, challenge) && r.state != GatingState::Open)
            .cloned()
            .collect())
    }
}

/// Reconciliation is prefix-scoped: `prism` also matches composite keys like
/// `prism:train:<arch_id>` so watcher resets cover training-only entries.
fn challenge_matches(row_challenge: &str, query: &str) -> bool {
    row_challenge == query || row_challenge.starts_with(&format!("{query}:"))
}

/// Postgres-backed store over the shared `submission_gating` table (migration
/// 0008). Runtime SQL only — no compile-time DB needed.
#[derive(Debug, Clone)]
pub struct PgGatingStore {
    pool: sqlx::PgPool,
}

impl PgGatingStore {
    /// From an existing pool (same DB as the challenge stores).
    #[must_use]
    pub fn new(pool: sqlx::PgPool) -> Self {
        Self { pool }
    }
}

const GATING_COLS: &str = "challenge, hotkey, uid, state, attempt_count, last_error_class, \
     (EXTRACT(EPOCH FROM created_at)::BIGINT * 1000) AS created_at_ms, \
     (EXTRACT(EPOCH FROM updated_at)::BIGINT * 1000) AS updated_at_ms";

fn row_from(r: &sqlx::postgres::PgRow) -> GatingRow {
    use sqlx::Row as _;
    GatingRow {
        challenge: r.get::<String, _>("challenge"),
        hotkey: r.get::<String, _>("hotkey"),
        uid: r
            .get::<Option<i32>, _>("uid")
            .and_then(|u| u32::try_from(u).ok()),
        state: GatingState::parse(&r.get::<String, _>("state")).unwrap_or(GatingState::Open),
        attempt_count: u32::try_from(r.get::<i32, _>("attempt_count").max(0)).unwrap_or(0),
        last_error_class: r.get::<Option<String>, _>("last_error_class"),
        created_at_ms: r.get::<i64, _>("created_at_ms").max(0).cast_unsigned(),
        updated_at_ms: r.get::<i64, _>("updated_at_ms").max(0).cast_unsigned(),
    }
}

#[allow(clippy::needless_pass_by_value)] // map_err passes the error by value
fn map_sqlx(e: sqlx::Error) -> GatingError {
    GatingError::Backend(e.to_string())
}

#[async_trait]
impl GatingStore for PgGatingStore {
    async fn get(&self, challenge: &str, hotkey: &str) -> Result<Option<GatingRow>, GatingError> {
        let q = format!(
            "SELECT {GATING_COLS} FROM submission_gating WHERE challenge = $1 AND hotkey = $2"
        );
        let row = sqlx::query(&q)
            .bind(challenge)
            .bind(hotkey)
            .fetch_optional(&self.pool)
            .await
            .map_err(map_sqlx)?;
        Ok(row.as_ref().map(row_from))
    }

    async fn mark_registered(
        &self,
        challenge: &str,
        hotkey: &str,
        uid: Option<u32>,
    ) -> Result<(), GatingError> {
        sqlx::query(
            "INSERT INTO submission_gating (challenge, hotkey, uid, state) \
             VALUES ($1, $2, $3, 'registered') \
             ON CONFLICT (challenge, hotkey) DO UPDATE SET \
               state = 'registered', uid = EXCLUDED.uid, updated_at = now()",
        )
        .bind(challenge)
        .bind(hotkey)
        .bind(uid.map(|u| i32::try_from(u).unwrap_or(i32::MAX)))
        .execute(&self.pool)
        .await
        .map_err(map_sqlx)?;
        Ok(())
    }

    async fn set_terminal(
        &self,
        challenge: &str,
        hotkey: &str,
        state: GatingState,
        error_class: Option<&str>,
    ) -> Result<(), GatingError> {
        debug_assert!(matches!(
            state,
            GatingState::Blocked | GatingState::Rejected
        ));
        sqlx::query(
            "UPDATE submission_gating SET state = $3, \
               last_error_class = COALESCE($4, last_error_class), updated_at = now() \
             WHERE challenge = $1 AND hotkey = $2",
        )
        .bind(challenge)
        .bind(hotkey)
        .bind(state.as_str())
        .bind(error_class)
        .execute(&self.pool)
        .await
        .map_err(map_sqlx)?;
        Ok(())
    }

    async fn bump_attempt(
        &self,
        challenge: &str,
        hotkey: &str,
        error_class: &str,
    ) -> Result<u32, GatingError> {
        use sqlx::Row as _;
        let row = sqlx::query(
            "UPDATE submission_gating SET attempt_count = attempt_count + 1, \
               last_error_class = $3, updated_at = now() \
             WHERE challenge = $1 AND hotkey = $2 RETURNING attempt_count",
        )
        .bind(challenge)
        .bind(hotkey)
        .bind(error_class)
        .fetch_one(&self.pool)
        .await
        .map_err(map_sqlx)?;
        Ok(u32::try_from(row.get::<i32, _>("attempt_count").max(0)).unwrap_or(0))
    }

    async fn reset_open(&self, challenge: &str, hotkey: &str) -> Result<(), GatingError> {
        sqlx::query(
            "UPDATE submission_gating SET state = 'open', attempt_count = 0, \
               last_error_class = NULL, updated_at = now() \
             WHERE challenge = $1 AND hotkey = $2",
        )
        .bind(challenge)
        .bind(hotkey)
        .execute(&self.pool)
        .await
        .map_err(map_sqlx)?;
        Ok(())
    }

    async fn list_non_open(&self, challenge: &str) -> Result<Vec<GatingRow>, GatingError> {
        let q = format!(
            "SELECT {GATING_COLS} FROM submission_gating \
             WHERE (challenge = $1 OR challenge LIKE $1 || ':%') AND state <> 'open' \
             ORDER BY updated_at ASC"
        );
        let rows = sqlx::query(&q)
            .bind(challenge)
            .fetch_all(&self.pool)
            .await
            .map_err(map_sqlx)?;
        Ok(rows.iter().map(row_from).collect())
    }
}

/// Point-in-time metagraph view (UID = position in `hotkeys`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetagraphView {
    /// Subnet netuid.
    pub netuid: u16,
    /// Fetch time (unix secs).
    pub fetched_at_secs: u64,
    /// Hotkeys in UID order.
    pub hotkeys: Vec<[u8; 32]>,
    /// Coldkeys UID-aligned with [`Self::hotkeys`] (`SubtensorModule.Owner`).
    /// All-zero entries mean unknown / chain default.
    pub coldkeys: Vec<[u8; 32]>,
}

impl MetagraphView {
    /// UID of a lowercase-hex hotkey, when present.
    #[must_use]
    pub fn uid_of_hex(&self, hotkey_hex: &str) -> Option<u32> {
        let raw = hex::decode(hotkey_hex).ok()?;
        let key: [u8; 32] = raw.try_into().ok()?;
        self.hotkeys
            .iter()
            .position(|h| h == &key)
            .and_then(|i| u32::try_from(i).ok())
    }

    /// Whether a lowercase-hex hotkey is in the metagraph.
    #[must_use]
    pub fn contains_hex(&self, hotkey_hex: &str) -> bool {
        self.uid_of_hex(hotkey_hex).is_some()
    }

    /// Lowercase-hex coldkey for a hotkey, when the Owner entry is non-zero.
    #[must_use]
    pub fn coldkey_hex_of(&self, hotkey_hex: &str) -> Option<String> {
        let uid = self.uid_of_hex(hotkey_hex)? as usize;
        let ck = self.coldkeys.get(uid)?;
        if ck.iter().all(|&b| b == 0) {
            return None;
        }
        Some(hex::encode(ck))
    }
}

/// Bulk metagraph snapshot max age (15m) before intake fail-closes.
pub const METAGRAPH_CACHE_TTL_SECS: u64 = 900;

/// Cached metagraph snapshot refreshed by the watcher; intake reads only this
/// cache (no chain call on the request path).
#[derive(Debug, Default)]
pub struct MetagraphCache {
    inner: RwLock<Option<MetagraphView>>,
}

impl MetagraphCache {
    /// Empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Install a fresh snapshot from raw (possibly non-32-byte) hotkeys.
    ///
    /// `coldkeys` should be UID-aligned with `hotkeys` when provided; shorter
    /// / empty slices leave unknown (zero) coldkeys for missing UIDs.
    pub fn update(&self, netuid: u16, hotkeys: &[Vec<u8>], coldkeys: &[Vec<u8>]) {
        let keys: Vec<[u8; 32]> = hotkeys
            .iter()
            .filter_map(|h| <[u8; 32]>::try_from(h.as_slice()).ok())
            .collect();
        let mut cks: Vec<[u8; 32]> = coldkeys
            .iter()
            .take(keys.len())
            .map(|c| <[u8; 32]>::try_from(c.as_slice()).unwrap_or([0; 32]))
            .collect();
        cks.resize(keys.len(), [0; 32]);
        let view = MetagraphView {
            netuid,
            fetched_at_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            hotkeys: keys,
            coldkeys: cks,
        };
        if let Ok(mut g) = self.inner.write() {
            *g = Some(view);
        }
    }

    /// Latest snapshot, when any refresh ever succeeded.
    #[must_use]
    pub fn snapshot(&self) -> Option<MetagraphView> {
        self.inner.read().ok().and_then(|g| g.clone())
    }

    /// Snapshot if younger than `max_age_secs`; else `None`.
    #[must_use]
    pub fn snapshot_fresh(&self, max_age_secs: u64) -> Option<MetagraphView> {
        let view = self.snapshot()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        if now.saturating_sub(view.fetched_at_secs) > max_age_secs {
            return None;
        }
        Some(view)
    }
}

/// Reopen every non-open gating row whose hotkey is absent from `view`.
///
/// A hotkey that left the metagraph (uid deregistered or replaced by a new
/// hotkey) regains eligibility; its owner may resubmit under a new uid.
/// Returns the reset hotkeys.
pub async fn reconcile_metagraph(
    store: &dyn GatingStore,
    challenge: &str,
    view: &MetagraphView,
) -> Result<Vec<String>, GatingError> {
    let mut reset = Vec::new();
    for row in store.list_non_open(challenge).await? {
        if !view.contains_hex(&row.hotkey) {
            store.reset_open(challenge, &row.hotkey).await?;
            reset.push(row.hotkey);
        }
    }
    Ok(reset)
}

/// One watcher tick: refresh the metagraph snapshot into `cache`, then
/// reconcile every `(challenge, store)` pair. Returns total rows reset.
///
/// On chain read failure the cache keeps its last-good snapshot and no
/// reconciliation happens (fail-closed: eligibility unchanged).
pub async fn watch_once<C: ChainClient + Send>(
    chain: &C,
    netuid: u16,
    cache: &MetagraphCache,
    stores: &[(String, Arc<dyn GatingStore>)],
) -> Result<usize, GatingError> {
    let tip = chain
        .current_block()
        .map_err(|e| GatingError::Backend(format!("current_block: {e}")))?;
    let hash = chain
        .block_hash(tip)
        .map_err(|e| GatingError::Backend(format!("block_hash: {e}")))?;
    let mg = chain
        .metagraph_at(&hash)
        .map_err(|e| GatingError::Backend(format!("metagraph: {e}")))?;
    cache.update(netuid, &mg.hotkeys, &mg.coldkeys);
    let Some(view) = cache.snapshot() else {
        return Ok(0);
    };
    let mut resets = 0usize;
    for (challenge, store) in stores {
        let r = reconcile_metagraph(store.as_ref(), challenge, &view).await?;
        if !r.is_empty() {
            tracing::info!(challenge = %challenge, count = r.len(), "gating eligibility reset");
        }
        resets = resets.saturating_add(r.len());
    }
    Ok(resets)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn hk(byte: u8) -> String {
        hex::encode([byte; 32])
    }

    #[tokio::test]
    async fn lifecycle_register_block_reset() {
        let s = MemoryGatingStore::new();
        assert!(s.get("design", &hk(1)).await.unwrap().is_none());
        s.mark_registered("design", &hk(1), Some(7)).await.unwrap();
        let row = s.get("design", &hk(1)).await.unwrap().unwrap();
        assert_eq!(row.state, GatingState::Registered);
        assert_eq!(row.uid, Some(7));
        assert_eq!(
            s.bump_attempt("design", &hk(1), "install").await.unwrap(),
            1
        );
        s.set_terminal("design", &hk(1), GatingState::Blocked, Some("install"))
            .await
            .unwrap();
        let row = s.get("design", &hk(1)).await.unwrap().unwrap();
        assert_eq!(row.state, GatingState::Blocked);
        assert_eq!(row.last_error_class.as_deref(), Some("install"));
        assert_eq!(s.list_non_open("design").await.unwrap().len(), 1);
        s.reset_open("design", &hk(1)).await.unwrap();
        let row = s.get("design", &hk(1)).await.unwrap().unwrap();
        assert_eq!(row.state, GatingState::Open);
        assert_eq!(row.attempt_count, 0);
        assert!(s.list_non_open("design").await.unwrap().is_empty());
    }

    #[test]
    fn metagraph_view_lookup() {
        let view = MetagraphView {
            netuid: 541,
            fetched_at_secs: 0,
            hotkeys: vec![[0xAA; 32], [0xBB; 32]],
            coldkeys: vec![[0x11; 32], [0x22; 32]],
        };
        assert_eq!(view.uid_of_hex(&hk(0xAA)), Some(0));
        assert_eq!(view.uid_of_hex(&hk(0xBB)), Some(1));
        assert_eq!(view.coldkey_hex_of(&hk(0xAA)), Some(hk(0x11)));
        assert!(!view.contains_hex(&hk(0xCC)));
        assert!(!view.contains_hex("not-hex"));
    }

    #[tokio::test]
    async fn reconcile_resets_only_departed_hotkeys() {
        let s = MemoryGatingStore::new();
        s.mark_registered("design", &hk(0xAA), Some(0))
            .await
            .unwrap();
        s.mark_registered("design", &hk(0xBB), Some(1))
            .await
            .unwrap();
        let view = MetagraphView {
            netuid: 541,
            fetched_at_secs: 0,
            hotkeys: vec![[0xAA; 32]], // 0xBB deregistered / replaced
            coldkeys: vec![[0xAA; 32]],
        };
        let reset = reconcile_metagraph(&s, "design", &view).await.unwrap();
        assert_eq!(reset, vec![hk(0xBB)]);
        assert_eq!(
            s.get("design", &hk(0xBB)).await.unwrap().unwrap().state,
            GatingState::Open
        );
        assert_eq!(
            s.get("design", &hk(0xAA)).await.unwrap().unwrap().state,
            GatingState::Registered
        );
    }

    #[tokio::test]
    async fn watch_once_updates_cache_and_reconciles() {
        let chain = chain::FakeChain::new(chain::FakeChainConfig::default());
        let cache = MetagraphCache::new();
        let store: Arc<dyn GatingStore> = Arc::new(MemoryGatingStore::new());
        // FakeChain default hotkeys: A1, B2, C3 at uids 0..2.
        let stale = hk(0xA1);
        store
            .mark_registered("design", &stale, Some(0))
            .await
            .unwrap();
        let stores = vec![("design".to_owned(), Arc::clone(&store))];
        let resets = watch_once(&chain, 1, &cache, &stores).await.unwrap();
        assert_eq!(resets, 0, "A1 still in metagraph");
        let snap = cache.snapshot().unwrap();
        assert!(snap.contains_hex(&stale));
    }
}
