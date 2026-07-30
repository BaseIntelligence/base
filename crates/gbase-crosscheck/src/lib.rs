//! Peer root cross-check with a minimum sample (task 31 / D26).
//!
//! Sample metagraph peers via `GET /v1/consensus/root/{epoch}`. Each peer returns
//! a signed `(epoch, merkle_root)` under [`gbase_crypto::domain::ROOT`].
//! Identity is the metagraph hotkey — never IP. Fail closed on disagreement or
//! insufficient sample. Task 32 consumes [`CrossCheckOutcome`] for dissent.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use gbase_chain::{ChainClient, Metagraph};
use gbase_config::DEFAULT_MIN_PEER_SAMPLE;
use gbase_crypto::{domain, sign_raw, verify_raw, KEY_LEN, SIGNATURE_LEN};
use parity_scale_codec::Encode;
use serde::{Deserialize, Serialize};

/// Default minimum agreeing peer sample (D26).
pub const DEFAULT_MIN_SAMPLE: u32 = DEFAULT_MIN_PEER_SAMPLE;

/// SCALE payload under [`domain::ROOT`]: `scale(epoch) ‖ scale(merkle_root)`.
#[must_use]
pub fn root_statement_payload(epoch: u64, merkle_root: &[u8; 32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(40);
    epoch.encode_to(&mut out);
    merkle_root.encode_to(&mut out);
    out
}

/// One signed peer root observation (evidence for task 32).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedRootStatement {
    pub epoch: u64,
    pub merkle_root: [u8; 32],
    pub hotkey: [u8; 32],
    pub signature: [u8; SIGNATURE_LEN],
}

impl SignedRootStatement {
    /// Sign `(epoch, merkle_root)` with a 32-byte mini-secret.
    ///
    /// # Errors
    ///
    /// Invalid secret key.
    pub fn sign(
        secret: &[u8; KEY_LEN],
        hotkey: [u8; KEY_LEN],
        epoch: u64,
        merkle_root: [u8; 32],
    ) -> Result<Self, gbase_crypto::CryptoError> {
        let payload = root_statement_payload(epoch, &merkle_root);
        let signature = sign_raw(secret, domain::ROOT, &payload)?;
        Ok(Self {
            epoch,
            merkle_root,
            hotkey,
            signature,
        })
    }

    /// Verify under [`domain::ROOT`] for `self.hotkey`.
    ///
    /// # Errors
    ///
    /// Crypto verification failures.
    pub fn verify(&self) -> Result<(), gbase_crypto::CryptoError> {
        let payload = root_statement_payload(self.epoch, &self.merkle_root);
        verify_raw(&self.hotkey, domain::ROOT, &payload, &self.signature)
    }
}

/// Wire JSON for `GET /v1/consensus/root/{epoch}` (hex fields).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootStatementJson {
    pub epoch: u64,
    pub merkle_root: String,
    pub hotkey: String,
    pub signature: String,
}

impl RootStatementJson {
    #[must_use]
    pub fn from_statement(s: &SignedRootStatement) -> Self {
        Self {
            epoch: s.epoch,
            merkle_root: hex::encode(s.merkle_root),
            hotkey: hex::encode(s.hotkey),
            signature: hex::encode(s.signature),
        }
    }

    /// # Errors
    ///
    /// Hex / length errors.
    pub fn into_statement(self) -> Result<SignedRootStatement, String> {
        Ok(SignedRootStatement {
            epoch: self.epoch,
            merkle_root: parse_hex32(&self.merkle_root, "merkle_root")?,
            hotkey: parse_hex32(&self.hotkey, "hotkey")?,
            signature: parse_hex64(&self.signature)?,
        })
    }
}

fn parse_hex32(s: &str, field: &str) -> Result<[u8; 32], String> {
    let s = strip0x(s);
    let bytes = hex::decode(s).map_err(|e| format!("{field}: {e}"))?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| format!("{field}: expected 32 bytes, got {}", v.len()))
}

fn parse_hex64(s: &str) -> Result<[u8; SIGNATURE_LEN], String> {
    let s = strip0x(s);
    let bytes = hex::decode(s).map_err(|e| format!("signature: {e}"))?;
    bytes
        .try_into()
        .map_err(|v: Vec<u8>| format!("signature: expected {SIGNATURE_LEN} bytes, got {}", v.len()))
}

fn strip0x(s: &str) -> &str {
    let s = s.trim();
    s.strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s)
}

/// In-memory local roots + evidence.
#[derive(Debug, Default)]
pub struct RootStatementStore {
    local_by_epoch: RwLock<BTreeMap<u64, SignedRootStatement>>,
    evidence: RwLock<Vec<SignedRootStatement>>,
}

pub type SharedRootStore = Arc<RootStatementStore>;

impl RootStatementStore {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn shared() -> SharedRootStore {
        Arc::new(Self::new())
    }

    pub fn put_local(&self, statement: SignedRootStatement) {
        {
            let mut map = self
                .local_by_epoch
                .write()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            map.insert(statement.epoch, statement.clone());
        }
        self.push_evidence(statement);
    }

    pub fn push_evidence(&self, statement: SignedRootStatement) {
        let mut ev = self
            .evidence
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !ev.iter().any(|e| e == &statement) {
            ev.push(statement);
        }
    }

    #[must_use]
    pub fn get_local(&self, epoch: u64) -> Option<SignedRootStatement> {
        self.local_by_epoch
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&epoch)
            .cloned()
    }

    #[must_use]
    pub fn evidence(&self) -> Vec<SignedRootStatement> {
        self.evidence
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    #[must_use]
    pub fn evidence_for_epoch(&self, epoch: u64) -> Vec<SignedRootStatement> {
        self.evidence()
            .into_iter()
            .filter(|s| s.epoch == epoch)
            .collect()
    }
}

/// `GET /v1/consensus/root/{epoch}`.
pub fn consensus_root_router(store: SharedRootStore) -> Router {
    Router::new()
        .route("/v1/consensus/root/{epoch}", get(get_consensus_root))
        .with_state(store)
}

async fn get_consensus_root(
    State(store): State<SharedRootStore>,
    Path(epoch): Path<u64>,
) -> Response {
    match store.get_local(epoch) {
        Some(stmt) => (
            StatusCode::OK,
            Json(RootStatementJson::from_statement(&stmt)),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "no root statement for epoch" })),
        )
            .into_response(),
    }
}

/// Failure kinds for task 32 dissent hooks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossCheckFailKind {
    /// Below `min_peer_sample` while other validators exist (`Degraded`).
    InsufficientPeers {
        reachable: usize,
        required: u32,
        others_on_metagraph: usize,
    },
    /// Verified peers disagree on root (class B / `CrossCheckFailed`).
    RootDisagreement { candidate: [u8; 32] },
}

/// Cross-check result. Task 32 maps [`Self::Failed`] to dissent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossCheckOutcome {
    Agreed {
        epoch: u64,
        merkle_root: [u8; 32],
        sample_size: usize,
        statements: Vec<SignedRootStatement>,
    },
    /// No other validators on metagraph — waived (D26).
    SingleValidatorExempt { epoch: u64, merkle_root: [u8; 32] },
    Failed {
        epoch: u64,
        kind: CrossCheckFailKind,
        statements: Vec<SignedRootStatement>,
    },
}

impl CrossCheckOutcome {
    #[must_use]
    pub const fn allows_submission(&self) -> bool {
        matches!(
            self,
            Self::Agreed { .. } | Self::SingleValidatorExempt { .. }
        )
    }

    #[must_use]
    pub const fn is_class_b_hook(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    #[must_use]
    pub fn is_cross_check_failed(&self) -> bool {
        matches!(
            self,
            Self::Failed {
                kind: CrossCheckFailKind::RootDisagreement { .. },
                ..
            }
        )
    }
}

/// Cross-check knobs.
#[derive(Debug, Clone)]
pub struct CrossCheckConfig {
    pub min_peer_sample: u32,
    pub own_hotkey: Option<Vec<u8>>,
}

impl Default for CrossCheckConfig {
    fn default() -> Self {
        Self {
            min_peer_sample: DEFAULT_MIN_SAMPLE,
            own_hotkey: None,
        }
    }
}

/// Peer HTTP endpoint (hotkey-bound).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerRef {
    pub hotkey: Vec<u8>,
    pub base_url: String,
}

/// Count other metagraph hotkeys excluding self (all treated as permitted).
#[must_use]
pub fn other_validators_on_metagraph(metagraph: &Metagraph, own_hotkey: Option<&[u8]>) -> usize {
    metagraph
        .hotkeys
        .iter()
        .filter(|h| own_hotkey.is_none_or(|own| h.as_slice() != own))
        .count()
}

/// Pure sample evaluation (no I/O).
#[must_use]
pub fn evaluate_sample(
    epoch: u64,
    candidate: [u8; 32],
    peer_statements: &[SignedRootStatement],
    others_on_metagraph: usize,
    min_peer_sample: u32,
) -> CrossCheckOutcome {
    if others_on_metagraph == 0 {
        return CrossCheckOutcome::SingleValidatorExempt {
            epoch,
            merkle_root: candidate,
        };
    }
    let statements = peer_statements.to_vec();
    let agreeing = peer_statements
        .iter()
        .filter(|s| s.epoch == epoch && s.merkle_root == candidate)
        .count();
    let has_disagree = peer_statements
        .iter()
        .any(|s| s.epoch == epoch && s.merkle_root != candidate);
    if has_disagree {
        return CrossCheckOutcome::Failed {
            epoch,
            kind: CrossCheckFailKind::RootDisagreement { candidate },
            statements,
        };
    }
    if agreeing < min_peer_sample as usize {
        return CrossCheckOutcome::Failed {
            epoch,
            kind: CrossCheckFailKind::InsufficientPeers {
                reachable: agreeing,
                required: min_peer_sample,
                others_on_metagraph,
            },
            statements,
        };
    }
    CrossCheckOutcome::Agreed {
        epoch,
        merkle_root: candidate,
        sample_size: agreeing,
        statements,
    }
}

/// Fetch + verify peer roots, then [`evaluate_sample`].
pub async fn crosscheck_peer_roots<C: ChainClient>(
    http: &reqwest::Client,
    chain: &C,
    peers: &[PeerRef],
    store: &SharedRootStore,
    cfg: &CrossCheckConfig,
    epoch: u64,
    candidate: [u8; 32],
) -> CrossCheckOutcome {
    let metagraph = match load_metagraph(chain) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(error = %e, "crosscheck: metagraph load failed");
            return CrossCheckOutcome::Failed {
                epoch,
                kind: CrossCheckFailKind::InsufficientPeers {
                    reachable: 0,
                    required: cfg.min_peer_sample,
                    others_on_metagraph: 0,
                },
                statements: vec![],
            };
        }
    };
    let own = cfg.own_hotkey.as_deref();
    let others = other_validators_on_metagraph(&metagraph, own);
    if others == 0 {
        return CrossCheckOutcome::SingleValidatorExempt {
            epoch,
            merkle_root: candidate,
        };
    }

    let mut verified = Vec::new();
    for peer in peers {
        if own.is_some_and(|o| peer.hotkey.as_slice() == o) {
            continue;
        }
        if !metagraph.hotkeys.iter().any(|h| h == &peer.hotkey) {
            continue;
        }
        let Ok(stmt) = fetch_peer_root(http, &peer.base_url, epoch).await else {
            continue;
        };
        if stmt.epoch != epoch {
            continue;
        }
        let Ok(expected_hk) = <[u8; 32]>::try_from(peer.hotkey.as_slice()) else {
            continue;
        };
        if stmt.hotkey != expected_hk || stmt.verify().is_err() {
            continue;
        }
        store.push_evidence(stmt.clone());
        verified.push(stmt);
    }

    evaluate_sample(epoch, candidate, &verified, others, cfg.min_peer_sample)
}

fn load_metagraph<C: ChainClient>(chain: &C) -> Result<Metagraph, String> {
    let tip = chain.current_block().map_err(|e| e.to_string())?;
    let hash = chain.block_hash(tip).map_err(|e| e.to_string())?;
    chain.metagraph_at(&hash).map_err(|e| e.to_string())
}

/// HTTP fetch of a peer root statement.
///
/// # Errors
///
/// Transport, HTTP status, or JSON/hex parse failures.
pub async fn fetch_peer_root(
    http: &reqwest::Client,
    base_url: &str,
    epoch: u64,
) -> Result<SignedRootStatement, String> {
    let base = base_url.trim().trim_end_matches('/');
    if base.is_empty() {
        return Err("empty peer base url".into());
    }
    let url = format!("{base}/v1/consensus/root/{epoch}");
    let resp = http.get(&url).send().await.map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status().as_u16()));
    }
    let body: RootStatementJson = resp.json().await.map_err(|e| e.to_string())?;
    body.into_statement()
}

/// Sign and store a local root.
///
/// # Errors
///
/// Signing failures.
pub fn record_local_root(
    store: &SharedRootStore,
    secret: &[u8; KEY_LEN],
    hotkey: [u8; KEY_LEN],
    epoch: u64,
    merkle_root: [u8; 32],
) -> Result<SignedRootStatement, gbase_crypto::CryptoError> {
    let stmt = SignedRootStatement::sign(secret, hotkey, epoch, merkle_root)?;
    store.put_local(stmt.clone());
    Ok(stmt)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
mod tests {
    use super::*;
    use gbase_crypto::secret_from_bytes;
    use sha2::{Digest, Sha256};

    fn sk(tag: u8) -> [u8; 32] {
        let dig = Sha256::digest([0xCC, tag, 0x33, tag]);
        let mut s = [0u8; 32];
        s.copy_from_slice(&dig);
        s
    }

    fn pk(secret: &[u8; 32]) -> [u8; 32] {
        secret_from_bytes(secret).unwrap().to_public().to_bytes()
    }

    #[test]
    fn sign_verify_roundtrip_domain_root() {
        let secret = sk(1);
        let hotkey = pk(&secret);
        let root = [0xABu8; 32];
        let stmt = SignedRootStatement::sign(&secret, hotkey, 7, root).unwrap();
        stmt.verify().unwrap();
        let mut bad = stmt.clone();
        bad.merkle_root[0] ^= 1;
        assert!(bad.verify().is_err());
    }

    #[test]
    fn evaluate_unanimous_pass() {
        let root = [1u8; 32];
        let sa = SignedRootStatement::sign(&sk(2), pk(&sk(2)), 9, root).unwrap();
        let sb = SignedRootStatement::sign(&sk(3), pk(&sk(3)), 9, root).unwrap();
        let out = evaluate_sample(9, root, &[sa, sb], 2, 1);
        assert!(matches!(
            out,
            CrossCheckOutcome::Agreed { sample_size: 2, .. }
        ));
        assert!(out.allows_submission());
    }

    #[test]
    fn evaluate_minority_disagree_fail_closed() {
        let root = [1u8; 32];
        let other = [2u8; 32];
        let sa = SignedRootStatement::sign(&sk(2), pk(&sk(2)), 9, root).unwrap();
        let sb = SignedRootStatement::sign(&sk(3), pk(&sk(3)), 9, other).unwrap();
        let out = evaluate_sample(9, root, &[sa, sb], 2, 1);
        assert!(out.is_cross_check_failed());
        assert!(!out.allows_submission());
    }

    #[test]
    fn evaluate_insufficient_and_single() {
        let out = evaluate_sample(5, [9u8; 32], &[], 3, 1);
        assert!(!out.allows_submission());
        let out2 = evaluate_sample(5, [9u8; 32], &[], 0, 1);
        assert!(matches!(
            out2,
            CrossCheckOutcome::SingleValidatorExempt { .. }
        ));
    }

    #[test]
    fn json_roundtrip() {
        let secret = sk(4);
        let stmt = SignedRootStatement::sign(&secret, pk(&secret), 3, [7u8; 32]).unwrap();
        let back = RootStatementJson::from_statement(&stmt)
            .into_statement()
            .unwrap();
        assert_eq!(back, stmt);
        back.verify().unwrap();
    }
}
