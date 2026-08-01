//! Announced miner endpoint registry (`AGENT_CHALLENGE` §9.3 step 5).
//!
//! A miner publishes its public CVM base URL by signing a request to our own
//! gateway, which persists the announcement to the control plane. Dispatch
//! reads it back from there. Chain axons are deliberately not consulted: the
//! subnet does not require miners to serve one, and an axon says nothing about
//! the CVM the pack actually has to run in.
//!
//! Like the attestation channel (§3.3), this reader can only ever withhold
//! work: an unresolvable endpoint is an absence, never a fabricated dispatch.

use std::collections::{BTreeMap, BTreeSet};

use agent_challenge::{ExpectedSet, KEY_LEN};

use crate::chainsnap::Hotkey;

/// How many epochs past the one it was made in an announcement stays usable.
///
/// An announcement is bound to its epoch, so a live miner has to re-announce
/// the way a neuron has to re-serve an axon. Three epochs is the smallest
/// window that survives both ways an honest miner would otherwise vanish at a
/// boundary — announcing into epoch `N` while the validator's pin has already
/// advanced, and missing one re-announcement to a restart or a gateway blip —
/// while still retiring a dead miner within a handful of epochs rather than
/// dialling a corpse forever.
const MAX_ANNOUNCEMENT_AGE_EPOCHS: u64 = 3;

/// Oldest announcement epoch still dispatchable at `epoch`.
fn min_epoch(epoch: u64) -> i64 {
    i64::try_from(epoch.saturating_sub(MAX_ANNOUNCEMENT_AGE_EPOCHS)).unwrap_or(i64::MAX)
}

/// One announced endpoint.
///
/// Mirrors `db::MinerEndpointRow` field-for-field so [`query_miner_endpoints`]
/// stays a one-line adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinerEndpointRow {
    /// Lowercase unprefixed 64-char hex, as written by the gateway.
    pub miner_hotkey: String,
    /// Public CVM base URL the packs are dispatched to.
    pub base_url: String,
    /// Epoch the announcement was made in.
    pub epoch: i64,
}

/// Adapter over the shared control-plane query.
async fn query_miner_endpoints(
    pool: &db::PgPool,
    netuid: i32,
    min_epoch: i64,
) -> Result<Vec<MinerEndpointRow>, String> {
    db::miner_endpoints(pool, netuid, min_epoch)
        .await
        .map(|rows| {
            rows.into_iter()
                .map(|r| MinerEndpointRow {
                    miner_hotkey: r.miner_hotkey,
                    base_url: r.base_url,
                    epoch: r.epoch,
                })
                .collect()
        })
        .map_err(|e| e.to_string())
}

/// Decode the control plane's hotkey encoding (`hex::encode(&[u8; 32])`).
fn decode_hotkey(hex_hotkey: &str) -> Option<Hotkey> {
    let bytes = hex::decode(hex_hotkey).ok()?;
    <[u8; KEY_LEN]>::try_from(bytes.as_slice()).ok()
}

/// Project announced rows onto the expected set.
///
/// The age filter is re-applied here rather than trusted to the query: a stale
/// row that slipped through would keep a departed miner dispatchable, which is
/// exactly the failure the window exists to prevent. Rows for hotkeys outside
/// `E` are dropped so a deregistered miner cannot manufacture work.
fn project(
    rows: &[MinerEndpointRow],
    min_epoch: i64,
    expected: &ExpectedSet,
) -> BTreeMap<Hotkey, String> {
    let allowed: BTreeSet<Hotkey> = expected.participants.iter().map(|p| p.hotkey).collect();
    let mut out = BTreeMap::new();
    for row in rows {
        if row.epoch < min_epoch || row.base_url.is_empty() {
            continue;
        }
        let Some(hotkey) = decode_hotkey(&row.miner_hotkey) else {
            tracing::warn!(
                event = "miner_endpoint_hotkey_undecodable",
                hotkey = %row.miner_hotkey,
                "announced hotkey is not 32-byte hex; announcement ignored"
            );
            continue;
        };
        if allowed.contains(&hotkey) {
            out.insert(hotkey, row.base_url.clone());
        }
    }
    out
}

/// Turn a registry read into the dispatch endpoint map.
///
/// A failed read yields an empty map: every miner is then absent and scored
/// through the ordinary not-attempted path, so `E` stays totally covered (D24)
/// and no outage can invent a dispatch or a score.
fn endpoints_from(
    rows: Result<Vec<MinerEndpointRow>, String>,
    min_epoch: i64,
    expected: &ExpectedSet,
) -> BTreeMap<Hotkey, String> {
    match rows {
        Ok(rows) => project(&rows, min_epoch, expected),
        Err(e) => {
            tracing::warn!(
                event = "miner_endpoint_registry_read_failed",
                error = %e,
                "endpoint registry unavailable; every miner treated as not dispatchable"
            );
            BTreeMap::new()
        }
    }
}

/// Control-plane reader for announced miner endpoints.
#[derive(Debug, Clone)]
pub struct EndpointRegistry {
    pool: db::PgPool,
    netuid: i32,
}

impl EndpointRegistry {
    /// Build over an already-connected control-plane pool.
    #[must_use]
    pub fn new(pool: db::PgPool, netuid: u16) -> Self {
        Self {
            pool,
            netuid: i32::from(netuid),
        }
    }

    /// Resolve the dispatch endpoint of every expected miner at `epoch`.
    ///
    /// Never fails: absence is the only degraded state this path can produce.
    pub async fn resolve_endpoints(
        &self,
        epoch: u64,
        expected: &ExpectedSet,
    ) -> BTreeMap<Hotkey, String> {
        let min = min_epoch(epoch);
        let rows = query_miner_endpoints(&self.pool, self.netuid, min).await;
        let out = endpoints_from(rows, min, expected);
        tracing::info!(
            event = "miner_endpoints_resolved",
            epoch,
            netuid = self.netuid,
            min_epoch = min,
            expected = expected.participants.len(),
            announced = out.len(),
            "announced endpoints resolved for the expected set"
        );
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_challenge::{score_map_covering_expected, ExpectedParticipant};

    const MINER: Hotkey = [0xB1; KEY_LEN];
    const OTHER: Hotkey = [0xC2; KEY_LEN];
    const STRANGER: Hotkey = [0xD3; KEY_LEN];
    const EPOCH: u64 = 10;
    const URL: &str = "https://miner.cvm.invalid";

    fn expected_set() -> ExpectedSet {
        ExpectedSet {
            block_hash: [0x11; 32],
            participants: vec![
                ExpectedParticipant {
                    hotkey: MINER,
                    uid: 1,
                },
                ExpectedParticipant {
                    hotkey: OTHER,
                    uid: 2,
                },
            ],
        }
    }

    fn row(hotkey: Hotkey, epoch: i64) -> MinerEndpointRow {
        MinerEndpointRow {
            miner_hotkey: hex::encode(hotkey),
            base_url: URL.to_owned(),
            epoch,
        }
    }

    /// Every expected hotkey must still be scored, whatever the registry said.
    fn covers_all_of_e(expected: &ExpectedSet) {
        let keys = expected.hotkeys();
        let covered = score_map_covering_expected(&keys, &BTreeMap::new(), &BTreeMap::new());
        assert_eq!(covered.len(), expected.participants.len());
        for p in &expected.participants {
            assert!(covered.contains_key(&p.hotkey), "D24: {} uncovered", p.uid);
        }
    }

    #[test]
    fn announced_miner_becomes_dispatchable() {
        let expected = expected_set();
        let map = endpoints_from(
            Ok(vec![row(MINER, i64::try_from(EPOCH).expect("epoch"))]),
            min_epoch(EPOCH),
            &expected,
        );
        assert_eq!(map.get(&MINER).map(String::as_str), Some(URL));
        assert!(!map.contains_key(&OTHER), "OTHER announced nothing");
    }

    #[test]
    fn miner_without_an_announcement_is_an_absence() {
        let expected = expected_set();
        let map = endpoints_from(Ok(Vec::new()), min_epoch(EPOCH), &expected);
        assert!(map.is_empty());
        covers_all_of_e(&expected);
    }

    /// The window is bounded: an announcement older than it is not a dial target.
    #[test]
    fn announcement_older_than_the_window_is_ignored() {
        let expected = expected_set();
        let min = min_epoch(EPOCH);
        assert_eq!(min, 7, "window is {MAX_ANNOUNCEMENT_AGE_EPOCHS} epochs");
        assert!(endpoints_from(Ok(vec![row(MINER, min - 1)]), min, &expected).is_empty());
        // The oldest still-valid epoch stays dispatchable.
        assert!(endpoints_from(Ok(vec![row(MINER, min)]), min, &expected).contains_key(&MINER));
    }

    /// A deregistered miner's announcement must never create work.
    #[test]
    fn announcement_from_outside_the_expected_set_is_dropped() {
        let expected = expected_set();
        let map = endpoints_from(
            Ok(vec![row(STRANGER, i64::try_from(EPOCH).expect("epoch"))]),
            min_epoch(EPOCH),
            &expected,
        );
        assert!(map.is_empty(), "hotkey outside E entered the map");
    }

    #[test]
    fn registry_read_failure_leaves_the_expected_set_covered() {
        let expected = expected_set();
        let map = endpoints_from(
            Err("connection reset".to_owned()),
            min_epoch(EPOCH),
            &expected,
        );
        assert!(
            map.is_empty(),
            "a read failure must not fabricate endpoints"
        );
        covers_all_of_e(&expected);
    }

    /// A malformed or empty announcement is discarded, not dialled.
    #[test]
    fn unusable_rows_are_discarded() {
        let expected = expected_set();
        let min = min_epoch(EPOCH);
        let epoch = i64::try_from(EPOCH).expect("epoch");
        let mut empty_url = row(MINER, epoch);
        empty_url.base_url = String::new();
        let mut bad_hex = row(OTHER, epoch);
        bad_hex.miner_hotkey = "0xnot-hex".to_owned();
        assert!(endpoints_from(Ok(vec![empty_url, bad_hex]), min, &expected).is_empty());
    }

    /// Epoch 0 must not underflow into a window that admits everything.
    #[test]
    fn early_epochs_do_not_underflow_the_window() {
        assert_eq!(min_epoch(0), 0);
        assert_eq!(min_epoch(1), 0);
    }
}
