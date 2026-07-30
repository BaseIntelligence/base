//! Expected participant set `E` sealed at `block_B` (I7 / BUNDLE_SPEC §7 / D24).
//!
//! Derivation is pure in `(block_hash, policy, metagraph_at(block_hash))`.
//! A moving tip is never an input — callers must pass an explicit pin.

use std::collections::BTreeSet;
use std::fmt;

use bundle::{expected_participants, metagraph_rows_from_chain};
use chain::{ChainClient, ChainError, Metagraph};
use crypto::KEY_LEN;
use thiserror::Error;
use trustroot::ParticipantPolicy;

/// Explicit 32-byte pin of the metagraph block (`block_hash` of `block_B`).
///
/// Tip is intentionally not representable here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PinnedBlockHash(pub [u8; 32]);

impl PinnedBlockHash {
    /// Wrap a known block hash.
    #[must_use]
    pub const fn new(hash: [u8; 32]) -> Self {
        Self(hash)
    }

    /// Raw hash bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// How the caller names the metagraph block for E derivation.
///
/// [`BlockSource::Tip`] is always refused (I7: seal at `block_B`, never a moving tip).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockSource {
    /// Explicit `block_hash` of `block_B`.
    Pinned(PinnedBlockHash),
    /// Moving chain tip — **not** allowed for expected-set derivation.
    Tip,
}

/// One hotkey in `E` with its UID at the sealed metagraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedParticipant {
    /// Miner / neuron hotkey.
    pub hotkey: [u8; KEY_LEN],
    /// UID in metagraph order at the sealed block.
    pub uid: u16,
}

/// Sealed expected set for one challenge at `block_B`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedSet {
    /// Pin used for derivation (`block_hash`).
    pub block_hash: [u8; 32],
    /// Participants ordered by ascending hotkey bytes (BUNDLE_SPEC §7.2).
    pub participants: Vec<ExpectedParticipant>,
}

impl ExpectedSet {
    /// Hotkey set view (sorted).
    #[must_use]
    pub fn hotkeys(&self) -> BTreeSet<[u8; KEY_LEN]> {
        self.participants.iter().map(|p| p.hotkey).collect()
    }

    /// Whether `hotkey` is in `E`.
    #[must_use]
    pub fn contains(&self, hotkey: &[u8; KEY_LEN]) -> bool {
        self.participants.iter().any(|p| p.hotkey == *hotkey)
    }
}

/// Errors from sealed expected-set derivation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExpectedSetError {
    /// Caller did not pin `block_hash` (tip / `None` / unpinned).
    #[error(
        "expected set requires an explicit pinned block_hash; refusing tip or unpinned derivation (I7)"
    )]
    BlockNotPinned,
    /// `metagraph_at(block_hash)` failed or hash unknown.
    #[error("metagraph unavailable for pinned block_hash: {0}")]
    MetagraphUnavailable(String),
    /// Hotkey lengths / uid overflow when projecting rows.
    #[error("invalid metagraph for expected set: {0}")]
    InvalidMetagraph(String),
}

impl From<ChainError> for ExpectedSetError {
    fn from(err: ChainError) -> Self {
        match err {
            ChainError::UnknownMetagraph | ChainError::UnknownBlock { .. } => {
                Self::MetagraphUnavailable(err.to_string())
            }
            other => Self::MetagraphUnavailable(other.to_string()),
        }
    }
}

/// Pure derivation: `E = expected_participants(policy, rows)` at a known pin.
///
/// `metagraph` MUST be the snapshot from `metagraph_at(block_hash)` — this function
/// does not read the chain and never defaults to tip.
///
/// # Errors
///
/// Invalid hotkey lengths in the metagraph projection.
pub fn expected_set_from_pinned_metagraph(
    policy: &ParticipantPolicy,
    pin: PinnedBlockHash,
    metagraph: &Metagraph,
) -> Result<ExpectedSet, ExpectedSetError> {
    let rows = metagraph_rows_from_chain(&metagraph.hotkeys, None)
        .map_err(|e| ExpectedSetError::InvalidMetagraph(e.to_string()))?;
    let keys = expected_participants(policy, &rows);
    let participants: Vec<ExpectedParticipant> = rows
        .into_iter()
        .filter(|r| keys.contains(&r.hotkey))
        .map(|r| ExpectedParticipant {
            hotkey: r.hotkey,
            uid: r.uid,
        })
        .collect();
    Ok(ExpectedSet {
        block_hash: pin.0,
        participants,
    })
}

/// Fetch metagraph at an explicit pin and derive `E`. Never uses tip.
///
/// # Errors
///
/// Unpinned tip, unknown hash, or invalid metagraph rows.
pub fn expected_set_at_chain(
    policy: &ParticipantPolicy,
    pin: PinnedBlockHash,
    chain: &impl ChainClient,
) -> Result<ExpectedSet, ExpectedSetError> {
    let metagraph = chain.metagraph_at(pin.as_bytes())?;
    expected_set_from_pinned_metagraph(policy, pin, &metagraph)
}

/// Derive `E` from a [`BlockSource`]. [`BlockSource::Tip`] → [`ExpectedSetError::BlockNotPinned`].
///
/// # Errors
///
/// Tip/unpinned, chain metagraph failure, or invalid rows.
pub fn expected_set_at(
    policy: &ParticipantPolicy,
    source: BlockSource,
    chain: &impl ChainClient,
) -> Result<ExpectedSet, ExpectedSetError> {
    match source {
        BlockSource::Tip => Err(ExpectedSetError::BlockNotPinned),
        BlockSource::Pinned(pin) => expected_set_at_chain(policy, pin, chain),
    }
}

/// Derive `E` when the pin is optional. `None` is a typed refusal (never silent tip).
///
/// # Errors
///
/// Missing pin, chain failure, or invalid rows.
pub fn expected_set_from_optional_pin(
    policy: &ParticipantPolicy,
    block_hash: Option<[u8; 32]>,
    chain: &impl ChainClient,
) -> Result<ExpectedSet, ExpectedSetError> {
    let Some(hash) = block_hash else {
        return Err(ExpectedSetError::BlockNotPinned);
    };
    expected_set_at_chain(policy, PinnedBlockHash::new(hash), chain)
}

/// Hex encode a 32-byte hash / hotkey for evidence logs.
#[must_use]
pub fn hex32(bytes: &[u8; 32]) -> String {
    hex::encode(bytes)
}

impl fmt::Display for ExpectedSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "block_hash={}", hex32(&self.block_hash))?;
        writeln!(f, "|E|={}", self.participants.len())?;
        for p in &self.participants {
            writeln!(
                f,
                "  uid={} hotkey={}",
                p.uid,
                hex32(&p.hotkey)
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use chain::{FakeChain, FakeChainConfig};

    /// Netuid 541 testnet role UIDs (Metis S10): owner=0, miner=1, validator=2.
    const NETUID_541: u16 = 541;
    fn owner_hk() -> [u8; 32] {
        [0xA0; 32]
    }
    fn miner_hk() -> [u8; 32] {
        [0xB1; 32]
    }
    fn validator_hk() -> [u8; 32] {
        [0xC2; 32]
    }
    /// Miner registered after `block_B` (mid-epoch) — must not enter E this epoch.
    fn late_miner_hk() -> [u8; 32] {
        [0xD3; 32]
    }

    fn block_b_hash() -> [u8; 32] {
        // Distinct fixed pin (not tip-derived).
        let mut h = [0u8; 32];
        h[0] = 0xBB;
        h[1] = 0x54;
        h[2] = 0x01;
        h
    }

    fn meta_at_block_b() -> Metagraph {
        Metagraph {
            netuid: NETUID_541,
            // UID order: 0 owner, 1 miner, 2 validator
            hotkeys: vec![
                owner_hk().to_vec(),
                miner_hk().to_vec(),
                validator_hk().to_vec(),
            ],
            owner_hotkey: owner_hk().to_vec(),
        }
    }

    fn meta_after_late_registration() -> Metagraph {
        let mut m = meta_at_block_b();
        m.hotkeys.push(late_miner_hk().to_vec());
        m
    }

    /// S1: same `(block_hash, policy, metagraph)` → identical ordered E (two runs).
    #[test]
    fn s1_expected_set_deterministic_at_pinned_block() {
        let pin = PinnedBlockHash::new(block_b_hash());
        let policy = ParticipantPolicy::AllMetagraphHotkeys;
        let mg = meta_at_block_b();

        let e1 = expected_set_from_pinned_metagraph(&policy, pin, &mg).expect("run1");
        let e2 = expected_set_from_pinned_metagraph(&policy, pin, &mg).expect("run2");
        assert_eq!(e1, e2, "E must be byte-identical across runs at fixed pin");
        assert_eq!(e1.block_hash, block_b_hash());
        assert_eq!(e1.participants.len(), 3);

        // UID order preserved from metagraph projection (sorted by hotkey in set display).
        let by_uid: BTreeSet<u16> = e1.participants.iter().map(|p| p.uid).collect();
        assert_eq!(by_uid, BTreeSet::from([0, 1, 2]));

        let printed = e1.to_string();
        assert!(printed.contains("uid=0"), "{printed}");
        assert!(printed.contains("uid=1"), "{printed}");
        assert!(printed.contains("uid=2"), "{printed}");
        assert!(printed.contains(&hex32(&owner_hk())), "{printed}");
        assert!(printed.contains(&hex32(&miner_hk())), "{printed}");
        assert!(printed.contains(&hex32(&validator_hk())), "{printed}");
    }

    /// S2: tip / missing pin → typed refusal; never a silent tip read.
    #[test]
    fn s2_unpinned_and_tip_refused() {
        let chain = FakeChain::new(FakeChainConfig {
            netuid: NETUID_541,
            hotkeys: vec![
                owner_hk().to_vec(),
                miner_hk().to_vec(),
                validator_hk().to_vec(),
            ],
            owner_hotkey: owner_hk().to_vec(),
            ..FakeChainConfig::default()
        });
        let policy = ParticipantPolicy::AllMetagraphHotkeys;

        let tip_err = expected_set_at(&policy, BlockSource::Tip, &chain).expect_err("tip");
        assert_eq!(tip_err, ExpectedSetError::BlockNotPinned);
        assert!(
            tip_err.to_string().contains("pinned block_hash"),
            "{tip_err}"
        );

        let none_err =
            expected_set_from_optional_pin(&policy, None, &chain).expect_err("none pin");
        assert_eq!(none_err, ExpectedSetError::BlockNotPinned);

        // Prove we did not fall through to tip: FakeChain tip metagraph would succeed if called.
        let tip_n = chain.current_block().unwrap();
        let tip_h = chain.block_hash(tip_n).unwrap();
        let ok = expected_set_at(
            &policy,
            BlockSource::Pinned(PinnedBlockHash::new(tip_h)),
            &chain,
        )
        .expect("explicit pin of tip height hash is still a pin");
        assert_eq!(ok.participants.len(), 3);
    }

    /// S3: miner registered after `block_B` is absent from sealed E (I7 / not a D24 shrink).
    #[test]
    fn s3_mid_epoch_registration_not_in_e() {
        let pin = PinnedBlockHash::new(block_b_hash());
        let policy = ParticipantPolicy::AllMetagraphHotkeys;

        let e_sealed =
            expected_set_from_pinned_metagraph(&policy, pin, &meta_at_block_b()).expect("sealed");
        let e_tip_view = expected_set_from_pinned_metagraph(
            &policy,
            // Different pin representing a later block's snapshot — late miner appears only here.
            PinnedBlockHash::new({
                let mut h = block_b_hash();
                h[31] = 0xFF;
                h
            }),
            &meta_after_late_registration(),
        )
        .expect("later");

        assert!(
            !e_sealed.contains(&late_miner_hk()),
            "late miner must not be in E sealed at block_B"
        );
        assert!(e_sealed.contains(&miner_hk()));
        assert_eq!(e_sealed.participants.len(), 3);

        assert!(
            e_tip_view.contains(&late_miner_hk()),
            "control: later snapshot includes late miner"
        );
        assert_eq!(e_tip_view.participants.len(), 4);
    }

    #[test]
    fn stake_policy_filters_rows() {
        let pin = PinnedBlockHash::new(block_b_hash());
        let mg = meta_at_block_b();
        // Default stakes are 0 via metagraph_rows_from_chain — StakeAtLeast{1} → empty.
        let e = expected_set_from_pinned_metagraph(
            &ParticipantPolicy::StakeAtLeast { min_stake: 1 },
            pin,
            &mg,
        )
        .expect("stake");
        assert!(e.participants.is_empty());
    }

    #[test]
    fn unknown_pin_on_chain_is_metagraph_unavailable() {
        let chain = FakeChain::with_defaults();
        let policy = ParticipantPolicy::AllMetagraphHotkeys;
        let bad = PinnedBlockHash::new([0xDE; 32]);
        let err = expected_set_at_chain(&policy, bad, &chain).expect_err("unknown");
        assert!(
            matches!(err, ExpectedSetError::MetagraphUnavailable(_)),
            "{err:?}"
        );
    }
}
