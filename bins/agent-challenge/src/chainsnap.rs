//! Per-tick chain snapshot: epoch pin, expected set `E`, and axon endpoints.
//!
//! Every value here is re-read from Subtensor on every tick. None of it is
//! configurable: a wrong uid or a stale epoch silently corrupts the weight
//! vector, so the tick fails rather than falling back to a local guess.

use std::collections::BTreeMap;

use agent_challenge::{expected_set_at_chain, ExpectedSet, PinnedBlockHash, KEY_LEN};
use chain::{current_epoch_pre_run_coinbase, gather_schedule_state, ChainClient};
use trustroot::ParticipantPolicy;

/// Miner hotkey bytes.
pub type Hotkey = [u8; KEY_LEN];

/// Chain-derived epoch identity for one tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpochPin {
    /// Subtensor's own epoch counter for the subnet.
    pub epoch: u64,
    /// Block at which the current epoch started (`block_B`).
    pub block_b: u64,
    /// `block_hash(block_B)` — the pin `E` is sealed at.
    pub block_hash: [u8; 32],
}

/// Everything one tick needs from the chain, read at a single pin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainSnapshot {
    /// Epoch identity.
    pub pin: EpochPin,
    /// Sealed expected participant set.
    pub expected: ExpectedSet,
    /// Reachable base URL per hotkey. A hotkey absent here published no axon.
    pub endpoints: BTreeMap<Hotkey, String>,
}

/// Read `(epoch, block_B, block_hash)` from the chain.
///
/// The epoch is [`current_epoch_pre_run_coinbase`], the repo's port of
/// Subtensor's own `run_coinbase` counter — the same function the CRV4 reveal
/// scheduler uses, so challenge leaves and weight commits number epochs alike.
/// `block_B` is the epoch's own start block, which makes the pin stable for the
/// whole epoch and advance exactly when the epoch does.
///
/// # Errors
///
/// Any chain read failure (tempo, schedule inputs, or block hash).
pub fn read_epoch_pin<C: ChainClient>(chain: &C, netuid: u16) -> Result<EpochPin, String> {
    let state = gather_schedule_state(chain, netuid)
        .map_err(|e| format!("epoch schedule read (netuid {netuid}): {e}"))?;
    let epoch = current_epoch_pre_run_coinbase(&state, state.current_block);
    let block_b = state.last_epoch_block;
    let block_hash = chain
        .block_hash(block_b)
        .map_err(|e| format!("block_hash({block_b}): {e}"))?;
    Ok(EpochPin {
        epoch,
        block_b,
        block_hash,
    })
}

/// Resolve every expected hotkey to its published axon base URL.
///
/// A neuron that never called `serve_axon` is simply missing from the result;
/// per-hotkey read failures are logged and treated the same way, because an
/// unresolvable endpoint and an unpublished one are the same thing to dispatch.
fn read_endpoints<C: ChainClient>(
    chain: &C,
    netuid: u16,
    expected: &ExpectedSet,
) -> BTreeMap<Hotkey, String> {
    let mut out = BTreeMap::new();
    for p in &expected.participants {
        match chain.axon(netuid, &p.hotkey) {
            Ok(Some(info)) => {
                if let Some(url) = info.base_url() {
                    out.insert(p.hotkey, url);
                }
            }
            Ok(None) => {}
            Err(e) => tracing::warn!(
                event = "axon_read_failed",
                uid = p.uid,
                hotkey = %hex::encode(p.hotkey),
                error = %e,
                "axon unresolved; miner treated as not dispatchable"
            ),
        }
    }
    out
}

/// Read the full per-tick snapshot at a freshly derived pin.
///
/// # Errors
///
/// Chain read failure, or a metagraph that cannot be projected onto `E`.
pub fn read_snapshot<C: ChainClient>(
    chain: &C,
    netuid: u16,
    policy: &ParticipantPolicy,
) -> Result<ChainSnapshot, String> {
    let pin = read_epoch_pin(chain, netuid)?;
    let expected = expected_set_at_chain(policy, PinnedBlockHash::new(pin.block_hash), chain)
        .map_err(|e| format!("expected set at block {}: {e}", pin.block_b))?;
    let endpoints = read_endpoints(chain, netuid, &expected);
    Ok(ChainSnapshot {
        pin,
        expected,
        endpoints,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chain::{AxonInfo, FakeChain, FakeChainConfig};

    const NETUID: u16 = 541;

    fn owner() -> Vec<u8> {
        vec![0xA0; 32]
    }
    fn miner() -> Vec<u8> {
        vec![0xB1; 32]
    }
    fn validator() -> Vec<u8> {
        vec![0xC2; 32]
    }

    fn axon(ip: u128, port: u16) -> AxonInfo {
        AxonInfo {
            block: 1,
            version: 1,
            ip,
            port,
            ip_type: 4,
            protocol: 4,
            placeholder1: 0,
            placeholder2: 0,
        }
    }

    fn cfg(epoch_index: u64, last_epoch_block: u64) -> FakeChainConfig {
        FakeChainConfig {
            netuid: NETUID,
            current_block: last_epoch_block + 1,
            tempo: 360,
            last_epoch_block,
            blocks_since_last_step: 1,
            subnet_epoch_index: epoch_index,
            owner_hotkey: owner(),
            // Deliberately not sorted by hotkey: uid must come from this order.
            hotkeys: vec![validator(), owner(), miner()],
            ..FakeChainConfig::default()
        }
    }

    /// Regression: uids used to be the enumeration index of an env var. They
    /// must be the metagraph position, which here disagrees with hotkey sort
    /// order in every slot.
    #[test]
    fn uids_come_from_the_metagraph_not_enumeration_order() {
        let chain = FakeChain::new(cfg(7, 3_600));
        let snap =
            read_snapshot(&chain, NETUID, &ParticipantPolicy::AllMetagraphHotkeys).expect("snap");

        let by_hotkey: BTreeMap<Vec<u8>, u16> = snap
            .expected
            .participants
            .iter()
            .map(|p| (p.hotkey.to_vec(), p.uid))
            .collect();
        assert_eq!(by_hotkey.get(&validator()), Some(&0));
        assert_eq!(by_hotkey.get(&owner()), Some(&1));
        assert_eq!(by_hotkey.get(&miner()), Some(&2));

        // Sorted-by-hotkey order would have handed out 0,1,2 in the other direction.
        let sorted_uids: Vec<u16> = {
            let mut keys: Vec<Vec<u8>> = by_hotkey.keys().cloned().collect();
            keys.sort();
            keys.iter().map(|k| by_hotkey[k]).collect()
        };
        assert_ne!(
            sorted_uids,
            vec![0, 1, 2],
            "uid must not track hotkey order"
        );
    }

    /// The pin is the real `block_hash(block_B)`, never a hardcoded constant.
    #[test]
    fn pin_is_the_real_block_hash_of_block_b() {
        let chain = FakeChain::new(cfg(7, 3_600));
        let pin = read_epoch_pin(&chain, NETUID).expect("pin");
        assert_eq!(pin.block_b, 3_600);
        assert_eq!(pin.block_hash, chain.block_hash(3_600).expect("hash"));
        assert_ne!(pin.block_hash[0], 0xBD, "placeholder pin must be gone");
    }

    /// Regression: the epoch used to be captured once from env and never move.
    #[test]
    fn epoch_advances_when_the_chain_advances() {
        let tick1 = read_epoch_pin(&FakeChain::new(cfg(7, 3_600)), NETUID).expect("tick1");
        let tick2 = read_epoch_pin(&FakeChain::new(cfg(8, 3_960)), NETUID).expect("tick2");
        assert_eq!(tick1.epoch, 7);
        assert_eq!(tick2.epoch, 8);
        assert_ne!(tick1.block_hash, tick2.block_hash);
    }

    /// A tempo boundary bumps the epoch before the chain records the step.
    #[test]
    fn epoch_includes_the_pending_step() {
        let chain = FakeChain::new(FakeChainConfig {
            current_block: 3_960,
            blocks_since_last_step: 360,
            ..cfg(7, 3_600)
        });
        assert_eq!(read_epoch_pin(&chain, NETUID).expect("pin").epoch, 8);
    }

    #[test]
    fn only_published_axons_become_endpoints() {
        let chain = FakeChain::new(cfg(7, 3_600));
        chain.set_axon(&miner(), axon(3_717_915_933, 8080));
        // Served then cleared: stored as all-zero, must not become a dial target.
        chain.set_axon(&validator(), axon(0, 0));

        let snap =
            read_snapshot(&chain, NETUID, &ParticipantPolicy::AllMetagraphHotkeys).expect("snap");
        assert_eq!(snap.endpoints.len(), 1);
        assert_eq!(
            snap.endpoints.get(&[0xB1_u8; 32]).map(String::as_str),
            Some("http://221.154.229.29:8080")
        );
    }

    /// Live probe that the epoch counter and pin are populated on our subnet.
    ///
    /// `SubnetEpochIndex` is a `ValueQuery` read: an absent key decodes to 0 and
    /// would silently freeze the epoch, so assert it against the real chain.
    #[test]
    #[ignore = "requires network access to finney testnet"]
    fn testnet_541_epoch_pin_is_live() {
        let mut chain = chain_live::LiveChainClient::connect("wss://test.finney.opentensor.ai:443")
            .expect("connect");
        chain.set_netuid(NETUID);
        let state = gather_schedule_state(&chain, NETUID).expect("schedule inputs");
        println!("netuid {NETUID} schedule: {state:?}");
        let pin = read_epoch_pin(&chain, NETUID).expect("pin");
        println!("netuid {NETUID} pin: {pin:?}");
        assert!(pin.block_b > 0, "block_B must be a real epoch boundary");
        assert!(
            pin.epoch > 0,
            "SubnetEpochIndex must be a live counter, got {}",
            pin.epoch
        );
        assert_ne!(pin.block_hash, [0_u8; 32]);
    }

    #[test]
    fn no_axons_at_all_is_an_empty_map_not_an_error() {
        let chain = FakeChain::new(cfg(7, 3_600));
        let snap =
            read_snapshot(&chain, NETUID, &ParticipantPolicy::AllMetagraphHotkeys).expect("snap");
        assert_eq!(snap.expected.participants.len(), 3);
        assert!(snap.endpoints.is_empty());
    }
}
