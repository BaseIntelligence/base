//! Per-epoch attestation credit book — D13 no carry-forward.

use std::collections::BTreeMap;

use gbase_crypto::KEY_LEN;

use crate::AttestOutcome;

/// Key for one miner attestation slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CreditKey {
    /// Subnet.
    pub netuid: u16,
    /// Epoch.
    pub epoch: u64,
    /// Miner hotkey.
    pub miner: [u8; KEY_LEN],
}

/// Records outcomes per epoch. Looking up credit never consults other epochs.
#[derive(Debug, Default, Clone)]
pub struct AttestCreditBook {
    /// `(netuid, epoch, miner) → outcome`
    slots: BTreeMap<CreditKey, AttestOutcome>,
}

impl AttestCreditBook {
    /// Empty book.
    #[must_use]
    pub fn new() -> Self {
        Self {
            slots: BTreeMap::new(),
        }
    }

    /// Record the outcome for this epoch slot (overwrites).
    pub fn record(&mut self, key: CreditKey, outcome: AttestOutcome) {
        self.slots.insert(key, outcome);
    }

    /// Whether the miner has attestation credit **for this exact epoch**.
    ///
    /// Prior epochs are never consulted (D13: park/reject never carry `Verified`).
    #[must_use]
    pub fn has_credit(&self, key: &CreditKey) -> bool {
        self.slots
            .get(key)
            .is_some_and(|o| matches!(o, AttestOutcome::Verified))
    }

    /// Stored outcome for the slot, if any.
    #[must_use]
    pub fn get(&self, key: &CreditKey) -> Option<&AttestOutcome> {
        self.slots.get(key)
    }
}
