//! Epoch clock derived from chain tip and configured epoch length.

use chain::{ChainClient, ChainError};
use serde::Serialize;

/// Snapshot of the validator's epoch clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct EpochSnapshot {
    /// Chain tip block number.
    pub current_block: u64,
    /// Configured epoch length in blocks.
    pub epoch_length: u64,
    /// Floor division `current_block / epoch_length`.
    pub epoch_index: u64,
    /// Block at the start of the current epoch window.
    pub epoch_start_block: u64,
    /// Inclusive end block of the current epoch window (or tip if mid-epoch).
    pub epoch_end_block: u64,
}

/// Compute epoch fields from tip + length.
///
/// # Errors
///
/// Returns an error string when `epoch_length` is zero.
pub fn epoch_from_block(current_block: u64, epoch_length: u64) -> Result<EpochSnapshot, String> {
    if epoch_length == 0 {
        return Err("epoch_length must be > 0".to_owned());
    }
    let epoch_index = current_block / epoch_length;
    let epoch_start_block = epoch_index.saturating_mul(epoch_length);
    let epoch_end_block = epoch_start_block
        .saturating_add(epoch_length)
        .saturating_sub(1);
    Ok(EpochSnapshot {
        current_block,
        epoch_length,
        epoch_index,
        epoch_start_block,
        epoch_end_block,
    })
}

/// Read tip from `chain` and build an [`EpochSnapshot`].
///
/// # Errors
///
/// Chain transport errors or zero epoch length.
pub fn epoch_from_chain(
    chain: &dyn ChainClient,
    epoch_length: u64,
) -> Result<EpochSnapshot, ChainError> {
    let current_block = chain.current_block()?;
    epoch_from_block(current_block, epoch_length).map_err(ChainError::Other)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chain::{fake_defaults, FakeChain};

    #[test]
    fn s4_epoch_clock_from_fake_chain_current_block() {
        let chain = FakeChain::with_defaults();
        let tip = chain.current_block().expect("tip");
        assert_eq!(tip, fake_defaults::CURRENT_BLOCK);
        let snap = epoch_from_chain(&chain, 360).expect("epoch");
        assert_eq!(snap.current_block, tip);
        assert_eq!(snap.epoch_length, 360);
        assert_eq!(snap.epoch_index, tip / 360);
        assert_eq!(snap.epoch_start_block, (tip / 360) * 360);
        assert_eq!(snap.epoch_end_block, snap.epoch_start_block + 360 - 1);
    }

    #[test]
    fn epoch_length_zero_rejected() {
        assert!(epoch_from_block(10, 0).is_err());
    }
}
