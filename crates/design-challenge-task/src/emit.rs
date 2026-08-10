//! Design leaf-emit scheduling (late-tempo filler + catch-up).

/// How many blocks before epoch end the NotAttempted filler may run.
///
/// Wider than the historical 48-block window so `base-real-seal` (10 min) still
/// has time to seal after design emits, while leaving most of the epoch for
/// `award_round` to land Score leaves first (first-write-wins).
pub const DESIGN_EMIT_LATE_BLOCKS: u64 = 96;

/// Planned design leaf emission for one emitter tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DesignEmitPlan {
    /// Epoch label for the leaf set.
    pub epoch: u64,
    /// Metagraph pin block (epoch start).
    pub pin_block: u64,
}

/// Decide whether/which epoch the design filler should emit.
///
/// - Catch up `last_emitted+1` when behind by more than one epoch (repairs the
///   end-of-epoch relabel race that skipped alternate epochs in prod).
/// - Otherwise wait until the last [`DESIGN_EMIT_LATE_BLOCKS`] of the current
///   epoch so admin awards can submit Score leaves first.
#[must_use]
pub fn design_emit_plan(
    last_emitted: u64,
    current_epoch: u64,
    blocks_since_last_step: u64,
    tempo: u64,
    current_last_epoch_block: u64,
) -> Option<DesignEmitPlan> {
    if current_epoch == 0 {
        return None;
    }
    let tempo = tempo.max(1);
    if last_emitted >= current_epoch {
        return None;
    }
    // Sequential catch-up for skipped epochs (award path / boundary race).
    if last_emitted + 1 < current_epoch {
        let target = last_emitted + 1;
        let epochs_back = current_epoch.saturating_sub(target);
        let pin_block = current_last_epoch_block.saturating_sub(epochs_back.saturating_mul(tempo));
        return Some(DesignEmitPlan {
            epoch: target,
            pin_block,
        });
    }
    // Current epoch: late-tempo filler only.
    if blocks_since_last_step.saturating_add(DESIGN_EMIT_LATE_BLOCKS) < tempo {
        return None;
    }
    Some(DesignEmitPlan {
        epoch: current_epoch,
        pin_block: current_last_epoch_block,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn emit_plan_waits_until_late_tempo_for_current_epoch() {
        assert!(design_emit_plan(10, 11, 200, 360, 1000).is_none());
        let p = design_emit_plan(10, 11, 360 - DESIGN_EMIT_LATE_BLOCKS, 360, 1000).unwrap();
        assert_eq!(
            p,
            DesignEmitPlan {
                epoch: 11,
                pin_block: 1000
            }
        );
    }

    #[test]
    fn emit_plan_catches_up_skipped_epochs_without_waiting() {
        let p = design_emit_plan(24412, 24423, 50, 360, 8_815_687).unwrap();
        assert_eq!(p.epoch, 24413);
        assert_eq!(p.pin_block, 8_815_687 - (24423 - 24413) * 360);
    }

    #[test]
    fn emit_plan_noop_when_already_emitted_current() {
        assert!(design_emit_plan(11, 11, 350, 360, 1000).is_none());
    }
}
