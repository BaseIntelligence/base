//! Vesting ledger: release `1/V` per subsequent segment; clawback unvested on regression.
//!
//! ```text
//! release  = 1/V per segment over V segments
//! clawback = unpaid (unvested) remainder suspended on detected regression
//! ```
//!
//! All balances are integer milliseconds of payable Δ (or any integer reward unit).
//! Per-segment release uses floor division; the final open segment receives the
//! remainder so the sum of releases equals `total` exactly (no dust loss).

use crate::error::PayError;

/// Default vesting horizon `V` (segments) when callers do not override.
pub const DEFAULT_VESTING_SEGMENTS: u32 = 4;

/// Opaque grant identifier on a [`VestingLedger`].
pub type GrantId = u64;

/// One vesting grant opened after a successful marginal pay event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VestingGrant {
    /// Ledger-local id.
    pub id: GrantId,
    /// Total reward units granted at open (ms of payable Δ).
    pub total: u64,
    /// Units already released (vested).
    pub vested: u64,
    /// Segments already advanced for this grant (`0..V`).
    pub segments_done: u32,
    /// Vesting horizon `V`.
    pub v: u32,
    /// True after clawback zeroed the unvested remainder.
    pub clawed_back: bool,
}

impl VestingGrant {
    /// Unvested (still locked) balance.
    #[must_use]
    pub fn unvested(&self) -> u64 {
        if self.clawed_back {
            return 0;
        }
        self.total.saturating_sub(self.vested)
    }

    /// Whether the grant still has segments that can vest.
    #[must_use]
    pub fn is_active(&self) -> bool {
        !self.clawed_back && self.segments_done < self.v && self.unvested() > 0
    }
}

/// In-memory vesting ledger for one miner (or one challenge-local account).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VestingLedger {
    next_id: GrantId,
    grants: Vec<VestingGrant>,
}

impl VestingLedger {
    /// Empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a grant of `total` reward units vesting over `v` segments (`1/V` each).
    ///
    /// `total == 0` is allowed (no-op grant that never pays). `v` must be ≥ 1.
    ///
    /// # Errors
    ///
    /// [`PayError::InvalidVestingSegments`] when `v == 0`.
    pub fn open_grant(&mut self, total: u64, v: u32) -> Result<GrantId, PayError> {
        if v == 0 {
            return Err(PayError::InvalidVestingSegments(0));
        }
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.grants.push(VestingGrant {
            id,
            total,
            vested: 0,
            segments_done: 0,
            v,
            clawed_back: false,
        });
        Ok(id)
    }

    /// Open a grant with [`DEFAULT_VESTING_SEGMENTS`].
    ///
    /// # Errors
    ///
    /// Propagates [`PayError`] from [`Self::open_grant`].
    pub fn open_grant_default_v(&mut self, total: u64) -> Result<GrantId, PayError> {
        self.open_grant(total, DEFAULT_VESTING_SEGMENTS)
    }

    /// Borrow a grant by id.
    #[must_use]
    pub fn grant(&self, id: GrantId) -> Option<&VestingGrant> {
        self.grants.iter().find(|g| g.id == id)
    }

    /// Advance one global segment: each active grant releases its next `1/V` slice.
    ///
    /// Returns the **newly** vested units across all grants this step.
    pub fn advance_segment(&mut self) -> u64 {
        let mut newly = 0_u64;
        for g in &mut self.grants {
            if !g.is_active() {
                continue;
            }
            let release = segment_release(g.total, g.v, g.segments_done);
            // Cap by remaining unvested (defensive).
            let release = release.min(g.unvested());
            g.vested = g.vested.saturating_add(release);
            g.segments_done = g.segments_done.saturating_add(1);
            newly = newly.saturating_add(release);
        }
        newly
    }

    /// Claw back all unvested balance on `id` (regression). Vested stays vested.
    ///
    /// Returns the amount clawed (former unvested). After this, `unvested() == 0`.
    ///
    /// # Errors
    ///
    /// - [`PayError::UnknownGrant`]
    /// - [`PayError::NothingToClawback`] when already fully vested or already clawed
    pub fn clawback_unvested(&mut self, id: GrantId) -> Result<u64, PayError> {
        let g = self
            .grants
            .iter_mut()
            .find(|g| g.id == id)
            .ok_or(PayError::UnknownGrant(id))?;
        if g.clawed_back {
            return Err(PayError::NothingToClawback(id));
        }
        let u = g.total.saturating_sub(g.vested);
        if u == 0 {
            return Err(PayError::NothingToClawback(id));
        }
        // Freeze: mark clawed; unvested reads as 0; no further segment releases.
        g.clawed_back = true;
        // Keep vested as-is; conceptually total effective = vested only.
        // Adjust total down to vested so accounting stays consistent.
        g.total = g.vested;
        g.segments_done = g.v;
        Ok(u)
    }

    /// Claw back unvested on **all** active grants (champion regression event).
    ///
    /// Returns total units clawed.
    pub fn clawback_all_unvested_on_regression(&mut self) -> u64 {
        let ids: Vec<GrantId> = self
            .grants
            .iter()
            .filter(|g| !g.clawed_back && g.total.saturating_sub(g.vested) > 0)
            .map(|g| g.id)
            .collect();
        let mut total = 0_u64;
        for id in ids {
            if let Ok(u) = self.clawback_unvested(id) {
                total = total.saturating_add(u);
            }
        }
        total
    }

    /// Sum of vested units across all grants.
    #[must_use]
    pub fn total_vested(&self) -> u64 {
        self.grants.iter().map(|g| g.vested).sum()
    }

    /// Sum of unvested units across all grants.
    #[must_use]
    pub fn total_unvested(&self) -> u64 {
        self.grants.iter().map(VestingGrant::unvested).sum()
    }
}

/// Release amount for segment index `segments_done` (`0`-based) of a grant.
///
/// Floor `total/V` for segments `0..V-1`; last segment gets the remainder.
#[must_use]
pub fn segment_release(total: u64, v: u32, segments_done: u32) -> u64 {
    if v == 0 || segments_done >= v {
        return 0;
    }
    let v_u = u64::from(v);
    let base = total / v_u;
    if segments_done + 1 == v {
        // Last segment: remainder so sum == total.
        total.saturating_sub(base.saturating_mul(u64::from(v - 1)))
    } else {
        base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_release_sums_to_total() {
        for total in [0_u64, 1, 3, 7, 100, 1_000_000] {
            for v in [1_u32, 2, 3, 4, 7, 10] {
                let mut sum = 0_u64;
                for s in 0..v {
                    sum = sum.saturating_add(segment_release(total, v, s));
                }
                assert_eq!(sum, total, "total={total} v={v}");
            }
        }
    }

    #[test]
    fn open_rejects_v_zero() {
        let mut led = VestingLedger::new();
        assert_eq!(
            led.open_grant(100, 0),
            Err(PayError::InvalidVestingSegments(0))
        );
    }

    #[test]
    fn vest_one_over_v_per_segment() {
        let mut led = VestingLedger::new();
        let id = led.open_grant(100, 4).expect("open");
        assert_eq!(led.grant(id).map(|g| g.vested), Some(0));
        assert_eq!(led.advance_segment(), 25);
        assert_eq!(led.grant(id).map(|g| g.vested), Some(25));
        assert_eq!(led.advance_segment(), 25);
        assert_eq!(led.advance_segment(), 25);
        assert_eq!(led.advance_segment(), 25);
        assert_eq!(led.grant(id).map(|g| g.vested), Some(100));
        assert_eq!(led.advance_segment(), 0);
    }

    #[test]
    fn clawback_zeros_unvested_keeps_vested() {
        let mut led = VestingLedger::new();
        let id = led.open_grant(100, 4).expect("open");
        assert_eq!(led.advance_segment(), 25);
        let clawed = led.clawback_unvested(id).expect("claw");
        assert_eq!(clawed, 75);
        let g = led.grant(id).expect("grant");
        assert_eq!(g.vested, 25);
        assert_eq!(g.unvested(), 0);
        assert!(g.clawed_back);
        // Further advances must not revive clawed balance.
        assert_eq!(led.advance_segment(), 0);
        assert_eq!(led.grant(id).map(|g| g.vested), Some(25));
        assert_eq!(led.grant(id).map(VestingGrant::unvested), Some(0));
    }

    #[test]
    fn clawback_all_on_regression() {
        let mut led = VestingLedger::new();
        let a = led.open_grant(40, 2).expect("a");
        let b = led.open_grant(60, 2).expect("b");
        led.advance_segment();
        let clawed = led.clawback_all_unvested_on_regression();
        assert_eq!(clawed, 20 + 30);
        assert_eq!(led.grant(a).map(VestingGrant::unvested), Some(0));
        assert_eq!(led.grant(b).map(VestingGrant::unvested), Some(0));
        assert_eq!(led.total_vested(), 20 + 30);
    }

    #[test]
    fn clawback_unknown_grant() {
        let mut led = VestingLedger::new();
        assert_eq!(led.clawback_unvested(99), Err(PayError::UnknownGrant(99)));
    }

    #[test]
    fn clawback_when_fully_vested_errors() {
        let mut led = VestingLedger::new();
        let id = led.open_grant(10, 1).expect("open");
        assert_eq!(led.advance_segment(), 10);
        assert_eq!(
            led.clawback_unvested(id),
            Err(PayError::NothingToClawback(id))
        );
    }
}
