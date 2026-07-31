//! Promotion lifecycle states (brief §9.4).

/// Opaque challenger identity (validator-assigned).
pub type ChallengerId = u64;

/// SHA-256 checkpoint content digest (public artifact).
pub type CheckpointHash = [u8; 32];

/// Why a challenger landed in [`PromoState::Rejected`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectReason {
    /// K=3 screen: slower wallclock or incoherent `d_i` sign.
    ScreenFailed,
    /// K=5 duel: non-inferiority, BH, or physical plausibility failed.
    DuelFailed,
    /// Private holdout disagreed in sign with the main duel.
    HoldoutDisagreed,
}

/// Promotion state machine nodes (brief §9.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromoState {
    /// Sealed admission done; ready for K=3 screen (kernel gate is a precondition).
    Admitted,
    /// Passed K=3 screen (median wallclock + sign coherence).
    Screened,
    /// Passed K=5 duel (non-inf + BH + plausibility).
    Duelled,
    /// Passed private holdout confirmation.
    Confirmed,
    /// Installed as current champion C(n).
    Champion,
    /// Terminal reject for this challenger.
    Rejected(RejectReason),
    /// Champion rolled back; prior checkpoint restored as C(n-1).
    RolledBack,
}

impl PromoState {
    /// Terminal states that accept no further forward transitions.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Rejected(_) | Self::RolledBack | Self::Champion)
    }
}

/// Screen K (criblage).
pub const SCREEN_K: usize = 3;
/// Promotion duel K.
pub const PROMOTION_K: usize = 5;
/// Family-wise α before BH (config default).
pub const ALPHA: f64 = 0.05;
/// Calibration K (config pin; not used by the state machine itself).
pub const CALIBRATION_K: usize = 10;
