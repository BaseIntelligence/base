//! Promotion machine: transitions, BH cohort duel, champion install, rollback.

use crate::bh::benjamini_hochberg;
use crate::error::PromoError;
use crate::lineage::CheckpointLineage;
use crate::state::{
    ChallengerId, CheckpointHash, PromoState, RejectReason, ALPHA, PROMOTION_K, SCREEN_K,
};

/// Evidence for the K=3 screen stage (brief §9.1 criblage).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenEvidence {
    /// Kernel gate already passed (Guard 1).
    pub kernel_passed: bool,
    /// Median wallclock of the candidate over K=3 (ms).
    pub candidate_median_ms: u64,
    /// Median wallclock of the champion over the same seeds (ms).
    pub champion_median_ms: u64,
    /// Sign of `d_i` coherent across the K screen runs.
    pub sign_coherent: bool,
    /// Must equal [`SCREEN_K`].
    pub k: usize,
}

/// Per-challenger duel inputs (K=5 stage).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DuelEvidence {
    /// Raw one-sided paired-test p-value on wallclock / quality.
    pub p_value: f64,
    /// Guard 2 non-inferiority passed.
    pub non_inferiority: bool,
    /// Guard 3 physical plausibility passed.
    pub physical_plausible: bool,
    /// Must equal [`PROMOTION_K`] (or elevated K from anti-noise).
    pub k: usize,
}

/// Private holdout confirmation (brief §9.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HoldoutEvidence {
    /// Sign of improvement agrees with the main duel.
    pub sign_agrees: bool,
}

/// One challenger tracked by the machine.
#[derive(Debug, Clone, PartialEq)]
pub struct Challenger {
    /// Validator-assigned id.
    pub id: ChallengerId,
    /// Lifecycle state.
    pub state: PromoState,
    /// Candidate checkpoint hash (artifact under evaluation).
    pub checkpoint_hash: CheckpointHash,
}

/// In-memory promotion controller (validator-side).
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionMachine {
    /// Active / historical challengers by insertion order.
    challengers: Vec<Challenger>,
    /// Public hashed champion lineage.
    lineage: CheckpointLineage,
    /// FDR level (default [`ALPHA`]).
    alpha: f64,
    /// Next challenger id allocator.
    next_id: ChallengerId,
}

impl Default for PromotionMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl PromotionMachine {
    /// Fresh machine with empty lineage (no champion).
    #[must_use]
    pub fn new() -> Self {
        Self {
            challengers: Vec::new(),
            lineage: CheckpointLineage::new(),
            alpha: ALPHA,
            next_id: 1,
        }
    }

    /// Override FDR α (tests / config).
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Install a genesis champion C(0) without a challenger path (bootstrap).
    ///
    /// Errors if a champion already exists.
    pub fn bootstrap_champion(
        &mut self,
        checkpoint_hash: CheckpointHash,
    ) -> Result<(), PromoError> {
        if !self.lineage.is_empty() {
            return Err(PromoError::InvalidTransition {
                from: PromoState::Champion,
                action: "bootstrap",
            });
        }
        self.lineage.append(checkpoint_hash, None);
        Ok(())
    }

    /// Public lineage (hashed).
    #[must_use]
    pub fn lineage(&self) -> &CheckpointLineage {
        &self.lineage
    }

    /// Current champion checkpoint, if any.
    #[must_use]
    pub fn champion_hash(&self) -> Option<CheckpointHash> {
        self.lineage.tip_hash()
    }

    /// Borrow a challenger by id.
    #[must_use]
    pub fn challenger(&self, id: ChallengerId) -> Option<&Challenger> {
        self.challengers.iter().find(|c| c.id == id)
    }

    fn challenger_mut(&mut self, id: ChallengerId) -> Option<&mut Challenger> {
        self.challengers.iter_mut().find(|c| c.id == id)
    }

    /// Admit a new challenger in [`PromoState::Admitted`].
    pub fn admit(&mut self, checkpoint_hash: CheckpointHash) -> ChallengerId {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        self.challengers.push(Challenger {
            id,
            state: PromoState::Admitted,
            checkpoint_hash,
        });
        id
    }

    /// ADMITTED → SCREENED | REJECTED (K=3 screen).
    pub fn advance_screen(
        &mut self,
        id: ChallengerId,
        ev: ScreenEvidence,
    ) -> Result<PromoState, PromoError> {
        if self.lineage.tip_hash().is_none() {
            return Err(PromoError::NoChampion);
        }
        if ev.k != SCREEN_K {
            return Err(PromoError::StageRejected {
                reason: "screen_k_mismatch",
            });
        }
        let ch = self
            .challenger_mut(id)
            .ok_or(PromoError::InvalidTransition {
                from: PromoState::Admitted,
                action: "screen",
            })?;
        if ch.state != PromoState::Admitted {
            return Err(PromoError::InvalidTransition {
                from: ch.state,
                action: "screen",
            });
        }
        if !ev.kernel_passed {
            ch.state = PromoState::Rejected(RejectReason::ScreenFailed);
            return Ok(ch.state);
        }
        let faster = ev.candidate_median_ms < ev.champion_median_ms;
        if faster && ev.sign_coherent {
            ch.state = PromoState::Screened;
        } else {
            ch.state = PromoState::Rejected(RejectReason::ScreenFailed);
        }
        Ok(ch.state)
    }

    /// SCREENED → DUELLED | REJECTED for one challenger (solo BH cohort of 1).
    pub fn advance_duel(
        &mut self,
        id: ChallengerId,
        ev: DuelEvidence,
    ) -> Result<PromoState, PromoError> {
        self.advance_duel_cohort(&[(id, ev)])?;
        self.challenger(id)
            .map(|c| c.state)
            .ok_or(PromoError::InvalidTransition {
                from: PromoState::Screened,
                action: "duel",
            })
    }

    /// Apply K=5 duel + BH across a cohort of screened challengers.
    ///
    /// Each entry must be in [`PromoState::Screened`]. BH runs on the cohort
    /// p-values; a challenger advances only if BH rejects H0 **and** guards pass.
    pub fn advance_duel_cohort(
        &mut self,
        cohort: &[(ChallengerId, DuelEvidence)],
    ) -> Result<Vec<(ChallengerId, PromoState)>, PromoError> {
        if cohort.is_empty() {
            return Ok(Vec::new());
        }
        // Validate states first (no partial mutation on error).
        for &(id, ref ev) in cohort {
            let ch = self.challenger(id).ok_or(PromoError::InvalidTransition {
                from: PromoState::Screened,
                action: "duel",
            })?;
            if ch.state != PromoState::Screened {
                return Err(PromoError::InvalidTransition {
                    from: ch.state,
                    action: "duel",
                });
            }
            if ev.k < PROMOTION_K {
                return Err(PromoError::StageRejected {
                    reason: "duel_k_too_small",
                });
            }
        }

        let p_values: Vec<f64> = cohort.iter().map(|(_, e)| e.p_value).collect();
        let bh = benjamini_hochberg(&p_values, self.alpha)?;

        let mut out = Vec::with_capacity(cohort.len());
        for (i, &(id, ref ev)) in cohort.iter().enumerate() {
            let ch = self
                .challenger_mut(id)
                .ok_or(PromoError::InvalidTransition {
                    from: PromoState::Screened,
                    action: "duel",
                })?;
            let guards = ev.non_inferiority && ev.physical_plausible;
            let sig = bh[i];
            if guards && sig {
                ch.state = PromoState::Duelled;
            } else {
                ch.state = PromoState::Rejected(RejectReason::DuelFailed);
            }
            out.push((id, ch.state));
        }
        Ok(out)
    }

    /// DUELLED → CONFIRMED | REJECTED (holdout).
    pub fn advance_holdout(
        &mut self,
        id: ChallengerId,
        ev: HoldoutEvidence,
    ) -> Result<PromoState, PromoError> {
        let ch = self
            .challenger_mut(id)
            .ok_or(PromoError::InvalidTransition {
                from: PromoState::Duelled,
                action: "holdout",
            })?;
        if ch.state != PromoState::Duelled {
            return Err(PromoError::InvalidTransition {
                from: ch.state,
                action: "holdout",
            });
        }
        if ev.sign_agrees {
            ch.state = PromoState::Confirmed;
        } else {
            ch.state = PromoState::Rejected(RejectReason::HoldoutDisagreed);
        }
        Ok(ch.state)
    }

    /// CONFIRMED → CHAMPION; append public lineage tip.
    pub fn promote(&mut self, id: ChallengerId) -> Result<PromoState, PromoError> {
        let (hash, _) = {
            let ch = self
                .challenger(id)
                .ok_or(PromoError::InvalidTransition {
                    from: PromoState::Confirmed,
                    action: "promote",
                })?;
            if ch.state != PromoState::Confirmed {
                return Err(PromoError::InvalidTransition {
                    from: ch.state,
                    action: "promote",
                });
            }
            (ch.checkpoint_hash, ch.id)
        };
        self.lineage.append(hash, Some(id));
        let ch = self.challenger_mut(id).ok_or(PromoError::InvalidTransition {
            from: PromoState::Confirmed,
            action: "promote",
        })?;
        ch.state = PromoState::Champion;
        Ok(ch.state)
    }

    /// CHAMPION regression → roll lineage tip back to C(n-1).
    ///
    /// The current champion challenger (if tracked) moves to [`PromoState::RolledBack`].
    /// Returns the restored prior checkpoint hash.
    pub fn rollback(&mut self) -> Result<CheckpointHash, PromoError> {
        let tip_hash = self.lineage.tip_hash().ok_or(PromoError::NoChampion)?;
        let restored = self
            .lineage
            .rollback_tip()
            .ok_or(PromoError::NoPriorChampion)?;

        // Mark any challenger that held the rolled-back tip as RolledBack.
        for ch in &mut self.challengers {
            if ch.state == PromoState::Champion && ch.checkpoint_hash == tip_hash {
                ch.state = PromoState::RolledBack;
            }
        }
        Ok(restored.checkpoint_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ck(b: u8) -> CheckpointHash {
        [b; 32]
    }

    fn screen_pass() -> ScreenEvidence {
        ScreenEvidence {
            kernel_passed: true,
            candidate_median_ms: 900,
            champion_median_ms: 1000,
            sign_coherent: true,
            k: SCREEN_K,
        }
    }

    fn duel_pass(p: f64) -> DuelEvidence {
        DuelEvidence {
            p_value: p,
            non_inferiority: true,
            physical_plausible: true,
            k: PROMOTION_K,
        }
    }

    #[test]
    fn happy_path_to_champion() {
        let mut m = PromotionMachine::new();
        m.bootstrap_champion(ck(1)).expect("bootstrap");
        let id = m.admit(ck(2));
        assert_eq!(m.advance_screen(id, screen_pass()).unwrap(), PromoState::Screened);
        assert_eq!(
            m.advance_duel(id, duel_pass(0.01)).unwrap(),
            PromoState::Duelled
        );
        assert_eq!(
            m.advance_holdout(id, HoldoutEvidence { sign_agrees: true })
                .unwrap(),
            PromoState::Confirmed
        );
        assert_eq!(m.promote(id).unwrap(), PromoState::Champion);
        assert_eq!(m.champion_hash(), Some(ck(2)));
        assert_eq!(m.lineage().len(), 2);
        assert_eq!(m.lineage().prior().unwrap().checkpoint_hash, ck(1));
    }

    #[test]
    fn screen_fail_rejects() {
        let mut m = PromotionMachine::new();
        m.bootstrap_champion(ck(1)).expect("bootstrap");
        let id = m.admit(ck(2));
        let st = m
            .advance_screen(
                id,
                ScreenEvidence {
                    kernel_passed: true,
                    candidate_median_ms: 1100,
                    champion_median_ms: 1000,
                    sign_coherent: true,
                    k: SCREEN_K,
                },
            )
            .unwrap();
        assert_eq!(st, PromoState::Rejected(RejectReason::ScreenFailed));
    }

    #[test]
    fn holdout_disagree_rejects() {
        let mut m = PromotionMachine::new();
        m.bootstrap_champion(ck(1)).expect("bootstrap");
        let id = m.admit(ck(2));
        m.advance_screen(id, screen_pass()).unwrap();
        m.advance_duel(id, duel_pass(0.01)).unwrap();
        let st = m
            .advance_holdout(id, HoldoutEvidence { sign_agrees: false })
            .unwrap();
        assert_eq!(st, PromoState::Rejected(RejectReason::HoldoutDisagreed));
    }

    #[test]
    fn rollback_restores_prior_champion_hash() {
        let mut m = PromotionMachine::new();
        m.bootstrap_champion(ck(1)).expect("bootstrap");
        let id = m.admit(ck(2));
        m.advance_screen(id, screen_pass()).unwrap();
        m.advance_duel(id, duel_pass(0.01)).unwrap();
        m.advance_holdout(id, HoldoutEvidence { sign_agrees: true })
            .unwrap();
        m.promote(id).unwrap();
        let restored = m.rollback().unwrap();
        assert_eq!(restored, ck(1));
        assert_eq!(m.champion_hash(), Some(ck(1)));
        assert_eq!(
            m.challenger(id).unwrap().state,
            PromoState::RolledBack
        );
    }

    #[test]
    fn invalid_skip_duel_errors() {
        let mut m = PromotionMachine::new();
        m.bootstrap_champion(ck(1)).expect("bootstrap");
        let id = m.admit(ck(2));
        let err = m.advance_duel(id, duel_pass(0.01)).unwrap_err();
        assert!(matches!(
            err,
            PromoError::InvalidTransition {
                action: "duel",
                ..
            }
        ));
    }

    #[test]
    fn bh_cohort_only_significant_advances() {
        let mut m = PromotionMachine::new();
        m.bootstrap_champion(ck(1)).expect("bootstrap");
        let a = m.admit(ck(2));
        let b = m.admit(ck(3));
        m.advance_screen(a, screen_pass()).unwrap();
        m.advance_screen(b, screen_pass()).unwrap();
        let out = m
            .advance_duel_cohort(&[(a, duel_pass(0.01)), (b, duel_pass(0.20))])
            .unwrap();
        assert_eq!(out[0], (a, PromoState::Duelled));
        assert_eq!(out[1], (b, PromoState::Rejected(RejectReason::DuelFailed)));
    }
}
