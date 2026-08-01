//! Control-plane attestation lookup (`AGENT_CHALLENGE` §3.1 I1, §3.3).
//!
//! The daemon never invents `Verified`: outcomes come from the shared
//! control-plane database written by the validator attestation path. Per §3.3 a
//! channel that is unavailable at scoring time is treated as Missing, which the
//! pure scorer turns into `NoScore{AttestationNotVerified}` — a lookup outage
//! can therefore never manufacture a score, only withhold one.

use std::collections::BTreeMap;

use agent_challenge::{AttestationLookup, AttestationStatus, Hotkey, KEY_LEN};

/// Outcome string written by the attestation control plane (table CHECK).
const OUTCOME_VERIFIED: &str = "verified";
/// Parked outcome — D13: grants no credit, never carries a prior `Verified`.
const OUTCOME_PARK: &str = "park";
/// Rejected outcome.
const OUTCOME_REJECT: &str = "reject";

/// Latest attestation row for one miner at one epoch.
///
/// Mirrors `db::AttestationRecord` field-for-field so [`query_attestation`]
/// stays a one-line adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttestationRecord {
    /// `verified` | `park` | `reject`.
    pub outcome: String,
    /// sr25519 receipt key from the measured `app_compose`; `verified` only.
    pub receipt_pk: Option<[u8; KEY_LEN]>,
}

/// Adapter over the shared control-plane query.
async fn query_attestation(
    pool: &db::PgPool,
    epoch: i64,
    miner_hotkey: &str,
) -> Result<Option<AttestationRecord>, String> {
    db::attestation_for_miner(pool, epoch, miner_hotkey)
        .await
        .map(|row| {
            row.map(|r| AttestationRecord {
                outcome: r.outcome,
                receipt_pk: r.receipt_pk,
            })
        })
        .map_err(|e| e.to_string())
}

/// One miner's attestation state for the epoch being scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinerAttestation {
    /// Status fed to the pure scorer's I1 gate.
    pub status: AttestationStatus,
    /// Receipt key to verify the work receipt against; `Verified` only.
    pub receipt_pk: Option<[u8; KEY_LEN]>,
}

impl MinerAttestation {
    /// Missing / undecided — the safe default for every miner (§3.3).
    #[must_use]
    pub const fn missing() -> Self {
        Self {
            status: AttestationStatus::Missing,
            receipt_pk: None,
        }
    }

    /// Verified with a pinned receipt key.
    #[must_use]
    pub const fn verified(receipt_pk: [u8; KEY_LEN]) -> Self {
        Self {
            status: AttestationStatus::Verified,
            receipt_pk: Some(receipt_pk),
        }
    }

    /// Map a control-plane row onto a scorer status.
    ///
    /// A `verified` row without a receipt key cannot bind any work receipt, so
    /// it is downgraded to Missing rather than trusted on its outcome string.
    #[must_use]
    fn from_record(record: &AttestationRecord) -> Self {
        match record.outcome.as_str() {
            OUTCOME_VERIFIED => record.receipt_pk.map_or_else(Self::missing, Self::verified),
            OUTCOME_PARK => Self {
                status: AttestationStatus::Parked,
                receipt_pk: None,
            },
            OUTCOME_REJECT => Self {
                status: AttestationStatus::Rejected,
                receipt_pk: None,
            },
            _ => Self::missing(),
        }
    }
}

/// Attestation outcomes for the expected set at one epoch.
///
/// Absence from the map is Missing, so a miner the control plane has never seen
/// is gated exactly like a rejected one.
#[derive(Debug, Clone, Default)]
pub struct EpochAttestations {
    by_miner: BTreeMap<Hotkey, MinerAttestation>,
}

impl EpochAttestations {
    /// Build from explicit per-miner rows (tests / injected control plane).
    #[must_use]
    pub fn new(by_miner: BTreeMap<Hotkey, MinerAttestation>) -> Self {
        Self { by_miner }
    }

    /// Attestation state for one miner, Missing when unseen.
    #[must_use]
    pub fn get(&self, miner: &Hotkey) -> MinerAttestation {
        self.by_miner
            .get(miner)
            .cloned()
            .unwrap_or_else(MinerAttestation::missing)
    }
}

impl AttestationLookup for EpochAttestations {
    fn status(&self, _netuid: u16, _epoch: u64, miner: &Hotkey) -> AttestationStatus {
        self.get(miner).status
    }
}

/// Postgres-backed control-plane reader.
#[derive(Debug, Clone)]
pub struct ControlPlane {
    pool: db::PgPool,
}

impl ControlPlane {
    /// Connect and prove the control plane is reachable before serving.
    ///
    /// # Errors
    ///
    /// Connection failure. Degrading to "no attestations" here would silently
    /// zero the whole subnet, so the daemon refuses to start instead.
    pub async fn connect(database_url: &str) -> Result<Self, String> {
        let pool = db::connect(database_url)
            .await
            .map_err(|e| format!("attestation control plane connect failed: {e}"))?;
        Ok(Self { pool })
    }

    /// Read every expected miner's attestation for `epoch`.
    ///
    /// Per-miner lookup failures are logged and left Missing (§3.3); this call
    /// never fails, because a partial read must still cover all of `E` (D24).
    pub async fn epoch_attestations(&self, epoch: u64, expected: &[Hotkey]) -> EpochAttestations {
        let epoch_i64 = i64::try_from(epoch).unwrap_or(i64::MAX);
        let mut by_miner = BTreeMap::new();
        for miner in expected {
            let hotkey_hex = hex::encode(miner);
            match query_attestation(&self.pool, epoch_i64, &hotkey_hex).await {
                Ok(Some(record)) => {
                    by_miner.insert(*miner, MinerAttestation::from_record(&record));
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(
                        event = "attestation_lookup_failed",
                        epoch,
                        hotkey = %hotkey_hex,
                        error = %e,
                        "attestation channel unavailable; miner treated as Missing"
                    );
                }
            }
        }
        EpochAttestations::new(by_miner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINER: Hotkey = [0xA1; KEY_LEN];
    const PK: [u8; KEY_LEN] = [0x77; KEY_LEN];

    fn record(outcome: &str, receipt_pk: Option<[u8; KEY_LEN]>) -> AttestationRecord {
        AttestationRecord {
            outcome: outcome.to_owned(),
            receipt_pk,
        }
    }

    #[test]
    fn verified_row_carries_the_receipt_key() {
        let a = MinerAttestation::from_record(&record(OUTCOME_VERIFIED, Some(PK)));
        assert_eq!(a.status, AttestationStatus::Verified);
        assert_eq!(a.receipt_pk, Some(PK));
    }

    /// A `verified` outcome we cannot bind a receipt to is not usable.
    #[test]
    fn verified_without_receipt_key_is_missing() {
        let a = MinerAttestation::from_record(&record(OUTCOME_VERIFIED, None));
        assert_eq!(a.status, AttestationStatus::Missing);
        assert!(a.receipt_pk.is_none());
    }

    #[test]
    fn park_and_reject_never_carry_a_key() {
        for outcome in [OUTCOME_PARK, OUTCOME_REJECT] {
            let a = MinerAttestation::from_record(&record(outcome, Some(PK)));
            assert_ne!(a.status, AttestationStatus::Verified);
            assert!(a.receipt_pk.is_none(), "{outcome} must not pin a key");
        }
    }

    #[test]
    fn unknown_outcome_and_unseen_miner_are_missing() {
        assert_eq!(
            MinerAttestation::from_record(&record("banana", Some(PK))).status,
            AttestationStatus::Missing
        );
        let empty = EpochAttestations::default();
        assert_eq!(empty.get(&MINER).status, AttestationStatus::Missing);
        assert_eq!(empty.status(1, 7, &MINER), AttestationStatus::Missing);
    }
}
