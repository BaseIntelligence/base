//! TOML-facing and SCALE-facing trust-root types (D23/D24).

use parity_scale_codec::{Decode, Encode};
use serde::{Deserialize, Serialize};

use crate::error::TrustRootError;
use crate::hexutil::{decode_hex_array, encode_hex};
use gbase_crypto::KEY_LEN;

/// SHA-384 measurement register width (TDX MRTD / RTMR).
pub const REGISTER_LEN: usize = 48;
/// Compose-hash width (SHA-256).
pub const COMPOSE_HASH_LEN: usize = 32;

/// Maximum emission share mass in basis points (must sum exactly).
pub const BPS_DENOM: u16 = 10_000;

/// Kind of trust-root document (for load helpers / ceremony).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustRootKind {
    /// `config/challenges.toml` family.
    Challenges,
    /// `config/measurements.toml` family.
    Measurements,
}

/// Participant policy (`BUNDLE_SPEC` §7.1 / D24). SCALE discriminant matches enum order.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
#[allow(clippy::cast_possible_truncation)] // SCALE enum index is u8
pub enum ParticipantPolicy {
    /// Every hotkey in `metagraph_at(block_hash)`.
    AllMetagraphHotkeys,
    /// Stake ≥ `min_stake` at metagraph.
    StakeAtLeast {
        /// Minimum stake (chain units).
        min_stake: u64,
    },
    /// Explicit allowlist ∩ metagraph.
    ExplicitAllowlist {
        /// Sorted unique hotkeys (validated on load).
        hotkeys: Vec<[u8; KEY_LEN]>,
    },
    /// Metagraph minus deny list.
    AllExceptDenyList {
        /// Sorted unique hotkeys (validated on load).
        hotkeys: Vec<[u8; KEY_LEN]>,
    },
}

/// One challenge row in the owner-signed challenges body.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct ChallengeEntry {
    /// Challenge id bytes (UTF-8 in TOML).
    pub id: Vec<u8>,
    /// sr25519 public key that signs raw-weight leaves.
    pub public_key: [u8; KEY_LEN],
    /// Emission share in basis points (D23).
    pub emission_share_bps: u16,
    /// Declared participant policy (D24).
    pub policy: ParticipantPolicy,
}

/// Canonical signed challenges body (without outer version wrapper).
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, Default)]
pub struct ChallengesBody {
    /// Challenge rows sorted by ascending `id` bytes.
    pub challenges: Vec<ChallengeEntry>,
}

/// One allowed measurement profile (populated in task 35+).
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct MeasurementEntry {
    /// MRTD (48 bytes).
    pub mr_td: [u8; REGISTER_LEN],
    /// RTMR0.
    pub rtmr0: [u8; REGISTER_LEN],
    /// RTMR1.
    pub rtmr1: [u8; REGISTER_LEN],
    /// RTMR2.
    pub rtmr2: [u8; REGISTER_LEN],
    /// RTMR3.
    pub rtmr3: [u8; REGISTER_LEN],
    /// Canonical compose hash (32 bytes).
    pub compose_hash: [u8; COMPOSE_HASH_LEN],
}

/// Canonical signed measurements body. Empty ⇒ fail-closed (every quote rejected).
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, Default)]
pub struct MeasurementsBody {
    /// Allowlisted measurement profiles.
    pub entries: Vec<MeasurementEntry>,
}

impl MeasurementsBody {
    /// Whether any entry matches the quote measurements exactly.
    ///
    /// Empty body always returns `false` (fail closed).
    #[must_use]
    pub fn allows_quote(
        &self,
        mr_td: &[u8; REGISTER_LEN],
        rtmr0: &[u8; REGISTER_LEN],
        rtmr1: &[u8; REGISTER_LEN],
        rtmr2: &[u8; REGISTER_LEN],
        rtmr3: &[u8; REGISTER_LEN],
        compose_hash: &[u8; COMPOSE_HASH_LEN],
    ) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        self.entries.iter().any(|e| {
            e.mr_td == *mr_td
                && e.rtmr0 == *rtmr0
                && e.rtmr1 == *rtmr1
                && e.rtmr2 == *rtmr2
                && e.rtmr3 == *rtmr3
                && e.compose_hash == *compose_hash
        })
    }
}

impl ChallengesBody {
    /// Validate bps sum, unique ids, sorted hotkeys in policies.
    ///
    /// # Errors
    ///
    /// [`TrustRootError::InvalidBody`] on semantic failure.
    pub fn validate(&self) -> Result<(), TrustRootError> {
        let mut sum: u32 = 0;
        let mut prev_id: Option<&[u8]> = None;
        for c in &self.challenges {
            if c.id.is_empty() {
                return Err(TrustRootError::InvalidBody(
                    "challenge id must be non-empty".into(),
                ));
            }
            if let Some(prev) = prev_id {
                if c.id.as_slice() <= prev {
                    return Err(TrustRootError::InvalidBody(
                        "challenges must be strictly sorted by unique id".into(),
                    ));
                }
            }
            prev_id = Some(c.id.as_slice());
            sum = sum
                .checked_add(u32::from(c.emission_share_bps))
                .ok_or_else(|| TrustRootError::InvalidBody("bps overflow".into()))?;
            validate_policy(&c.policy)?;
        }
        if sum != u32::from(BPS_DENOM) {
            return Err(TrustRootError::InvalidBody(format!(
                "emission_share_bps sum must be {BPS_DENOM}, got {sum}"
            )));
        }
        Ok(())
    }

    /// Look up a challenge by id bytes.
    #[must_use]
    pub fn get(&self, id: &[u8]) -> Option<&ChallengeEntry> {
        self.challenges.iter().find(|c| c.id.as_slice() == id)
    }

    /// Emission shares as `(id, bps)` in body order (already sorted by id).
    #[must_use]
    pub fn emission_shares(&self) -> Vec<(Vec<u8>, u16)> {
        self.challenges
            .iter()
            .map(|c| (c.id.clone(), c.emission_share_bps))
            .collect()
    }
}

fn validate_policy(policy: &ParticipantPolicy) -> Result<(), TrustRootError> {
    match policy {
        ParticipantPolicy::AllMetagraphHotkeys | ParticipantPolicy::StakeAtLeast { .. } => Ok(()),
        ParticipantPolicy::ExplicitAllowlist { hotkeys }
        | ParticipantPolicy::AllExceptDenyList { hotkeys } => {
            let mut prev: Option<&[u8; KEY_LEN]> = None;
            for hk in hotkeys {
                if let Some(p) = prev {
                    if hk.as_slice() <= p.as_slice() {
                        return Err(TrustRootError::InvalidBody(
                            "policy hotkeys must be strictly sorted unique".into(),
                        ));
                    }
                }
                prev = Some(hk);
            }
            Ok(())
        }
    }
}

// --- TOML wire format -------------------------------------------------------

/// On-disk challenges.toml document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChallengesToml {
    /// Document format version (signed).
    pub version: u32,
    /// Epoch at which this version becomes eligible (D21).
    #[serde(default)]
    pub introduced_epoch: u64,
    /// Challenge rows.
    pub challenges: Vec<ChallengeToml>,
}

/// TOML challenge row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChallengeToml {
    /// Challenge id string.
    pub id: String,
    /// 32-byte public key as hex (optional `0x`).
    pub public_key: String,
    /// Emission share bps.
    pub emission_share_bps: u16,
    /// Participant policy.
    pub policy: PolicyToml,
}

/// TOML policy: string shorthand or table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum PolicyToml {
    /// Shorthand: `"all_metagraph_hotkeys"`.
    Name(String),
    /// Full table form.
    Table(PolicyTable),
}

/// Structured policy table.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PolicyTable {
    /// All metagraph hotkeys.
    AllMetagraphHotkeys,
    /// Stake threshold.
    StakeAtLeast {
        /// Minimum stake.
        min_stake: u64,
    },
    /// Allowlist.
    ExplicitAllowlist {
        /// Hotkeys as hex.
        hotkeys: Vec<String>,
    },
    /// Deny list.
    AllExceptDenyList {
        /// Hotkeys as hex.
        hotkeys: Vec<String>,
    },
}

/// On-disk measurements.toml document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeasurementsToml {
    /// Document format version (signed).
    pub version: u32,
    /// Epoch at which this version becomes eligible (D21).
    #[serde(default)]
    pub introduced_epoch: u64,
    /// Allowlist entries (empty = fail closed).
    #[serde(default)]
    pub measurements: Vec<MeasurementToml>,
}

/// TOML measurement row (hex fields).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeasurementToml {
    /// MRTD hex.
    pub mr_td: String,
    /// RTMR0 hex.
    pub rtmr0: String,
    /// RTMR1 hex.
    pub rtmr1: String,
    /// RTMR2 hex.
    pub rtmr2: String,
    /// RTMR3 hex.
    pub rtmr3: String,
    /// Compose hash hex (32 bytes).
    pub compose_hash: String,
}

impl ChallengesToml {
    /// Convert TOML → validated SCALE body.
    ///
    /// # Errors
    ///
    /// Encoding or semantic validation errors.
    pub fn to_body(&self) -> Result<ChallengesBody, TrustRootError> {
        let mut challenges = Vec::with_capacity(self.challenges.len());
        for row in &self.challenges {
            challenges.push(ChallengeEntry {
                id: row.id.as_bytes().to_vec(),
                public_key: decode_hex_array(&row.public_key)?,
                emission_share_bps: row.emission_share_bps,
                policy: policy_from_toml(&row.policy)?,
            });
        }
        challenges.sort_by(|a, b| a.id.cmp(&b.id));
        let body = ChallengesBody { challenges };
        body.validate()?;
        Ok(body)
    }
}

impl MeasurementsToml {
    /// Convert TOML → SCALE body (empty allowed).
    ///
    /// # Errors
    ///
    /// Hex decode errors.
    pub fn to_body(&self) -> Result<MeasurementsBody, TrustRootError> {
        let mut entries = Vec::with_capacity(self.measurements.len());
        for row in &self.measurements {
            entries.push(MeasurementEntry {
                mr_td: decode_hex_array(&row.mr_td)?,
                rtmr0: decode_hex_array(&row.rtmr0)?,
                rtmr1: decode_hex_array(&row.rtmr1)?,
                rtmr2: decode_hex_array(&row.rtmr2)?,
                rtmr3: decode_hex_array(&row.rtmr3)?,
                compose_hash: decode_hex_array(&row.compose_hash)?,
            });
        }
        Ok(MeasurementsBody { entries })
    }
}

fn policy_from_toml(p: &PolicyToml) -> Result<ParticipantPolicy, TrustRootError> {
    match p {
        PolicyToml::Name(name) => match name.as_str() {
            "all_metagraph_hotkeys" | "AllMetagraphHotkeys" => {
                Ok(ParticipantPolicy::AllMetagraphHotkeys)
            }
            other => Err(TrustRootError::InvalidBody(format!(
                "unknown policy shorthand {other:?}"
            ))),
        },
        PolicyToml::Table(PolicyTable::AllMetagraphHotkeys) => {
            Ok(ParticipantPolicy::AllMetagraphHotkeys)
        }
        PolicyToml::Table(PolicyTable::StakeAtLeast { min_stake }) => {
            Ok(ParticipantPolicy::StakeAtLeast {
                min_stake: *min_stake,
            })
        }
        PolicyToml::Table(PolicyTable::ExplicitAllowlist { hotkeys }) => {
            Ok(ParticipantPolicy::ExplicitAllowlist {
                hotkeys: decode_hotkey_list(hotkeys)?,
            })
        }
        PolicyToml::Table(PolicyTable::AllExceptDenyList { hotkeys }) => {
            Ok(ParticipantPolicy::AllExceptDenyList {
                hotkeys: decode_hotkey_list(hotkeys)?,
            })
        }
    }
}

fn decode_hotkey_list(hotkeys: &[String]) -> Result<Vec<[u8; KEY_LEN]>, TrustRootError> {
    let mut out = Vec::with_capacity(hotkeys.len());
    for h in hotkeys {
        out.push(decode_hex_array(h)?);
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Build a TOML challenge row from a SCALE entry (for ceremony helpers / tests).
#[must_use]
pub fn challenge_to_toml(entry: &ChallengeEntry) -> ChallengeToml {
    ChallengeToml {
        id: String::from_utf8_lossy(&entry.id).into_owned(),
        public_key: encode_hex(&entry.public_key),
        emission_share_bps: entry.emission_share_bps,
        policy: PolicyToml::Table(policy_to_table(&entry.policy)),
    }
}

fn policy_to_table(p: &ParticipantPolicy) -> PolicyTable {
    match p {
        ParticipantPolicy::AllMetagraphHotkeys => PolicyTable::AllMetagraphHotkeys,
        ParticipantPolicy::StakeAtLeast { min_stake } => PolicyTable::StakeAtLeast {
            min_stake: *min_stake,
        },
        ParticipantPolicy::ExplicitAllowlist { hotkeys } => PolicyTable::ExplicitAllowlist {
            hotkeys: hotkeys.iter().map(|h| encode_hex(h)).collect(),
        },
        ParticipantPolicy::AllExceptDenyList { hotkeys } => PolicyTable::AllExceptDenyList {
            hotkeys: hotkeys.iter().map(|h| encode_hex(h)).collect(),
        },
    }
}
