//! Shared funding types.

use serde::{Deserialize, Serialize};

/// USD economics for a challenge GPU prepay quote.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FundingEconomics {
    /// GPU USD per hour.
    pub rate_usd_per_hour: f64,
    /// Billable hours (e.g. Prism train cap).
    pub hours: f64,
    /// Fractional buffer on top of USD cost (default 0.10).
    pub buffer: f64,
}

impl FundingEconomics {
    /// Prism defaults: $0.67/h × 6h × 1.10.
    #[must_use]
    pub const fn prism_default() -> Self {
        Self {
            rate_usd_per_hour: 0.67,
            hours: 6.0,
            buffer: 0.10,
        }
    }

    /// `rate * hours * (1 + buffer)`.
    #[must_use]
    pub fn usd_cost(&self) -> f64 {
        self.rate_usd_per_hour * self.hours * (1.0 + self.buffer)
    }
}

/// Lifecycle of a funding credit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreditState {
    /// Payment confirmed; unused.
    Unspent,
    /// Consumed on successful pod provision.
    Spent,
    /// Quote expired or superseded.
    Void,
}

/// Miner-facing quote + deposit instructions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingQuote {
    /// Opaque quote id.
    pub quote_id: String,
    /// Challenge id (`prism`, …).
    pub challenge_id: String,
    /// Miner hotkey (ss58 or hex as provided).
    pub hotkey: String,
    /// USD cost after buffer.
    pub usd_cost: f64,
    /// Economics used.
    pub rate_usd_per_hour: f64,
    /// Hours used.
    pub hours: f64,
    /// Buffer used.
    pub buffer: f64,
    /// Oracle USD per 1 TAO at quote time.
    pub tao_usd_price: f64,
    /// TAO the miner must send.
    pub tao_amount: f64,
    /// Operator deposit address (SS58).
    pub deposit_address: String,
    /// On-chain memo / reference.
    pub memo: String,
    /// Unix ms when the quote expires.
    pub expires_at_ms: u64,
}

/// Expected / observed deposit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingDeposit {
    /// Quote id this deposit satisfies.
    pub quote_id: String,
    /// Challenge id.
    pub challenge_id: String,
    /// Hotkey.
    pub hotkey: String,
    /// Expected TAO.
    pub expected_tao: f64,
    /// Observed TAO (0 until confirmed).
    pub observed_tao: f64,
    /// Deposit address.
    pub deposit_address: String,
    /// Memo.
    pub memo: String,
    /// Optional extrinsic / tx hash.
    pub tx_hash: Option<String>,
    /// Confirmed.
    pub confirmed: bool,
}

/// Granted credit after payment confirmation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingCredit {
    /// Credit id.
    pub credit_id: String,
    /// Quote that produced this credit.
    pub quote_id: String,
    /// Challenge.
    pub challenge_id: String,
    /// Hotkey.
    pub hotkey: String,
    /// USD covered.
    pub usd_cost: f64,
    /// TAO paid.
    pub tao_paid: f64,
    /// State.
    pub state: CreditState,
    /// Created unix ms.
    pub created_at_ms: u64,
    /// Spent unix ms (if any).
    pub spent_at_ms: Option<u64>,
}

/// Miner status projection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingStatus {
    /// Challenge.
    pub challenge_id: String,
    /// Hotkey.
    pub hotkey: String,
    /// Latest open quote (if any).
    pub quote: Option<FundingQuote>,
    /// Latest deposit tracking.
    pub deposit: Option<FundingDeposit>,
    /// Active unspent credit (if any).
    pub credit: Option<FundingCredit>,
    /// Whether the rent gate would pass when require-funding is on.
    pub rent_allowed: bool,
}
