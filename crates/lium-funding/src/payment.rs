//! TAO deposit verification (fake/testnet first; live watcher later).

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::error::FundingError;
use crate::types::FundingDeposit;

/// Observe TAO payments to the operator deposit address.
#[async_trait]
pub trait TaoPaymentVerifier: Send + Sync {
    /// Check whether `deposit` has been satisfied on-chain (or fake).
    async fn check_payment(&self, deposit: &FundingDeposit)
        -> Result<FundingDeposit, FundingError>;
}

/// In-memory fake: tests call [`FakeTaoVerifier::confirm`] to simulate payment.
#[derive(Debug, Default)]
pub struct FakeTaoVerifier {
    /// memo → observed TAO (+ optional tx).
    paid: Mutex<HashMap<String, (f64, Option<String>)>>,
}

impl FakeTaoVerifier {
    /// Record a fake payment for `memo`.
    ///
    /// # Errors
    /// Lock poisoned.
    pub fn confirm(
        &self,
        memo: &str,
        tao: f64,
        tx_hash: Option<String>,
    ) -> Result<(), FundingError> {
        let mut g = self
            .paid
            .lock()
            .map_err(|_| FundingError::Store("payment lock".into()))?;
        g.insert(memo.to_owned(), (tao, tx_hash));
        Ok(())
    }
}

#[async_trait]
impl TaoPaymentVerifier for FakeTaoVerifier {
    async fn check_payment(
        &self,
        deposit: &FundingDeposit,
    ) -> Result<FundingDeposit, FundingError> {
        let g = self
            .paid
            .lock()
            .map_err(|_| FundingError::Store("payment lock".into()))?;
        let mut out = deposit.clone();
        if let Some((tao, tx)) = g.get(&deposit.memo) {
            out.observed_tao = *tao;
            out.tx_hash = tx.clone();
            // Small underpay tolerance for float dust.
            out.confirmed = *tao + 1e-9 >= deposit.expected_tao;
        }
        Ok(out)
    }
}

/// Build a stable memo: `basefund:<challenge>:<hotkey_prefix>:<quote_prefix>`.
#[must_use]
pub fn funding_memo(challenge_id: &str, hotkey: &str, quote_id: &str) -> String {
    let hk = truncate(hotkey, 16);
    let q = truncate(quote_id, 12);
    format!("basefund:{challenge_id}:{hk}:{q}")
}

fn truncate(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}
