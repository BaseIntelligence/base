//! Funding credit ledger (memory first; DB later).

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::error::FundingError;
use crate::types::{CreditState, FundingCredit, FundingDeposit, FundingQuote};

/// Persist quotes, deposits, and credits.
#[async_trait]
pub trait FundingStore: Send + Sync {
    async fn put_quote(&self, quote: FundingQuote) -> Result<(), FundingError>;
    async fn get_quote(&self, quote_id: &str) -> Result<Option<FundingQuote>, FundingError>;
    async fn latest_quote(
        &self,
        challenge_id: &str,
        hotkey: &str,
    ) -> Result<Option<FundingQuote>, FundingError>;

    async fn put_deposit(&self, deposit: FundingDeposit) -> Result<(), FundingError>;
    async fn get_deposit(&self, quote_id: &str) -> Result<Option<FundingDeposit>, FundingError>;

    async fn put_credit(&self, credit: FundingCredit) -> Result<(), FundingError>;
    async fn get_unspent(
        &self,
        challenge_id: &str,
        hotkey: &str,
    ) -> Result<Option<FundingCredit>, FundingError>;
    async fn any_credit(
        &self,
        challenge_id: &str,
        hotkey: &str,
    ) -> Result<Option<FundingCredit>, FundingError>;
    async fn consume(
        &self,
        challenge_id: &str,
        hotkey: &str,
        now_ms: u64,
    ) -> Result<FundingCredit, FundingError>;
    async fn list_credits(&self) -> Result<Vec<FundingCredit>, FundingError>;
}

/// In-memory store for tests and local scaffolding.
#[derive(Debug, Default)]
pub struct MemoryFundingStore {
    quotes: Mutex<HashMap<String, FundingQuote>>,
    deposits: Mutex<HashMap<String, FundingDeposit>>,
    credits: Mutex<Vec<FundingCredit>>,
}

#[async_trait]
impl FundingStore for MemoryFundingStore {
    async fn put_quote(&self, quote: FundingQuote) -> Result<(), FundingError> {
        let mut g = self
            .quotes
            .lock()
            .map_err(|_| FundingError::Store("quotes lock".into()))?;
        g.insert(quote.quote_id.clone(), quote);
        Ok(())
    }

    async fn get_quote(&self, quote_id: &str) -> Result<Option<FundingQuote>, FundingError> {
        let g = self
            .quotes
            .lock()
            .map_err(|_| FundingError::Store("quotes lock".into()))?;
        Ok(g.get(quote_id).cloned())
    }

    async fn latest_quote(
        &self,
        challenge_id: &str,
        hotkey: &str,
    ) -> Result<Option<FundingQuote>, FundingError> {
        let g = self
            .quotes
            .lock()
            .map_err(|_| FundingError::Store("quotes lock".into()))?;
        Ok(g.values()
            .filter(|q| q.challenge_id == challenge_id && q.hotkey == hotkey)
            .max_by_key(|q| q.expires_at_ms)
            .cloned())
    }

    async fn put_deposit(&self, deposit: FundingDeposit) -> Result<(), FundingError> {
        let mut g = self
            .deposits
            .lock()
            .map_err(|_| FundingError::Store("deposits lock".into()))?;
        g.insert(deposit.quote_id.clone(), deposit);
        Ok(())
    }

    async fn get_deposit(&self, quote_id: &str) -> Result<Option<FundingDeposit>, FundingError> {
        let g = self
            .deposits
            .lock()
            .map_err(|_| FundingError::Store("deposits lock".into()))?;
        Ok(g.get(quote_id).cloned())
    }

    async fn put_credit(&self, credit: FundingCredit) -> Result<(), FundingError> {
        let mut g = self
            .credits
            .lock()
            .map_err(|_| FundingError::Store("credits lock".into()))?;
        g.push(credit);
        Ok(())
    }

    async fn get_unspent(
        &self,
        challenge_id: &str,
        hotkey: &str,
    ) -> Result<Option<FundingCredit>, FundingError> {
        let g = self
            .credits
            .lock()
            .map_err(|_| FundingError::Store("credits lock".into()))?;
        Ok(g.iter()
            .find(|c| {
                c.challenge_id == challenge_id
                    && c.hotkey == hotkey
                    && c.state == CreditState::Unspent
            })
            .cloned())
    }

    async fn any_credit(
        &self,
        challenge_id: &str,
        hotkey: &str,
    ) -> Result<Option<FundingCredit>, FundingError> {
        let g = self
            .credits
            .lock()
            .map_err(|_| FundingError::Store("credits lock".into()))?;
        Ok(g.iter()
            .filter(|c| c.challenge_id == challenge_id && c.hotkey == hotkey)
            .max_by_key(|c| c.created_at_ms)
            .cloned())
    }

    async fn consume(
        &self,
        challenge_id: &str,
        hotkey: &str,
        now_ms: u64,
    ) -> Result<FundingCredit, FundingError> {
        let mut g = self
            .credits
            .lock()
            .map_err(|_| FundingError::Store("credits lock".into()))?;
        let credit = g.iter_mut().find(|c| {
            c.challenge_id == challenge_id && c.hotkey == hotkey && c.state == CreditState::Unspent
        });
        let Some(c) = credit else {
            return Err(FundingError::Credit("no unspent credit".into()));
        };
        c.state = CreditState::Spent;
        c.spent_at_ms = Some(now_ms);
        Ok(c.clone())
    }

    async fn list_credits(&self) -> Result<Vec<FundingCredit>, FundingError> {
        let g = self
            .credits
            .lock()
            .map_err(|_| FundingError::Store("credits lock".into()))?;
        Ok(g.clone())
    }
}
