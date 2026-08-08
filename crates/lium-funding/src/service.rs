//! Quote → expect payment → confirm → grant credit → consume.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::FundingConfig;
use crate::credit::FundingStore;
use crate::error::FundingError;
use crate::lium_account::LiumAccountClient;
use crate::oracle::{tao_from_usd, TaoPriceOracle};
use crate::payment::{funding_memo, TaoPaymentVerifier};
use crate::policy::ChallengeFundingPolicy;
use crate::types::{CreditState, FundingCredit, FundingDeposit, FundingQuote, FundingStatus};

/// Shared funding service (challenge-agnostic).
pub struct FundingService {
    cfg: FundingConfig,
    policy: Arc<dyn ChallengeFundingPolicy>,
    store: Arc<dyn FundingStore>,
    oracle: Arc<dyn TaoPriceOracle>,
    payments: Arc<dyn TaoPaymentVerifier>,
    lium: Arc<dyn LiumAccountClient>,
}

impl FundingService {
    /// Construct.
    #[must_use]
    pub fn new(
        cfg: FundingConfig,
        policy: Arc<dyn ChallengeFundingPolicy>,
        store: Arc<dyn FundingStore>,
        oracle: Arc<dyn TaoPriceOracle>,
        payments: Arc<dyn TaoPaymentVerifier>,
        lium: Arc<dyn LiumAccountClient>,
    ) -> Self {
        Self {
            cfg,
            policy,
            store,
            oracle,
            payments,
            lium,
        }
    }

    /// Config accessor.
    #[must_use]
    pub const fn cfg(&self) -> &FundingConfig {
        &self.cfg
    }

    /// Bound challenge id from the active policy.
    #[must_use]
    pub fn challenge_id(&self) -> &str {
        self.policy.challenge_id()
    }

    /// Issue a quote after eligibility checks.
    pub async fn quote(&self, hotkey: &str) -> Result<FundingQuote, FundingError> {
        let challenge_id = self.policy.challenge_id().to_owned();
        self.policy.ensure_eligible(hotkey).await?;
        if self.policy.one_funding_per_hotkey() {
            if let Some(c) = self.store.any_credit(&challenge_id, hotkey).await? {
                if c.state != CreditState::Void {
                    return Err(FundingError::Ineligible(
                        "hotkey already funded for this challenge".into(),
                    ));
                }
            }
        }
        let econ = self.cfg.economics;
        let usd = econ.usd_cost();
        let tao_usd = self.oracle.tao_usd_price().await?;
        let tao_amount = tao_from_usd(usd, tao_usd)?;
        let quote_id = uuid::Uuid::new_v4().to_string();
        let memo = funding_memo(&challenge_id, hotkey, &quote_id);
        let now = now_ms();
        let quote = FundingQuote {
            quote_id: quote_id.clone(),
            challenge_id: challenge_id.clone(),
            hotkey: hotkey.to_owned(),
            usd_cost: usd,
            rate_usd_per_hour: econ.rate_usd_per_hour,
            hours: econ.hours,
            buffer: econ.buffer,
            tao_usd_price: tao_usd,
            tao_amount,
            deposit_address: self.cfg.deposit_address.clone(),
            memo: memo.clone(),
            expires_at_ms: now.saturating_add(self.cfg.quote_ttl_secs.saturating_mul(1000)),
        };
        self.store.put_quote(quote.clone()).await?;
        let deposit = FundingDeposit {
            quote_id,
            challenge_id,
            hotkey: hotkey.to_owned(),
            expected_tao: tao_amount,
            observed_tao: 0.0,
            deposit_address: self.cfg.deposit_address.clone(),
            memo,
            tx_hash: None,
            confirmed: false,
        };
        self.store.put_deposit(deposit).await?;
        Ok(quote)
    }

    /// Refresh payment status; grant credit when confirmed.
    pub async fn confirm_if_paid(&self, hotkey: &str) -> Result<FundingStatus, FundingError> {
        let challenge_id = self.policy.challenge_id().to_owned();
        let quote = self.store.latest_quote(&challenge_id, hotkey).await?;
        let Some(quote) = quote else {
            let credit = self.store.get_unspent(&challenge_id, hotkey).await?;
            let rent_allowed = self.rent_allowed_inner(&challenge_id, hotkey).await?;
            return Ok(FundingStatus {
                challenge_id,
                hotkey: hotkey.to_owned(),
                quote: None,
                deposit: None,
                credit,
                rent_allowed,
            });
        };
        let Some(mut deposit) = self.store.get_deposit(&quote.quote_id).await? else {
            return Err(FundingError::Payment("deposit missing for quote".into()));
        };
        if !deposit.confirmed {
            deposit = self.payments.check_payment(&deposit).await?;
            self.store.put_deposit(deposit.clone()).await?;
            if deposit.confirmed {
                // Optional operator Lium balance probe (informational; not a hard gate).
                let _ = self.lium.balance_usd().await;
                if self
                    .store
                    .get_unspent(&challenge_id, hotkey)
                    .await?
                    .is_none()
                {
                    let credit = FundingCredit {
                        credit_id: uuid::Uuid::new_v4().to_string(),
                        quote_id: quote.quote_id.clone(),
                        challenge_id: challenge_id.clone(),
                        hotkey: hotkey.to_owned(),
                        usd_cost: quote.usd_cost,
                        tao_paid: deposit.observed_tao,
                        state: CreditState::Unspent,
                        created_at_ms: now_ms(),
                        spent_at_ms: None,
                    };
                    self.store.put_credit(credit).await?;
                }
            }
        }
        let credit = self.store.any_credit(&challenge_id, hotkey).await?;
        Ok(FundingStatus {
            rent_allowed: self.rent_allowed_inner(&challenge_id, hotkey).await?,
            challenge_id,
            hotkey: hotkey.to_owned(),
            quote: Some(quote),
            deposit: Some(deposit),
            credit,
        })
    }

    /// Status without forcing a payment check.
    pub async fn status(&self, hotkey: &str) -> Result<FundingStatus, FundingError> {
        self.confirm_if_paid(hotkey).await
    }

    /// Rent gate: no-op when `require_funding` is false.
    pub async fn before_rent(&self, hotkey: &str) -> Result<(), FundingError> {
        if !self.cfg.require_funding {
            return Ok(());
        }
        let challenge_id = self.policy.challenge_id();
        match self.store.get_unspent(challenge_id, hotkey).await? {
            Some(_) => Ok(()),
            None => Err(FundingError::Credit(
                "unspent funding credit required before Lium rent".into(),
            )),
        }
    }

    /// Consume credit after successful provision (no-op when gate off and no credit).
    pub async fn consume_on_provision(&self, hotkey: &str) -> Result<(), FundingError> {
        if !self.cfg.require_funding {
            // Still consume if a credit exists (keeps ledger honest in tests with flag off).
            if self
                .store
                .get_unspent(self.policy.challenge_id(), hotkey)
                .await?
                .is_none()
            {
                return Ok(());
            }
        }
        let _ = self
            .store
            .consume(self.policy.challenge_id(), hotkey, now_ms())
            .await?;
        Ok(())
    }

    /// Admin list.
    pub async fn list_credits(&self) -> Result<Vec<FundingCredit>, FundingError> {
        self.store.list_credits().await
    }

    async fn rent_allowed_inner(
        &self,
        challenge_id: &str,
        hotkey: &str,
    ) -> Result<bool, FundingError> {
        if !self.cfg.require_funding {
            return Ok(true);
        }
        Ok(self
            .store
            .get_unspent(challenge_id, hotkey)
            .await?
            .is_some())
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}
