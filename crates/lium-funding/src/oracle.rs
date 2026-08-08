//! TAO/USD price oracle (pluggable; fixed/env for testnet first).

use async_trait::async_trait;

use crate::error::FundingError;

/// USD price of one TAO.
#[async_trait]
pub trait TaoPriceOracle: Send + Sync {
    /// Current USD per 1 TAO.
    async fn tao_usd_price(&self) -> Result<f64, FundingError>;
}

/// Constant price (tests / local).
#[derive(Debug, Clone)]
pub struct FixedTaoOracle {
    /// USD per TAO.
    pub usd_per_tao: f64,
}

#[async_trait]
impl TaoPriceOracle for FixedTaoOracle {
    async fn tao_usd_price(&self) -> Result<f64, FundingError> {
        if self.usd_per_tao <= 0.0 || !self.usd_per_tao.is_finite() {
            return Err(FundingError::Quote("invalid fixed TAO/USD".into()));
        }
        Ok(self.usd_per_tao)
    }
}

/// Read `LIUM_FUNDING_TAO_USD` (or constructor value) each call.
#[derive(Debug, Clone)]
pub struct EnvTaoOracle {
    /// Fallback when env unset.
    pub fallback_usd_per_tao: f64,
}

#[async_trait]
impl TaoPriceOracle for EnvTaoOracle {
    async fn tao_usd_price(&self) -> Result<f64, FundingError> {
        let v = std::env::var("LIUM_FUNDING_TAO_USD")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(self.fallback_usd_per_tao);
        if v <= 0.0 || !v.is_finite() {
            return Err(FundingError::Quote(
                "LIUM_FUNDING_TAO_USD missing or invalid".into(),
            ));
        }
        Ok(v)
    }
}

/// `usd_cost / tao_usd_price`.
pub fn tao_from_usd(usd_cost: f64, tao_usd_price: f64) -> Result<f64, FundingError> {
    if usd_cost < 0.0 || !usd_cost.is_finite() {
        return Err(FundingError::Quote("usd_cost invalid".into()));
    }
    if tao_usd_price <= 0.0 || !tao_usd_price.is_finite() {
        return Err(FundingError::Quote("tao_usd_price invalid".into()));
    }
    let tao = usd_cost / tao_usd_price;
    if !tao.is_finite() || tao <= 0.0 {
        return Err(FundingError::Quote("tao_amount invalid".into()));
    }
    Ok(tao)
}
