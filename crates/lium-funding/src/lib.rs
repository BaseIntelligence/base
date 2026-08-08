//! Shared challenge GPU prepay via operator Lium wallet + TAO deposits.
//!
//! See [`docs/LIUM_FUNDING.md`](../../docs/LIUM_FUNDING.md).

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_fields_in_debug)] // api_key intentionally redacted
#![allow(clippy::result_large_err)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::redundant_closure_for_method_calls)]

mod config;
mod credit;
mod error;
mod http;
mod lium_account;
mod oracle;
mod payment;
mod policy;
mod service;
mod types;

pub use config::FundingConfig;
pub use credit::{FundingStore, MemoryFundingStore};
pub use error::FundingError;
pub use http::{funding_router, FundingHttpState};
pub use lium_account::{FakeLiumAccount, HttpLiumAccount, LiumAccountClient, LIUM_API_BASE_URL};
pub use oracle::{tao_from_usd, EnvTaoOracle, FixedTaoOracle, TaoPriceOracle};
pub use payment::{funding_memo, FakeTaoVerifier, TaoPaymentVerifier};
pub use policy::{
    AllowlistEligibility, ChallengeFundingPolicy, EligibilityChecker, OpenFundingPolicy,
    PrismFundingPolicy,
};
pub use service::FundingService;
pub use types::{
    CreditState, FundingCredit, FundingDeposit, FundingEconomics, FundingQuote, FundingStatus,
};

/// Crate identity.
#[must_use]
pub fn crate_name() -> &'static str {
    "lium-funding"
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    #![allow(clippy::expect_used)]
    #![allow(clippy::float_cmp)]

    use std::sync::Arc;

    use super::*;

    fn prism_stack(
        require: bool,
    ) -> (
        Arc<FundingService>,
        Arc<FakeTaoVerifier>,
        Arc<AllowlistEligibility>,
    ) {
        let elig = Arc::new(AllowlistEligibility::default());
        elig.allowed.lock().unwrap().push("hk1".into());
        let payments = Arc::new(FakeTaoVerifier::default());
        let mut cfg = FundingConfig::default();
        cfg.require_funding = require;
        cfg.deposit_address = "5TestDepositAddress".into();
        cfg.economics = FundingEconomics::prism_default();
        let policy = Arc::new(PrismFundingPolicy::new(
            elig.clone() as Arc<dyn EligibilityChecker>
        ));
        let svc = Arc::new(FundingService::new(
            cfg,
            policy,
            Arc::new(MemoryFundingStore::default()),
            Arc::new(FixedTaoOracle { usd_per_tao: 400.0 }),
            payments.clone() as Arc<dyn TaoPaymentVerifier>,
            Arc::new(FakeLiumAccount { balance_usd: 100.0 }),
        ));
        (svc, payments, elig)
    }

    #[test]
    fn quote_math_prism_default() {
        let e = FundingEconomics::prism_default();
        let usd = e.usd_cost();
        assert!((usd - 4.422).abs() < 1e-9, "usd={usd}");
        let tao = tao_from_usd(usd, 400.0).unwrap();
        assert!((tao - 0.011_055).abs() < 1e-9, "tao={tao}");
    }

    #[tokio::test]
    async fn eligibility_rejects_unknown_hotkey() {
        let (svc, _, _) = prism_stack(false);
        let err = svc.quote("unknown").await.unwrap_err();
        assert!(matches!(err, FundingError::Ineligible(_)));
    }

    #[tokio::test]
    async fn credit_consume_once() {
        let (svc, payments, _) = prism_stack(true);
        let q = svc.quote("hk1").await.unwrap();
        payments
            .confirm(&q.memo, q.tao_amount, Some("0xabc".into()))
            .unwrap();
        let st = svc.confirm_if_paid("hk1").await.unwrap();
        assert!(st.credit.as_ref().unwrap().state == CreditState::Unspent);
        svc.before_rent("hk1").await.unwrap();
        svc.consume_on_provision("hk1").await.unwrap();
        let err = svc.before_rent("hk1").await.unwrap_err();
        assert!(matches!(err, FundingError::Credit(_)));
        // Second consume fails.
        let err = svc.consume_on_provision("hk1").await.unwrap_err();
        assert!(matches!(err, FundingError::Credit(_)));
    }

    #[tokio::test]
    async fn feature_flag_off_skips_rent_gate() {
        let (svc, _, _) = prism_stack(false);
        // No credit, but require_funding=false → Ok.
        svc.before_rent("hk1").await.unwrap();
        svc.consume_on_provision("hk1").await.unwrap();
    }

    #[tokio::test]
    async fn one_funding_per_hotkey() {
        let (svc, payments, _) = prism_stack(false);
        let q = svc.quote("hk1").await.unwrap();
        payments.confirm(&q.memo, q.tao_amount, None).unwrap();
        let _ = svc.confirm_if_paid("hk1").await.unwrap();
        let err = svc.quote("hk1").await.unwrap_err();
        assert!(matches!(err, FundingError::Ineligible(_)));
    }

    #[test]
    fn identity() {
        assert_eq!(crate_name(), "lium-funding");
    }
}
