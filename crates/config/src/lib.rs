//! Typed layered configuration for base binaries.
//!
//! # Layers (later wins)
//!
//! 1. **Defaults** — chain endpoint, epoch length, D6/D21/D26 knobs.
//! 2. **`base.toml`** (or path from `BASE_CONFIG`) — optional file.
//! 3. **`BASE_*` environment variables** — highest precedence.
//!
//! # Validation
//!
//! [`Config::validate`] and the load path accumulate **all** problems into a
//! [`ValidationReport`] instead of failing on the first error.

#![forbid(unsafe_code)]

mod error;
mod load;
mod netuid;
mod role;

pub use error::{Error, Issue, ValidationReport};
pub use load::{
    base_env_from_process, database_url_from, keys, load, load_from, load_from_toml_str,
    resolve_toml_path, DEFAULT_CHAIN_ENDPOINT, DEFAULT_EPOCH_LENGTH,
    DEFAULT_MAX_COLLATERAL_AGE_SECS, DEFAULT_MIN_PEER_SAMPLE, DEFAULT_MIN_SHARE_MASS_BPS,
    DEFAULT_ROTATION_EPOCHS, MAX_SHARE_MASS_BPS,
};
pub use netuid::{NetUid, ParseNetUidError};
pub use role::{ParseRoleError, Role};

use std::path::PathBuf;

/// Fully resolved runtime configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Process role.
    pub role: Role,
    /// Subnet netuid.
    pub netuid: NetUid,
    /// Chain WebSocket endpoint. May hold a comma-separated ordered failover
    /// list (`BASE_CHAIN_ENDPOINTS` wins over `BASE_CHAIN_ENDPOINT`);
    /// `chain-live` splits on `,` and tries endpoints in order.
    pub chain_endpoint: String,
    /// Optional gateway base URL (used by validators / miners).
    pub gateway_endpoint: Option<String>,
    /// Postgres URL (mutually exclusive with [`Self::database_url_file`]).
    pub database_url: Option<String>,
    /// Path to a file containing the Postgres URL.
    pub database_url_file: Option<PathBuf>,
    /// Epoch length in blocks.
    pub epoch_length: u64,
    /// Quarantine surviving-share floor in basis points (D6).
    pub min_share_mass_bps: u16,
    /// Trust-root dual-accept window in epochs (D21).
    pub rotation_epochs: u32,
    /// Minimum peer sample for cross-check (D26).
    pub min_peer_sample: u32,
    /// Max DCAP collateral age in seconds before park (D13).
    pub max_collateral_age_secs: u64,
    /// Public DNS zone (`BASE_DOMAIN`, D25). Required for [`Role::Gateway`].
    pub domain: Option<String>,
}

impl Config {
    /// Re-check semantic invariants; returns **all** problems at once.
    ///
    /// # Errors
    ///
    /// [`ValidationReport`] when one or more rules fail.
    pub fn validate(&self) -> Result<(), ValidationReport> {
        let mut issues = Vec::new();

        if self.min_share_mass_bps > MAX_SHARE_MASS_BPS {
            issues.push(Issue::MinShareMassBpsOutOfRange {
                value: self.min_share_mass_bps,
            });
        }
        if self.epoch_length == 0 {
            issues.push(Issue::EpochLengthZero);
        }
        if self.chain_endpoint.trim().is_empty() {
            issues.push(Issue::MissingField {
                field: "chain_endpoint",
            });
        }

        let has_db_url = self
            .database_url
            .as_ref()
            .is_some_and(|s| !s.trim().is_empty());
        let has_db_file = self.database_url_file.is_some();
        if has_db_url && has_db_file {
            issues.push(Issue::MutuallyExclusiveDatabaseUrlSources);
        }
        if self.role.requires_database() && !has_db_url && !has_db_file {
            issues.push(Issue::DatabaseRequired {
                role: self.role.to_string(),
            });
        }
        if self.role.requires_domain() {
            let domain_ok = self.domain.as_ref().is_some_and(|d| !d.trim().is_empty());
            if !domain_ok {
                issues.push(Issue::DomainRequiredForGateway);
            }
        }

        match ValidationReport::from_issues(issues) {
            None => Ok(()),
            Some(report) => Err(report),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use tempfile::tempdir;

    fn base_env() -> BTreeMap<String, String> {
        BTreeMap::from([
            (keys::ROLE.to_owned(), "validator".to_owned()),
            (keys::NETUID.to_owned(), "7".to_owned()),
            (
                keys::DATABASE_URL.to_owned(),
                "postgres://u:p@localhost/db".to_owned(),
            ),
        ])
    }

    #[test]
    fn s1_env_overrides_toml_and_defaults() {
        let toml = r#"
role = "miner"
netuid = 1
min_share_mass_bps = 4000
database_url = "postgres://file/db"
"#;
        let mut env = BTreeMap::new();
        env.insert(keys::ROLE.to_owned(), "validator".to_owned());
        env.insert(keys::NETUID.to_owned(), "42".to_owned());
        env.insert(keys::MIN_SHARE_MASS_BPS.to_owned(), "6000".to_owned());
        env.insert(
            keys::DATABASE_URL.to_owned(),
            "postgres://env/db".to_owned(),
        );
        env.insert(
            keys::CHAIN_ENDPOINT.to_owned(),
            "wss://env.example/ws".to_owned(),
        );

        let cfg = load_from_toml_str(Some(toml), &env).expect("load");

        assert_eq!(cfg.role, Role::Validator);
        assert_eq!(cfg.netuid.get(), 42);
        assert_eq!(cfg.min_share_mass_bps, 6000);
        assert_eq!(cfg.database_url.as_deref(), Some("postgres://env/db"));
        assert_eq!(cfg.chain_endpoint, "wss://env.example/ws");
        assert_eq!(cfg.rotation_epochs, DEFAULT_ROTATION_EPOCHS);
        assert_eq!(cfg.min_peer_sample, DEFAULT_MIN_PEER_SAMPLE);
        assert_eq!(cfg.epoch_length, DEFAULT_EPOCH_LENGTH);
        cfg.validate().expect("valid");
    }

    #[test]
    fn s1_toml_overrides_defaults_when_env_silent() {
        let toml = r#"
role = "miner"
netuid = 3
epoch_length = 100
"#;
        let cfg = load_from_toml_str(Some(toml), &BTreeMap::new()).expect("miner needs no db");
        assert_eq!(cfg.role, Role::Miner);
        assert_eq!(cfg.netuid.get(), 3);
        assert_eq!(cfg.epoch_length, 100);
        assert_eq!(cfg.chain_endpoint, DEFAULT_CHAIN_ENDPOINT);
        assert_eq!(cfg.min_share_mass_bps, DEFAULT_MIN_SHARE_MASS_BPS);
    }

    #[test]
    fn s1_chain_endpoints_list_wins_over_singular() {
        let mut env = base_env();
        env.insert(
            keys::CHAIN_ENDPOINT.to_owned(),
            "wss://one.example/ws".to_owned(),
        );
        env.insert(
            keys::CHAIN_ENDPOINTS.to_owned(),
            "wss://a.example/ws, wss://b.example/ws".to_owned(),
        );
        let cfg = load_from_toml_str(None, &env).expect("load");
        assert_eq!(cfg.chain_endpoint, "wss://a.example/ws, wss://b.example/ws");
    }

    #[test]
    fn s1_chain_endpoint_singular_when_list_absent() {
        let mut env = base_env();
        env.insert(
            keys::CHAIN_ENDPOINT.to_owned(),
            "wss://one.example/ws".to_owned(),
        );
        let cfg = load_from_toml_str(None, &env).expect("load");
        assert_eq!(cfg.chain_endpoint, "wss://one.example/ws");
    }

    #[test]
    fn s2_missing_domain_for_gateway() {
        let mut env = base_env();
        env.insert(keys::ROLE.to_owned(), "gateway".to_owned());

        let err = load_from_toml_str(None, &env).expect_err("gateway needs domain");
        let Error::Validation(report) = err;
        assert!(
            report.any(|i| matches!(i, Issue::DomainRequiredForGateway)),
            "report={report}"
        );
    }

    #[test]
    fn s2_gateway_with_domain_ok() {
        let mut env = base_env();
        env.insert(keys::ROLE.to_owned(), "gateway".to_owned());
        env.insert(keys::DOMAIN.to_owned(), "example.com".to_owned());
        let cfg = load_from_toml_str(None, &env).expect("ok");
        assert_eq!(cfg.domain.as_deref(), Some("example.com"));
        cfg.validate().expect("valid");
    }

    #[test]
    fn s3_invalid_netuid_rejected() {
        let mut env = base_env();
        env.insert(keys::NETUID.to_owned(), "not-a-number".to_owned());
        let err = load_from_toml_str(None, &env).expect_err("bad netuid");
        let Error::Validation(report) = err;
        assert!(report.any(|i| matches!(
            i,
            Issue::InvalidValue {
                field: "netuid",
                ..
            } | Issue::MissingField { field: "netuid" }
        )));
    }

    #[test]
    fn s3_netuid_overflow_u16() {
        let mut env = base_env();
        env.insert(keys::NETUID.to_owned(), "70000".to_owned());
        let err = load_from_toml_str(None, &env).expect_err("overflow");
        let Error::Validation(report) = err;
        assert!(report.any(|i| matches!(
            i,
            Issue::InvalidValue {
                field: "netuid",
                ..
            }
        )));
    }

    #[test]
    fn s4_min_share_mass_bps_out_of_range() {
        let mut env = base_env();
        env.insert(keys::MIN_SHARE_MASS_BPS.to_owned(), "10001".to_owned());
        let err = load_from_toml_str(None, &env).expect_err("bps");
        let Error::Validation(report) = err;
        assert!(report.any(|i| matches!(i, Issue::MinShareMassBpsOutOfRange { value: 10001 })));
    }

    #[test]
    fn s5_mutually_exclusive_database_url_sources() {
        let mut env = base_env();
        env.insert(
            keys::DATABASE_URL_FILE.to_owned(),
            "/run/secrets/db_url".to_owned(),
        );

        let err = load_from_toml_str(None, &env).expect_err("mutex");
        let Error::Validation(report) = err;
        assert!(
            report.any(|i| matches!(i, Issue::MutuallyExclusiveDatabaseUrlSources)),
            "report={report}"
        );
    }

    #[test]
    fn s5_database_url_file_alone_ok_for_validator() {
        let env = BTreeMap::from([
            (keys::ROLE.to_owned(), "validator".to_owned()),
            (keys::NETUID.to_owned(), "1".to_owned()),
            (
                keys::DATABASE_URL_FILE.to_owned(),
                "/run/secrets/db_url".to_owned(),
            ),
        ]);
        let cfg = load_from_toml_str(None, &env).expect("file only");
        assert!(cfg.database_url.is_none());
        assert_eq!(
            cfg.database_url_file.as_deref(),
            Some(std::path::Path::new("/run/secrets/db_url"))
        );
    }

    #[test]
    fn s6_validate_accumulates_multiple_issues() {
        let cfg = Config {
            role: Role::Gateway,
            netuid: NetUid::new(1),
            chain_endpoint: String::new(),
            gateway_endpoint: None,
            database_url: Some("postgres://x".to_owned()),
            database_url_file: Some(PathBuf::from("/secret")),
            epoch_length: 0,
            min_share_mass_bps: 12_000,
            rotation_epochs: 3,
            min_peer_sample: 1,
            max_collateral_age_secs: 1,
            domain: None,
        };
        let report = cfg.validate().expect_err("many issues");
        assert!(report.len() >= 4, "report={report}");
        assert!(report.any(|i| matches!(i, Issue::DomainRequiredForGateway)));
        assert!(report.any(|i| matches!(i, Issue::MutuallyExclusiveDatabaseUrlSources)));
        assert!(report.any(|i| matches!(i, Issue::EpochLengthZero)));
        assert!(report.any(|i| matches!(i, Issue::MinShareMassBpsOutOfRange { .. })));
        assert!(report.any(|i| matches!(
            i,
            Issue::MissingField {
                field: "chain_endpoint"
            }
        )));
    }

    #[test]
    fn s6_missing_role_and_netuid_together() {
        let err = load_from_toml_str(None, &BTreeMap::new()).expect_err("missing");
        let Error::Validation(report) = err;
        assert!(report.any(|i| matches!(i, Issue::MissingField { field: "role" })));
        assert!(report.any(|i| matches!(i, Issue::MissingField { field: "netuid" })));
        assert!(report.len() >= 2);
    }

    #[test]
    fn load_from_real_toml_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("base.toml");
        std::fs::write(
            &path,
            r#"
role = "updater"
netuid = 5
database_url = "postgres://u:p@localhost/base"
domain = "ignored-for-updater.example"
"#,
        )
        .unwrap();
        let cfg = load_from(Some(&path), &BTreeMap::new()).expect("file load");
        assert_eq!(cfg.role, Role::Updater);
        assert_eq!(cfg.netuid.get(), 5);
    }

    /// The challenge daemons persist to Postgres but have no [`Role`], so
    /// asking them for one just to locate the database made dispatch
    /// impossible to enable with the shipped compose.
    #[test]
    fn database_url_resolves_without_a_role() {
        let mut env = BTreeMap::new();
        env.insert(
            keys::DATABASE_URL.to_owned(),
            "postgres://base@db/base".to_owned(),
        );
        assert!(load_from(None, &env).is_err(), "role is still required");
        assert_eq!(
            database_url_from(None, &env).expect("resolve"),
            Some("postgres://base@db/base".to_owned())
        );
    }

    #[test]
    fn database_url_is_absent_rather_than_an_error_when_unset() {
        assert_eq!(
            database_url_from(None, &BTreeMap::new()).expect("resolve"),
            None
        );
    }
}
