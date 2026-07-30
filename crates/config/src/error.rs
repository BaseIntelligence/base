//! Config load and validation errors.

use std::fmt;
use std::path::PathBuf;

/// One discrete problem found while loading or validating configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Issue {
    /// Required field was not set by defaults, file, or environment.
    MissingField {
        /// Field name (`snake_case`, matches TOML keys).
        field: &'static str,
    },
    /// Environment or TOML value could not be parsed.
    InvalidValue {
        /// Field or env key.
        field: &'static str,
        /// Human-readable reason.
        reason: String,
    },
    /// `min_share_mass_bps` must be in `0..=10_000`.
    MinShareMassBpsOutOfRange {
        /// Provided value.
        value: u16,
    },
    /// `epoch_length` must be non-zero.
    EpochLengthZero,
    /// `GBASE_DOMAIN` / `domain` required when `role = gateway` (D25).
    DomainRequiredForGateway,
    /// Server roles need a database URL source.
    DatabaseRequired {
        /// Role that needs the database.
        role: String,
    },
    /// `database_url` and `database_url_file` must not both be set.
    MutuallyExclusiveDatabaseUrlSources,
    /// TOML file could not be read or parsed.
    TomlIo {
        /// Path that failed.
        path: PathBuf,
        /// Underlying message.
        message: String,
    },
}

impl fmt::Display for Issue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingField { field } => write!(f, "missing required field `{field}`"),
            Self::InvalidValue { field, reason } => {
                write!(f, "invalid value for `{field}`: {reason}")
            }
            Self::MinShareMassBpsOutOfRange { value } => {
                write!(
                    f,
                    "min_share_mass_bps={value} out of range (expected 0..=10000)"
                )
            }
            Self::EpochLengthZero => write!(f, "epoch_length must be > 0"),
            Self::DomainRequiredForGateway => {
                write!(f, "domain (GBASE_DOMAIN) is required when role is gateway")
            }
            Self::DatabaseRequired { role } => {
                write!(
                    f,
                    "database_url or database_url_file is required for role `{role}`"
                )
            }
            Self::MutuallyExclusiveDatabaseUrlSources => write!(
                f,
                "database_url and database_url_file are mutually exclusive; set only one"
            ),
            Self::TomlIo { path, message } => {
                write!(
                    f,
                    "failed to read config file {}: {message}",
                    path.display()
                )
            }
        }
    }
}

/// Non-empty collection of configuration issues.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    issues: Vec<Issue>,
}

impl ValidationReport {
    /// Construct a report. Panics are avoided: empty input yields a single internal marker
    /// only via [`Self::invariant_violation`]; prefer [`Self::from_issues`].
    #[must_use]
    pub fn from_issues(issues: Vec<Issue>) -> Option<Self> {
        if issues.is_empty() {
            None
        } else {
            Some(Self { issues })
        }
    }

    /// Report used when an internal invariant is broken (should never surface in tests).
    #[must_use]
    pub fn invariant_violation() -> Self {
        Self {
            issues: vec![Issue::MissingField { field: "internal" }],
        }
    }

    /// All collected issues, in discovery order.
    #[must_use]
    pub fn issues(&self) -> &[Issue] {
        &self.issues
    }

    /// Number of issues.
    #[must_use]
    pub fn len(&self) -> usize {
        self.issues.len()
    }

    /// Whether the report has no issues.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.issues.is_empty()
    }

    /// True if any issue matches the predicate.
    #[must_use]
    pub fn any(&self, mut pred: impl FnMut(&Issue) -> bool) -> bool {
        self.issues.iter().any(&mut pred)
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} configuration problem(s):", self.issues.len())?;
        for (i, issue) in self.issues.iter().enumerate() {
            write!(f, "\n  {}. {issue}", i + 1)?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationReport {}

/// Top-level error for config loading.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// One or more validation / merge problems.
    #[error(transparent)]
    Validation(#[from] ValidationReport),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_issues_rejects_empty() {
        assert!(ValidationReport::from_issues(vec![]).is_none());
    }

    #[test]
    fn display_lists_all_issues() {
        let report = ValidationReport::from_issues(vec![
            Issue::MissingField { field: "role" },
            Issue::EpochLengthZero,
        ])
        .expect("non-empty");
        let text = report.to_string();
        assert!(text.contains("role"));
        assert!(text.contains("epoch_length"));
        assert!(text.starts_with("2 configuration problem"));
    }
}
