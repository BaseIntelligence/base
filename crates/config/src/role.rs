//! Process role for a gbase binary.

use std::fmt;
use std::str::FromStr;

/// Which binary / service this process is running as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Role {
    /// Epoch validation, peer cross-check, weight submission.
    Validator,
    /// Master-only gateway (subnet owner hotkey).
    Gateway,
    /// Digest-pinned auto-updater container.
    Updater,
    /// Miner / agent participant.
    Miner,
}

impl Role {
    /// Stable lowercase wire / env / TOML name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Validator => "validator",
            Self::Gateway => "gateway",
            Self::Updater => "updater",
            Self::Miner => "miner",
        }
    }

    /// Roles that require a Postgres `database_url` (or url file).
    #[must_use]
    pub const fn requires_database(self) -> bool {
        matches!(self, Self::Validator | Self::Gateway | Self::Updater)
    }

    /// Gateway owns TLS hostnames and must have `GBASE_DOMAIN` (D25).
    #[must_use]
    pub const fn requires_domain(self) -> bool {
        matches!(self, Self::Gateway)
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Role {
    type Err = ParseRoleError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "validator" => Ok(Self::Validator),
            "gateway" => Ok(Self::Gateway),
            "updater" => Ok(Self::Updater),
            "miner" => Ok(Self::Miner),
            other => Err(ParseRoleError {
                got: other.to_owned(),
            }),
        }
    }
}

/// Unknown role string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseRoleError {
    got: String,
}

impl fmt::Display for ParseRoleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unknown role `{}` (expected validator|gateway|updater|miner)",
            self.got
        )
    }
}

impl std::error::Error for ParseRoleError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_case_insensitive() {
        assert_eq!("Gateway".parse::<Role>().unwrap(), Role::Gateway);
        assert_eq!("VALIDATOR".parse::<Role>().unwrap(), Role::Validator);
    }

    #[test]
    fn rejects_unknown() {
        assert!("oracle".parse::<Role>().is_err());
    }
}
