//! Bittensor network UID newtype.

use std::fmt;
use std::num::ParseIntError;
use std::str::FromStr;

/// Subnet network UID (`u16` on chain).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NetUid(u16);

impl NetUid {
    /// Wrap a raw on-chain netuid.
    #[must_use]
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    /// Borrow the raw `u16`.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

impl fmt::Display for NetUid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl FromStr for NetUid {
    type Err = ParseNetUidError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let raw: u16 = s.trim().parse().map_err(ParseNetUidError::Parse)?;
        Ok(Self::new(raw))
    }
}

impl From<u16> for NetUid {
    fn from(value: u16) -> Self {
        Self::new(value)
    }
}

/// Failed to parse a netuid string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseNetUidError {
    /// Not a valid `u16`.
    Parse(ParseIntError),
}

impl fmt::Display for ParseNetUidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid netuid: {e}"),
        }
    }
}

impl std::error::Error for ParseNetUidError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(e) => Some(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_u16() {
        assert_eq!("42".parse::<NetUid>().unwrap().get(), 42);
    }

    #[test]
    fn rejects_overflow() {
        assert!("70000".parse::<NetUid>().is_err());
    }

    #[test]
    fn rejects_negative() {
        assert!("-1".parse::<NetUid>().is_err());
    }
}
