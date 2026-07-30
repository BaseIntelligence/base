//! Registration stub — full on-chain registration is a later task.
//!
//! The skeleton only tracks a local view of whether this hotkey is known to
//! hold a UID, so readiness and logs can surface registration state without
//! talking to the chain yet.

use serde::Serialize;

/// Local registration view for the validator process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RegistrationStatus {
    /// Whether a UID has been assigned in this process.
    pub registered: bool,
    /// Assigned UID when registered.
    pub uid: Option<u16>,
}

impl RegistrationStatus {
    /// Fresh process: not yet registered on-chain (or not loaded).
    #[must_use]
    pub const fn unregistered() -> Self {
        Self {
            registered: false,
            uid: None,
        }
    }

    /// Mark this process as holding `uid` (stub for later chain sync).
    #[must_use]
    pub const fn with_uid(uid: u16) -> Self {
        Self {
            registered: true,
            uid: Some(uid),
        }
    }
}

/// Mutable registration stub held by the running validator.
#[derive(Debug, Clone)]
pub struct RegistrationStub {
    status: RegistrationStatus,
}

impl RegistrationStub {
    /// Start unregistered.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            status: RegistrationStatus::unregistered(),
        }
    }

    /// Current status snapshot.
    #[must_use]
    pub fn status(&self) -> RegistrationStatus {
        self.status.clone()
    }

    /// Record a UID (local stub only — no chain extrinsic).
    pub fn set_uid(&mut self, uid: u16) {
        self.status = RegistrationStatus::with_uid(uid);
    }

    /// Clear registration (e.g. after a failed permit check later).
    pub fn clear(&mut self) {
        self.status = RegistrationStatus::unregistered();
    }
}

impl Default for RegistrationStub {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s6_registration_stub_default_and_set_uid() {
        let mut stub = RegistrationStub::new();
        assert!(!stub.status().registered);
        assert_eq!(stub.status().uid, None);
        stub.set_uid(7);
        assert!(stub.status().registered);
        assert_eq!(stub.status().uid, Some(7));
        stub.clear();
        assert!(!stub.status().registered);
    }
}
