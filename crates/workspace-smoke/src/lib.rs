//! Smoke crate: proves the cargo workspace compiles and inherits lints.
//! Real config lives in `config` (later task). Do not grow this crate.

/// Workspace identity string used by the smoke unit test.
#[must_use]
pub fn workspace_name() -> &'static str {
    "gbase"
}

#[cfg(test)]
mod tests {
    use super::workspace_name;

    #[test]
    fn workspace_name_is_gbase() {
        assert_eq!(workspace_name(), "gbase");
    }
}
