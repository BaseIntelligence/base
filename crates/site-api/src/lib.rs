//! Marketing site aggregator: `GET /v1/site/*` (camelCase) over challenge backends.
//!
//! Reads only. Resolves design/prism upstreams via the gateway registry (same
//! pick path as `/challenge/{id}/*`). Never invents scores or duel fixtures.
//!
//! Contract: frontend `BaseApi` / `types.ts`. Coding arena is paused.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::doc_markdown)]

mod handlers;
mod map;
mod state;
mod upstream;

pub use handlers::site_router;
pub use site_types::*;
pub use state::SiteState;

/// Crate identity smoke.
#[must_use]
pub fn crate_name() -> &'static str {
    "site-api"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        assert_eq!(crate_name(), "site-api");
    }
}
