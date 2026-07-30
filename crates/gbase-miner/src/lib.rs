//! Miner CVM deploy helpers (task 37).
//!
//! Renders the measured `app-compose.json` mandated by
//! [`docs/AGENT_CHALLENGE.md`](../../../docs/AGENT_CHALLENGE.md) §9, computes
//! the offline compose-hash via [`gbase_compose_hash`], and optionally shells
//! out to `phala deploy`.
//!
//! **D11 / secrets:** only env *names* and the `LAUNCH_TOKEN` *hash* appear in
//! the measured compose. Secret material is file-mounted under `/run/gbase/`,
//! never as compose `environment:` values. The miner funds their own Phala
//! account (R3).

#![forbid(unsafe_code)]

mod deploy;
mod inspect;
mod template;

pub use deploy::{
    deploy_or_dry_run, empty_launch_token_hash_hex, render_app_compose, render_app_compose_bytes,
    run_phala_deploy, DeployError, DeployMode, DeployParams, DeployResult, DEFAULT_AGENT_IMAGE,
    DEFAULT_ATTEST_HELPER_IMAGE,
};
pub use inspect::{docker_compose_from_app_compose_json, environment_block_has_no_secrets};
pub use template::{
    docker_compose_yaml, ComposeTemplateInput, AGENT_PORT, AGENT_SERVICE, ATTEST_HELPER_PORT,
    ATTEST_HELPER_SERVICE, HOTKEY_FILE_IN_CVM, LAUNCH_TOKEN_FILE_IN_CVM, RUN_GBASE_DIR,
};
