//! Miner CVM deploy helpers (task 37).
//!
//! Renders the measured `app-compose.json` mandated by
//! [`docs/AGENT_CHALLENGE.md`](../../../docs/AGENT_CHALLENGE.md) §9, computes
//! the offline compose-hash via [`compose_hash`], and optionally shells
//! out to `phala deploy`.
//!
//! **D11 / secrets:** only env *names* and the `LAUNCH_TOKEN` *hash* appear in
//! the measured compose. Secret material is file-mounted under `/run/base/`,
//! never as compose `environment:` values; the files are materialised at boot
//! by the measured pre-launch script from Phala encrypted secrets, so no value
//! ever enters the compose that validators and Phala publish. The miner funds
//! their own Phala account (R3).
//!
//! **§9.1.1 / D4:** Docker access is only via a measured `socket-proxy`; the
//! `agent` service must never mount raw `/var/run/docker.sock`.

#![forbid(unsafe_code)]

mod announce;
mod certify;
mod deploy;
mod inspect;
mod template;

pub use announce::{announce, AnnounceError, AnnounceOutcome, AnnounceParams};
pub use certify::{
    certify, parse_hotkey_hex, CertifyError, CertifyParams, CertifyResult, QuoteSource,
};
pub use deploy::{
    deploy_or_dry_run, empty_launch_token_hash_hex, launch_token_hash_hex, render_app_compose,
    render_app_compose_bytes, run_phala_deploy, DeployError, DeployMode, DeployParams,
    DeployResult, DeploySecrets, PhalaDeployInvocation, DEFAULT_AGENT_IMAGE,
    DEFAULT_ATTEST_HELPER_IMAGE, DEFAULT_PACK_CATALOG_URL, DEFAULT_SOCKET_PROXY_IMAGE,
    DEFAULT_TRUSTED_CHALLENGE_PUBKEY_HEX,
};
pub use inspect::{
    agent_service_mounts_docker_sock, docker_compose_from_app_compose_json,
    environment_block_has_no_secrets, reject_raw_docker_sock_on_agent, RawDockerSockOnAgent,
};
pub use template::{
    docker_compose_yaml, pre_launch_script, ComposeTemplateInput, AGENT_PORT, AGENT_SERVICE,
    ATTEST_HELPER_PORT, ATTEST_HELPER_SERVICE, CVM_BIND_SOURCE_DIR, DEFAULT_ENVIRONMENT_IMAGE,
    DEFAULT_PACK_ROOT_IN_CVM, DOCKER_BASE_ENV, ENV_IMAGE_ENV, HOTKEY_FILE_IN_CVM, LAUNCH_TOKEN_ENV,
    LAUNCH_TOKEN_FILE_IN_CVM, MINER_HOTKEY_HEX_ENV, PACK_CATALOG_URL_ENV, PACK_ROOT_ENV,
    RECEIPT_SK_FILE_IN_CVM, RECEIPT_SK_HEX_ENV, RUN_BASE_DIR, SOCKET_PROXY_PORT,
    SOCKET_PROXY_SERVICE, TRUSTED_CHALLENGE_PUBKEY_ENV,
};
