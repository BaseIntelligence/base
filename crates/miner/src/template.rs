//! Normative miner CVM docker-compose fragment (`AGENT_CHALLENGE.md` §9).

/// Public agent HTTP service name.
pub const AGENT_SERVICE: &str = "agent";
/// Attest helper service name (loopback only).
pub const ATTEST_HELPER_SERVICE: &str = "attest-helper";
/// Measured allowlisted Docker Engine proxy (not public).
pub const SOCKET_PROXY_SERVICE: &str = "socket-proxy";
/// Agent container / published port.
pub const AGENT_PORT: u16 = 8080;
/// Attest-helper container port (bound to 127.0.0.1 only).
pub const ATTEST_HELPER_PORT: u16 = 8081;
/// Socket-proxy Docker Engine HTTP port (compose-internal only).
pub const SOCKET_PROXY_PORT: u16 = 2375;
/// In-CVM directory for secret file mounts (never env values).
pub const RUN_BASE_DIR: &str = "/run/base";
/// Hotkey secret path inside the CVM.
pub const HOTKEY_FILE_IN_CVM: &str = "/run/base/miner_hotkey";
/// Optional raw launch-token file path (hash is what is measured).
pub const LAUNCH_TOKEN_FILE_IN_CVM: &str = "/run/base/launch_token";
/// CVM-local work-receipt mini-secret path (never the challenge sk).
pub const RECEIPT_SK_FILE_IN_CVM: &str = "/run/base/receipt_sk";
/// Env name for the runner Docker Engine HTTP base (socket-proxy URL).
pub const DOCKER_BASE_ENV: &str = "BASE_DOCKER_BASE";
/// Env name for the digest-pinned Harbor environment image.
pub const ENV_IMAGE_ENV: &str = "BASE_ENVIRONMENT_IMAGE";
/// Env name for the on-disk Harbor pack root inside the agent.
pub const PACK_ROOT_ENV: &str = "BASE_PACK_ROOT";
/// Env name for the pack catalog HTTP base URL.
pub const PACK_CATALOG_URL_ENV: &str = "BASE_PACK_CATALOG_URL";
/// Env name for the trusted challenge public key (64 lowercase hex).
pub const TRUSTED_CHALLENGE_PUBKEY_ENV: &str = "BASE_TRUSTED_CHALLENGE_PUBKEY";
/// Default pack root path inside the CVM agent container.
pub const DEFAULT_PACK_ROOT_IN_CVM: &str = "/var/lib/base/packs";
/// Default digest-pinned environment image for pack runs (frozen pin).
pub const DEFAULT_ENVIRONMENT_IMAGE: &str =
    "bash@sha256:3bee76a96d86d5d2d5efc7c1c570e5a7c95db22348a26944e0e546fa174e3324";
/// Spike-proven digest pin for the measured socket-proxy image.
pub const DEFAULT_SOCKET_PROXY_IMAGE: &str = concat!(
    "tecnativa/docker-socket-proxy@sha256:",
    "1f5038b54f06c3e18422902cf00ba21803d1c97805aae032e5e6673d532d3459"
);

/// Inputs that shape the measured docker-compose YAML string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposeTemplateInput<'a> {
    /// Digest-pinned agent image (`repo@sha256:<64 hex>`).
    pub agent_image: &'a str,
    /// Digest-pinned attest-helper image.
    pub attest_helper_image: &'a str,
    /// Digest-pinned socket-proxy image.
    pub socket_proxy_image: &'a str,
    /// Lowercase hex SHA-256 of the launch token (measured; not the raw token).
    pub launch_token_hash: &'a str,
    /// Subnet netuid written as a non-secret env value.
    pub netuid: u16,
    /// Work-receipt public key as 64 lowercase hex chars (measured; not secret).
    pub receipt_public_key_hex: &'a str,
    /// Digest-pinned Harbor environment image for pack execution.
    pub environment_image: &'a str,
    /// Pack root path inside the agent container.
    pub pack_root: &'a str,
    /// Pack catalog HTTP base URL (gateway overlay in production).
    pub pack_catalog_url: &'a str,
    /// Trusted challenge public key (64 lowercase hex).
    pub trusted_challenge_pubkey_hex: &'a str,
}

/// Render the docker-compose YAML embedded in `app-compose.json`.
///
/// Contract:
/// - services `socket-proxy` (internal), `agent` (:8080 public), `attest-helper` (127.0.0.1:8081)
/// - only `socket-proxy` mounts `/var/run/docker.sock` (read-only) with an explicit allowlist
/// - agent reaches Docker via `BASE_DOCKER_BASE=http://socket-proxy:2375` (no raw sock)
/// - agent has pack triad env + named `packs` volume at `pack_root` (writable for catalog fetch)
/// - digest-pinned images only (caller must not pass `:latest`)
/// - `environment:` holds only non-secret config + launch-token **hash** + receipt **pubkey**
/// - secrets are bind-mounted files under `/run/base/`
// YAML template is intentionally monolithic for measured compose hash stability.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn docker_compose_yaml(input: &ComposeTemplateInput<'_>) -> String {
    // YAML is hand-built so key order and spacing stay stable for hashing.
    let docker_base = format!("http://{SOCKET_PROXY_SERVICE}:{SOCKET_PROXY_PORT}");
    format!(
        r#"# base miner CVM — AGENT_CHALLENGE.md §9 (challenge_scoring_version=2)
# Secrets: file mounts under {run_dir} only. Never put secret values in environment.
# LAUNCH_TOKEN: only the hash is measured (D11). Miner funds their own Phala account.
# Work-receipt: private key file-mounted; public key published for challenge pin (D19).
# Docker: measured socket-proxy only; agent must not mount docker.sock (D4 / §9.1.1).
# Packs: named volume at pack_root; catalog URL for on-demand fetch.
services:
  {proxy}:
    image: {proxy_image}
    restart: unless-stopped
    environment:
      CONTAINERS: "1"
      IMAGES: "1"
      POST: "1"
      ALLOW_START: "1"
      ALLOW_STOP: "1"
      NETWORKS: "1"
      INFO: "1"
      AUTH: "0"
      BUILD: "0"
      EXEC: "0"
      VOLUMES: "0"
      SWARM: "0"
      SERVICES: "0"
      SYSTEM: "0"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
  {agent}:
    image: {agent_image}
    restart: unless-stopped
    depends_on:
      - {proxy}
    ports:
      - "{agent_port}:{agent_port}"
    environment:
      BASE_NETUID: "{netuid}"
      BASE_MINER_HOTKEY_FILE: "{hotkey_file}"
      BASE_LAUNCH_TOKEN_HASH: "{launch_hash}"
      BASE_RECEIPT_SK_FILE: "{receipt_sk_file}"
      BASE_RECEIPT_PUBLIC_KEY: "{receipt_pk}"
      {docker_base_env}: "{docker_base}"
      {env_image_env}: "{environment_image}"
      {pack_root_env}: "{pack_root}"
      {pack_catalog_url_env}: "{pack_catalog_url}"
      {trusted_pubkey_env}: "{trusted_pubkey}"
    volumes:
      - type: bind
        source: miner_hotkey
        target: {hotkey_file}
        read_only: true
      - type: bind
        source: launch_token
        target: {launch_token_file}
        read_only: true
      - type: bind
        source: receipt_sk
        target: {receipt_sk_file}
        read_only: true
      - packs:{pack_root}
  {attest}:
    image: {attest_image}
    restart: unless-stopped
    ports:
      - "127.0.0.1:{attest_port}:{attest_port}"
    environment:
      BASE_LAUNCH_TOKEN_HASH: "{launch_hash}"
      BASE_MINER_HOTKEY_FILE: "{hotkey_file}"
    volumes:
      - type: bind
        source: miner_hotkey
        target: {hotkey_file}
        read_only: true
      - type: bind
        source: launch_token
        target: {launch_token_file}
        read_only: true
      - /var/run/dstack.sock:/var/run/dstack.sock
volumes:
  packs:
"#,
        run_dir = RUN_BASE_DIR,
        proxy = SOCKET_PROXY_SERVICE,
        proxy_image = input.socket_proxy_image,
        agent = AGENT_SERVICE,
        agent_image = input.agent_image,
        agent_port = AGENT_PORT,
        netuid = input.netuid,
        hotkey_file = HOTKEY_FILE_IN_CVM,
        launch_hash = input.launch_token_hash,
        launch_token_file = LAUNCH_TOKEN_FILE_IN_CVM,
        receipt_sk_file = RECEIPT_SK_FILE_IN_CVM,
        receipt_pk = input.receipt_public_key_hex,
        docker_base_env = DOCKER_BASE_ENV,
        docker_base = docker_base,
        env_image_env = ENV_IMAGE_ENV,
        environment_image = input.environment_image,
        pack_root_env = PACK_ROOT_ENV,
        pack_root = input.pack_root,
        pack_catalog_url_env = PACK_CATALOG_URL_ENV,
        pack_catalog_url = input.pack_catalog_url,
        trusted_pubkey_env = TRUSTED_CHALLENGE_PUBKEY_ENV,
        trusted_pubkey = input.trusted_challenge_pubkey_hex,
        attest = ATTEST_HELPER_SERVICE,
        attest_image = input.attest_helper_image,
        attest_port = ATTEST_HELPER_PORT,
    )
}
