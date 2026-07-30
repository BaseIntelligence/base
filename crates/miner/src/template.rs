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
pub const RUN_GBASE_DIR: &str = "/run/gbase";
/// Hotkey secret path inside the CVM.
pub const HOTKEY_FILE_IN_CVM: &str = "/run/gbase/miner_hotkey";
/// Optional raw launch-token file path (hash is what is measured).
pub const LAUNCH_TOKEN_FILE_IN_CVM: &str = "/run/gbase/launch_token";
/// CVM-local work-receipt mini-secret path (never the challenge sk).
pub const RECEIPT_SK_FILE_IN_CVM: &str = "/run/gbase/receipt_sk";
/// Env name for the runner Docker Engine HTTP base (socket-proxy URL).
pub const DOCKER_BASE_ENV: &str = "GBASE_DOCKER_BASE";
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
}

/// Render the docker-compose YAML embedded in `app-compose.json`.
///
/// Contract:
/// - services `socket-proxy` (internal), `agent` (:8080 public), `attest-helper` (127.0.0.1:8081)
/// - only `socket-proxy` mounts `/var/run/docker.sock` (read-only) with an explicit allowlist
/// - agent reaches Docker via `GBASE_DOCKER_BASE=http://socket-proxy:2375` (no raw sock)
/// - digest-pinned images only (caller must not pass `:latest`)
/// - `environment:` holds only non-secret config + launch-token **hash** + receipt **pubkey**
/// - secrets are bind-mounted files under `/run/gbase/`
#[must_use]
pub fn docker_compose_yaml(input: &ComposeTemplateInput<'_>) -> String {
    // YAML is hand-built so key order and spacing stay stable for hashing.
    let docker_base = format!(
        "http://{proxy}:{port}",
        proxy = SOCKET_PROXY_SERVICE,
        port = SOCKET_PROXY_PORT
    );
    format!(
        r#"# gbase miner CVM — AGENT_CHALLENGE.md §9 (challenge_scoring_version=2)
# Secrets: file mounts under {run_dir} only. Never put secret values in environment.
# LAUNCH_TOKEN: only the hash is measured (D11). Miner funds their own Phala account.
# Work-receipt: private key file-mounted; public key published for challenge pin (D19).
# Docker: measured socket-proxy only; agent must not mount docker.sock (D4 / §9.1.1).
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
      GBASE_NETUID: "{netuid}"
      GBASE_MINER_HOTKEY_FILE: "{hotkey_file}"
      GBASE_LAUNCH_TOKEN_HASH: "{launch_hash}"
      GBASE_RECEIPT_SK_FILE: "{receipt_sk_file}"
      GBASE_RECEIPT_PUBLIC_KEY: "{receipt_pk}"
      {docker_base_env}: "{docker_base}"
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
  {attest}:
    image: {attest_image}
    restart: unless-stopped
    ports:
      - "127.0.0.1:{attest_port}:{attest_port}"
    environment:
      GBASE_LAUNCH_TOKEN_HASH: "{launch_hash}"
      GBASE_MINER_HOTKEY_FILE: "{hotkey_file}"
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
"#,
        run_dir = RUN_GBASE_DIR,
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
        attest = ATTEST_HELPER_SERVICE,
        attest_image = input.attest_helper_image,
        attest_port = ATTEST_HELPER_PORT,
    )
}
