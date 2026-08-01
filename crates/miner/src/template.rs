//! Normative miner CVM docker-compose fragment (`AGENT_CHALLENGE.md` §9).

/// Public agent HTTP service name.
pub const AGENT_SERVICE: &str = "agent";
/// Attest helper service name.
pub const ATTEST_HELPER_SERVICE: &str = "attest-helper";
/// Measured allowlisted Docker Engine proxy (not public).
pub const SOCKET_PROXY_SERVICE: &str = "socket-proxy";
/// Agent container / published port.
pub const AGENT_PORT: u16 = 8080;
/// Attest-helper container / published port.
///
/// Published so the documented remote certify flow can reach `/v1/quote`
/// through the dstack gateway; the helper gates that path on the launch-token
/// bearer credential whose hash is measured, not on a loopback bind.
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
/// Env name carrying the raw work-receipt mini-secret (Phala encrypted secret).
///
/// Only the *name* is measured; the value is decrypted inside the TEE and the
/// pre-launch script turns it into [`RECEIPT_SK_FILE_IN_CVM`].
pub const RECEIPT_SK_HEX_ENV: &str = "BASE_RECEIPT_SK_HEX";
/// Env name carrying the raw launch token (Phala encrypted secret).
pub const LAUNCH_TOKEN_ENV: &str = "BASE_LAUNCH_TOKEN";
/// Env name carrying the public miner hotkey hex (Phala encrypted secret).
pub const MINER_HOTKEY_HEX_ENV: &str = "BASE_MINER_HOTKEY_HEX";
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
/// Env name for the staging root the executor binds from (sibling containers).
pub const WORK_ROOT_ENV: &str = "BASE_AGENT_WORK_ROOT";
/// Env name for an optional operator agent command, as a JSON array string.
pub const AGENT_CMD_ENV: &str = "BASE_AGENT_CMD";
/// Default pack root path inside the CVM agent container.
pub const DEFAULT_PACK_ROOT_IN_CVM: &str = "/var/lib/base/packs";
/// Default exec staging root. It must be a host bind with identical source and
/// target: the VM's Docker daemon resolves bind sources on the VM filesystem,
/// while the executor stages them from inside the agent container, so only an
/// identical string is seen the same way by both.
pub const DEFAULT_WORK_ROOT_IN_CVM: &str = "/var/lib/base/agent-work";
/// Default digest-pinned environment image for pack runs (frozen pin).
pub const DEFAULT_ENVIRONMENT_IMAGE: &str =
    "bash@sha256:3bee76a96d86d5d2d5efc7c1c570e5a7c95db22348a26944e0e546fa174e3324";
/// Spike-proven digest pin for the measured socket-proxy image.
pub const DEFAULT_SOCKET_PROXY_IMAGE: &str = concat!(
    "tecnativa/docker-socket-proxy@sha256:",
    "1f5038b54f06c3e18422902cf00ba21803d1c97805aae032e5e6673d532d3459"
);

/// Directory the compose bind `source:` entries resolve against inside the CVM.
///
/// dstack runs the pre-launch script with this as the compose working
/// directory, so a relative `source: receipt_sk` lands here.
pub const CVM_BIND_SOURCE_DIR: &str = "/dstack";
/// Uid of the runner images; the secret files must be readable by it.
const RUNNER_UID: u32 = 65532;
/// Gid of the runner images.
const RUNNER_GID: u32 = 65532;

/// Render the measured `pre_launch_script`.
///
/// The script text is part of the measured compose and therefore public, so it
/// may only contain variable *references*. The values arrive as Phala
/// encrypted secrets (X25519 + AES-256-GCM, decrypted inside the TEE); only the
/// names appear in `allowed_envs`, which is what stops a miner from smuggling
/// an unmeasured variable into the guest.
///
/// This replaces Phala's default pre-launch script, which exists to do
/// private-registry login and `DSTACK_APP_ID` / `DSTACK_GATEWAY_DOMAIN`
/// substitution. base images are public and the compose references neither
/// variable, so nothing is lost.
#[must_use]
pub fn pre_launch_script() -> String {
    let dir = CVM_BIND_SOURCE_DIR;
    let uid_gid = format!("{RUNNER_UID}:{RUNNER_GID}");
    let uid = RUNNER_UID.to_string();
    let gid = RUNNER_GID.to_string();
    let work_root = DEFAULT_WORK_ROOT_IN_CVM;
    let sk_env = RECEIPT_SK_HEX_ENV;
    let token_env = LAUNCH_TOKEN_ENV;
    let hotkey_env = MINER_HOTKEY_HEX_ENV;
    format!(
        r#"#!/bin/bash
set -euo pipefail
umask 077
mkdir -p {dir}
# ${{VAR:?}} aborts the boot instead of writing an empty file: an empty launch
# token combined with the measured empty-token hash would authenticate anyone.
printf '%s' "${{{sk_env}:?{sk_env} was not supplied as an encrypted secret}}" > {dir}/receipt_sk
printf '%s' "${{{token_env}:?{token_env} was not supplied as an encrypted secret}}" > {dir}/launch_token
printf '%s' "${{{hotkey_env}:?{hotkey_env} was not supplied as an encrypted secret}}" > {dir}/miner_hotkey
if chown {uid_gid} {dir}/receipt_sk {dir}/launch_token {dir}/miner_hotkey 2>/dev/null; then
  chmod 0400 {dir}/receipt_sk
  chmod 0444 {dir}/launch_token {dir}/miner_hotkey
else
  chmod 0444 {dir}/receipt_sk {dir}/launch_token {dir}/miner_hotkey
fi
# The executor stages bind sources for sibling containers under the work root.
# Docker would auto-create an unknown bind source as root-owned, and the agent
# runs as {uid_gid}, so create it here with the right owner instead.
install -d -m 0775 -o {uid} -g {gid} {work_root}
# Existence proof only — never echo a value.
ls -la {dir}/receipt_sk {dir}/launch_token {dir}/miner_hotkey
"#
    )
}

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
    /// Exec staging root; rendered as both the env value and the bind whose
    /// source equals its target (see [`DEFAULT_WORK_ROOT_IN_CVM`]).
    pub work_root: &'a str,
    /// Optional `BASE_AGENT_CMD` JSON array. `None` keeps the reference agent
    /// command, which is a placeholder no honest scoring depends on.
    pub agent_cmd_json: Option<&'a str>,
}

/// Render the docker-compose YAML embedded in `app-compose.json`.
///
/// Contract:
/// - services `socket-proxy` (internal), `agent` (:8080 public), `attest-helper`
///   (:8081 public, launch-token authenticated)
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
    let agent_cmd_line = input.agent_cmd_json.map_or_else(String::new, |json| {
        format!("\n      {AGENT_CMD_ENV}: '{}'", json.replace('\'', "''"))
    });
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
      {work_root_env}: "{work_root}"{agent_cmd_line}
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
      # The executor stages bind sources under this path and hands them to the
      # VM's Docker daemon, which resolves bind sources on the VM, not in this
      # container. source == target keeps staging path and daemon path the same
      # string, which is the only arrangement that can work for both.
      - type: bind
        source: {work_root}
        target: {work_root}
  {attest}:
    image: {attest_image}
    restart: unless-stopped
    ports:
      - "{attest_port}:{attest_port}"
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
        work_root_env = WORK_ROOT_ENV,
        work_root = input.work_root,
        agent_cmd_line = agent_cmd_line,
        attest = ATTEST_HELPER_SERVICE,
        attest_image = input.attest_helper_image,
        attest_port = ATTEST_HELPER_PORT,
    )
}
