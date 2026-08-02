//! Normative non-TEE local miner runtime compose (`AGENT_CHALLENGE.md` §9.6).
//!
//! The measured CVM path ([`crate::template`]) needs a Phala/dstack node whose
//! sealed-env pipeline is healthy. When none is available (prod9 sealed-env
//! wedge of 2026-08-01), the same runner can execute on a plain Docker host:
//! same images, same env names, same socket-proxy ACL, only the delivery of
//! secrets changes — file binds from the operator host instead of measured
//! pre-launch materialisation inside a TEE.
//!
//! **Trust model (D11 suspended):** nothing here is measured — no compose-hash,
//! no quote, no allowlist row. Credit for a local runtime comes from the
//! master-only `POST /v1/admin/attest-grant` route, which records
//! `reason: admin-exempt` in the shared control plane. This stack is therefore
//! a *testnet* runtime; production miners run the measured CVM path.

use crate::template::{
    AGENT_CMD_ENV, AGENT_PORT, AGENT_SERVICE, DEFAULT_PACK_ROOT_IN_CVM, DEFAULT_WORK_ROOT_IN_CVM,
    DOCKER_BASE_ENV, ENV_IMAGE_ENV, PACK_CATALOG_URL_ENV, PACK_ROOT_ENV, RECEIPT_SK_FILE_IN_CVM,
    SOCKET_PROXY_PORT, SOCKET_PROXY_SERVICE, TRUSTED_CHALLENGE_PUBKEY_ENV, WORK_ROOT_ENV,
};

/// Inputs that shape the local (non-TEE) docker-compose YAML string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalComposeInput<'a> {
    /// Digest-pinned agent image (`repo@sha256:<64 hex>`) or a locally built tag.
    pub agent_image: &'a str,
    /// Digest-pinned socket-proxy image.
    pub socket_proxy_image: &'a str,
    /// Subnet netuid written as a non-secret env value.
    pub netuid: u16,
    /// Digest-pinned Harbor environment image for pack execution.
    pub environment_image: &'a str,
    /// Pack root path inside the agent container.
    pub pack_root: &'a str,
    /// Pack catalog HTTP base URL (gateway overlay on the control plane).
    pub pack_catalog_url: &'a str,
    /// Trusted challenge public key (64 lowercase hex).
    pub trusted_challenge_pubkey_hex: &'a str,
    /// Exec staging root; rendered as both the env value and the bind whose
    /// source equals its target (see [`DEFAULT_WORK_ROOT_IN_CVM`]).
    pub work_root: &'a str,
    /// Host directory holding `receipt_sk` (mode 0400, uid 65532).
    pub secrets_dir: &'a str,
    /// Host port published for the signed endpoint announcement.
    pub publish_port: u16,
    /// Optional `BASE_AGENT_CMD` JSON array. `None` keeps the reference agent
    /// command, which is a placeholder no honest scoring depends on.
    pub agent_cmd_json: Option<&'a str>,
}

impl Default for LocalComposeInput<'_> {
    /// Placeholder-shaped defaults; callers always override. Trusted-pubkey
    /// and images here are sentinel values with no matching running system.
    fn default() -> Self {
        Self {
            agent_image: "base-agent:local",
            socket_proxy_image: "tecnativa/docker-socket-proxy:latest",
            netuid: 1,
            environment_image: "bash:latest",
            pack_root: DEFAULT_PACK_ROOT_IN_CVM,
            pack_catalog_url: "http://gateway:8080/challenge/agent-v1",
            trusted_challenge_pubkey_hex:
                "0000000000000000000000000000000000000000000000000000000000000000",
            work_root: DEFAULT_WORK_ROOT_IN_CVM,
            secrets_dir: "/var/lib/base/secrets",
            publish_port: 8080,
            agent_cmd_json: None,
        }
    }
}

/// Render the local (non-TEE) docker-compose YAML.
///
/// Contract:
/// - services `socket-proxy` (internal ACL identical to the measured CVM one)
///   and `agent` (`:{publish_port}` → container 8080)
/// - only `socket-proxy` mounts `/var/run/docker.sock` (read-only allowlist)
/// - agent reaches Docker via `BASE_DOCKER_BASE=http://socket-proxy:2375`
/// - secrets are bind-mounted files from the operator host, never env values
/// - staging root renders as bind with source == target (daemon path model)
/// - no `pre_launch_script`, no `attest-helper`, no dstack socket: nothing is
///   measured and credit comes only from the master-only admin grant
#[must_use]
pub fn local_compose_yaml(input: &LocalComposeInput<'_>) -> String {
    let docker_base = format!("http://{SOCKET_PROXY_SERVICE}:{SOCKET_PROXY_PORT}");
    let agent_cmd_line = input.agent_cmd_json.map_or_else(String::new, |json| {
        format!("\n      {AGENT_CMD_ENV}: '{}'", json.replace('\'', "''"))
    });
    format!(
        r#"# base miner LOCAL runtime — AGENT_CHALLENGE.md §9.6 (testnet, no TEE)
# Nothing in this file is measured: credit comes from POST /v1/admin/attest-grant
# (reason: admin-exempt). Production miners run the measured CVM stack (§9).
# Secrets: file binds from {secrets_dir} only. Never put secret values in environment.
# Docker: socket-proxy only; the agent must not mount raw /var/run/docker.sock.
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
      - "{publish_port}:{agent_port}"
    environment:
      BASE_NETUID: "{netuid}"
      BASE_RECEIPT_SK_FILE: "{receipt_sk_file}"
      {docker_base_env}: "{docker_base}"
      {env_image_env}: "{environment_image}"
      {pack_root_env}: "{pack_root}"
      {pack_catalog_url_env}: "{pack_catalog_url}"
      {trusted_pubkey_env}: "{trusted_pubkey}"
      {work_root_env}: "{work_root}"{agent_cmd_line}
    volumes:
      - type: bind
        source: {secrets_dir}/receipt_sk
        target: {receipt_sk_file}
        read_only: true
      - packs:{pack_root}
      # The executor stages bind sources under this path and hands them to the
      # host Docker daemon, which resolves bind sources on the host, not in this
      # container. source == target keeps staging path and daemon path the same
      # string, which is the only arrangement that can work for both.
      - type: bind
        source: {work_root}
        target: {work_root}
volumes:
  packs:
"#,
        proxy = SOCKET_PROXY_SERVICE,
        proxy_image = input.socket_proxy_image,
        agent = AGENT_SERVICE,
        agent_image = input.agent_image,
        agent_port = AGENT_PORT,
        publish_port = input.publish_port,
        netuid = input.netuid,
        receipt_sk_file = RECEIPT_SK_FILE_IN_CVM,
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
        secrets_dir = input.secrets_dir,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::{
        DEFAULT_ENVIRONMENT_IMAGE, DEFAULT_SOCKET_PROXY_IMAGE, TRUSTED_CHALLENGE_PUBKEY_ENV,
    };

    const PK_64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn input() -> LocalComposeInput<'static> {
        LocalComposeInput {
            agent_image: concat!(
                "ghcr.io/baseintelligence/base/agent@sha256:",
                "0000000000000000000000000000000000000000000000000000000000000001"
            ),
            socket_proxy_image: DEFAULT_SOCKET_PROXY_IMAGE,
            netuid: 541,
            environment_image: DEFAULT_ENVIRONMENT_IMAGE,
            pack_root: crate::template::DEFAULT_PACK_ROOT_IN_CVM,
            pack_catalog_url: "http://gateway:8080/challenge/agent-v1",
            trusted_challenge_pubkey_hex: PK_64,
            work_root: crate::template::DEFAULT_WORK_ROOT_IN_CVM,
            secrets_dir: "/var/lib/base/secrets",
            publish_port: 18081,
            agent_cmd_json: None,
        }
    }

    #[test]
    fn renders_socket_proxy_acl_identical_to_cvm() {
        let yaml = local_compose_yaml(&input());
        for kv in [
            "CONTAINERS: \"1\"",
            "IMAGES: \"1\"",
            "POST: \"1\"",
            "ALLOW_START: \"1\"",
            "ALLOW_STOP: \"1\"",
            "NETWORKS: \"1\"",
            "INFO: \"1\"",
            "AUTH: \"0\"",
            "BUILD: \"0\"",
            "EXEC: \"0\"",
            "VOLUMES: \"0\"",
            "SWARM: \"0\"",
            "SERVICES: \"0\"",
            "SYSTEM: \"0\"",
        ] {
            assert!(yaml.contains(kv), "proxy ACL entry missing: {kv}");
        }
    }

    #[test]
    fn only_socket_proxy_mounts_the_raw_docker_sock() {
        let yaml = local_compose_yaml(&input());
        // The bind itself appears exactly once, on socket-proxy; the header
        // comment mentioning the path is not a mount.
        assert_eq!(
            yaml.matches("/var/run/docker.sock:/var/run/docker.sock")
                .count(),
            1
        );
        let agent_section = yaml.split("  agent:").nth(1).expect("agent service");
        assert!(!agent_section.contains("/var/run/docker.sock"));
    }

    #[test]
    fn agent_has_runner_env_and_secret_bind_not_env() {
        let yaml = local_compose_yaml(&input());
        assert!(yaml.contains("BASE_RECEIPT_SK_FILE: \"/run/base/receipt_sk\""));
        assert!(yaml.contains("BASE_DOCKER_BASE: \"http://socket-proxy:2375\""));
        assert!(yaml.contains(PACK_ROOT_ENV));
        assert!(yaml.contains(PACK_CATALOG_URL_ENV));
        assert!(yaml.contains(TRUSTED_CHALLENGE_PUBKEY_ENV));
        assert!(yaml.contains("BASE_AGENT_WORK_ROOT"));
        assert!(yaml.contains("source: /var/lib/base/secrets/receipt_sk"));
        assert!(yaml.contains("target: /run/base/receipt_sk"));
        // No secret VALUE anywhere in the file.
        assert!(
            !yaml.contains("BASE_RECEIPT_SK:"),
            "receipt secret must stay a file"
        );
    }

    #[test]
    fn staging_root_bind_has_identical_source_and_target() {
        let yaml = local_compose_yaml(&input());
        assert!(yaml.contains("source: /var/lib/base/agent-work"));
        assert!(yaml.contains("target: /var/lib/base/agent-work"));
    }

    #[test]
    fn nothing_is_measured_or_phala_specific() {
        let yaml = local_compose_yaml(&input());
        for banned in [
            "pre_launch",
            "attest-helper",
            "dstack",
            "app_id",
            "LAUNCH_TOKEN",
            "mr_config",
        ] {
            assert!(
                !yaml.contains(banned),
                "local runtime must not reference {banned}"
            );
        }
    }

    #[test]
    fn publish_port_is_rendered() {
        let yaml = local_compose_yaml(&input());
        assert!(yaml.contains("- \"18081:8080\""));
    }

    #[test]
    fn agent_cmd_json_is_single_quote_escaped() {
        let mut i = input();
        i.agent_cmd_json = Some(r#"["python3","-c","x's"]"#);
        let yaml = local_compose_yaml(&i);
        assert!(
            yaml.contains("x''s"),
            "single quotes must be doubled: {yaml}"
        );
    }

    #[test]
    fn agent_cmd_absent_renders_no_agent_cmd_line() {
        let yaml = local_compose_yaml(&input());
        assert!(!yaml.contains("BASE_AGENT_CMD"));
    }

    #[test]
    fn default_secrets_dir_points_outside_the_repo() {
        let i = LocalComposeInput::default();
        assert!(i.secrets_dir.starts_with('/'));
    }
}
