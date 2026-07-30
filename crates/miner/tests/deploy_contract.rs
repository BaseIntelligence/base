//! Integration: `AGENT_CHALLENGE.md` §9 contract + offline compose-hash.

use compose_hash::compose_hash_hex;
use miner::{
    agent_service_mounts_docker_sock, deploy_or_dry_run, docker_compose_from_app_compose_json,
    docker_compose_yaml, environment_block_has_no_secrets, reject_raw_docker_sock_on_agent,
    render_app_compose_bytes, ComposeTemplateInput, DeployMode, DeployParams, AGENT_PORT,
    AGENT_SERVICE, ATTEST_HELPER_PORT, ATTEST_HELPER_SERVICE, DEFAULT_AGENT_IMAGE,
    DEFAULT_ATTEST_HELPER_IMAGE, DEFAULT_SOCKET_PROXY_IMAGE, DOCKER_BASE_ENV, SOCKET_PROXY_PORT,
    SOCKET_PROXY_SERVICE,
};

/// Frozen compose-hash of `DeployParams::default()` **before** measured socket-proxy
/// (todo 22). New renders must differ so RTMR3/compose-hash picks up the proxy.
const PRE_SOCKET_PROXY_DEFAULT_COMPOSE_HASH: &str =
    "31e9ea0199236c14972009a576c04d178f97e6aa1cc519ad85b62c62b0c82bd3";

#[test]
fn no_deploy_compose_hash_equals_gbase_compose_hash_of_rendered_bytes() {
    let params = DeployParams {
        mode: DeployMode::NoDeploy,
        ..DeployParams::default()
    };
    let rendered = render_app_compose_bytes(&params).expect("bytes");
    let expected = compose_hash_hex(&rendered).expect("hash");
    let result = deploy_or_dry_run(&params).expect("dry-run");
    assert_eq!(
        result.compose_hash_hex, expected,
        "CLI/library hash must equal compose_hash(rendered app-compose)"
    );
    assert!(!result.phala_invoked);
    assert_eq!(result.compose_hash_hex.len(), 64);
}

#[test]
fn rendered_services_match_agent_challenge_image_port_contract() {
    let result = deploy_or_dry_run(&DeployParams::default()).expect("render");
    let yaml = docker_compose_from_app_compose_json(&result.app_compose_json).expect("yaml");

    assert!(
        yaml.contains(&format!("  {AGENT_SERVICE}:")),
        "missing agent service"
    );
    assert!(
        yaml.contains(&format!("  {ATTEST_HELPER_SERVICE}:")),
        "missing attest-helper service"
    );
    assert!(
        yaml.contains(&format!("  {SOCKET_PROXY_SERVICE}:")),
        "missing socket-proxy service"
    );
    assert!(
        yaml.contains(&format!("\"{AGENT_PORT}:{AGENT_PORT}\"")),
        "agent must publish {AGENT_PORT}"
    );
    assert!(
        yaml.contains(&format!(
            "127.0.0.1:{ATTEST_HELPER_PORT}:{ATTEST_HELPER_PORT}"
        )),
        "attest-helper must be loopback-only on {ATTEST_HELPER_PORT}"
    );
    assert!(
        yaml.contains("ghcr.io/baseintelligence/gbase-agent@sha256:"),
        "agent image repository contract"
    );
    assert!(
        yaml.contains("ghcr.io/baseintelligence/gbase-attest-helper@sha256:"),
        "attest-helper image repository contract"
    );
    assert!(
        yaml.contains(DEFAULT_SOCKET_PROXY_IMAGE),
        "socket-proxy must use spike digest pin"
    );
    assert!(
        !yaml.contains(":latest"),
        "digest pins only — zero :latest tags"
    );
    assert!(DEFAULT_AGENT_IMAGE.contains("@sha256:"));
    assert!(DEFAULT_ATTEST_HELPER_IMAGE.contains("@sha256:"));
    assert!(DEFAULT_SOCKET_PROXY_IMAGE.contains("@sha256:"));

    let doc: serde_json::Value = serde_json::from_str(&result.app_compose_json).expect("json");
    let allowed = doc["allowed_envs"].as_array().expect("allowed_envs array");
    let names: Vec<&str> = allowed.iter().filter_map(|v| v.as_str()).collect();
    assert!(names.contains(&"GBASE_NETUID"));
    assert!(names.contains(&"GBASE_MINER_HOTKEY_FILE"));
    assert!(names.contains(&"GBASE_LAUNCH_TOKEN_HASH"));
    assert!(names.contains(&DOCKER_BASE_ENV));
}

#[test]
fn measured_socket_proxy_allowlist_and_agent_docker_base() {
    let result = deploy_or_dry_run(&DeployParams::default()).expect("render");
    let yaml = docker_compose_from_app_compose_json(&result.app_compose_json).expect("yaml");

    for key in [
        "CONTAINERS",
        "IMAGES",
        "POST",
        "ALLOW_START",
        "ALLOW_STOP",
        "NETWORKS",
        "INFO",
        "AUTH",
        "BUILD",
        "EXEC",
        "VOLUMES",
        "SWARM",
        "SERVICES",
        "SYSTEM",
    ] {
        assert!(
            yaml.contains(&format!("{key}:")),
            "socket-proxy allowlist missing {key}:\n{yaml}"
        );
    }
    assert!(
        yaml.contains("/var/run/docker.sock:/var/run/docker.sock:ro"),
        "only socket-proxy mounts docker.sock ro"
    );
    assert!(
        yaml.contains(&format!("depends_on:\n      - {SOCKET_PROXY_SERVICE}")),
        "agent must depend_on socket-proxy:\n{yaml}"
    );
    let expected_base = format!("http://{SOCKET_PROXY_SERVICE}:{SOCKET_PROXY_PORT}");
    assert!(
        yaml.contains(&format!("{DOCKER_BASE_ENV}: \"{expected_base}\"")),
        "agent must point GBASE_DOCKER_BASE at proxy:\n{yaml}"
    );
    // No host ports on socket-proxy (not published publicly)
    let proxy_block = yaml
        .split(&format!("  {SOCKET_PROXY_SERVICE}:"))
        .nth(1)
        .and_then(|rest| rest.split("  agent:").next())
        .expect("proxy block");
    assert!(
        !proxy_block.contains("ports:"),
        "socket-proxy must not publish host ports:\n{proxy_block}"
    );
    reject_raw_docker_sock_on_agent(&yaml).expect("canonical template must pass");
    assert!(!agent_service_mounts_docker_sock(&yaml));
}

#[test]
fn agent_must_not_mount_raw_docker_sock() {
    let result = deploy_or_dry_run(&DeployParams::default()).expect("render");
    let yaml = docker_compose_from_app_compose_json(&result.app_compose_json).expect("yaml");
    // Agent volumes must not include docker.sock
    let agent_block = yaml
        .split(&format!("  {AGENT_SERVICE}:"))
        .nth(1)
        .and_then(|rest| rest.split(&format!("  {ATTEST_HELPER_SERVICE}:")).next())
        .expect("agent block");
    assert!(
        !agent_block.contains("docker.sock"),
        "agent must not mount docker.sock:\n{agent_block}"
    );
}

#[test]
fn raw_docker_sock_on_agent_is_rejected() {
    // Hand-edited YAML: mount raw sock into agent (forbidden).
    let bad = r#"services:
  agent:
    image: alpine@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
  socket-proxy:
    image: tecnativa/docker-socket-proxy@sha256:1f5038b54f06c3e18422902cf00ba21803d1c97805aae032e5e6673d532d3459
"#;
    assert!(agent_service_mounts_docker_sock(bad));
    let err = reject_raw_docker_sock_on_agent(bad).expect_err("must reject");
    let msg = err.to_string();
    assert!(
        msg.contains("docker.sock") || msg.contains("socket-proxy"),
        "unexpected err: {msg}"
    );

    let good = r#"services:
  socket-proxy:
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
  agent:
    image: alpine@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    environment:
      GBASE_DOCKER_BASE: "http://socket-proxy:2375"
"#;
    reject_raw_docker_sock_on_agent(good).expect("proxy-only sock ok");
}

#[test]
fn compose_hash_deterministic_and_differs_from_pre_proxy_baseline() {
    let a = deploy_or_dry_run(&DeployParams::default()).expect("a");
    let b = deploy_or_dry_run(&DeployParams::default()).expect("b");
    assert_eq!(a.compose_hash_hex, b.compose_hash_hex, "deterministic");
    assert_ne!(
        a.compose_hash_hex, PRE_SOCKET_PROXY_DEFAULT_COMPOSE_HASH,
        "measured socket-proxy must change compose-hash from pre-change baseline"
    );
    assert_eq!(a.compose_hash_hex.len(), 64);
}

#[test]
fn no_secret_appears_in_rendered_environment_block() {
    let params = DeployParams {
        // Even with a realistic launch-token hash, environment must not hold raw secrets.
        launch_token_hash: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
            .into(),
        ..DeployParams::default()
    };
    let result = deploy_or_dry_run(&params).expect("render");
    let yaml = docker_compose_from_app_compose_json(&result.app_compose_json).expect("yaml");

    assert!(
        environment_block_has_no_secrets(&yaml),
        "environment block leaked a secret:\n{yaml}"
    );
    // File mounts present
    assert!(yaml.contains("target: /run/gbase/miner_hotkey"));
    assert!(yaml.contains("target: /run/gbase/launch_token"));
    // No PEM / mnemonic markers anywhere in compose YAML
    let lower = yaml.to_ascii_lowercase();
    assert!(!lower.contains("begin private"));
    assert!(!lower.contains("secretphrase"));
    assert!(!lower.contains("mnemonic"));
}

#[test]
fn hash_changes_when_launch_token_hash_changes() {
    let a = deploy_or_dry_run(&DeployParams::default()).expect("a");
    let p = DeployParams {
        launch_token_hash: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            .into(),
        ..DeployParams::default()
    };
    let b = deploy_or_dry_run(&p).expect("b");
    assert_ne!(a.compose_hash_hex, b.compose_hash_hex);
}

#[test]
fn write_out_compose_roundtrip_hash() {
    let dir = tempfile::tempdir().expect("tmp");
    let path = dir.path().join("app-compose.json");
    let params = DeployParams {
        out_compose: Some(path.clone()),
        mode: DeployMode::NoDeploy,
        ..DeployParams::default()
    };
    let result = deploy_or_dry_run(&params).expect("write");
    let on_disk = std::fs::read(&path).expect("read");
    let from_disk = compose_hash_hex(&on_disk).expect("hash disk");
    assert_eq!(
        from_disk, result.compose_hash_hex,
        "hash of written file must match printed compose-hash"
    );
}

#[test]
fn rejects_latest_tag_on_agent_image() {
    let params = DeployParams {
        agent_image: "ghcr.io/baseintelligence/gbase-agent:latest".into(),
        ..DeployParams::default()
    };
    let err = miner::render_app_compose(&params).expect_err("latest");
    let msg = err.to_string();
    assert!(
        msg.contains("digest-pinned") || msg.contains("latest"),
        "unexpected err: {msg}"
    );
}

#[test]
fn rejects_latest_tag_on_socket_proxy_image() {
    let params = DeployParams {
        socket_proxy_image: "tecnativa/docker-socket-proxy:latest".into(),
        ..DeployParams::default()
    };
    let err = miner::render_app_compose(&params).expect_err("latest");
    let msg = err.to_string();
    assert!(
        msg.contains("digest-pinned") || msg.contains("latest"),
        "unexpected err: {msg}"
    );
}

#[test]
fn receipt_public_key_published_and_stable_across_renders() {
    let pk = "ab".repeat(32);
    let params = DeployParams {
        receipt_public_key_hex: pk.clone(),
        ..DeployParams::default()
    };
    let a = deploy_or_dry_run(&params).expect("a");
    let b = deploy_or_dry_run(&params).expect("b");
    assert_eq!(a.compose_hash_hex, b.compose_hash_hex, "stable compose-hash");
    let yaml = docker_compose_from_app_compose_json(&a.app_compose_json).expect("yaml");
    assert!(
        yaml.contains(&format!("GBASE_RECEIPT_PUBLIC_KEY: \"{pk}\"")),
        "pubkey published in compose env:
{yaml}"
    );
    assert!(yaml.contains("target: /run/gbase/receipt_sk"));
    assert!(yaml.contains("GBASE_RECEIPT_SK_FILE:"));
    // Public key printed surface for challenge pin
    assert!(a.app_compose_json.contains(&pk) || yaml.contains(&pk));
}

#[test]
fn receipt_private_key_never_leaks_into_compose_or_env() {
    // Use a distinctive secret-looking blob that must never appear.
    let sk_hex = "deadbeefcafebabe0123456789abcdefdeadbeefcafebabe0123456789abcdef";
    let pk = "cd".repeat(32);
    let params = DeployParams {
        receipt_public_key_hex: pk,
        ..DeployParams::default()
    };
    let result = deploy_or_dry_run(&params).expect("render");
    let yaml = docker_compose_from_app_compose_json(&result.app_compose_json).expect("yaml");
    let full = format!("{}\n{}", result.app_compose_json, yaml);
    assert!(
        !full.contains(sk_hex),
        "private key material must not appear in compose"
    );
    assert!(
        !full.to_ascii_lowercase().contains("begin private"),
        "no PEM private markers"
    );
    // Env must not carry a secret-looking _SK value (only the path).
    assert!(environment_block_has_no_secrets(&yaml), "env secrets leaked:
{yaml}");
    // Path only — not raw key bytes
    for line in yaml.lines() {
        if line.contains("GBASE_RECEIPT_SK_FILE") {
            assert!(
                line.contains("/run/gbase/receipt_sk"),
                "sk env must be path only: {line}"
            );
            assert!(
                !line.to_ascii_lowercase().contains("deadbeef"),
                "sk env leaked key: {line}"
            );
        }
    }
    // allowed_envs lists path + pubkey names only
    let doc: serde_json::Value = serde_json::from_str(&result.app_compose_json).expect("json");
    let names: Vec<&str> = doc["allowed_envs"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(names.contains(&"GBASE_RECEIPT_SK_FILE"));
    assert!(names.contains(&"GBASE_RECEIPT_PUBLIC_KEY"));
}

#[test]
fn template_input_includes_socket_proxy_image() {
    let launch = "cc".repeat(32);
    let pk = "dd".repeat(32);
    let yaml = docker_compose_yaml(&ComposeTemplateInput {
        agent_image: DEFAULT_AGENT_IMAGE,
        attest_helper_image: DEFAULT_ATTEST_HELPER_IMAGE,
        socket_proxy_image: DEFAULT_SOCKET_PROXY_IMAGE,
        launch_token_hash: &launch,
        netuid: 541,
        receipt_public_key_hex: &pk,
    });
    assert!(yaml.contains("challenge_scoring_version=2"));
    assert!(yaml.contains(DEFAULT_SOCKET_PROXY_IMAGE));
    reject_raw_docker_sock_on_agent(&yaml).expect("ok");
}
