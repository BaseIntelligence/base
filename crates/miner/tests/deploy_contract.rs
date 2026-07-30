//! Integration: `AGENT_CHALLENGE.md` §9 contract + offline compose-hash.

use compose_hash::compose_hash_hex;
use miner::{
    deploy_or_dry_run, docker_compose_from_app_compose_json, environment_block_has_no_secrets,
    render_app_compose_bytes, DeployMode, DeployParams, AGENT_PORT, AGENT_SERVICE,
    ATTEST_HELPER_PORT, ATTEST_HELPER_SERVICE, DEFAULT_AGENT_IMAGE, DEFAULT_ATTEST_HELPER_IMAGE,
};

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
        !yaml.contains(":latest"),
        "digest pins only — zero :latest tags"
    );
    assert!(DEFAULT_AGENT_IMAGE.contains("@sha256:"));
    assert!(DEFAULT_ATTEST_HELPER_IMAGE.contains("@sha256:"));

    let doc: serde_json::Value = serde_json::from_str(&result.app_compose_json).expect("json");
    let allowed = doc["allowed_envs"].as_array().expect("allowed_envs array");
    let names: Vec<&str> = allowed.iter().filter_map(|v| v.as_str()).collect();
    assert!(names.contains(&"GBASE_NETUID"));
    assert!(names.contains(&"GBASE_MINER_HOTKEY_FILE"));
    assert!(names.contains(&"GBASE_LAUNCH_TOKEN_HASH"));
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
