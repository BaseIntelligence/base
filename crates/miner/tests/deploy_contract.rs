//! Integration: `AGENT_CHALLENGE.md` §9 contract + offline compose-hash.

use compose_hash::compose_hash_hex;
use miner::DeploySecrets;
use miner::{
    agent_service_mounts_docker_sock, deploy_or_dry_run, docker_compose_from_app_compose_json,
    docker_compose_yaml, environment_block_has_no_secrets, reject_raw_docker_sock_on_agent,
    render_app_compose_bytes, ComposeTemplateInput, DeployMode, DeployParams, AGENT_PORT,
    AGENT_SERVICE, ATTEST_HELPER_PORT, ATTEST_HELPER_SERVICE, DEFAULT_AGENT_IMAGE,
    DEFAULT_ATTEST_HELPER_IMAGE, DEFAULT_ENVIRONMENT_IMAGE, DEFAULT_PACK_CATALOG_URL,
    DEFAULT_PACK_ROOT_IN_CVM, DEFAULT_SOCKET_PROXY_IMAGE, DEFAULT_TRUSTED_CHALLENGE_PUBKEY_HEX,
    DOCKER_BASE_ENV, ENV_IMAGE_ENV, LAUNCH_TOKEN_ENV, MINER_HOTKEY_HEX_ENV, PACK_CATALOG_URL_ENV,
    PACK_ROOT_ENV, RECEIPT_SK_HEX_ENV, SOCKET_PROXY_PORT, SOCKET_PROXY_SERVICE,
    TRUSTED_CHALLENGE_PUBKEY_ENV,
};

/// Frozen compose-hash of `DeployParams::default()` **before** measured socket-proxy
/// (todo 22). New renders must differ so RTMR3/compose-hash picks up the proxy.
const PRE_SOCKET_PROXY_DEFAULT_COMPOSE_HASH: &str =
    "31e9ea0199236c14972009a576c04d178f97e6aa1cc519ad85b62c62b0c82bd3";

#[test]
fn no_deploy_compose_hash_equals_base_compose_hash_of_rendered_bytes() {
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
        yaml.contains(&format!("\"{ATTEST_HELPER_PORT}:{ATTEST_HELPER_PORT}\"")),
        "attest-helper must publish {ATTEST_HELPER_PORT} for remote certify"
    );
    assert!(
        !yaml.contains(&format!("127.0.0.1:{ATTEST_HELPER_PORT}")),
        "attest-helper is guarded by the launch token, not by a loopback bind:\n{yaml}"
    );
    assert!(
        yaml.contains("ghcr.io/baseintelligence/base/base-agent@sha256:"),
        "agent image repository contract"
    );
    assert!(
        yaml.contains("ghcr.io/baseintelligence/base/base-attest-helper@sha256:"),
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

    // These settings are literals in the measured YAML, not injected values,
    // so they belong in the compose and must stay out of allowed_envs.
    for env in [
        "BASE_NETUID",
        "BASE_MINER_HOTKEY_FILE",
        "BASE_LAUNCH_TOKEN_HASH",
        DOCKER_BASE_ENV,
        ENV_IMAGE_ENV,
        PACK_ROOT_ENV,
        PACK_CATALOG_URL_ENV,
        TRUSTED_CHALLENGE_PUBKEY_ENV,
    ] {
        assert!(yaml.contains(env), "{env} must be a literal in the compose");
    }
}

#[test]
fn pack_triad_env_and_packs_volume_in_measured_compose() {
    let result = deploy_or_dry_run(&DeployParams::default()).expect("render");
    let yaml = docker_compose_from_app_compose_json(&result.app_compose_json).expect("yaml");

    assert!(
        yaml.contains(&format!("{ENV_IMAGE_ENV}: \"{DEFAULT_ENVIRONMENT_IMAGE}\"")),
        "missing digest-pinned environment image:\n{yaml}"
    );
    assert!(
        yaml.contains(&format!("{PACK_ROOT_ENV}: \"{DEFAULT_PACK_ROOT_IN_CVM}\"")),
        "missing pack root:\n{yaml}"
    );
    assert!(
        yaml.contains(&format!(
            "{PACK_CATALOG_URL_ENV}: \"{DEFAULT_PACK_CATALOG_URL}\""
        )),
        "missing pack catalog url:\n{yaml}"
    );
    assert!(
        yaml.contains(&format!(
            "{TRUSTED_CHALLENGE_PUBKEY_ENV}: \"{DEFAULT_TRUSTED_CHALLENGE_PUBKEY_HEX}\""
        )),
        "missing trusted challenge pubkey:\n{yaml}"
    );
    assert!(
        yaml.contains(&format!("packs:{DEFAULT_PACK_ROOT_IN_CVM}")),
        "agent must mount packs volume at pack_root:\n{yaml}"
    );
    assert!(
        yaml.contains("volumes:\n  packs:"),
        "named packs volume declaration missing:\n{yaml}"
    );
    // Agent must not mount docker.sock; packs volume is writable (no :ro suffix on packs mount).
    let agent_block = yaml
        .split(&format!("  {AGENT_SERVICE}:"))
        .nth(1)
        .and_then(|rest| rest.split(&format!("  {ATTEST_HELPER_SERVICE}:")).next())
        .expect("agent block");
    assert!(!agent_block.contains("docker.sock"));
    assert!(
        agent_block.contains(&format!("packs:{DEFAULT_PACK_ROOT_IN_CVM}")),
        "packs mount in agent:\n{agent_block}"
    );
    assert!(
        !agent_block.contains(&format!("packs:{DEFAULT_PACK_ROOT_IN_CVM}:ro")),
        "packs volume must be writable for catalog fetch:\n{agent_block}"
    );
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
        "agent must point BASE_DOCKER_BASE at proxy:\n{yaml}"
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
    let bad = r"services:
  agent:
    image: alpine@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
  socket-proxy:
    image: tecnativa/docker-socket-proxy@sha256:1f5038b54f06c3e18422902cf00ba21803d1c97805aae032e5e6673d532d3459
";
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
      BASE_DOCKER_BASE: "http://socket-proxy:2375"
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
    assert!(yaml.contains("target: /run/base/miner_hotkey"));
    assert!(yaml.contains("target: /run/base/launch_token"));
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
        agent_image: "ghcr.io/baseintelligence/base/base-agent:latest".into(),
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
    assert_eq!(
        a.compose_hash_hex, b.compose_hash_hex,
        "stable compose-hash"
    );
    let yaml = docker_compose_from_app_compose_json(&a.app_compose_json).expect("yaml");
    assert!(
        yaml.contains(&format!("BASE_RECEIPT_PUBLIC_KEY: \"{pk}\"")),
        "pubkey published in compose env:
{yaml}"
    );
    assert!(yaml.contains("target: /run/base/receipt_sk"));
    assert!(yaml.contains("BASE_RECEIPT_SK_FILE:"));
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
    assert!(
        environment_block_has_no_secrets(&yaml),
        "env secrets leaked:
{yaml}"
    );
    // Path only — not raw key bytes
    for line in yaml.lines() {
        if line.contains("BASE_RECEIPT_SK_FILE") {
            assert!(
                line.contains("/run/base/receipt_sk"),
                "sk env must be path only: {line}"
            );
            assert!(
                !line.to_ascii_lowercase().contains("deadbeef"),
                "sk env leaked key: {line}"
            );
        }
    }
    // The path and the public half are literals in the compose; only the
    // secret's carrier name is an injected value.
    assert!(yaml.contains("BASE_RECEIPT_SK_FILE"));
    assert!(yaml.contains("BASE_RECEIPT_PUBLIC_KEY"));
}

/// Distinctive values a leaking renderer would bake into the measured compose.
fn probe_secrets() -> DeploySecrets {
    DeploySecrets {
        receipt_sk_hex: "deadbeefcafebabe0123456789abcdefdeadbeefcafebabe0123456789abcdef".into(),
        launch_token: "tok-4f1c9a2b-never-measured".into(),
        miner_hotkey_hex: "fe".repeat(32),
    }
}

#[test]
fn pre_launch_script_materialises_secrets_without_baking_any_in() {
    let secrets = probe_secrets();
    let params = DeployParams {
        secrets: secrets.clone(),
        ..DeployParams::default()
    };
    let result = deploy_or_dry_run(&params).expect("render");
    let doc: serde_json::Value = serde_json::from_str(&result.app_compose_json).expect("json");
    let script = doc["pre_launch_script"]
        .as_str()
        .expect("pre_launch_script");

    for path in [
        "/dstack/receipt_sk",
        "/dstack/launch_token",
        "/dstack/miner_hotkey",
    ] {
        assert!(
            script.contains(path),
            "script must create {path}:\n{script}"
        );
    }
    for env in [RECEIPT_SK_HEX_ENV, LAUNCH_TOKEN_ENV, MINER_HOTKEY_HEX_ENV] {
        assert!(
            script.contains(&format!("${{{env}:?")),
            "{env} must be read fail-closed as ${{VAR:?...}}:\n{script}"
        );
    }
    assert!(
        !script.contains(r"printf '\x"),
        "byte-literal printf is how a key gets baked into the measurement:\n{script}"
    );
    assert!(
        !script.lines().any(|l| {
            let l = l.trim();
            l.starts_with("echo") && !l.starts_with("echo \"pre")
        }),
        "the script must never echo a value:\n{script}"
    );
    let has_hex64_run = script
        .split(|c: char| !c.is_ascii_hexdigit())
        .any(|w| w.len() >= 64);
    assert!(
        !has_hex64_run,
        "no key-shaped literal may appear:\n{script}"
    );

    // The whole measured document, not just the script.
    for value in [
        secrets.receipt_sk_hex.as_str(),
        secrets.launch_token.as_str(),
        secrets.miner_hotkey_hex.as_str(),
    ] {
        assert!(
            !result.app_compose_json.contains(value),
            "secret value leaked into the measured app-compose"
        );
    }
}

#[test]
fn allowed_envs_names_the_secrets_and_the_hash_ignores_their_values() {
    let a = deploy_or_dry_run(&DeployParams {
        secrets: probe_secrets(),
        ..DeployParams::default()
    })
    .expect("a");
    let b = deploy_or_dry_run(&DeployParams {
        secrets: DeploySecrets {
            receipt_sk_hex: "11".repeat(32),
            launch_token: "a-completely-different-token".into(),
            miner_hotkey_hex: "22".repeat(32),
        },
        ..DeployParams::default()
    })
    .expect("b");
    assert_eq!(
        a.compose_hash_hex, b.compose_hash_hex,
        "only secret *names* are measured, so an owner can sign the measurement \
         before any miner picks their values"
    );

    let doc: serde_json::Value = serde_json::from_str(&a.app_compose_json).expect("json");
    let names: Vec<&str> = doc["allowed_envs"]
        .as_array()
        .expect("allowed_envs")
        .iter()
        .filter_map(serde_json::Value::as_str)
        .collect();
    for env in [RECEIPT_SK_HEX_ENV, LAUNCH_TOKEN_ENV, MINER_HOTKEY_HEX_ENV] {
        assert!(
            names.contains(&env),
            "allowed_envs missing {env}: {names:?}"
        );
    }
    // Order and membership are not ours to choose: `phala deploy` rewrites
    // allowed_envs from the `-e` env file in file order and appends its own
    // entry. Predicting it wrong is what makes the printed compose-hash differ
    // from the one the hardware measures.
    assert_eq!(
        names,
        vec![
            RECEIPT_SK_HEX_ENV,
            LAUNCH_TOKEN_ENV,
            MINER_HOTKEY_HEX_ENV,
            "DSTACK_AUTHORIZED_KEYS",
        ],
        "allowed_envs must mirror what the Phala CLI writes"
    );
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
        environment_image: DEFAULT_ENVIRONMENT_IMAGE,
        pack_root: DEFAULT_PACK_ROOT_IN_CVM,
        pack_catalog_url: DEFAULT_PACK_CATALOG_URL,
        trusted_challenge_pubkey_hex: DEFAULT_TRUSTED_CHALLENGE_PUBKEY_HEX,
    });
    assert!(yaml.contains("challenge_scoring_version=2"));
    assert!(yaml.contains(DEFAULT_SOCKET_PROXY_IMAGE));
    reject_raw_docker_sock_on_agent(&yaml).expect("ok");
}

/// The compose-hash printed here is what an owner signs into
/// `config/measurements.toml` before any miner deploys, so it has to equal the
/// value Phala measures. Pinned against the live CVM `base-miner-541`
/// (`app_id` `340ead2af2ff1d950d47a6fae0ffa473854b5d96`): its hardware
/// `mr_config_id` carries `01 || this hash || 15 zero bytes`.
#[test]
fn offline_hash_equals_the_hash_measured_by_real_tdx_hardware() {
    let params = DeployParams {
        name: "base-miner-541".into(),
        netuid: 541,
        launch_token_hash: "8070605dc9b99dd5ad90f2bc8eed7aef29f76fe07139e3b3eb4a59fad4189957"
            .into(),
        receipt_public_key_hex: "da3dc84f6f64b86b0cfc26bd2370f4f6e159e589435f0ac52ae4c51e15169a7c"
            .into(),
        pack_catalog_url: "http://68.183.23.51:8090".into(),
        ..DeployParams::default()
    };
    let result = deploy_or_dry_run(&params).expect("render");
    assert_eq!(
        result.compose_hash_hex, "f3dd0224a37f70b4c534effe091b5548c2732d7c0ecafd35257487e5a06f580a",
        "renderer drifted from the app-compose Phala actually measured"
    );
}
