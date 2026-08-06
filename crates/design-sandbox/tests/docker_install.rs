//! Docker-backed install/run proof for miner dependencies (real engine, real
//! egress proxy, real `PyPI`). Ignored by default; run on a host with:
//!
//! - a Docker Engine HTTP endpoint (`DESIGN_TEST_DOCKER_BASE`, default
//!   `http://127.0.0.1:2375`) reachable with the verifier allowlist,
//! - the `design-runtime:0.1.0` image (or `DESIGN_TEST_RUNTIME_IMAGE`),
//! - the egress proxy reachable from sandbox containers
//!   (`DESIGN_TEST_EGRESS_PROXY`, default `http://design-egress-proxy:8094`
//!   on the `design-sandbox-egress` network; override
//!   `DESIGN_TEST_NETWORK_MODE` when needed).
//!
//! Example (local compose stack up):
//! `cargo test -p design-sandbox --test docker_install -- --ignored`

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

use design_harness::HarnessBundle;
use design_sandbox::{DockerSandbox, SandboxBackend, SandboxError};

const DEP_PYPROJECT: &str = r#"[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "design-dep-fixture"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["cowsay==6.1"]

[tool.setuptools]
py-modules = ["agent"]
"#;

/// Agent that proves (a) an installed third-party dep imports and runs,
/// (b) a public HTTP endpoint is reachable from the run phase via the egress
/// proxy, and (c) cloud metadata is refused by the proxy blocklist.
const DEP_AGENT: &str = r#"import json
import urllib.request

import cowsay


def _probe(url: str, timeout: int):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except Exception:
        return "refused"


def run(task, llm, out) -> None:
    said = cowsay.get_output_string("cow", "cowsay-dep-ok")
    probes = {
        "ext_http": _probe("http://example.com/", 20),
        "metadata": _probe("http://169.254.169.254/latest/meta-data/", 10),
    }
    for page in ("index.html", "pricing.html", "components.html"):
        out.write_page(
            page,
            "<html><body><pre>{}</pre><p>probes={}</p></body></html>".format(
                said, json.dumps(probes, sort_keys=True)
            ),
        )
"#;

const BROKEN_PYPROJECT: &str = r#"[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "design-broken-dep-fixture"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["definitely-not-a-real-pypi-package-xyz==0.0.0"]

[tool.setuptools]
py-modules = ["agent"]
"#;

fn bundle(pyproject: &str, agent: &str) -> HarnessBundle {
    let mut env_vars = BTreeMap::new();
    env_vars.insert("MINER_SECRET".into(), "supersecret-value".into());
    HarnessBundle {
        miner_hotkey: "ab".repeat(32),
        agent_py: agent.into(),
        pyproject_toml: pyproject.into(),
        extra_files: BTreeMap::new(),
        env_vars,
    }
}

fn sandbox() -> DockerSandbox {
    let base =
        std::env::var("DESIGN_TEST_DOCKER_BASE").unwrap_or_else(|_| "http://127.0.0.1:2375".into());
    let mut sb = DockerSandbox::new(&base, std::env::temp_dir().join("design-docker-test"))
        .unwrap_or_else(|e| panic!("docker sandbox init ({base}): {e}"));
    if let Ok(img) = std::env::var("DESIGN_TEST_RUNTIME_IMAGE") {
        sb.image = img;
    }
    if let Ok(net) = std::env::var("DESIGN_TEST_NETWORK_MODE") {
        sb.network_mode = net;
    }
    sb.client
        .list_owned()
        .unwrap_or_else(|e| panic!("docker engine unreachable at {base}: {e}"));
    sb
}

fn egress_proxy() -> String {
    std::env::var("DESIGN_TEST_EGRESS_PROXY")
        .unwrap_or_else(|_| "http://design-egress-proxy:8094".into())
}

fn run_id(tag: &str) -> String {
    format!("t{tag}{}", std::process::id())
}

#[test]
#[ignore = "requires docker HTTP endpoint + design-runtime image + egress proxy"]
fn install_run_with_pure_python_dep_and_open_egress() {
    let sb = sandbox();
    let out = sb
        .execute(
            &bundle(DEP_PYPROJECT, DEP_AGENT),
            1,
            &run_id("dep"),
            "fixture prompt",
            &egress_proxy(),
        )
        .unwrap_or_else(|e| panic!("install+run with cowsay dep failed: {e}"));
    assert!(
        out.install_logs.contains("pip-install-ok"),
        "install logs: {}",
        out.install_logs
    );
    let index = &out.pages["index.html"];
    assert!(
        index.contains("cowsay-dep-ok"),
        "dep output missing: {index}"
    );
    assert!(
        index.contains(r#""ext_http": 200"#),
        "public endpoint unreachable from run phase: {index}"
    );
    assert!(
        index.contains(r#""metadata": "refused""#),
        "cloud metadata must be refused by the proxy: {index}"
    );
    // Miner env is run-phase only; install logs must never leak it.
    assert!(
        !out.install_logs.contains("supersecret-value"),
        "install logs leaked miner env"
    );
    let _ = std::fs::remove_dir_all(std::env::temp_dir().join("design-docker-test"));
}

#[test]
#[ignore = "requires docker HTTP endpoint + design-runtime image + egress proxy"]
fn broken_dep_is_install_class_error() {
    let sb = sandbox();
    let agent = "def run(task, llm, out):\n    pass\n";
    let err = sb
        .execute(
            &bundle(BROKEN_PYPROJECT, agent),
            1,
            &run_id("brk"),
            "fixture prompt",
            &egress_proxy(),
        )
        .expect_err("nonexistent package must fail the install phase");
    match err {
        SandboxError::PhaseFailed { phase, logs, .. } => {
            assert_eq!(phase, "install", "logs: {logs}");
        }
        other => panic!("expected PhaseFailed{{install}}, got: {other}"),
    }
}
