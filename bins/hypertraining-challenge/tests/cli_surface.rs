//! Process-level surface for hypertraining-challenge binary (todo 13).
//!
//! S1: serve + GET /health → 200
//! S2: missing sk file → non-zero exit

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::io::Write;
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_hypertraining-challenge")
}

fn write_sk(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("challenge_sk");
    let m = schnorrkel::MiniSecretKey::generate_with(rand_core::OsRng);
    let mut f = std::fs::File::create(&path).expect("create sk");
    f.write_all(&m.to_bytes()).expect("write sk");
    path
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .expect("bind ephemeral")
        .local_addr()
        .expect("local_addr")
        .port()
}

#[test]
fn s2_missing_sk_file_exits_nonzero() {
    let status = Command::new(bin())
        .env_remove("BASE_CHALLENGE_SK_FILE")
        .env_remove("HYPERTRAINING_CHALLENGE_SK_FILE")
        .args(["--bind", "127.0.0.1:0", "serve"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .expect("spawn");
    assert!(
        !status.success(),
        "expected non-zero exit when sk file missing, got {status}"
    );
}

#[test]
fn s2b_sk_path_missing_on_disk_exits_nonzero() {
    let dir = tempfile::tempdir().expect("tempdir");
    let missing = dir.path().join("no-such-sk");
    let status = Command::new(bin())
        .env_remove("HYPERTRAINING_CHALLENGE_SK_FILE")
        .env("BASE_CHALLENGE_SK_FILE", &missing)
        .args(["--bind", "127.0.0.1:0", "serve"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .status()
        .expect("spawn");
    assert!(
        !status.success(),
        "expected non-zero when sk path does not exist, got {status}"
    );
}

#[test]
fn s1_health_returns_200() {
    let dir = tempfile::tempdir().expect("tempdir");
    let sk = write_sk(dir.path());
    let port = free_port();
    let bind = format!("127.0.0.1:{port}");

    let mut child = Command::new(bin())
        .env_remove("HYPERTRAINING_CHALLENGE_SK_FILE")
        .env("BASE_CHALLENGE_SK_FILE", &sk)
        .args(["--bind", &bind, "serve"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn serve");

    let url = format!("http://{bind}/health");
    let deadline = Instant::now() + Duration::from_secs(15);
    let mut last_err = String::new();
    let body = loop {
        if Instant::now() > deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("health never became ready: {last_err}");
        }
        match http_get(&url) {
            Ok((200, body)) => break body,
            Ok((code, body)) => last_err = format!("status {code} body {body:?}"),
            Err(e) => last_err = e,
        }
        thread::sleep(Duration::from_millis(50));
    };

    assert_eq!(body.trim(), "ok");

    let _ = child.kill();
    let _ = child.wait();
}

fn http_get(url: &str) -> Result<(u16, String), String> {
    let resp = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|e| e.to_string())?
        .get(url)
        .send()
        .map_err(|e| e.to_string())?;
    let status = resp.status().as_u16();
    let body = resp.text().map_err(|e| e.to_string())?;
    Ok((status, body))
}
