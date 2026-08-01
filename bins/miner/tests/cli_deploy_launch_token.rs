//! `miner deploy --launch-token-file`: 0600 provisioning, reuse, no leaks.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

fn miner_bin() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_miner") {
        return PathBuf::from(p);
    }
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../target/debug/miner");
    p
}

fn work_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("miner-launch-token-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn deploy(dir: &Path, token_file: Option<&Path>) -> String {
    let mut cmd = Command::new(miner_bin());
    cmd.args([
        "deploy",
        "--no-deploy",
        "--receipt-sk-host-path",
        dir.join("receipt_sk").to_str().unwrap(),
    ]);
    // The env fallbacks would otherwise smuggle an operator token into the test.
    cmd.env_remove("BASE_LAUNCH_TOKEN_FILE")
        .env_remove("BASE_LAUNCH_TOKEN_HASH");
    if let Some(path) = token_file {
        cmd.args(["--launch-token-file", path.to_str().unwrap()]);
    }
    let out = cmd.output().expect("spawn miner");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "deploy failed: {stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

fn field<'a>(stdout: &'a str, key: &str) -> &'a str {
    stdout
        .lines()
        .find_map(|l| l.strip_prefix(key))
        .unwrap_or_else(|| panic!("missing {key} in:\n{stdout}"))
}

#[test]
fn generated_launch_token_is_0600_reused_and_never_printed() {
    let dir = work_dir("gen");
    let token_path = dir.join("nested").join("launch_token");

    let first = deploy(&dir, Some(&token_path));
    let token = std::fs::read_to_string(&token_path).unwrap();
    assert_eq!(token.len(), 64, "32 random bytes hex-encoded");
    assert!(token.chars().all(|c| c.is_ascii_hexdigit()));
    let mode = std::fs::metadata(&token_path).unwrap().permissions().mode();
    assert_eq!(mode & 0o777, 0o600, "token file must be owner-only");
    assert!(!first.contains(&token), "token leaked to stdout:\n{first}");
    assert!(first.contains(&format!("launch-token-host-path={}", token_path.display())));
    assert!(!first.contains("warning=no_launch_token_configured"));

    let second = deploy(&dir, Some(&token_path));
    assert_eq!(
        std::fs::read_to_string(&token_path).unwrap(),
        token,
        "an existing token file must be reused, not overwritten"
    );
    assert_eq!(
        field(&first, "compose-hash="),
        field(&second, "compose-hash="),
        "same token ⇒ same measured compose"
    );

    let other = dir.join("other_token");
    let third = deploy(&dir, Some(&other));
    assert_ne!(
        field(&first, "compose-hash="),
        field(&third, "compose-hash=")
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn deploy_without_a_token_warns_that_quotes_cannot_be_served() {
    let dir = work_dir("empty");
    let stdout = deploy(&dir, None);
    assert!(
        stdout.contains("warning=no_launch_token_configured"),
        "missing warning:\n{stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn trailing_newline_is_not_part_of_the_token() {
    let dir = work_dir("trim");
    let a = dir.join("plain");
    let b = dir.join("newline");
    std::fs::write(&a, "opaque-token").unwrap();
    std::fs::write(&b, "opaque-token\n").unwrap();
    assert_eq!(
        field(&deploy(&dir, Some(&a)), "compose-hash="),
        field(&deploy(&dir, Some(&b)), "compose-hash=")
    );
    let _ = std::fs::remove_dir_all(&dir);
}
