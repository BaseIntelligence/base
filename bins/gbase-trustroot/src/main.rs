//! `gbase-trustroot` — offline ceremony CLI (keygen / sign / verify).
//!
//! Secrets are written outside the git tree (operator chooses path). Prefer
//! `/root/.gbase-secrets/` with mode 0700. Age encryption is applied via the
//! system `age` binary when `--age-recipient` is set.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use clap::{Parser, Subcommand};
use gbase_crypto::{KEY_LEN, SIGNATURE_LEN};
use gbase_trustroot::{
    encode_challenges_body, encode_hex, encode_measurements_body, load_challenges_file_with_sig,
    load_measurements_file_with_sig, load_owner_public_key, sign_trust_root_raw, ChallengesToml,
    MeasurementsToml, TrustRootError,
};
use rand_core::OsRng;
use schnorrkel::MiniSecretKey;

#[derive(Debug, Parser)]
#[command(name = "gbase-trustroot", about = "Owner-signed trust root ceremony tools")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Generate an sr25519 mini-secret + public key (challenge or owner throwaway).
    Keygen {
        /// Write 32-byte public key as hex here (safe to commit when owner).
        #[arg(long)]
        out_pub: PathBuf,
        /// Write secret material here (NEVER commit). Raw 32 bytes, or age-encrypted if --age-recipient.
        #[arg(long)]
        out_secret: PathBuf,
        /// Optional age recipient (`age1...`). Encrypts secret to `out_secret`.
        #[arg(long)]
        age_recipient: Option<String>,
    },
    /// Produce a detached owner signature over a challenges or measurements TOML body.
    Sign {
        /// Path to 32-byte mini-secret (raw) or age-encrypted ciphertext.
        #[arg(long)]
        key: PathBuf,
        /// Decrypt age ciphertext with this identity file (age -d -i).
        #[arg(long)]
        age_identity: Option<PathBuf>,
        /// Input TOML (challenges.toml or measurements.toml).
        #[arg(long)]
        input: PathBuf,
        /// Output signature path (hex). Default: `<input>.sig`.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Document kind.
        #[arg(long, value_parser = ["challenges", "measurements"])]
        kind: String,
    },
    /// Verify a detached owner signature.
    Verify {
        /// Owner public key hex file.
        #[arg(long)]
        owner_pub: PathBuf,
        /// Input TOML.
        #[arg(long)]
        input: PathBuf,
        /// Signature path. Default: `<input>.sig`.
        #[arg(long)]
        sig: Option<PathBuf>,
        /// Document kind.
        #[arg(long, value_parser = ["challenges", "measurements"])]
        kind: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("gbase-trustroot: {msg}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.cmd {
        Cmd::Keygen {
            out_pub,
            out_secret,
            age_recipient,
        } => cmd_keygen(&out_pub, &out_secret, age_recipient.as_deref()),
        Cmd::Sign {
            key,
            age_identity,
            input,
            out,
            kind,
        } => cmd_sign(&key, age_identity.as_deref(), &input, out.as_deref(), &kind),
        Cmd::Verify {
            owner_pub,
            input,
            sig,
            kind,
        } => cmd_verify(&owner_pub, &input, sig.as_deref(), &kind),
    }
}

fn cmd_keygen(
    out_pub: &Path,
    out_secret: &Path,
    age_recipient: Option<&str>,
) -> Result<(), String> {
    refuse_git_secret_path(out_secret)?;
    let mini = MiniSecretKey::generate_with(OsRng);
    let secret = mini.to_bytes();
    let public = mini
        .expand(schnorrkel::ExpansionMode::Ed25519)
        .to_public()
        .to_bytes();

    if let Some(parent) = out_pub.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    if let Some(parent) = out_secret.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    fs::write(out_pub, format!("{}\n", encode_hex(&public))).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(out_pub, fs::Permissions::from_mode(0o644));
    }

    if let Some(recipient) = age_recipient {
        age_encrypt(&secret, recipient, out_secret)?;
    } else {
        fs::write(out_secret, secret).map_err(|e| e.to_string())?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(out_secret, fs::Permissions::from_mode(0o600));
    }

    println!("public_key={}", encode_hex(&public));
    println!("pub_path={}", out_pub.display());
    println!("secret_path={}", out_secret.display());
    if age_recipient.is_some() {
        println!("secret_format=age");
    } else {
        println!("secret_format=raw32");
    }
    Ok(())
}

fn cmd_sign(
    key_path: &Path,
    age_identity: Option<&Path>,
    input: &Path,
    out: Option<&Path>,
    kind: &str,
) -> Result<(), String> {
    let secret = load_secret(key_path, age_identity)?;
    let text = fs::read_to_string(input).map_err(|e| format!("read input: {e}"))?;
    let (version, introduced_epoch, body_scale) = match kind {
        "challenges" => {
            let doc: ChallengesToml =
                toml::from_str(&text).map_err(|e| format!("parse challenges: {e}"))?;
            let body = doc.to_body().map_err(|e| err_str(&e))?;
            (
                doc.version,
                doc.introduced_epoch,
                encode_challenges_body(&body),
            )
        }
        "measurements" => {
            let doc: MeasurementsToml =
                toml::from_str(&text).map_err(|e| format!("parse measurements: {e}"))?;
            let body = doc.to_body().map_err(|e| err_str(&e))?;
            (
                doc.version,
                doc.introduced_epoch,
                encode_measurements_body(&body),
            )
        }
        _ => return Err(format!("unknown kind {kind}")),
    };
    let sig = sign_trust_root_raw(&secret, version, introduced_epoch, &body_scale).map_err(|e| err_str(&e))?;
    let out_path = out.map_or_else(
        || PathBuf::from(format!("{}.sig", input.display())),
        Path::to_path_buf,
    );
    fs::write(&out_path, format!("{}\n", encode_hex(&sig))).map_err(|e| e.to_string())?;
    println!("signature={}", encode_hex(&sig));
    println!("sig_path={}", out_path.display());
    Ok(())
}

fn cmd_verify(
    owner_pub_path: &Path,
    input: &Path,
    sig: Option<&Path>,
    kind: &str,
) -> Result<(), String> {
    let owner = load_owner_public_key(owner_pub_path).map_err(|e| err_str(&e))?;
    let sig_path = sig.map_or_else(
        || PathBuf::from(format!("{}.sig", input.display())),
        Path::to_path_buf,
    );
    match kind {
        "challenges" => {
            load_challenges_file_with_sig(input, &sig_path, &owner).map_err(|e| err_str(&e))?;
        }
        "measurements" => {
            load_measurements_file_with_sig(input, &sig_path, &owner).map_err(|e| err_str(&e))?;
        }
        _ => return Err(format!("unknown kind {kind}")),
    }
    println!("OK verified under owner {}", encode_hex(&owner));
    Ok(())
}

fn load_secret(path: &Path, age_identity: Option<&Path>) -> Result<[u8; KEY_LEN], String> {
    let raw = fs::read(path).map_err(|e| format!("read key: {e}"))?;
    if let Some(id) = age_identity {
        return age_decrypt(&raw, id);
    }
    // Age armor / binary starts with "age-encryption.org" or binary header.
    if raw.starts_with(b"age-encryption.org") || raw.starts_with(b"-----BEGIN AGE") {
        return Err("age-encrypted secret requires --age-identity".into());
    }
    if raw.len() == KEY_LEN {
        let mut out = [0u8; KEY_LEN];
        out.copy_from_slice(&raw);
        return Ok(out);
    }
    // Hex secret file.
    let text = std::str::from_utf8(&raw).map_err(|e| e.to_string())?;
    gbase_trustroot::decode_hex_array(text.trim()).map_err(|e| err_str(&e))
}

fn age_encrypt(secret: &[u8; KEY_LEN], recipient: &str, out: &Path) -> Result<(), String> {
    let mut child = Command::new("age")
        .arg("-r")
        .arg(recipient)
        .arg("-o")
        .arg(out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn age: {e} (install age CLI)"))?;
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().ok_or("age stdin")?;
        stdin.write_all(secret).map_err(|e| e.to_string())?;
    }
    let outp = child.wait_with_output().map_err(|e| e.to_string())?;
    if !outp.status.success() {
        return Err(format!(
            "age encrypt failed: {}",
            String::from_utf8_lossy(&outp.stderr)
        ));
    }
    Ok(())
}

fn age_decrypt(ciphertext: &[u8], identity: &Path) -> Result<[u8; KEY_LEN], String> {
    let mut child = Command::new("age")
        .arg("-d")
        .arg("-i")
        .arg(identity)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn age: {e}"))?;
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().ok_or("age stdin")?;
        stdin.write_all(ciphertext).map_err(|e| e.to_string())?;
    }
    let outp = child.wait_with_output().map_err(|e| e.to_string())?;
    if !outp.status.success() {
        return Err(format!(
            "age decrypt failed: {}",
            String::from_utf8_lossy(&outp.stderr)
        ));
    }
    if outp.stdout.len() != KEY_LEN {
        return Err(format!(
            "decrypted secret length {}, expected {KEY_LEN}",
            outp.stdout.len()
        ));
    }
    let mut key = [0u8; KEY_LEN];
    key.copy_from_slice(&outp.stdout);
    let _ = SIGNATURE_LEN; // keep import used if optimized
    Ok(key)
}

fn refuse_git_secret_path(path: &Path) -> Result<(), String> {
    let s = path.to_string_lossy();
    if s.contains("/gbase/config/") || s.contains("/gbase/crates/") || s.ends_with(".toml") {
        return Err(format!(
            "refusing to write secret into likely-git path: {}",
            path.display()
        ));
    }
    Ok(())
}

fn err_str(e: &TrustRootError) -> String {
    e.to_string()
}
