//! gbase workspace maintenance binary.
//!
//! Subcommands:
//! - `loc-cap` — fail if any crate under `crates/` or `bins/` exceeds 1500 non-test LOC
//! - `consensus-lint` — fail if listed consensus crates use forbidden tokens (D8)
//! - `metadata-snapshot` — fetch testnet metadata + epoch-schedule sources into `metadata/testnet.lock`
//! - `spec-check` — fail if `docs/BUNDLE_SPEC.md` is missing plan pins (a)–(l)
//! - `agent-challenge-check` — fail if `docs/AGENT_CHALLENGE.md` is missing plan task 9 pins
//! - `external-docs-check` — fail if external miner docs `protocol_version` ≠ bundle, or D19 drifts
#![allow(clippy::print_stdout, clippy::print_stderr)]

mod agent_challenge_check;
mod consensus_lint;
mod external_docs_check;
mod loc_cap;
mod metadata_snapshot;
mod spec_check;

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Debug, Parser)]
#[command(name = "xtask", about = "gbase workspace maintenance tasks")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Fail if any package under crates/ or bins/ exceeds 1500 non-test Rust LOC.
    LocCap,
    /// Fail if consensus crates contain `HashMap`, `f32`/`f64`, `wrapping_*`, or bare `u128` ops.
    ConsensusLint,
    /// Snapshot Finney testnet metadata + epoch-schedule read paths into a lockfile.
    MetadataSnapshot {
        /// JSON-RPC endpoint (`wss://` is rewritten to `https://`).
        #[arg(long, default_value = metadata_snapshot::DEFAULT_ENDPOINT)]
        endpoint: String,
        /// Netuid used when probing per-subnet schedule storage (default 1).
        #[arg(long, default_value_t = metadata_snapshot::DEFAULT_SNAPSHOT_NETUID)]
        netuid: u16,
        /// Lockfile path relative to workspace root (or absolute).
        #[arg(long, default_value = "metadata/testnet.lock")]
        out: PathBuf,
        /// Compare live snapshot to the committed lockfile; exit 1 on drift.
        #[arg(long)]
        check: bool,
    },
    /// Fail if `docs/BUNDLE_SPEC.md` is missing required (a)–(l) pins (task 8).
    SpecCheck,
    /// Fail if `docs/AGENT_CHALLENGE.md` is missing required task 9 pins.
    AgentChallengeCheck,
    /// Fail if external miner docs `protocol_version` differs from `bundle`, or `THREAT_MODEL` D19 drifts.
    ExternalDocsCheck,
}

fn workspace_root() -> Result<PathBuf, String> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "xtask crate must live one level under the workspace root".into())
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let root = match workspace_root() {
        Ok(p) => p,
        Err(err) => {
            eprintln!("xtask error: {err}");
            return ExitCode::FAILURE;
        }
    };
    let result = match cli.command {
        Command::LocCap => loc_cap::run(&root),
        Command::ConsensusLint => consensus_lint::run(&root),
        Command::MetadataSnapshot {
            endpoint,
            netuid,
            out,
            check,
        } => {
            let args = metadata_snapshot::SnapshotArgs {
                endpoint,
                netuid,
                out,
                check,
            };
            metadata_snapshot::run(&root, &args)
        }
        Command::SpecCheck => spec_check::run(&root),
        Command::AgentChallengeCheck => agent_challenge_check::run(&root),
        Command::ExternalDocsCheck => external_docs_check::run(&root),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask error: {err}");
            ExitCode::FAILURE
        }
    }
}
