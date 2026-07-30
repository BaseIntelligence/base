//! gbase workspace maintenance binary.
//!
//! Subcommands:
//! - `loc-cap` — fail if any crate under `crates/` or `bins/` exceeds 1500 non-test LOC
//! - `consensus-lint` — fail if listed consensus crates use forbidden tokens (D8)
//! - `metadata-snapshot` — fetch testnet metadata + epoch-schedule sources into `metadata/testnet.lock`

#![allow(clippy::print_stdout, clippy::print_stderr)]

mod consensus_lint;
mod loc_cap;
mod metadata_snapshot;

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
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask error: {err}");
            ExitCode::FAILURE
        }
    }
}
