//! gbase workspace maintenance binary.
//!
//! Subcommands:
//! - `loc-cap` — fail if any crate under `crates/` or `bins/` exceeds 1500 non-test LOC
//! - `consensus-lint` — fail if listed consensus crates use forbidden tokens (D8)

#![allow(clippy::print_stdout, clippy::print_stderr)]

mod consensus_lint;
mod loc_cap;

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
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("xtask error: {err}");
            ExitCode::FAILURE
        }
    }
}
