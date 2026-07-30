//! `gbase-miner` — miner self-deployment CLI (task 37).
//!
//! ```text
//! gbase-miner deploy --no-deploy          # render + print compose-hash (default)
//! gbase-miner deploy --deploy             # also run `phala deploy`
//! ```
//!
//! Secrets are file mounts under `/run/gbase/` only. The miner funds their own
//! Phala account. See `docs/AGENT_CHALLENGE.md` §9.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use gbase_miner::{
    deploy_or_dry_run, empty_launch_token_hash_hex, DeployMode, DeployParams, DEFAULT_AGENT_IMAGE,
    DEFAULT_ATTEST_HELPER_IMAGE,
};

#[derive(Debug, Parser)]
#[command(
    name = "gbase-miner",
    about = "Miner CVM deploy/certify CLI (AGENT_CHALLENGE.md §9)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Render measured app-compose, print offline compose-hash, optionally phala deploy.
    ///
    /// Default is dry-run (`--no-deploy`): no Phala API calls. Pass `--deploy` to
    /// invoke `phala deploy` after printing the hash. Miner must fund their own
    /// Phala account.
    Deploy {
        /// CVM / app-compose name.
        #[arg(long, default_value = "gbase-miner")]
        name: String,
        /// Digest-pinned agent image (`repo@sha256:<64 hex>`).
        #[arg(long, default_value = DEFAULT_AGENT_IMAGE, env = "GBASE_AGENT_IMAGE")]
        agent_image: String,
        /// Digest-pinned attest-helper image.
        #[arg(
            long,
            default_value = DEFAULT_ATTEST_HELPER_IMAGE,
            env = "GBASE_ATTEST_HELPER_IMAGE"
        )]
        attest_helper_image: String,
        /// Lowercase hex SHA-256 of the launch token (measured; not the raw token).
        #[arg(long, env = "GBASE_LAUNCH_TOKEN_HASH")]
        launch_token_hash: Option<String>,
        /// Subnet netuid embedded as non-secret env.
        #[arg(long, default_value_t = 1, env = "GBASE_NETUID")]
        netuid: u16,
        /// Write rendered app-compose.json here.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Skip `phala deploy` (default). Print compose-hash only.
        #[arg(long, conflicts_with = "deploy")]
        no_deploy: bool,
        /// Actually invoke `phala deploy` after hashing (requires Phala CLI + funded account).
        #[arg(long)]
        deploy: bool,
        /// Path to `phala` binary.
        #[arg(long, default_value = "phala", env = "GBASE_PHALA_BIN")]
        phala_bin: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("gbase-miner: {msg}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.cmd {
        Cmd::Deploy {
            name,
            agent_image,
            attest_helper_image,
            launch_token_hash,
            netuid,
            out,
            no_deploy: _,
            deploy,
            phala_bin,
        } => {
            // Default: dry-run. Only `--deploy` invokes Phala.
            let mode = if deploy {
                DeployMode::Deploy
            } else {
                DeployMode::NoDeploy
            };
            let params = DeployParams {
                name,
                agent_image,
                attest_helper_image,
                launch_token_hash: launch_token_hash.unwrap_or_else(empty_launch_token_hash_hex),
                netuid,
                mode,
                out_compose: out,
                phala_bin,
            };
            let result = deploy_or_dry_run(&params).map_err(|e| e.to_string())?;
            // Stable machine-readable lines for tests / operators.
            println!("compose-hash={}", result.compose_hash_hex);
            println!("phala_invoked={}", result.phala_invoked);
            println!("mode={mode:?}");
            println!(
                "note=miner_funds_own_phala_account secrets_are_file_mounts_under_/run/gbase"
            );
            Ok(())
        }
    }
}
