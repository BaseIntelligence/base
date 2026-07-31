//! CLI: materialize a deepagent catalog into a local pack cache.
//!
//! Usage: `cargo run -p agent-pack --example materialize_cli -- <src> <cache>`

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(src) = args.next() else {
        eprintln!("usage: materialize_cli <src> <cache>");
        return ExitCode::FAILURE;
    };
    let Some(cache) = args.next() else {
        eprintln!("usage: materialize_cli <src> <cache>");
        return ExitCode::FAILURE;
    };
    let src = PathBuf::from(src);
    let cache = PathBuf::from(cache);
    let man = match agent_pack::materialize_catalog(&src, &cache) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("materialize: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "entries={} pin={} catalog_digest={}",
        man.entries.len(),
        man.pin,
        man.catalog_digest
    );
    for e in &man.entries {
        println!(
            "{} {} {}",
            e.pack_id, e.pack_digest, e.environment_image_digest
        );
    }
    ExitCode::SUCCESS
}
