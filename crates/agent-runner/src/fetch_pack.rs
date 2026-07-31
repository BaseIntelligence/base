//! On-demand Harbor pack fetch from a catalog HTTP base.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use agent_pack::load_pack;
use flate2::read::GzDecoder;
use thiserror::Error;

/// Failures while ensuring a pack is present under `pack_root`.
#[derive(Debug, Error)]
pub enum PackFetchError {
    /// Pack missing and no catalog URL configured.
    #[error("pack {pack_id} missing under {pack_root} and BASE_PACK_CATALOG_URL unset")]
    MissingNoCatalog { pack_id: String, pack_root: String },
    /// Invalid pack id (path traversal / empty).
    #[error("invalid pack_id: {0}")]
    InvalidPackId(String),
    /// HTTP or I/O failure while downloading/extracting.
    #[error("pack fetch: {0}")]
    Fetch(String),
    /// Downloaded archive did not yield a loadable Harbor pack.
    #[error("pack after fetch invalid: {0}")]
    InvalidAfterFetch(String),
}

/// Ensure `{pack_root}/{pack_id}` is a loadable Harbor pack.
///
/// If already loadable via [`load_pack`], returns immediately.
/// Otherwise, when `catalog_url` is `Some` non-empty, GETs
/// `{catalog_url}/v1/packs/{pack_id}` as a gzipped tar and extracts
/// atomically into `pack_root/pack_id`.
///
/// # Errors
/// Missing pack without catalog, bad id, HTTP/IO failures, or invalid archive.
pub fn ensure_pack(
    pack_root: &Path,
    catalog_url: Option<&str>,
    pack_id: &str,
) -> Result<(), PackFetchError> {
    validate_pack_id(pack_id)?;
    let dest = pack_root.join(pack_id);
    if pack_loadable(&dest) {
        return Ok(());
    }

    let catalog = catalog_url.map(str::trim).filter(|s| !s.is_empty());
    let Some(base) = catalog else {
        return Err(PackFetchError::MissingNoCatalog {
            pack_id: pack_id.to_owned(),
            pack_root: pack_root.display().to_string(),
        });
    };

    let url = format!("{}/v1/packs/{}", base.trim_end_matches('/'), pack_id);
    let bytes = http_get_bytes(&url)?;
    extract_tar_gz_atomic(pack_root, pack_id, &bytes)?;
    if !pack_loadable(&dest) {
        return Err(PackFetchError::InvalidAfterFetch(format!(
            "{} not a Harbor pack after extract",
            dest.display()
        )));
    }
    Ok(())
}

fn validate_pack_id(pack_id: &str) -> Result<(), PackFetchError> {
    if pack_id.is_empty()
        || pack_id.contains("..")
        || pack_id.contains('/')
        || pack_id.contains('\\')
        || pack_id.contains('\0')
    {
        return Err(PackFetchError::InvalidPackId(pack_id.to_owned()));
    }
    Ok(())
}

fn pack_loadable(dir: &Path) -> bool {
    load_pack(dir).is_ok()
}

fn http_get_bytes(url: &str) -> Result<Vec<u8>, PackFetchError> {
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| PackFetchError::Fetch(e.to_string()))?;
    let resp = client
        .get(url)
        .send()
        .map_err(|e| PackFetchError::Fetch(e.to_string()))?;
    if !resp.status().is_success() {
        return Err(PackFetchError::Fetch(format!(
            "GET {url} -> HTTP {}",
            resp.status()
        )));
    }
    resp.bytes()
        .map(|b| b.to_vec())
        .map_err(|e| PackFetchError::Fetch(e.to_string()))
}

fn extract_tar_gz_atomic(
    pack_root: &Path,
    pack_id: &str,
    gz_tar: &[u8],
) -> Result<(), PackFetchError> {
    fs::create_dir_all(pack_root).map_err(|e| PackFetchError::Fetch(e.to_string()))?;
    let ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let staging = pack_root.join(format!(".fetch-{pack_id}-{ns}"));
    let final_dir = pack_root.join(pack_id);

    let _ = fs::remove_dir_all(&staging);
    if final_dir.exists() && !pack_loadable(&final_dir) {
        let _ = fs::remove_dir_all(&final_dir);
    }

    fs::create_dir_all(&staging).map_err(|e| PackFetchError::Fetch(e.to_string()))?;

    let result = (|| -> Result<(), PackFetchError> {
        let dec = GzDecoder::new(gz_tar);
        let mut archive = tar::Archive::new(dec);
        archive
            .unpack(&staging)
            .map_err(|e| PackFetchError::Fetch(format!("tar unpack: {e}")))?;

        let content_root = resolve_extracted_root(&staging, pack_id)?;
        if content_root != staging {
            let nested = staging.join(pack_id);
            if nested.is_dir() && pack_loadable(&nested) {
                if final_dir.exists() {
                    let _ = fs::remove_dir_all(&final_dir);
                }
                fs::rename(&nested, &final_dir)
                    .map_err(|e| PackFetchError::Fetch(e.to_string()))?;
                let _ = fs::remove_dir_all(&staging);
                return Ok(());
            }
            if final_dir.exists() {
                let _ = fs::remove_dir_all(&final_dir);
            }
            fs::rename(&content_root, &final_dir)
                .map_err(|e| PackFetchError::Fetch(e.to_string()))?;
            let _ = fs::remove_dir_all(&staging);
            return Ok(());
        }

        if final_dir.exists() {
            let _ = fs::remove_dir_all(&final_dir);
        }
        fs::rename(&staging, &final_dir).map_err(|e| PackFetchError::Fetch(e.to_string()))?;
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_dir_all(&staging);
        if final_dir.exists() && !pack_loadable(&final_dir) {
            let _ = fs::remove_dir_all(&final_dir);
        }
    }
    result
}

fn resolve_extracted_root(staging: &Path, pack_id: &str) -> Result<PathBuf, PackFetchError> {
    if pack_loadable(staging) {
        return Ok(staging.to_path_buf());
    }
    let nested = staging.join(pack_id);
    if pack_loadable(&nested) {
        return Ok(nested);
    }
    let mut dirs = Vec::new();
    let rd = fs::read_dir(staging).map_err(|e| PackFetchError::Fetch(e.to_string()))?;
    for ent in rd {
        let ent = ent.map_err(|e| PackFetchError::Fetch(e.to_string()))?;
        let p = ent.path();
        if p.is_dir() {
            dirs.push(p);
        }
    }
    if dirs.len() == 1 && pack_loadable(&dirs[0]) {
        return Ok(dirs[0].clone());
    }
    Err(PackFetchError::InvalidAfterFetch(
        "archive has no Harbor pack layout (need task.toml + instruction.md + environment/Dockerfile)"
            .into(),
    ))
}

/// Build a gzipped tar of a Harbor pack directory (test helper).
#[cfg(test)]
pub fn pack_dir_to_tar_gz(dir: &Path) -> Result<Vec<u8>, String> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut builder = tar::Builder::new(Vec::new());
    builder
        .append_dir_all(".", dir)
        .map_err(|e| e.to_string())?;
    let tar_bytes = builder.into_inner().map_err(|e| e.to_string())?;
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&tar_bytes).map_err(|e| e.to_string())?;
    enc.finish().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;

    fn fixture_minimal_ok() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../agent-pack/tests/fixtures/minimal-ok")
    }

    fn copy_pack(src: &Path, dest: &Path) {
        fn copy_rec(from: &Path, to: &Path) {
            fs::create_dir_all(to).expect("mkdir");
            for ent in fs::read_dir(from).expect("rd") {
                let ent = ent.expect("ent");
                let p = ent.path();
                let name = ent.file_name();
                let dst = to.join(name);
                if p.is_dir() {
                    copy_rec(&p, &dst);
                } else {
                    fs::copy(&p, &dst).expect("cp");
                }
            }
        }
        copy_rec(src, dest);
    }

    #[test]
    fn ensure_pack_ok_when_already_on_disk() {
        let tmp = tempfile::tempdir().expect("tmp");
        let pack_root = tmp.path();
        let dest = pack_root.join("minimal-ok");
        copy_pack(&fixture_minimal_ok(), &dest);
        ensure_pack(pack_root, None, "minimal-ok").expect("already present");
        ensure_pack(pack_root, Some(""), "minimal-ok").expect("empty catalog ignored");
    }

    #[test]
    fn ensure_pack_missing_without_catalog_errors() {
        let tmp = tempfile::tempdir().expect("tmp");
        let err = ensure_pack(tmp.path(), None, "no-such-pack").expect_err("must fail");
        match err {
            PackFetchError::MissingNoCatalog { pack_id, .. } => {
                assert_eq!(pack_id, "no-such-pack");
            }
            other => panic!("unexpected: {other}"),
        }
    }

    #[test]
    fn ensure_pack_rejects_path_traversal_id() {
        let tmp = tempfile::tempdir().expect("tmp");
        let err = ensure_pack(tmp.path(), Some("http://127.0.0.1:9"), "../x").expect_err("bad id");
        assert!(matches!(err, PackFetchError::InvalidPackId(_)));
    }

    #[test]
    fn ensure_pack_fetches_from_catalog_http() {
        let fixture = fixture_minimal_ok();
        let body = pack_dir_to_tar_gz(&fixture).expect("tar.gz");
        let body = Arc::new(body);

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let body_srv = Arc::clone(&body);
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut req = [0u8; 4096];
            let _ = stream.read(&mut req);
            let req_s = String::from_utf8_lossy(&req);
            assert!(
                req_s.contains("GET /v1/packs/minimal-ok"),
                "request: {req_s}"
            );
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body_srv.len()
            );
            stream.write_all(header.as_bytes()).expect("hdr");
            stream.write_all(&body_srv).expect("body");
        });

        let tmp = tempfile::tempdir().expect("tmp");
        let catalog = format!("http://{addr}");
        ensure_pack(tmp.path(), Some(&catalog), "minimal-ok").expect("fetch");
        handle.join().expect("server");
        load_pack(tmp.path().join("minimal-ok")).expect("loadable after fetch");
    }

    #[test]
    fn ensure_pack_catalog_404_errors() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut req = [0u8; 1024];
            let _ = stream.read(&mut req);
            let resp = b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
            stream.write_all(resp).expect("write");
        });
        let tmp = tempfile::tempdir().expect("tmp");
        let catalog = format!("http://{addr}");
        let err = ensure_pack(tmp.path(), Some(&catalog), "missing").expect_err("404");
        assert!(matches!(err, PackFetchError::Fetch(_)), "{err}");
        handle.join().expect("server");
    }
}
