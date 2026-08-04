//! ZIP → architecture.py + training.py unpack.

use std::io::{Cursor, Read};

use thiserror::Error;
use zip::ZipArchive;

use crate::{check_contract, ContractError, MAX_SOURCE_BYTES};

/// ZIP submit errors.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ZipSubmitError {
    /// Archive failure.
    #[error("invalid zip submission: {0}")]
    Invalid(String),
    /// Contract shape.
    #[error(transparent)]
    Contract(#[from] ContractError),
}

/// Unpack ZIP bytes into `(architecture_py, training_py)`.
///
/// Accepts flat or one top-level folder. Rejects symlinks / traversal.
///
/// # Errors
/// Archive or contract failures.
pub fn sources_from_zip(zip_bytes: &[u8]) -> Result<(String, String), ZipSubmitError> {
    if zip_bytes.len() > MAX_SOURCE_BYTES.saturating_mul(4) {
        return Err(ZipSubmitError::Invalid("zip exceeds size budget".into()));
    }
    let mut archive = ZipArchive::new(Cursor::new(zip_bytes))
        .map_err(|e| ZipSubmitError::Invalid(format!("zip open: {e}")))?;

    let mut architecture_py = None;
    let mut training_py = None;

    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| ZipSubmitError::Invalid(format!("zip entry: {e}")))?;
        if file.is_dir() {
            continue;
        }
        #[allow(deprecated)]
        if file.unix_mode().is_some_and(|m| m & 0o170_000 == 0o120_000) {
            return Err(ZipSubmitError::Invalid("symlinks forbidden".into()));
        }
        let name = file
            .enclosed_name()
            .ok_or_else(|| ZipSubmitError::Invalid("unsafe zip path".into()))?
            .to_string_lossy()
            .replace('\\', "/");
        let name = name.trim_start_matches("./");
        if name.is_empty() || name.contains("..") || name.starts_with('/') {
            return Err(ZipSubmitError::Invalid(format!("bad path: {name}")));
        }
        let rel = normalize_rel(name);
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| ZipSubmitError::Invalid(format!("zip read: {e}")))?;
        if data.len() > MAX_SOURCE_BYTES {
            return Err(ZipSubmitError::Invalid(format!("file too large: {rel}")));
        }
        let text = String::from_utf8(data)
            .map_err(|_| ZipSubmitError::Invalid(format!("non-utf8: {rel}")))?;
        match rel.as_str() {
            "architecture.py" => architecture_py = Some(text),
            "training.py" => training_py = Some(text),
            _ => {}
        }
    }

    let architecture_py =
        architecture_py.ok_or_else(|| ZipSubmitError::Invalid("architecture.py missing".into()))?;
    let training_py =
        training_py.ok_or_else(|| ZipSubmitError::Invalid("training.py missing".into()))?;
    check_contract(&architecture_py, &training_py)?;
    Ok((architecture_py, training_py))
}

fn normalize_rel(name: &str) -> String {
    let parts: Vec<&str> = name.split('/').filter(|p| !p.is_empty()).collect();
    match parts.as_slice() {
        [] => String::new(),
        [one] => (*one).to_owned(),
        [root, rest @ ..]
            if root
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') =>
        {
            rest.join("/")
        }
        _ => name.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use std::io::Write;
    use zip::write::SimpleFileOptions;
    use zip::ZipWriter;

    fn zip_of(files: &[(&str, &str)]) -> Vec<u8> {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut w = ZipWriter::new(&mut buf);
            let opts = SimpleFileOptions::default();
            for (n, b) in files {
                w.start_file(*n, opts).unwrap();
                w.write_all(b.as_bytes()).unwrap();
            }
            w.finish().unwrap();
        }
        buf.into_inner()
    }

    #[test]
    fn unpacks_sources() {
        let z = zip_of(&[
            (
                "architecture.py",
                "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n",
            ),
            ("training.py", "def train(model, ctx):\n    return {}\n"),
        ]);
        let (a, t) = sources_from_zip(&z).unwrap();
        assert!(a.contains("build_model"));
        assert!(t.contains("def train"));
    }
}
