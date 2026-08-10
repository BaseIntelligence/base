//! ffmpeg video compress (720p, CRF ~28).

use std::path::{Path, PathBuf};
use std::process::Stdio;

use sha2::{Digest, Sha256};
use thiserror::Error;
use tokio::process::Command;

/// Compress outcome.
#[derive(Debug, Clone)]
pub struct CompressResult {
    /// Output path under artifacts volume.
    pub video_path: PathBuf,
    /// sha256 hex of compressed bytes.
    pub sha256: String,
    /// Byte length.
    pub bytes: u64,
}

/// Compress failures.
#[derive(Debug, Error)]
pub enum CompressError {
    /// Missing input / IO.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// ffmpeg non-zero exit.
    #[error("ffmpeg: {0}")]
    Ffmpeg(String),
    /// Empty output.
    #[error("empty compressed output")]
    EmptyOutput,
}

/// Compress `input` → `output` with libx264 720p CRF 28 + aac.
///
/// When `force_sim` is set, copies the input bytes (CI / no ffmpeg) and hashes
/// the copy — never invents success without writing a file.
///
/// # Errors
/// IO / ffmpeg failure / empty output.
pub async fn compress_video(
    input: &Path,
    output: &Path,
    force_sim: bool,
) -> Result<CompressResult, CompressError> {
    if let Some(parent) = output.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    if force_sim {
        tokio::fs::copy(input, output).await?;
    } else {
        let status = Command::new("ffmpeg")
            .args([
                "-y",
                "-i",
                input.to_str().unwrap_or(""),
                "-vf",
                "scale=-2:720",
                "-c:v",
                "libx264",
                "-crf",
                "28",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                output.to_str().unwrap_or(""),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .status()
            .await?;
        if !status.success() {
            return Err(CompressError::Ffmpeg(format!("exit {status}")));
        }
    }
    let bytes = tokio::fs::read(output).await?;
    if bytes.is_empty() {
        return Err(CompressError::EmptyOutput);
    }
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(CompressResult {
        video_path: output.to_path_buf(),
        sha256: hex::encode(h.finalize()),
        bytes: bytes.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[tokio::test]
    async fn sim_compress_copies_and_hashes() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("raw.bin");
        let output = dir.path().join("out.mp4");
        tokio::fs::write(&input, b"fake-video-bytes").await.unwrap();
        let r = compress_video(&input, &output, true).await.unwrap();
        assert_eq!(r.bytes, 16);
        assert_eq!(r.sha256.len(), 64);
        assert_eq!(tokio::fs::read(&output).await.unwrap(), b"fake-video-bytes");
    }
}
