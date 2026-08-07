//! Content sniffing for intake.
//!
//! Two questions intake needs answered before it can decide whether a file may
//! enter the tree: *is this text I can run the banned-pattern scan over?* and
//! *is this a prebuilt executable?* Extension checks alone answer neither — a
//! miner can name an ELF `tokenizer.json`.

/// File signatures that indicate machine code or a nested archive.
///
/// Matched at offset 0. Anything here is a prebuilt artifact, which the recipe
/// bans outright: every kernel must be compiled on the pod from submitted
/// source so the cheatguard and the reviewer can read what will run.
const MAGIC_AT_START: &[(&[u8], &str)] = &[
    (b"\x7fELF", "ELF executable"),
    (b"MZ", "DOS/PE executable"),
    (b"\xfe\xed\xfa\xce", "Mach-O executable"),
    (b"\xfe\xed\xfa\xcf", "Mach-O executable"),
    (b"\xce\xfa\xed\xfe", "Mach-O executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable"),
    (b"\xca\xfe\xba\xbe", "Mach-O fat or Java class file"),
    (b"\0asm", "WebAssembly module"),
    (b"!<arch>", "static library archive"),
    (b"PK\x03\x04", "zip archive"),
    (b"\x1f\x8b", "gzip stream"),
    (b"BZh", "bzip2 stream"),
    (b"\xfd7zXZ", "xz stream"),
    (b"\x28\xb5\x2f\xfd", "zstd stream"),
    (b"\x50\xed\x55\xba", "CUDA fatbin"),
];

/// Signatures searched for at *any* offset, so an executable cannot ride along
/// inside an otherwise plausible binary tokenizer asset.
///
/// Kept to signatures long and distinctive enough that a false positive inside
/// a real vocabulary file is implausible.
const MAGIC_ANYWHERE: &[(&[u8], &str)] = &[
    (b"\x7fELF\x02\x01\x01", "embedded ELF executable"),
    (b"\0asm\x01\0\0\0", "embedded WebAssembly module"),
    (b"\x50\xed\x55\xba", "embedded CUDA fatbin"),
];

/// Whether the bytes cannot be treated as source text.
///
/// Anything that is not valid UTF-8, or that embeds a NUL, is opaque to the
/// banned-pattern scan and therefore may only enter the tree through the
/// narrow tokenizer-asset allowance.
#[must_use]
pub fn looks_binary(bytes: &[u8]) -> bool {
    bytes.contains(&0) || std::str::from_utf8(bytes).is_err()
}

/// Describe the executable format `bytes` appear to be, if any.
#[must_use]
pub fn executable_magic(bytes: &[u8]) -> Option<&'static str> {
    for (magic, what) in MAGIC_AT_START {
        if bytes.starts_with(magic) {
            return Some(what);
        }
    }
    for (magic, what) in MAGIC_ANYWHERE {
        if bytes.windows(magic.len()).any(|w| w == *magic) {
            return Some(what);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_not_binary() {
        assert!(!looks_binary(b"import torch\n# \xc3\xa9t\xc3\xa9\n"));
        assert!(!looks_binary(b""));
    }

    #[test]
    fn nul_and_invalid_utf8_are_binary() {
        assert!(looks_binary(b"vocab\0\x01"));
        assert!(looks_binary(&[0xff, 0xfe, 0xfd]));
    }

    #[test]
    fn detects_executables_at_offset_zero() {
        assert_eq!(executable_magic(b"\x7fELF\x02..."), Some("ELF executable"));
        assert_eq!(executable_magic(b"MZ\x90\0"), Some("DOS/PE executable"));
        assert_eq!(executable_magic(b"PK\x03\x04rest"), Some("zip archive"));
        assert!(executable_magic(b"{\"model\":{}}").is_none());
    }

    #[test]
    fn detects_executables_hidden_inside_a_blob() {
        let mut blob = vec![9u8; 4096];
        blob.extend_from_slice(b"\x7fELF\x02\x01\x01\0payload");
        assert_eq!(executable_magic(&blob), Some("embedded ELF executable"));
    }

    #[test]
    fn plausible_tokenizer_bytes_pass() {
        let mut model = b"\x0a\x1f\n\x07<unk>\x15\0\0\0\0".to_vec();
        model.extend((0..4096u32).map(|i| u8::try_from(i % 251).unwrap_or(0)));
        assert!(executable_magic(&model).is_none());
    }
}
