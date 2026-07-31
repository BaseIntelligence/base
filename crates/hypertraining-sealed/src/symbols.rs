//! Sealed-symbol body extraction and simplified AST hashing.

use std::collections::BTreeMap;

use crate::error::AdmitError;
use crate::hash::sha256_hex;
use crate::paths::normalize_path;

/// Default sealed-symbol keys (`path:symbol`) from brief §6.4.
pub const DEFAULT_SEALED_SYMBOL_KEYS: &[&str] = &[
    "megatron/training/training.py:consumed_train_samples",
    "megatron/training/training.py:train_iters_loop",
    "megatron/training/training.py:num_floating_point_operations",
    "megatron/training/training.py:update_num_microbatches",
];

/// Marker substrings used when a seal is not a full `def` (pattern seals).
///
/// Keys are the symbol half of `path:symbol`. Values are unique source markers;
/// the normalized line containing the marker becomes the sealed body.
pub static SEALED_SYMBOL_MARKERS: &[(&str, &str)] = &[
    ("consumed_train_samples", "consumed_train_samples"),
    ("train_iters_loop", "while iteration < args.train_iters"),
];

/// Compute the simplified AST hash for `path:symbol` given file UTF-8 source.
///
/// # Errors
///
/// [`AdmitError::InvalidSymbolKey`], [`AdmitError::SealedSymbolNotFound`].
pub fn sealed_symbol_ast_hash(key: &str, source: &str) -> Result<String, AdmitError> {
    let body = extract_sealed_body(key, source)?;
    Ok(sha256_hex(body.as_bytes()))
}

/// Extract and normalize the sealed body for `path:symbol`.
///
/// # Errors
///
/// See [`sealed_symbol_ast_hash`].
pub fn extract_sealed_body(key: &str, source: &str) -> Result<String, AdmitError> {
    let (_path, symbol) = split_symbol_key(key)?;
    if let Some(body) = extract_function_body(source, &symbol) {
        return Ok(normalize_source(&body));
    }
    if let Some(marker) = marker_for_symbol(&symbol) {
        if let Some(line) = find_marker_line(source, marker) {
            return Ok(normalize_source(&line));
        }
    }
    Err(AdmitError::SealedSymbolNotFound {
        key: key.to_owned(),
    })
}

/// Split `path:symbol` on the last `:` so Windows drive letters are not an issue
/// (we only use repo-relative POSIX paths).
pub fn split_symbol_key(key: &str) -> Result<(String, String), AdmitError> {
    let Some((path, symbol)) = key.rsplit_once(':') else {
        return Err(AdmitError::InvalidSymbolKey {
            key: key.to_owned(),
        });
    };
    if path.is_empty() || symbol.is_empty() {
        return Err(AdmitError::InvalidSymbolKey {
            key: key.to_owned(),
        });
    }
    Ok((normalize_path(path), symbol.to_owned()))
}

fn marker_for_symbol(symbol: &str) -> Option<&'static str> {
    SEALED_SYMBOL_MARKERS
        .iter()
        .find(|(s, _)| *s == symbol)
        .map(|(_, m)| *m)
}

fn find_marker_line(source: &str, marker: &str) -> Option<String> {
    source
        .lines()
        .find(|l| l.contains(marker))
        .map(std::string::ToString::to_string)
}

/// Extract `def name(...):` body including the def line through the last body line.
fn extract_function_body(source: &str, name: &str) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    let def_prefix_paren = format!("def {name}(");
    let def_prefix_space = format!("def {name} ");
    let mut start = None;
    let mut def_indent = 0usize;
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with(&def_prefix_paren) || trimmed.starts_with(&def_prefix_space) {
            def_indent = line.len() - trimmed.len();
            start = Some(i);
            break;
        }
    }
    let start = start?;
    let mut end = lines.len();
    for (j, line) in lines.iter().enumerate().skip(start + 1) {
        if line.trim().is_empty() {
            continue;
        }
        let trimmed = line.trim_start();
        let indent = line.len() - trimmed.len();
        if indent <= def_indent && (trimmed.starts_with("def ") || trimmed.starts_with("class ")) {
            end = j;
            break;
        }
    }
    Some(lines[start..end].join("\n"))
}

/// Normalize source for hashing: drop full-line comments, trim, collapse ws.
#[must_use]
pub fn normalize_source(src: &str) -> String {
    let mut out_lines = Vec::new();
    for line in src.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        // Strip trailing inline comment only when preceded by space (naive).
        let without_inline = strip_inline_comment(t);
        let collapsed = collapse_ws(without_inline.trim());
        if !collapsed.is_empty() {
            out_lines.push(collapsed);
        }
    }
    out_lines.join("\n")
}

fn strip_inline_comment(line: &str) -> &str {
    // Keep `#` inside strings out of scope for this simplified hasher.
    if let Some(idx) = line.find('#') {
        if idx > 0 && line.as_bytes()[idx - 1].is_ascii_whitespace() {
            return &line[..idx];
        }
    }
    line
}

fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out
}

/// Build a map of default sealed keys → ast hashes from a single training.py source.
///
/// # Errors
///
/// Propagates extraction failures.
pub fn hash_default_symbols(training_py: &str) -> Result<BTreeMap<String, String>, AdmitError> {
    let mut map = BTreeMap::new();
    for key in DEFAULT_SEALED_SYMBOL_KEYS {
        let h = sealed_symbol_ast_hash(key, training_py)?;
        map.insert((*key).to_owned(), h);
    }
    Ok(map)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const SAMPLE: &str = r"
def num_floating_point_operations(args, batch_size):
    # comment
    return batch_size * 2

def other():
    pass
";

    #[test]
    fn extracts_function_and_is_stable() {
        let a = sealed_symbol_ast_hash(
            "megatron/training/training.py:num_floating_point_operations",
            SAMPLE,
        )
        .unwrap();
        let b = sealed_symbol_ast_hash(
            "megatron/training/training.py:num_floating_point_operations",
            SAMPLE,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn body_edit_changes_hash() {
        let edited = SAMPLE.replace("batch_size * 2", "batch_size * 3");
        let a = sealed_symbol_ast_hash(
            "megatron/training/training.py:num_floating_point_operations",
            SAMPLE,
        )
        .unwrap();
        let b = sealed_symbol_ast_hash(
            "megatron/training/training.py:num_floating_point_operations",
            &edited,
        )
        .unwrap();
        assert_ne!(a, b);
    }
}
