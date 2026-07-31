//! L0 — source normalization before any comparison (brief §12.3).

/// Strip full-line and trailing `#` comments, collapse whitespace, trim lines.
///
/// Does not alpha-rename; see [`normalize_with_alpha_rename`].
#[must_use]
pub fn normalize_source(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        let without_comment = strip_line_comment(line);
        let collapsed = collapse_ws(without_comment.trim());
        if collapsed.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(&collapsed);
    }
    out
}

/// L0 + simple alpha-rename of identifier-like tokens to positional names `v0`, `v1`, …
///
/// Keywords and pure-digit tokens are left unchanged. First-seen order defines indices.
#[must_use]
pub fn normalize_with_alpha_rename(src: &str) -> String {
    let base = normalize_source(src);
    alpha_rename(&base)
}

fn strip_line_comment(line: &str) -> &str {
    // Python / shell style: `#` starts a comment when not inside a simple string.
    // Fixture sources are training-code style; we only strip unquoted `#`.
    let mut in_single = false;
    let mut in_double = false;
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b'\'' if !in_double => in_single = !in_single,
            b'"' if !in_single => in_double = !in_double,
            b'#' if !in_single && !in_double => {
                return &line[..i];
            }
            _ => {}
        }
        i += 1;
    }
    line
}

fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_ws = false;
    for ch in s.chars() {
        if ch.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(ch);
            prev_ws = false;
        }
    }
    out
}

const KEYWORDS: &[&str] = &[
    "def", "class", "return", "if", "else", "elif", "for", "while", "import", "from", "as", "with",
    "try", "except", "finally", "raise", "yield", "lambda", "pass", "break", "continue", "and",
    "or", "not", "in", "is", "None", "True", "False", "global", "nonlocal", "assert", "async",
    "await", "del",
];

fn is_keyword(tok: &str) -> bool {
    KEYWORDS.contains(&tok)
}

fn alpha_rename(src: &str) -> String {
    use std::collections::BTreeMap;
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    let mut next = 0_u32;
    let mut out = String::with_capacity(src.len());
    let mut token = String::new();
    let flush =
        |tok: &mut String, map: &mut BTreeMap<String, String>, next: &mut u32, out: &mut String| {
            if tok.is_empty() {
                return;
            }
            if tok.chars().all(|c| c.is_ascii_digit()) || is_keyword(tok) {
                out.push_str(tok);
            } else if let Some(name) = map.get(tok.as_str()) {
                out.push_str(name);
            } else {
                let name = format!("v{next}");
                *next = next.saturating_add(1);
                map.insert(tok.clone(), name.clone());
                out.push_str(&name);
            }
            tok.clear();
        };

    for ch in src.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            token.push(ch);
        } else {
            flush(&mut token, &mut map, &mut next, &mut out);
            out.push(ch);
        }
    }
    flush(&mut token, &mut map, &mut next, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_comments_and_whitespace() {
        let src = "def foo():  # hi\n    x = 1\n\n    y = 2  \n";
        let n = normalize_source(src);
        assert_eq!(n, "def foo():\nx = 1\ny = 2");
    }

    #[test]
    fn alpha_rename_is_stable_for_rename_only() {
        let a = normalize_with_alpha_rename("def foo(x):\n    return x + 1\n");
        let b = normalize_with_alpha_rename("def bar(y):\n    return y + 1\n");
        assert_eq!(a, b);
    }
}
