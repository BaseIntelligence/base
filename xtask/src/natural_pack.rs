//! Operator-side builder for the G5 natural-document eval packs.
//!
//! Fetches two community long-context sources at a **pinned revision**,
//! subsamples them with a fixed seed, and writes the private scored pools
//! plus a disjoint `public_dev` mirror into the operator eval-assets dir —
//! which `prism-lium` already stages onto the pod after the miner's train
//! phase is dead, so no separate transport is needed:
//!
//! | Slice | Source | Scoring on-pod |
//! |-------|--------|----------------|
//! | `natural_mcq` | LongBench-v2 four-way MCQ | length-normalized logprob over answer texts |
//! | `helmet_rag` | HELMET RAG (`kilt` nq/triviaqa/hotpotqa/popqa) | non-chat few-shot completion + substring EM |
//!
//! Packs store **raw text + choices + gold** only. Nothing is tokenized
//! here: the miner chooses the tokenizer, so length measurement and
//! over-length truncation happen on-pod
//! (`crates/prism-recipe/harness/eval/natural_docs.py`). The character
//! window this tool applies to LongBench-v2 contexts is deliberately
//! [`CHARS_PER_TOKEN`]-times the token budget so any plausible tokenizer
//! still finds a full budget of material inside it.
//!
//! Reproducible by construction: same pin + same seed + same counts ⇒ the
//! same `pack_hash`. `--check` rebuilds beside an existing pack set and
//! fails on drift. Datasets are never committed; only the pins live here.
//!
//! Both upstream artifacts are cached under `--cache` and verified against
//! a pinned SHA-256 before use, so a rebuild (or a `--seed` rotation) needs
//! no network. HELMET ships one 10.5 GiB archive, which dominates that
//! cache and can be deleted once the packs exist.

use serde::Serialize;
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Pack directory relative to the operator eval-assets root.
pub const PACK_REL: &str = "g5/natural";
/// Mirror (public-side) subdirectory of [`PACK_REL`].
pub const MIRROR_REL: &str = "g5/natural/public_dev";
/// Token budget the on-pod adapter enforces per slice.
pub const SLICE_TOKENS: usize = 16384;
/// Conservative chars-per-token bound used to size the stored context
/// window: even a poorly-compressing tokenizer finds [`SLICE_TOKENS`]
/// tokens inside `SLICE_TOKENS * CHARS_PER_TOKEN` characters.
pub const CHARS_PER_TOKEN: usize = 8;
/// Default pack seed. Changing it reshuffles which rows are private and
/// which are the mirror — a deliberate rotation, never incidental.
pub const DEFAULT_PACK_SEED: &str = "prism-g5-natural-v1";

const HTTP_TIMEOUT: Duration = Duration::from_mins(30);

// ------------------------------------------------------------------ pins

/// LongBench-v2 (MCQ over real long documents).
pub const LONGBENCH_REPO: &str = "zai-org/LongBench-v2";
/// Pinned dataset revision (HF commit sha).
pub const LONGBENCH_REV: &str = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9";
/// Pinned artifact within the revision.
pub const LONGBENCH_FILE: &str = "data.json";
/// SHA-256 of [`LONGBENCH_FILE`] at [`LONGBENCH_REV`].
pub const LONGBENCH_SHA256: &str =
    "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2";
/// Declared license of the LongBench-v2 dataset card.
pub const LONGBENCH_LICENSE: &str = "apache-2.0";

/// HELMET (long-context benchmark suite; we take the RAG slice only).
pub const HELMET_REPO: &str = "princeton-nlp/HELMET";
/// Pinned dataset revision (HF commit sha).
pub const HELMET_REV: &str = "dddb209d03e38f1f0faf76d6d05ef4ccf96240ee";
/// Pinned archive within the revision.
pub const HELMET_FILE: &str = "data.tar.gz";
/// SHA-256 of [`HELMET_FILE`] at [`HELMET_REV`] (its git-lfs oid).
pub const HELMET_SHA256: &str = "9d693981aa3c065b8b2ff82ddf946141cdc4ece4524f18bff6f3fbd2a86982d9";
/// Size of [`HELMET_FILE`] in bytes — 10.5 GiB of cache the operator needs
/// once, and can delete after the packs are built.
pub const HELMET_BYTES: u64 = 11_271_916_108;
/// HELMET ships no license on its dataset card; the benchmark code repo is
/// MIT and the RAG slice redistributes KILT-derived corpora (Natural
/// Questions, `TriviaQA`, `HotpotQA`, `PopQA`) under their own terms. Recorded
/// verbatim in the manifest so the operator decision is auditable.
pub const HELMET_LICENSE: &str =
    "unspecified on dataset card; benchmark code MIT; RAG slice redistributes KILT-derived corpora (NQ, TriviaQA, HotpotQA, PopQA) under their upstream terms";

/// HELMET RAG members consumed by the builder: `(archive path, sha256)`.
///
/// From `configs/rag_short.yaml` at the pinned revision — the `k50` (≈8k
/// tokens) and `k105` (≈16k tokens) test files, which are the two grid
/// points at or under the [`SLICE_TOKENS`] budget, plus the `k3` few-shot
/// demo files. Digests pin the extracted content, which is a stronger pin
/// than the archive digest and cheap to re-verify.
pub const HELMET_MEMBERS: &[(&str, &str)] = &[
    ("data/kilt/hotpotqa-dev-multikilt_1000_k105_dep3.jsonl", ""),
    ("data/kilt/hotpotqa-dev-multikilt_1000_k50_dep3.jsonl", ""),
    ("data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl", ""),
    ("data/kilt/nq-dev-multikilt_1000_k105_dep6.jsonl", ""),
    ("data/kilt/nq-dev-multikilt_1000_k50_dep6.jsonl", ""),
    ("data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl", ""),
    ("data/kilt/popqa_test_1000_k105_dep6.jsonl", ""),
    ("data/kilt/popqa_test_1000_k3_dep6.jsonl", ""),
    ("data/kilt/popqa_test_1000_k50_dep6.jsonl", ""),
    ("data/kilt/triviaqa-dev-multikilt_1000_k105_dep6.jsonl", ""),
    ("data/kilt/triviaqa-dev-multikilt_1000_k50_dep6.jsonl", ""),
    ("data/kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl", ""),
];

/// RAG corpora: `(cluster, test member stem, demo member)`.
const RAG_SOURCES: &[(&str, &str, &str)] = &[
    (
        "nq",
        "data/kilt/nq-dev-multikilt_1000_k{k}_dep6.jsonl",
        "data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl",
    ),
    (
        "triviaqa",
        "data/kilt/triviaqa-dev-multikilt_1000_k{k}_dep6.jsonl",
        "data/kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl",
    ),
    (
        "hotpotqa",
        "data/kilt/hotpotqa-dev-multikilt_1000_k{k}_dep3.jsonl",
        "data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl",
    ),
    (
        "popqa",
        "data/kilt/popqa_test_1000_k{k}_dep6.jsonl",
        "data/kilt/popqa_test_1000_k3_dep6.jsonl",
    ),
];

/// RAG length grid, in the `k` (retrieved-passage count) HELMET indexes by.
const RAG_KS: &[usize] = &[50, 105];

// ------------------------------------------------------------------ args

/// `xtask natural-pack` arguments.
#[derive(Debug, Clone)]
pub struct PackArgs {
    /// Operator eval-assets root; packs land under `<out>/g5/natural/`.
    pub out: PathBuf,
    /// Source artifact cache (downloads and extracted members).
    pub cache: PathBuf,
    /// Pack seed string.
    pub seed: String,
    /// MCQ rows per side (private and mirror each get this many).
    pub mcq_pool: usize,
    /// RAG rows per side, per `(corpus, k)` cell.
    pub rag_per_cell: usize,
    /// Few-shot demo rows per side, per corpus.
    pub demos_per_corpus: usize,
    /// LongBench-v2 `length` bands to draw from (`short` keeps truncation loss lowest).
    pub lengths: Vec<String>,
    /// Never touch the network; every artifact must already be cached.
    pub offline: bool,
    /// Rebuild beside `<out>` and fail if the pack hash drifted.
    pub check: bool,
}

impl Default for PackArgs {
    fn default() -> Self {
        Self {
            out: PathBuf::from("prism-eval-assets"),
            cache: std::env::temp_dir().join("prism-natural-src"),
            seed: DEFAULT_PACK_SEED.to_owned(),
            mcq_pool: 64,
            rag_per_cell: 12,
            demos_per_corpus: 8,
            lengths: vec!["short".to_owned()],
            offline: false,
            check: false,
        }
    }
}

// -------------------------------------------------------------- manifest

/// One pinned upstream artifact.
#[derive(Debug, Clone, Serialize)]
pub struct SourcePin {
    /// Slice this artifact feeds.
    pub slice: String,
    /// Canonical dataset ref.
    pub dataset: String,
    /// Pinned revision.
    pub revision: String,
    /// Resolve URL actually fetched.
    pub url: String,
    /// Artifact path (archive member for tar sources).
    pub artifact: String,
    /// SHA-256 of the artifact bytes.
    pub sha256: String,
    /// Artifact size in bytes.
    pub bytes: u64,
    /// License as declared upstream, recorded verbatim.
    pub license: String,
}

/// Per-pack accounting: counts, a length histogram in characters (the
/// tokenizer is the miner's, so bytes/chars are the only honest unit
/// operator-side), and the file digest.
#[derive(Debug, Clone, Serialize)]
pub struct PackStat {
    /// Row count.
    pub rows: usize,
    /// File size in bytes.
    pub bytes: u64,
    /// SHA-256 of the pack file.
    pub sha256: String,
    /// Rows per character-length bucket (bucket label → count).
    pub char_hist: BTreeMap<String, usize>,
}

/// `manifest.json` written beside the packs.
#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Tool that produced the packs.
    pub builder: String,
    /// Pack seed string in force.
    pub pack_seed: String,
    /// Token budget the on-pod adapter enforces.
    pub slice_tokens: usize,
    /// Character window applied to MCQ contexts.
    pub context_char_window: usize,
    /// Pinned upstream artifacts.
    pub sources: Vec<SourcePin>,
    /// Per-pack stats, keyed by path relative to the assets root.
    pub packs: BTreeMap<String, PackStat>,
    /// SHA-256 over every pack file in sorted-path order.
    pub pack_hash: String,
    /// True when no row id appears on both the private and the mirror side.
    pub sides_disjoint: bool,
}

// ------------------------------------------------------------------- rng

/// `SplitMix64` seeded from SHA-256 of the pack seed plus a stream label.
/// Platform-independent, so two operators building the same pin agree.
struct Rng(u64);

impl Rng {
    fn new(seed: &str, stream: &str) -> Self {
        let digest = Sha256::digest(format!("{seed}\u{0}{stream}").as_bytes());
        let mut head = [0u8; 8];
        head.copy_from_slice(&digest[..8]);
        Self(u64::from_be_bytes(head))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// In-place Fisher-Yates over a canonically ordered slice.
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = usize::try_from(self.next_u64() % (i as u64 + 1)).unwrap_or(0);
            items.swap(i, j);
        }
    }
}

// -------------------------------------------------------------- fetching

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn hf_resolve(repo: &str, revision: &str, file: &str) -> String {
    format!("https://huggingface.co/datasets/{repo}/resolve/{revision}/{file}")
}

/// Cached download: reuse the cache entry when its digest matches the pin,
/// otherwise fetch once and verify before returning the bytes.
fn fetch_pinned(url: &str, dest: &Path, want_sha: &str, offline: bool) -> Result<Vec<u8>, String> {
    if let Ok(bytes) = fs::read(dest) {
        let got = sha256_hex(&bytes);
        if got == want_sha {
            return Ok(bytes);
        }
        if offline {
            return Err(format!(
                "cached {} has sha256 {got}, pin wants {want_sha}",
                dest.display()
            ));
        }
    }
    if offline {
        return Err(format!(
            "--offline but {} is not cached (fetch {url} first)",
            dest.display()
        ));
    }
    println!("fetching {url}");
    let client = reqwest::blocking::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .map_err(|e| format!("http client: {e}"))?;
    let mut resp = client
        .get(url)
        .send()
        .map_err(|e| format!("GET {url}: {e}"))?
        .error_for_status()
        .map_err(|e| format!("GET {url}: {e}"))?;
    let mut bytes = Vec::new();
    resp.read_to_end(&mut bytes)
        .map_err(|e| format!("read {url}: {e}"))?;
    let got = sha256_hex(&bytes);
    if got != want_sha {
        return Err(format!("{url}: sha256 {got}, pin wants {want_sha}"));
    }
    write_atomic(dest, &bytes)?;
    Ok(bytes)
}

/// SHA-256 of a file read in chunks — the HELMET archive does not fit in
/// memory, so it is never slurped like the LongBench-v2 JSON is.
fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        if n == 0 {
            return Ok(hex::encode(hasher.finalize()));
        }
        hasher.update(&buf[..n]);
    }
}

/// Extract the pinned HELMET RAG members out of the upstream archive.
///
/// The archive is 10.5 GiB and a `tar.gz` is not seekable, so there is no
/// way to pull members without reading up to them. It is therefore cached
/// as a whole artifact with a **resumable** download (`curl -C -`): a
/// single-shot stream that dies at 6 GiB has to start over and corrupts the
/// tar pipe, which is exactly what a resume avoids. The archive digest is
/// pinned, so the cache is self-verifying and a second build (or a `--seed`
/// rotation) re-extracts with no network at all.
///
/// Extracted members are digest-verified separately in [`member_pin`],
/// which is a stronger pin than the archive digest and cheap to re-check.
fn ensure_helmet_members(cache: &Path, offline: bool) -> Result<PathBuf, String> {
    let root = cache.join("helmet");
    let missing: Vec<&str> = HELMET_MEMBERS
        .iter()
        .map(|(m, _)| *m)
        .filter(|m| !root.join(m).is_file())
        .collect();
    if missing.is_empty() {
        return Ok(root);
    }
    let archive = cache.join("helmet_data.tar.gz");
    ensure_helmet_archive(&archive, offline)?;
    fs::create_dir_all(&root).map_err(|e| format!("create {}: {e}", root.display()))?;
    println!(
        "extracting {} member(s) from {}",
        missing.len(),
        archive.display()
    );
    let mut tar = std::process::Command::new("tar");
    tar.arg("-xzf").arg(&archive).arg("-C").arg(&root);
    for (member, _) in HELMET_MEMBERS {
        tar.arg(member);
    }
    let status = tar
        .status()
        .map_err(|e| format!("spawn tar: {e} (is tar on PATH?)"))?;
    if !status.success() {
        return Err(format!("tar -xzf {}: {status}", archive.display()));
    }
    Ok(root)
}

/// Resumable, digest-verified download of the HELMET archive.
fn ensure_helmet_archive(archive: &Path, offline: bool) -> Result<(), String> {
    if archive.is_file() && sha256_file(archive)? == HELMET_SHA256 {
        return Ok(());
    }
    if offline {
        return Err(format!(
            "--offline but {} is absent or fails its pin",
            archive.display()
        ));
    }
    if let Some(parent) = archive.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let url = hf_resolve(HELMET_REPO, HELMET_REV, HELMET_FILE);
    println!(
        "downloading {url} ({} GiB, resumable)",
        HELMET_BYTES / (1 << 30)
    );
    // `-C -` resumes a partial file, `--retry` covers mid-transfer drops.
    let status = std::process::Command::new("curl")
        .args(["-fL", "-C", "-", "--retry", "20", "--retry-delay", "5"])
        .args(["--retry-all-errors", "--progress-bar", "-o"])
        .arg(archive)
        .arg(&url)
        .status()
        .map_err(|e| format!("spawn curl: {e} (is curl on PATH?)"))?;
    if !status.success() {
        return Err(format!("curl {url}: {status}"));
    }
    let got = sha256_file(archive)?;
    if got != HELMET_SHA256 {
        return Err(format!(
            "{}: sha256 {got}, pin wants {HELMET_SHA256}",
            archive.display()
        ));
    }
    Ok(())
}

fn write_atomic(dest: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let tmp = dest.with_extension("part");
    fs::write(&tmp, bytes).map_err(|e| format!("write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, dest).map_err(|e| format!("rename into {}: {e}", dest.display()))
}

// ----------------------------------------------------------- pack rows

/// One MCQ pool row (field order is the on-disk JSONL order).
#[derive(Debug, Clone, Serialize)]
struct McqRow {
    id: String,
    slice: &'static str,
    cluster: String,
    question: String,
    choices: Vec<String>,
    gold: usize,
    context: String,
    meta: McqMeta,
}

#[derive(Debug, Clone, Serialize)]
struct McqMeta {
    domain: String,
    sub_domain: String,
    difficulty: String,
    length: String,
    chars: usize,
    source_chars: usize,
    truncated: bool,
}

/// One retrieved passage, rendered on-pod through HELMET's own template.
#[derive(Debug, Clone, Serialize)]
struct Passage {
    title: String,
    text: String,
}

/// One RAG pool row.
#[derive(Debug, Clone, Serialize)]
struct RagRow {
    id: String,
    slice: &'static str,
    cluster: String,
    question: String,
    answers: Vec<String>,
    passages: Vec<Passage>,
    meta: RagMeta,
}

#[derive(Debug, Clone, Serialize)]
struct RagMeta {
    dataset: String,
    k: usize,
    chars: usize,
}

/// One few-shot demo row.
#[derive(Debug, Clone, Serialize)]
struct DemoRow {
    id: String,
    cluster: String,
    question: String,
    answer: String,
    passages: Vec<Passage>,
}

trait Row {
    fn row_id(&self) -> &str;
    fn row_chars(&self) -> usize;
}

impl Row for McqRow {
    fn row_id(&self) -> &str {
        &self.id
    }
    fn row_chars(&self) -> usize {
        self.meta.chars
    }
}

impl Row for RagRow {
    fn row_id(&self) -> &str {
        &self.id
    }
    fn row_chars(&self) -> usize {
        self.meta.chars
    }
}

impl Row for DemoRow {
    fn row_id(&self) -> &str {
        &self.id
    }
    fn row_chars(&self) -> usize {
        self.passages.iter().map(|p| p.text.chars().count()).sum()
    }
}

// ------------------------------------------------------------ conversion

/// Keep the head and tail of a character window, dropping the middle —
/// `LongBench`'s official over-length rule, applied here in characters so
/// the pack stays tokenizer-agnostic.
fn head_tail(text: &str, window: usize) -> (String, bool) {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= window {
        return (text.to_owned(), false);
    }
    let half = window / 2;
    let mut out = String::new();
    out.extend(&chars[..half]);
    out.extend(&chars[chars.len() - (window - half)..]);
    (out, true)
}

fn field(value: &JsonValue, key: &str) -> String {
    value
        .get(key)
        .and_then(JsonValue::as_str)
        .unwrap_or_default()
        .to_owned()
}

fn build_mcq_rows(raw: &[u8], args: &PackArgs) -> Result<Vec<McqRow>, String> {
    let items: Vec<JsonValue> =
        serde_json::from_slice(raw).map_err(|e| format!("parse LongBench-v2 data.json: {e}"))?;
    let window = SLICE_TOKENS * CHARS_PER_TOKEN;
    let mut rows = Vec::new();
    for item in items {
        let length = field(&item, "length");
        if !args.lengths.contains(&length) {
            continue;
        }
        let letters = ["A", "B", "C", "D"];
        let choices: Vec<String> = letters
            .iter()
            .map(|l| field(&item, &format!("choice_{l}")))
            .collect();
        let answer = field(&item, "answer");
        let Some(gold) = letters.iter().position(|l| *l == answer) else {
            continue;
        };
        if choices.iter().any(String::is_empty) {
            continue;
        }
        let id = format!("lbv2:{}", field(&item, "_id"));
        // Per-row choice permutation: the pack never stores gold at the
        // upstream letter position. The scored order is redrawn again
        // on-pod from the secret seed.
        let mut order: Vec<usize> = (0..choices.len()).collect();
        Rng::new(&args.seed, &format!("mcq/choices/{id}")).shuffle(&mut order);
        let source = field(&item, "context");
        let (context, truncated) = head_tail(&source, window);
        rows.push(McqRow {
            id,
            slice: "natural_mcq",
            // `domain` (6 values, smallest 4 rows in the short band), not
            // `sub_domain` (12 values, two of them singletons) — the
            // clustered bootstrap wants few well-populated clusters.
            cluster: field(&item, "domain"),
            question: field(&item, "question"),
            choices: order.iter().map(|i| choices[*i].clone()).collect(),
            gold: order
                .iter()
                .position(|i| *i == gold)
                .ok_or_else(|| "gold lost in permutation".to_owned())?,
            meta: McqMeta {
                domain: field(&item, "domain"),
                sub_domain: field(&item, "sub_domain"),
                difficulty: field(&item, "difficulty"),
                length,
                chars: context.chars().count(),
                source_chars: source.chars().count(),
                truncated,
            },
            context,
        });
    }
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(rows)
}

fn read_jsonl(path: &Path) -> Result<Vec<JsonValue>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect())
}

fn passages(item: &JsonValue) -> Vec<Passage> {
    item.get("ctxs")
        .and_then(JsonValue::as_array)
        .map(|ctxs| {
            ctxs.iter()
                .map(|c| Passage {
                    title: field(c, "title"),
                    text: field(c, "text"),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn answers(item: &JsonValue) -> Vec<String> {
    item.get("answers")
        .and_then(JsonValue::as_array)
        .map(|a| {
            a.iter()
                .filter_map(JsonValue::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn build_rag_rows(
    root: &Path,
    cluster: &str,
    member: &str,
    k: usize,
) -> Result<Vec<RagRow>, String> {
    let mut rows = Vec::new();
    for (i, item) in read_jsonl(&root.join(member))?.into_iter().enumerate() {
        let question = field(&item, "question");
        let answers = answers(&item);
        let passages = passages(&item);
        if question.is_empty() || answers.is_empty() || passages.is_empty() {
            continue;
        }
        let native = item
            .get("id")
            .and_then(JsonValue::as_str)
            .map_or_else(|| format!("{i:05}"), str::to_owned);
        let chars = passages.iter().map(|p| p.text.chars().count()).sum();
        rows.push(RagRow {
            id: format!("helmet:{cluster}:k{k}:{native}"),
            slice: "helmet_rag",
            cluster: cluster.to_owned(),
            question,
            answers,
            passages,
            meta: RagMeta {
                dataset: cluster.to_owned(),
                k,
                chars,
            },
        });
    }
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(rows)
}

fn build_demo_rows(root: &Path, cluster: &str, member: &str) -> Result<Vec<DemoRow>, String> {
    let mut rows = Vec::new();
    for (i, item) in read_jsonl(&root.join(member))?.into_iter().enumerate() {
        let question = field(&item, "question");
        let answers = answers(&item);
        let passages = passages(&item);
        if question.is_empty() || answers.is_empty() || passages.is_empty() {
            continue;
        }
        let native = item
            .get("id")
            .and_then(JsonValue::as_str)
            .map_or_else(|| format!("{i:05}"), str::to_owned);
        rows.push(DemoRow {
            id: format!("helmet-demo:{cluster}:{native}"),
            cluster: cluster.to_owned(),
            question,
            answer: answers[0].clone(),
            passages,
        });
    }
    rows.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(rows)
}

// -------------------------------------------------------------- splitting

/// Seeded split of a canonically ordered pool into two disjoint sides of
/// `per_side` rows each: private first, mirror second.
fn split_sides<T: Clone>(
    rows: &[T],
    per_side: usize,
    seed: &str,
    stream: &str,
) -> (Vec<T>, Vec<T>) {
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    Rng::new(seed, stream).shuffle(&mut idx);
    let take = per_side.min(rows.len() / 2);
    let pick = |slice: &[usize]| slice.iter().map(|i| rows[*i].clone()).collect();
    (pick(&idx[..take]), pick(&idx[take..take * 2]))
}

fn char_bucket(chars: usize) -> String {
    for edge in [1_024, 4_096, 16_384, 65_536, 262_144] {
        if chars < edge {
            return format!("lt_{edge}");
        }
    }
    "ge_262144".to_owned()
}

fn write_pack<T: Serialize + Row>(root: &Path, rel: &str, rows: &[T]) -> Result<PackStat, String> {
    let mut body = String::new();
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for row in rows {
        body.push_str(&serde_json::to_string(row).map_err(|e| format!("serialize {rel}: {e}"))?);
        body.push('\n');
        *hist.entry(char_bucket(row.row_chars())).or_default() += 1;
    }
    let path = root.join(rel);
    write_atomic(&path, body.as_bytes())?;
    Ok(PackStat {
        rows: rows.len(),
        bytes: body.len() as u64,
        sha256: sha256_hex(body.as_bytes()),
        char_hist: hist,
    })
}

fn disjoint<T: Row>(private: &[T], mirror: &[T]) -> bool {
    let left: std::collections::BTreeSet<&str> = private.iter().map(Row::row_id).collect();
    mirror.iter().all(|r| !left.contains(r.row_id()))
}

// -------------------------------------------------------------- pipeline

fn build(args: &PackArgs, out: &Path) -> Result<Manifest, String> {
    let mut sources = Vec::new();
    let mut packs = BTreeMap::new();
    let mut sides_disjoint = true;

    // --- LongBench-v2 MCQ ---
    let url = hf_resolve(LONGBENCH_REPO, LONGBENCH_REV, LONGBENCH_FILE);
    let cached = args.cache.join("longbench_v2_data.json");
    let raw = fetch_pinned(&url, &cached, LONGBENCH_SHA256, args.offline)?;
    sources.push(SourcePin {
        slice: "natural_mcq".to_owned(),
        dataset: LONGBENCH_REPO.to_owned(),
        revision: LONGBENCH_REV.to_owned(),
        url,
        artifact: LONGBENCH_FILE.to_owned(),
        sha256: LONGBENCH_SHA256.to_owned(),
        bytes: raw.len() as u64,
        license: LONGBENCH_LICENSE.to_owned(),
    });
    let mcq = build_mcq_rows(&raw, args)?;
    drop(raw);
    println!("longbench-v2: {} candidate row(s)", mcq.len());
    let (mcq_private, mcq_mirror) = split_sides(&mcq, args.mcq_pool, &args.seed, "mcq/split");
    sides_disjoint &= disjoint(&mcq_private, &mcq_mirror);
    packs.insert(
        format!("{PACK_REL}/natural_mcq.jsonl"),
        write_pack(out, &format!("{PACK_REL}/natural_mcq.jsonl"), &mcq_private)?,
    );
    packs.insert(
        format!("{MIRROR_REL}/natural_mcq.jsonl"),
        write_pack(out, &format!("{MIRROR_REL}/natural_mcq.jsonl"), &mcq_mirror)?,
    );

    // --- HELMET RAG ---
    // The RAG slice pins the extracted members, not the archive rows: the
    // archive digest is recorded on every member pin so the provenance
    // chain is archive -> member -> pack.
    let root = ensure_helmet_members(&args.cache, args.offline)?;
    let helmet_url = hf_resolve(HELMET_REPO, HELMET_REV, HELMET_FILE);
    let (mut rag_private, mut rag_mirror) = (Vec::new(), Vec::new());
    let (mut demo_private, mut demo_mirror) = (Vec::new(), Vec::new());
    for (cluster, test_tmpl, demo_member) in RAG_SOURCES {
        for k in RAG_KS {
            let member = test_tmpl.replace("{k}", &k.to_string());
            let rows = build_rag_rows(&root, cluster, &member, *k)?;
            let (a, b) = split_sides(
                &rows,
                args.rag_per_cell,
                &args.seed,
                &format!("rag/{cluster}/k{k}"),
            );
            sides_disjoint &= disjoint(&a, &b);
            rag_private.extend(a);
            rag_mirror.extend(b);
            sources.push(member_pin("helmet_rag", &helmet_url, &root, &member)?);
        }
        let demos = build_demo_rows(&root, cluster, demo_member)?;
        let (a, b) = split_sides(
            &demos,
            args.demos_per_corpus,
            &args.seed,
            &format!("rag/demos/{cluster}"),
        );
        sides_disjoint &= disjoint(&a, &b);
        demo_private.extend(a);
        demo_mirror.extend(b);
        sources.push(member_pin("helmet_rag", &helmet_url, &root, demo_member)?);
    }
    for (rel, rows) in [
        (format!("{PACK_REL}/helmet_rag.jsonl"), &rag_private),
        (format!("{MIRROR_REL}/helmet_rag.jsonl"), &rag_mirror),
    ] {
        packs.insert(rel.clone(), write_pack(out, &rel, rows)?);
    }
    for (rel, rows) in [
        (format!("{PACK_REL}/helmet_rag.demos.jsonl"), &demo_private),
        (format!("{MIRROR_REL}/helmet_rag.demos.jsonl"), &demo_mirror),
    ] {
        packs.insert(rel.clone(), write_pack(out, &rel, rows)?);
    }

    sources.sort_by(|a, b| (&a.slice, &a.artifact).cmp(&(&b.slice, &b.artifact)));
    let mut hasher = Sha256::new();
    for (rel, stat) in &packs {
        hasher.update(rel.as_bytes());
        hasher.update([0x00]);
        hasher.update(stat.sha256.as_bytes());
        hasher.update([0xff]);
    }
    Ok(Manifest {
        schema_version: 1,
        builder: "xtask natural-pack".to_owned(),
        pack_seed: args.seed.clone(),
        slice_tokens: SLICE_TOKENS,
        context_char_window: SLICE_TOKENS * CHARS_PER_TOKEN,
        sources,
        packs,
        pack_hash: hex::encode(hasher.finalize()),
        sides_disjoint,
    })
}

fn member_pin(slice: &str, url: &str, root: &Path, member: &str) -> Result<SourcePin, String> {
    let path = root.join(member);
    let bytes = fs::metadata(&path)
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len();
    let raw = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let got = sha256_hex(&raw);
    if let Some((_, want)) = HELMET_MEMBERS.iter().find(|(m, _)| *m == member) {
        if !want.is_empty() && *want != got {
            return Err(format!("{member}: sha256 {got}, pin wants {want}"));
        }
    }
    Ok(SourcePin {
        slice: slice.to_owned(),
        dataset: HELMET_REPO.to_owned(),
        revision: HELMET_REV.to_owned(),
        url: url.to_owned(),
        artifact: member.to_owned(),
        sha256: got,
        bytes,
        license: HELMET_LICENSE.to_owned(),
    })
}

/// Build (or verify) the natural packs.
///
/// # Errors
/// Network / digest / parse failures, or a pack-hash drift under `--check`.
pub fn run(workspace_root: &Path, args: &PackArgs) -> Result<(), String> {
    let out = if args.out.is_absolute() {
        args.out.clone()
    } else {
        workspace_root.join(&args.out)
    };
    let manifest_rel = format!("{PACK_REL}/manifest.json");

    if args.check {
        let previous: JsonValue = serde_json::from_str(
            &fs::read_to_string(out.join(&manifest_rel))
                .map_err(|e| format!("read {manifest_rel}: {e} (build it first)"))?,
        )
        .map_err(|e| format!("parse {manifest_rel}: {e}"))?;
        let want = field(&previous, "pack_hash");
        let scratch =
            std::env::temp_dir().join(format!("prism-natural-check-{}", std::process::id()));
        let rebuilt = build(args, &scratch);
        let _ = fs::remove_dir_all(&scratch);
        let rebuilt = rebuilt?;
        if rebuilt.pack_hash != want {
            return Err(format!(
                "pack hash drift: rebuilt {} vs recorded {want} — same pin + seed must give the same packs",
                rebuilt.pack_hash
            ));
        }
        println!("natural-pack --check: OK (pack_hash {})", rebuilt.pack_hash);
        return Ok(());
    }

    let manifest = build(args, &out)?;
    if !manifest.sides_disjoint {
        return Err(
            "private and public_dev sides overlap — mirror gap would be meaningless".into(),
        );
    }
    let mut body =
        serde_json::to_string_pretty(&manifest).map_err(|e| format!("serialize manifest: {e}"))?;
    body.push('\n');
    write_atomic(&out.join(&manifest_rel), body.as_bytes())?;
    println!(
        "wrote {} pack(s) under {}",
        manifest.packs.len(),
        out.display()
    );
    for (rel, stat) in &manifest.packs {
        println!("  {rel}: {} rows, {} bytes", stat.rows, stat.bytes);
    }
    println!("pack_hash={}", manifest.pack_hash);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_is_deterministic_and_seed_sensitive() {
        let draw = |seed: &str, stream: &str| {
            let mut rng = Rng::new(seed, stream);
            (0..8).map(|_| rng.next_u64()).collect::<Vec<_>>()
        };
        assert_eq!(draw("a", "s"), draw("a", "s"));
        assert_ne!(draw("a", "s"), draw("b", "s"));
        assert_ne!(draw("a", "s"), draw("a", "t"));
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut items: Vec<u32> = (0..64).collect();
        Rng::new(DEFAULT_PACK_SEED, "t").shuffle(&mut items);
        let mut sorted = items.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
        assert_ne!(items, sorted, "a 64-element shuffle should move something");
    }

    #[test]
    fn split_sides_is_disjoint_and_stable() {
        let rows: Vec<String> = (0..40).map(|i| format!("id{i:02}")).collect();
        let (a, b) = split_sides(&rows, 12, DEFAULT_PACK_SEED, "t");
        assert_eq!((a.len(), b.len()), (12, 12));
        assert!(a.iter().all(|x| !b.contains(x)), "sides must not overlap");
        assert_eq!(
            (a.clone(), b.clone()),
            split_sides(&rows, 12, DEFAULT_PACK_SEED, "t")
        );
        let (c, _) = split_sides(&rows, 12, "other-seed", "t");
        assert_ne!(a, c, "a different pack seed must redraw the sides");
    }

    #[test]
    fn split_sides_never_overruns_a_small_pool() {
        let rows: Vec<u8> = (0..5).collect();
        let (a, b) = split_sides(&rows, 40, DEFAULT_PACK_SEED, "t");
        assert_eq!((a.len(), b.len()), (2, 2));
    }

    #[test]
    fn head_tail_keeps_both_ends_and_flags_truncation() {
        let text: String = (0u8..100).map(|i| char::from(b'a' + i % 26)).collect();
        let (kept, truncated) = head_tail(&text, 100);
        assert!(!truncated);
        assert_eq!(kept, text);
        let (cut, truncated) = head_tail(&text, 10);
        assert!(truncated);
        assert_eq!(cut.chars().count(), 10);
        assert!(text.starts_with(&cut[..5]));
        assert!(text.ends_with(&cut[5..]));
    }

    #[test]
    fn head_tail_is_char_safe_on_multibyte_text() {
        let text = "αβγδεζηθικλμνξοπρστυφχψω".repeat(4);
        let (cut, truncated) = head_tail(&text, 8);
        assert!(truncated);
        assert_eq!(cut.chars().count(), 8);
    }

    #[test]
    fn mcq_rows_permute_choices_and_carry_provenance() {
        let raw = br#"[{"_id":"x1","domain":"Single-Document QA","sub_domain":"Academic",
          "difficulty":"easy","length":"short","question":"q?","context":"ctx",
          "choice_A":"a","choice_B":"b","choice_C":"c","choice_D":"d","answer":"C"}]"#;
        let rows = build_mcq_rows(raw, &PackArgs::default()).expect("rows");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.id, "lbv2:x1");
        assert_eq!(row.cluster, "Single-Document QA");
        assert_eq!(row.meta.sub_domain, "Academic");
        assert_eq!(
            row.choices[row.gold], "c",
            "gold must follow the permutation"
        );
        let mut sorted = row.choices.clone();
        sorted.sort();
        assert_eq!(sorted, vec!["a", "b", "c", "d"]);
        assert!(!row.meta.truncated);
        assert_eq!(row.meta.source_chars, 3);
    }

    #[test]
    fn mcq_rows_drop_other_length_bands_and_bad_answers() {
        let raw = br#"[{"_id":"long1","length":"long","question":"q","context":"c",
            "choice_A":"a","choice_B":"b","choice_C":"c","choice_D":"d","answer":"A"},
           {"_id":"bad1","length":"short","question":"q","context":"c",
            "choice_A":"a","choice_B":"b","choice_C":"c","choice_D":"d","answer":"Z"},
           {"_id":"empty1","length":"short","question":"q","context":"c",
            "choice_A":"","choice_B":"b","choice_C":"c","choice_D":"d","answer":"B"}]"#;
        assert!(build_mcq_rows(raw, &PackArgs::default())
            .expect("rows")
            .is_empty());
    }

    #[test]
    fn char_buckets_cover_the_range() {
        assert_eq!(char_bucket(0), "lt_1024");
        assert_eq!(char_bucket(5_000), "lt_16384");
        assert_eq!(char_bucket(1_000_000), "ge_262144");
    }

    #[test]
    fn helmet_members_are_sorted_unique_and_cover_every_rag_cell() {
        let paths: Vec<&str> = HELMET_MEMBERS.iter().map(|(m, _)| *m).collect();
        let mut sorted = paths.clone();
        sorted.sort_unstable();
        assert_eq!(paths, sorted, "HELMET_MEMBERS must stay sorted");
        sorted.dedup();
        assert_eq!(sorted.len(), paths.len(), "duplicate HELMET member");
        for (_, test_tmpl, demo) in RAG_SOURCES {
            assert!(paths.contains(demo), "unpinned demo member {demo}");
            for k in RAG_KS {
                let member = test_tmpl.replace("{k}", &k.to_string());
                assert!(
                    paths.contains(&member.as_str()),
                    "unpinned test member {member}"
                );
            }
        }
    }

    #[test]
    fn sha256_file_matches_the_in_memory_digest() {
        let path = std::env::temp_dir().join(format!("prism-nat-sha-{}", std::process::id()));
        let body = b"pinned bytes".repeat(4096);
        fs::write(&path, &body).expect("write");
        let got = sha256_file(&path);
        let _ = fs::remove_file(&path);
        assert_eq!(got.expect("digest"), sha256_hex(&body));
    }

    #[test]
    fn helmet_archive_pin_is_hex_and_distinct_from_every_member_pin() {
        assert_eq!(HELMET_SHA256.len(), 64);
        assert!(HELMET_SHA256.chars().all(|c| c.is_ascii_hexdigit()));
        for (member, sha) in HELMET_MEMBERS {
            assert!(sha.is_empty() || sha.len() == 64, "{member}: bad pin {sha}");
            assert_ne!(*sha, HELMET_SHA256, "{member} pinned to the archive digest");
        }
    }
}
