# Vendored third-party eval protocols

Two community long-context protocols are vendored into the harness as pure
stdlib Python. The eval subprocess has **no network and no PyPI**, so nothing
here may import `numpy`, `nltk`, `wonderwords`, `datasets` or `torch`, and
nothing may read a file that `HARNESS_FILES` does not ship or that the
operator did not stage under the private eval-assets directory.

Machine-readable provenance lives in `UPSTREAM` at the top of each vendored
module (`eval/vendor_ruler.py`, `eval/vendor_babilong.py`) so a pod run can
report exactly what it scored. This file is the narrative version.

## RULER — `eval/vendor_ruler.py`, adapter `eval/g5_ruler.py`

| | |
|---|---|
| Upstream | <https://github.com/NVIDIA/RULER> |
| Commit | `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a` (2026-07-22) |
| License | Apache-2.0 |
| Paper | *RULER: What's the Real Context Size of Your Long-Context Language Models?* — arXiv:2404.06654 |
| Files drawn from | `scripts/data/synthetic/constants.py`, `scripts/data/synthetic/niah.py`, `scripts/data/synthetic/variable_tracking.py`, `scripts/data/synthetic/qa.py`, `scripts/eval/synthetic/constants.py`, `scripts/synthetic.yaml` |

Verbatim (checked against the pinned commit, value-for-value):

- the `TASKS` table for `niah` / `variable_tracking` / `qa` — `template`,
  `answer_prefix` and `tokens_to_generate`;
- the needle sentence `One of the special magic {type_needle_v} for {key} is:
  {value}.`;
- the `noise` haystack sentence (`The grass is green. …`);
- the `Document {i}:\n{document}` QA prompt;
- the `num_needle_k = max(num_needle_k, num_needle_q)` clamp and the
  `num_needle_q * num_needle_v == 1` singular template rewrite;
- the needle / chain splice into the haystack (`sorted(sample(range(n), m))`,
  newline join) and the query/answer join (`", ".join(q[:-1]) + ", and " +
  q[-1]`);
- the variable-assignment chain grammar, including the trailing space in
  `VAR {b} = VAR {a} `;
- `string_match_all` / `string_match_part` from the eval-side constants —
  used by `tests/smoke_protocols.py` to prove the generated golds score 100
  under upstream's own metric.

Subsets follow upstream `scripts/synthetic.yaml`: `niah_multikey` (4 keys),
`niah_multivalue` (4 values), `niah_multiquery` (4 queries), `vt`
(`num_hops: 4`) and `qa`.

Deviations, all forced by the harness contract:

1. **Seeding.** `random.Random(seed)` per call replaces upstream's global
   `random.seed` + `np.random`, so a generator call is a pure function of its
   `eval.common.task_seed` lattice position and instances regenerate per run
   from the private secret seed.
2. **Haystack type.** Upstream's `essay` haystack (`json/PaulGrahamEssays.json`,
   ~1 MB of public text) is *not* vendored — it would bloat the payload and it
   is public. Every probe uses an upstream haystack type that needs no
   corpus: `needle` (as in upstream `niah_multikey_2/3`) or `noise` (as in
   upstream `vt`). Consequently the essay-only pieces of upstream are absent:
   the `DEPTHS` insertion lattice, `sent_tokenize`, and the
   `". \n" -> ".\n"` VT normalization (a verified no-op for the `noise` and
   `needle` haystacks, whose sentences never end in `". "`).
3. **Key vocabulary.** Upstream draws `adjective-noun` keys from
   `wonderwords`; keys here come from the harness's seeded synthetic lexicon
   (`eval.generators.lexicon`).
4. **Length fitting.** Upstream binary-searches a haystack size against a
   fixed tokenizer, then truncates. Here the generator emits the spliced
   haystack plus a `filler()` of the same sentence grammar, and `eval.toklen`
   (`common.fit_to_tokens`) grows or trims **filler only** until the whole
   prompt is exactly the target in tokens of the *submitted* tokenizer. A
   needle therefore cannot be truncated away, and the achieved length is
   measured and reported (`g5.ruler.L*.len_err`) rather than assumed.
5. **Scoring.** Upstream generates free text and applies `string_match_*`.
   Base LMs are scored by log-probability over in-distribution candidate
   answers instead, so each instance also carries `slots` (per-reference
   candidate sets). The prompt the model sees is unchanged.
6. **VT chains.** Upstream `synthetic.yaml` pins `num_chains: 1`; a second
   chain is generated here so the non-queried chain supplies in-distribution
   negatives. With one chain every variable in the text is a gold answer and
   any choice-based scorer is trivially perfect.
7. **QA documents.** Upstream reads SQuAD / HotpotQA. The adapter reads an
   operator-staged private pack (`g5/ruler_qa.jsonl`, upstream `read_squad`
   row shape) and falls back to a seeded synthetic pool, reporting which one
   it scored via `g5.ruler.qa.pack`.

## BABILong — `eval/vendor_babilong.py`, adapter `eval/g5_babilong.py`

| | |
|---|---|
| Upstream | <https://github.com/booydar/babilong> |
| Commit | `7a6efee29f5cac03c3c410e6799c80fd2ffe3610` (2026-06-01) |
| License | Apache-2.0 |
| Paper | *BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack* — arXiv:2406.10149 |
| Files drawn from | `babilong/babilong_utils.py`, `babilong/prompts.py` |
| Task grammar | bAbI tasks v1.2 `qa1`–`qa5` (<https://github.com/facebookarchive/bAbI-tasks>, BSD-3-Clause, arXiv:1502.05698) |

Faithful to upstream:

- `SentenceSampler(shuffle=True)`: random document, random start offset
  inside it, sentence-length filter, resample until the quota is met;
- `USER_TEMPLATE` (`<context>…</context>\n\nQuestion: …`);
- `compare_answers` (lower-case, first sentence, cut at `<context>` /
  `<example>`, substring test);
- the bAbI v1.2 vocabularies and phrasings: actors, locations, objects,
  movement / take / drop / transfer verb sets, the question forms and the
  answer keys of `qa1`–`qa5`.

Deviations, all forced by the harness contract:

1. **Stories are generated, not replayed.** Upstream reads
   `tasks_1-20_v1-2/en-10k`. A fixed public instance set is memorizable
   across runs, which defeats the private-seed tier, so the qa1–qa5 grammar
   is reimplemented and seeded from the lattice. Uniqueness guards: one
   transfer per object in qa5, collision-free relation pairs in qa4, and the
   `qa3` "before" question only fires when the object's room history has two
   distinct entries. `tests/smoke_protocols.py` replays every story with an
   independent world model and asserts the gold.
2. **Filler corpus.** Upstream uses public PG-19. Filler here comes from the
   operator's **private pinned** pack (`g5/babilong_filler.jsonl` +
   `g5/babilong_filler.manifest.json`); with no pack staged a seeded
   synthetic prose pool is used and the adapter reports which source it
   scored via `g5.babilong.filler_pack`.
3. **No `numpy` / `nltk`.** `random.Random(seed)` replaces
   `np.random.default_rng`; a regex sentence splitter replaces
   `nltk.PunktSentenceTokenizer` (no NLTK data offline).
4. **Sampler units and fact placement.** Upstream's sentence-length filter is
   in tokens; here it is in words so the vendored code stays
   tokenizer-agnostic, and the adapter owns every token budget through
   `eval.toklen`. Fact placement moves with it: upstream
   `NoiseInjectionDataset` draws fact positions uniformly over the gaps of a
   fixed background list, whereas the shared fitter splices background
   sentences at uniformly random positions between the facts until the length
   target is met. Both realize the protocol's rule — a fact may sit anywhere
   in the context and story order is preserved (asserted in
   `tests/smoke_protocols.py`) — but the fitter's version additionally
   guarantees that no fact is ever truncated to hit the target, and it
   reports the achieved depth (`g5.babilong.mean_depth`) and length
   (`g5.babilong.L*.len_err`).
5. **No instruction prompting.** Upstream `DEFAULT_PROMPTS` (instruction +
   post-prompt + chat template) is deliberately *not* vendored: this battery
   scores pretrained base models. The adapter builds a few-shot completion
   from upstream's own `<context>`/`Question:` framing plus the bare
   `Answer:` prefix that upstream's `<example>` blocks use, and scores the
   closed answer vocabulary by log-probability instead of generating and
   applying `compare_answers`.

## Operator-side private assets

Both adapters run without them (seeded synthetic fallback, reported in the
metrics) but the intended production configuration stages:

| Asset | Consumer | Shape |
|---|---|---|
| `g5/babilong_filler.jsonl` | `g5_babilong` | one `{"text": "..."}` object per line; natural prose, ≥ ~40 k words total |
| `g5/babilong_filler.manifest.json` | provenance | `{source, license, sha256, n_docs, pinned_at}` |
| `g5/ruler_qa.jsonl` | `g5_ruler` (`qa`) | one `{"question": str, "answers": [str], "context": str}` object per line (upstream `read_squad` shape) |

Paths are resolved with `eval.common.assets_path`, i.e. relative to the
private eval-assets directory that `prism-lium` stages after training.
