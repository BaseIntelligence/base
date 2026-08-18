#!/usr/bin/env python3
"""Build a **public** PRISM eval-assets pack from Hugging Face sources.

Operator-side only — never commit the resulting JSONL. These are held-out
eval sets built from **public** HF datasets (FineMath, WikiText/Paloma,
codeparrot/stack-smol, FineWeb dump for fresh, official G2 val splits,
LongBench-v2 + HELMET for G5 natural). They are not secret; staging them
post-train only prevents in-process train contamination.

Default `eval_tier` for a staged pack is `public`. Optional
`PRISM_EVAL_TIER=private` / pack `tier.json` keeps the contamination-mirror
ceremony for operators who still want secret seeds.

Env:
  PRISM_EVAL_ASSETS_DIR  output root (default /tmp/prism-eval-assets)
  PACK_TIER              written to tier.json (default public)
  G1_N                   docs per G1 domain / fresh (default/max 400)
  G2_N                   max items per G2 task (default 400)
  G2_N_USABLE            discriminative G2 items (default/max 400)
  G5_FILLER_DOCS         PG-19 docs for babilong filler (default 8)
  G5_QA_N                SQuAD rows for ruler_qa (default 200)
  SKIP_G5                if 1, skip G5 assets
  SKIP_G5_NATURAL        if 1, skip LongBench/HELMET natural pack
  G5_NATURAL_SRC         optional existing g5/natural dir to copy
  MAX_PACKED_MIB         packed tar.gz cap check (default 256)
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

OUT = Path(os.environ.get("PRISM_EVAL_ASSETS_DIR", "/tmp/prism-eval-assets"))
# Hard governance cap: no generated JSONL asset may exceed 400 rows. Keeping
# this fixed prevents an operator typo from silently multiplying eval cost.
MAX_ASSET_ROWS = 400


def row_cap(name: str, default: int) -> int:
    try:
        requested = int(os.environ.get(name, str(default)))
    except ValueError:
        requested = default
    return max(1, min(MAX_ASSET_ROWS, requested))


G1_N = row_cap("G1_N", MAX_ASSET_ROWS)
G2_N = row_cap("G2_N", MAX_ASSET_ROWS)
G2_N_USABLE = row_cap("G2_N_USABLE", MAX_ASSET_ROWS)
G2_DISCRIMINATIVE = ("lambada", "hellaswag", "piqa", "arc_easy")


def g2_cap(task: str) -> int:
    """Rows to pack for one G2 task (mirrors `eval.common.eval_g2_cap`)."""
    if task in G2_DISCRIMINATIVE:
        return max(G2_N, G2_N_USABLE)
    return G2_N
G5_FILLER_DOCS = int(os.environ.get("G5_FILLER_DOCS", "8"))
G5_QA_N = row_cap("G5_QA_N", 200)
SKIP_G5 = os.environ.get("SKIP_G5", "0") == "1"
SKIP_G5_NATURAL = os.environ.get("SKIP_G5_NATURAL", "0") == "1"
G5_NATURAL_SRC = os.environ.get("G5_NATURAL_SRC", "").strip()
PACK_TIER = (os.environ.get("PACK_TIER") or "public").strip().lower()
MAX_PACKED_MIB = int(os.environ.get("MAX_PACKED_MIB", "256"))
MAX_TEXT = int(os.environ.get("MAX_TEXT_CHARS", "2048"))
MIN_TEXT = int(os.environ.get("MIN_TEXT_CHARS", "64"))
SEED = int(os.environ.get("PACK_SEED", "20260809"))
HERE = Path(__file__).resolve().parent
PUBLIC_DEV_NATURAL = HERE / "public_dev" / "g5" / "natural"

rng = random.Random(SEED)
manifest: list[dict[str, Any]] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def truncate(text: str, n: int = MAX_TEXT) -> str:
    text = text.replace("\x00", " ").strip()
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0] + "…"


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if n >= MAX_ASSET_ROWS:
                break
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def cap_jsonl_tree(root: Path) -> None:
    """Truncate copied asset pools to the same hard per-file row cap."""
    if not root.is_dir():
        return
    for path in root.rglob("*.jsonl"):
        rows = []
        with path.open(encoding="utf-8") as src:
            for line in src:
                if len(rows) >= MAX_ASSET_ROWS:
                    break
                rows.append(line)
        with path.open("w", encoding="utf-8") as dst:
            dst.writelines(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(slot: str, path: Path, **meta: Any) -> None:
    entry = {
        "slot": slot,
        "path": str(path.relative_to(OUT)),
        "rows": sum(1 for _ in path.open()) if path.exists() else 0,
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else None,
        **meta,
    }
    manifest.append(entry)
    log(f"  OK {entry['path']}: {entry['rows']} rows ({entry['bytes']} B)")


def stream_take(
    ds: Iterable[dict[str, Any]],
    n: int,
    text_key: str = "text",
    transform=None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in ds:
        try:
            if transform:
                item = transform(row)
            else:
                t = row.get(text_key) or ""
                if not isinstance(t, str) or len(t.strip()) < MIN_TEXT:
                    continue
                item = {"text": truncate(t)}
            if item is None:
                continue
            out.append(item)
        except Exception as exc:  # noqa: BLE001
            log(f"    skip row: {exc}")
            continue
        if len(out) >= n:
            break
    return out


def try_load_streaming(path: str, **kwargs):
    from datasets import load_dataset

    return load_dataset(path, streaming=True, trust_remote_code=False, **kwargs)


def build_g1_prose() -> None:
    log("== G1 prose ==")
    path = OUT / "g1/domains/prose.jsonl"
    meta: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    # Try Paloma (gated); fall back to wikitext-103-raw test.
    try:
        log("  trying allenai/paloma (may be gated)…")
        ds = try_load_streaming("allenai/paloma", split="validation")
        rows = stream_take(ds, G1_N, text_key="text")
        meta = {
            "dataset": "allenai/paloma",
            "split": "validation",
            "license": "AI2 ImpACT LR (gated)",
        }
    except Exception as exc:  # noqa: BLE001
        log(f"  paloma unavailable ({exc}); using Salesforce/wikitext")
        from datasets import load_dataset

        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-103-raw-v1",
            split="test",
            trust_remote_code=False,
        )
        rows = []
        for row in ds:
            t = (row.get("text") or "").strip()
            if len(t) < MIN_TEXT:
                continue
            rows.append({"text": truncate(t)})
            if len(rows) >= G1_N:
                break
        meta = {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-103-raw-v1",
            "split": "test",
            "license": "CC-BY-SA-3.0 + GFDL",
        }
    write_jsonl(path, rows)
    record("g1.domains.prose", path, **meta)


def build_g1_math() -> None:
    log("== G1 math ==")
    path = OUT / "g1/domains/math.jsonl"
    # Prefer finemath-3plus; fall back to open-web-math.
    last_err = None
    for name, kwargs, lic in (
        ("HuggingFaceTB/finemath", {"name": "finemath-3plus", "split": "train"}, "ODC-By"),
        ("open-web-math/open-web-math", {"split": "train"}, "ODC-By"),
    ):
        try:
            log(f"  trying {name} {kwargs}…")
            ds = try_load_streaming(name, **kwargs)
            rows = stream_take(ds, G1_N, text_key="text")
            if len(rows) < max(50, G1_N // 10):
                raise RuntimeError(f"too few rows: {len(rows)}")
            write_jsonl(path, rows)
            record(
                "g1.domains.math",
                path,
                dataset=name,
                config=kwargs.get("name"),
                split=kwargs.get("split"),
                license=lic,
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log(f"  failed: {exc}")
    raise RuntimeError(f"math sources failed: {last_err}")


def build_g1_code() -> None:
    log("== G1 code ==")
    path = OUT / "g1/domains/code.jsonl"
    last_err = None
    attempts = [
        ("bigcode/the-stack-smol", {"data_dir": "data/python", "split": "train"}, "content"),
        ("bigcode/the-stack-smol", {"split": "train"}, "content"),
        ("codeparrot/codeparrot-clean", {"split": "train"}, "content"),
    ]
    for name, kwargs, key in attempts:
        try:
            log(f"  trying {name} {kwargs}…")
            ds = try_load_streaming(name, **kwargs)

            def _xf(row, _key=key):
                t = row.get(_key) or row.get("text") or ""
                if not isinstance(t, str) or len(t.strip()) < MIN_TEXT:
                    return None
                return {"text": truncate(t)}

            rows = stream_take(ds, G1_N, transform=_xf)
            if len(rows) < max(50, G1_N // 10):
                raise RuntimeError(f"too few rows: {len(rows)}")
            write_jsonl(path, rows)
            record(
                "g1.domains.code",
                path,
                dataset=name,
                data_dir=kwargs.get("data_dir"),
                split=kwargs.get("split"),
                license="see dataset card / BigCode terms",
                text_field=key,
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log(f"  failed: {exc}")
    raise RuntimeError(f"code sources failed: {last_err}")


def build_g1_news() -> None:
    log("== G1 news (optional) ==")
    path = OUT / "g1/domains/news.jsonl"
    try:
        from datasets import load_dataset

        # Prefer small, non-streaming first for reliability.
        for name, kwargs, field, lic in (
            ("fancyzhx/ag_news", {"split": "test"}, "text", "unknown"),
            ("cc_news", {"split": "train"}, "text", "unknown"),
        ):
            try:
                log(f"  trying {name}…")
                if name == "cc_news":
                    ds = try_load_streaming(name, **kwargs)
                    rows = stream_take(ds, G1_N, text_key=field)
                else:
                    ds = load_dataset(name, trust_remote_code=False, **kwargs)
                    rows = []
                    for row in ds:
                        t = (row.get(field) or "").strip()
                        if len(t) < MIN_TEXT:
                            continue
                        rows.append({"text": truncate(t)})
                        if len(rows) >= G1_N:
                            break
                if len(rows) < 50:
                    continue
                write_jsonl(path, rows)
                record(
                    "g1.domains.news",
                    path,
                    dataset=name,
                    split=kwargs.get("split"),
                    license=lic,
                )
                return
            except Exception as exc:  # noqa: BLE001
                log(f"  failed {name}: {exc}")
        log("  WARN: news skipped")
    except Exception as exc:  # noqa: BLE001
        log(f"  WARN: news skipped ({exc})")


def build_g1_fresh() -> None:
    log("== G1 fresh (FineWeb CC-MAIN-2025-*, NOT fineweb-edu sample/10BT) ==")
    path = OUT / "g1/fresh.jsonl"
    last_err = None
    # Prefer newest dump; try a few configs.
    for cfg in ("CC-MAIN-2025-26", "CC-MAIN-2025-18", "CC-MAIN-2025-08"):
        try:
            log(f"  trying HuggingFaceFW/fineweb name={cfg}…")
            ds = try_load_streaming(
                "HuggingFaceFW/fineweb", name=cfg, split="train"
            )
            rows = stream_take(ds, G1_N, text_key="text")
            if len(rows) < max(50, G1_N // 10):
                raise RuntimeError(f"too few rows: {len(rows)}")
            write_jsonl(path, rows)
            record(
                "g1.fresh",
                path,
                dataset="HuggingFaceFW/fineweb",
                config=cfg,
                split="train",
                license="ODC-By",
                note="disjoint from HuggingFaceFW/fineweb-edu@sample/10BT train pin",
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log(f"  failed: {exc}")
    raise RuntimeError(f"fresh crawl sources failed: {last_err}")


def _gold_index(answer_key: Any, labels: list[str]) -> int | None:
    if answer_key is None:
        return None
    if isinstance(answer_key, int):
        return answer_key
    s = str(answer_key).strip()
    if s.isdigit():
        return int(s)
    try:
        return labels.index(s)
    except ValueError:
        return None


def build_g2() -> None:
    from datasets import load_dataset

    log("== G2 official validation mirrors ==")
    g2_dir = OUT / "g2"

    # --- lambada ---
    try:
        log("  lambada…")
        ds = load_dataset("EleutherAI/lambada_openai", split="test", trust_remote_code=False)
        words = []
        raw = []
        for row in ds:
            t = (row.get("text") or "").strip()
            if not t or " " not in t:
                continue
            *prefix, gold = t.rsplit(" ", 1)
            prompt = " ".join(prefix)
            if len(prompt) < 20:
                continue
            raw.append((prompt, gold))
            words.append(gold)
            if len(raw) >= g2_cap("lambada"):
                break
        rows = []
        for prompt, gold in raw:
            distractors = []
            while len(distractors) < 3:
                w = rng.choice(words)
                if w != gold and w not in distractors:
                    distractors.append(w)
            choices = [f" {gold}"] + [f" {d}" for d in distractors]
            # shuffle but track gold
            order = list(range(4))
            rng.shuffle(order)
            choices = [choices[i] for i in order]
            gold_i = order.index(0)
            rows.append({"prompt": prompt, "choices": choices, "gold": gold_i})
        p = g2_dir / "lambada.jsonl"
        write_jsonl(p, rows)
        record(
            "g2.lambada",
            p,
            dataset="EleutherAI/lambada_openai",
            split="test",
            license="MIT",
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  lambada FAILED: {exc}")

    # --- hellaswag ---
    try:
        log("  hellaswag…")
        ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=False)
        rows = []
        for row in ds:
            ctx = (row.get("ctx") or "").strip()
            endings = row.get("endings") or []
            label = row.get("label")
            if not ctx or len(endings) < 2 or label is None:
                continue
            try:
                gold = int(label)
            except (TypeError, ValueError):
                continue
            rows.append({"prompt": ctx, "choices": list(endings), "gold": gold})
            if len(rows) >= g2_cap("hellaswag"):
                break
        p = g2_dir / "hellaswag.jsonl"
        write_jsonl(p, rows)
        record(
            "g2.hellaswag",
            p,
            dataset="Rowan/hellaswag",
            split="validation",
            license="MIT (upstream)",
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  hellaswag FAILED: {exc}")

    # --- piqa (HF script dataset removed; use AI2 mosaic zip) ---
    try:
        log("  piqa…")
        import io
        import urllib.request
        import zipfile

        url = "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip"
        raw = urllib.request.urlopen(url, timeout=120).read()
        zf = zipfile.ZipFile(io.BytesIO(raw))
        labels = zf.read("physicaliqa-train-dev/dev-labels.lst").decode().splitlines()
        rows = []
        with zf.open("physicaliqa-train-dev/dev.jsonl") as fh:
            for i, line in enumerate(fh):
                if i >= g2_cap("piqa"):
                    break
                o = json.loads(line)
                rows.append(
                    {
                        "prompt": f"Question: {o['goal']}\nAnswer:",
                        "choices": [o["sol1"], o["sol2"]],
                        "gold": int(labels[i]),
                    }
                )
        p = g2_dir / "piqa.jsonl"
        write_jsonl(p, rows)
        record(
            "g2.piqa",
            p,
            dataset="ybisk/piqa (physicaliqa-train-dev.zip)",
            split="dev/validation",
            license="unknown / AI2 mosaic",
            source_url=url,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  piqa FAILED: {exc}")

    # --- ARC easy / challenge ---
    for task, cfg in (("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge")):
        try:
            log(f"  {task}…")
            ds = load_dataset(
                "allenai/ai2_arc", cfg, split="validation", trust_remote_code=False
            )
            rows = []
            for row in ds:
                q = (row.get("question") or "").strip()
                ch = row.get("choices") or {}
                texts = list(ch.get("text") or [])
                labels = list(ch.get("label") or [])
                key = row.get("answerKey")
                gold = _gold_index(key, labels)
                if not q or len(texts) < 2 or gold is None:
                    continue
                rows.append(
                    {
                        "prompt": f"Question: {q}\nAnswer:",
                        "choices": texts,
                        "gold": gold,
                    }
                )
                if len(rows) >= g2_cap(task):
                    break
            p = g2_dir / f"{task}.jsonl"
            write_jsonl(p, rows)
            record(
                f"g2.{task}",
                p,
                dataset="allenai/ai2_arc",
                config=cfg,
                split="validation",
                license="CC-BY-SA-4.0",
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  {task} FAILED: {exc}")

    # --- winogrande ---
    try:
        log("  winogrande…")
        ds = load_dataset(
            "allenai/winogrande",
            "winogrande_xl",
            split="validation",
            trust_remote_code=False,
        )
        rows = []
        for row in ds:
            sent = (row.get("sentence") or "").strip()
            o1, o2 = row.get("option1"), row.get("option2")
            ans = row.get("answer")
            if not sent or not o1 or not o2 or ans is None:
                continue
            gold = int(ans) - 1  # 1/2 → 0/1
            if gold not in (0, 1):
                continue
            rows.append(
                {"prompt": sent, "choices": [str(o1), str(o2)], "gold": gold}
            )
            if len(rows) >= g2_cap("winogrande"):
                break
        p = g2_dir / "winogrande.jsonl"
        write_jsonl(p, rows)
        record(
            "g2.winogrande",
            p,
            dataset="allenai/winogrande",
            config="winogrande_xl",
            split="validation",
            license="unspecified on card",
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  winogrande FAILED: {exc}")

    # --- boolq ---
    try:
        log("  boolq…")
        ds = load_dataset("google/boolq", split="validation", trust_remote_code=False)
        rows = []
        for row in ds:
            passage = (row.get("passage") or "").strip()
            question = (row.get("question") or "").strip()
            answer = row.get("answer")
            if not passage or not question or answer is None:
                continue
            gold = 0 if bool(answer) else 1
            rows.append(
                {
                    "prompt": f"Passage: {passage}\nQuestion: {question}?\nAnswer:",
                    "choices": [" yes", " no"],
                    "gold": gold,
                }
            )
            if len(rows) >= g2_cap("boolq"):
                break
        p = g2_dir / "boolq.jsonl"
        write_jsonl(p, rows)
        record(
            "g2.boolq",
            p,
            dataset="google/boolq",
            split="validation",
            license="CC-BY-SA-3.0",
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  boolq FAILED: {exc}")

    # --- openbookqa ---
    try:
        log("  openbookqa…")
        ds = load_dataset(
            "allenai/openbookqa", "main", split="validation", trust_remote_code=False
        )
        rows = []
        for row in ds:
            q = (row.get("question_stem") or "").strip()
            ch = row.get("choices") or {}
            texts = list(ch.get("text") or [])
            labels = list(ch.get("label") or [])
            key = row.get("answerKey")
            gold = _gold_index(key, labels)
            if not q or len(texts) < 2 or gold is None:
                continue
            rows.append(
                {
                    "prompt": f"Question: {q}\nAnswer:",
                    "choices": texts,
                    "gold": gold,
                }
            )
            if len(rows) >= g2_cap("openbookqa"):
                break
        p = g2_dir / "openbookqa.jsonl"
        write_jsonl(p, rows)
        record(
            "g2.openbookqa",
            p,
            dataset="allenai/openbookqa",
            config="main",
            split="validation",
            license="unknown",
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  openbookqa FAILED: {exc}")


def build_g5() -> None:
    if SKIP_G5:
        log("== G5 skipped (SKIP_G5=1) ==")
        return
    from datasets import load_dataset

    log("== G5 filler (PG-19) + ruler_qa (SQuAD) ==")
    # filler
    try:
        log("  pg19 filler…")
        last_err = None
        rows: list[dict[str, Any]] = []
        for name in ("emozilla/pg19-test", "pg19", "deepmind/pg19"):
            try:
                log(f"    trying {name}…")
                # pg19 is huge — prefer tiny test mirror; else stream.
                if name == "emozilla/pg19-test":
                    ds = load_dataset(name, split="test", trust_remote_code=False)
                    for row in ds:
                        t = (row.get("text") or "").strip()
                        if len(t) < 500:
                            continue
                        rows.append({"text": truncate(t, 50_000)})
                        if len(rows) >= G5_FILLER_DOCS:
                            break
                else:
                    ds = try_load_streaming(name, split="test")
                    for row in ds:
                        t = (row.get("text") or "").strip()
                        if len(t) < 500:
                            continue
                        rows.append({"text": truncate(t, 50_000)})
                        if len(rows) >= G5_FILLER_DOCS:
                            break
                if rows:
                    break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log(f"    failed: {exc}")
                rows = []
        if not rows:
            raise RuntimeError(f"pg19 failed: {last_err}")
        words = sum(len(r["text"].split()) for r in rows)
        p = OUT / "g5/babilong_filler.jsonl"
        write_jsonl(p, rows)
        mf = {
            "source": "pg19 family",
            "license": "Apache-2.0 (deepmind/pg19)",
            "n_docs": len(rows),
            "approx_words": words,
            "pinned_at": time.strftime("%Y-%m-%d"),
            "sha256": None,
        }
        write_jsonl(p, rows)  # ensure exists
        mf["sha256"] = sha256_file(p)
        (OUT / "g5/babilong_filler.manifest.json").write_text(
            json.dumps(mf, indent=2) + "\n", encoding="utf-8"
        )
        record(
            "g5.babilong_filler",
            p,
            dataset="emozilla/pg19-test or pg19",
            license="Apache-2.0",
            approx_words=words,
        )
        if words < 40_000:
            log(f"  WARN: filler words={words} < 40k target")
    except Exception as exc:  # noqa: BLE001
        log(f"  filler FAILED: {exc}")

    # ruler_qa
    try:
        log("  squad ruler_qa…")
        ds = load_dataset("rajpurkar/squad", split="validation", trust_remote_code=False)
        rows = []
        for row in ds:
            q = (row.get("question") or "").strip()
            ctx = (row.get("context") or "").strip()
            ans = row.get("answers") or {}
            texts = list(ans.get("text") or [])
            if not q or not ctx or not texts:
                continue
            rows.append({"question": q, "answers": texts, "context": truncate(ctx, 4000)})
            if len(rows) >= G5_QA_N:
                break
        p = OUT / "g5/ruler_qa.jsonl"
        write_jsonl(p, rows)
        record(
            "g5.ruler_qa",
            p,
            dataset="rajpurkar/squad",
            split="validation",
            license="CC-BY-SA-4.0",
        )
    except Exception as exc:  # noqa: BLE001
        log(f"  ruler_qa FAILED: {exc}")

    if SKIP_G5_NATURAL:
        log("  SKIP_G5_NATURAL=1 — omitting LongBench/HELMET natural pools")
        return
    try:
        import shutil

        dst = OUT / "g5/natural"
        src: Path | None = None
        note = ""
        if G5_NATURAL_SRC:
            cand = Path(G5_NATURAL_SRC)
            if cand.is_dir() and (cand / "natural_mcq.jsonl").is_file():
                src = cand
                note = f"copied from G5_NATURAL_SRC={cand}"
        if src is None:
            # Prefer a previously built operator pack under common paths.
            for cand in (
                Path("/tmp/natural-packs/g5/natural"),
                Path.home() / "prism-eval-assets" / "g5" / "natural",
            ):
                if cand.is_dir() and (cand / "natural_mcq.jsonl").is_file():
                    # Skip tiny public_dev smoke fixtures (≤8 rows).
                    n = sum(1 for _ in (cand / "natural_mcq.jsonl").open())
                    if n >= 16:
                        src = cand
                        note = f"copied existing natural pack ({n} mcq rows) from {cand}"
                        break
        if src is None:
            # Try xtask natural-pack when a repo checkout is available.
            repo = HERE.parents[3]  # .../crates/prism-recipe/harness/eval → repo root
            xtask_ok = (repo / "xtask" / "src" / "natural_pack.rs").is_file()
            if xtask_ok:
                log("  invoking cargo run -p xtask -- natural-pack …")
                import subprocess

                cache = Path(os.environ.get("PRISM_NATURAL_CACHE", "/tmp/natural-cache"))
                cmd = [
                    "cargo",
                    "run",
                    "-q",
                    "-p",
                    "xtask",
                    "--",
                    "natural-pack",
                    "--out",
                    str(OUT),
                    "--cache",
                    str(cache),
                    "--mcq-pool",
                    os.environ.get("G5_MCQ_POOL", "64"),
                    "--rag-per-cell",
                    os.environ.get("G5_RAG_PER_CELL", "12"),
                ]
                if os.environ.get("PRISM_NATURAL_OFFLINE", "0") == "1":
                    cmd.append("--offline")
                try:
                    subprocess.run(cmd, cwd=str(repo), check=True, timeout=7200)
                    if (dst / "natural_mcq.jsonl").is_file():
                        src = dst
                        note = "built via xtask natural-pack (LongBench-v2 + HELMET)"
                except Exception as exc:  # noqa: BLE001
                    log(f"  xtask natural-pack FAILED: {exc}")
        if src is None and PUBLIC_DEV_NATURAL.is_dir():
            src = PUBLIC_DEV_NATURAL
            note = "fallback: public_dev tiny natural fixtures (run xtask natural-pack for full G5)"
        if src is None:
            log("  natural pack SKIPPED — no source available")
            return
        if src.resolve() != dst.resolve():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, dirs_exist_ok=True)
        cap_jsonl_tree(dst)
        mcq = dst / "natural_mcq.jsonl"
        if mcq.is_file():
            log(f"  natural: {note}")
            record(
                "g5.natural",
                mcq,
                dataset="zai-org/LongBench-v2 + princeton-nlp/HELMET (public)",
                note=note,
            )
    except Exception as exc:  # noqa: BLE001
        log(f"  natural copy FAILED: {exc}")


def write_manifest() -> None:
    md = OUT / "MANIFEST.md"
    lines = [
        f"# PRISM {PACK_TIER} eval-assets pack (HF held-out)",
        "",
        f"- Built: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"- Out: `{OUT}`",
        f"- Pack seed: `{SEED}`",
        f"- Pack tier: `{PACK_TIER}` (default public — not secret)",
        f"- G1_N={G1_N} G2_N={G2_N} G2_N_USABLE={G2_N_USABLE}",
        "",
        "**Held-out note:** G1 fresh uses `HuggingFaceFW/fineweb` CC-MAIN-2025-* dumps, "
        "**not** `HuggingFaceFW/fineweb-edu@sample/10BT` (train pin). Benchmarks are public HF "
        "datasets; staging post-train only blocks in-process contamination.",
        "",
        "| Slot | Path | Rows | Bytes | Dataset | Split | License |",
        "|------|------|------|-------|---------|-------|---------|",
    ]
    for e in manifest:
        lines.append(
            f"| {e.get('slot')} | `{e.get('path')}` | {e.get('rows')} | {e.get('bytes')} | "
            f"{e.get('dataset', e.get('config', ''))} | {e.get('split', '')} | {e.get('license', '')} |"
        )
    lines.append("")
    lines.append("## SHA-256")
    lines.append("")
    for e in manifest:
        if e.get("sha256"):
            lines.append(f"- `{e['path']}`: `{e['sha256']}`")
    lines.append("")
    lines.append("## JSON provenance")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(manifest, indent=2))
    lines.append("```")
    lines.append("")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (OUT / "tier.json").write_text(
        json.dumps(
            {
                "tier": PACK_TIER,
                "kind": (
                    "hf_held_out_public"
                    if PACK_TIER == "public"
                    else "hf_held_out_plus_secret_seed_mirrors"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log(f"Wrote {md} + tier.json ({PACK_TIER})")


def main() -> int:
    t0 = time.time()
    if PACK_TIER not in ("public", "private"):
        log(f"ERROR: PACK_TIER must be public|private, got {PACK_TIER!r}")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    # Clean previous pack content (keep dir).
    for sub in ("g1", "g2", "g5"):
        p = OUT / sub
        if p.exists():
            import shutil

            shutil.rmtree(p)
    for f in ("MANIFEST.md", "manifest.json", "tier.json"):
        (OUT / f).unlink(missing_ok=True)

    build_g1_prose()
    build_g1_math()
    build_g1_code()
    build_g1_news()
    build_g1_fresh()
    build_g2()
    build_g5()
    write_manifest()

    # Packed size check vs recipe cap (default 256 MiB).
    import subprocess

    tar = subprocess.check_output(["tar", "-cz", "-C", str(OUT), "."])
    cap = MAX_PACKED_MIB * 1024 * 1024
    log(f"Packed tar.gz size: {len(tar)} bytes (cap {MAX_PACKED_MIB} MiB = {cap})")
    if len(tar) > cap:
        log("ERROR: pack exceeds MAX_EVAL_ASSETS_PACKED_BYTES")
        return 2
    log(f"Done in {time.time()-t0:.1f}s → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
