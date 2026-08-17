"""Pinned shard materialization + text loading (harness-side only).

The parent downloads + SHA-256 verifies the pinned fineweb-edu parquet
before any miner code runs; the miner subprocess only reads the verified
file from disk.
"""

import hashlib
import os
import time

from . import VAL_ROWS
from .envutil import log


def dataset_dst():
    """Where the verified parquet lives (test override via PRISM_DATASET_PATH)."""
    return os.environ.get("PRISM_DATASET_PATH", "/tmp/prism_dataset.parquet")


def _download_url(url, dst):
    import urllib.error
    import urllib.request

    last_err = None
    for attempt in range(1, 6):
        try:
            log(f"downloading {url} (attempt {attempt}/5)")
            t0 = time.time()
            tmp = dst + ".part"
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 22)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(tmp, dst)
            log(f"downloaded in {time.time()-t0:.0f}s")
            return
        except urllib.error.HTTPError as exc:
            last_err = exc
            if os.path.isfile(dst + ".part"):
                try:
                    os.remove(dst + ".part")
                except OSError:
                    pass
            if exc.code in (408, 429, 500, 502, 503, 504) and attempt < 5:
                wait = min(90, 8 * (2 ** (attempt - 1)))
                log(f"download {exc.code}; retry in {wait}s")
                time.sleep(wait)
                continue
            raise
        except (TimeoutError, OSError) as exc:
            last_err = exc
            if attempt < 5:
                wait = min(90, 8 * (2 ** (attempt - 1)))
                log(f"download error ({exc}); retry in {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"dataset download failed: {last_err}")


def materialize_dataset(url, sha256_hex):
    dst = dataset_dst()
    if os.path.isfile(dst):
        log("dataset already materialized")
    else:
        try:
            _download_url(url, dst)
        except Exception as first:  # noqa: BLE001
            # HuggingFace CDN 429s from Lium egress; official mirror keeps the pin.
            mirror = url.replace("https://huggingface.co/", "https://hf-mirror.com/", 1)
            if mirror != url:
                log(f"primary download failed ({first}); trying mirror")
                _download_url(mirror, dst)
            else:
                raise
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    got = h.hexdigest()
    if got != sha256_hex:
        raise RuntimeError(f"dataset sha256 mismatch: got {got}, want {sha256_hex}")
    log("dataset sha256 ok")
    return dst


def load_texts(parquet_path):
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["text"])
    texts = table.column("text").to_pylist()
    texts = [t for t in texts if isinstance(t, str) and len(t) >= 100]
    if len(texts) < VAL_ROWS + 64:
        raise RuntimeError(f"shard too small after filter: {len(texts)}")
    return texts
