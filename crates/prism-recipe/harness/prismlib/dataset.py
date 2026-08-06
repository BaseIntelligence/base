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


def materialize_dataset(url, sha256_hex):
    import urllib.request

    dst = dataset_dst()
    if os.path.isfile(dst):
        log("dataset already materialized")
    else:
        log(f"downloading {url}")
        t0 = time.time()
        with urllib.request.urlopen(url, timeout=600) as r, open(dst, "wb") as f:
            while True:
                chunk = r.read(1 << 22)
                if not chunk:
                    break
                f.write(chunk)
        log(f"downloaded in {time.time()-t0:.0f}s")
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
