#!/usr/bin/env python
"""CLI entrypoint to stage the offline reference tokenizers at a fixed path.

Network is required (this is a one-time prep/build step that runs OUTSIDE the eval sandbox).
It bakes the gpt2 tiktoken BPE cache and the non-gated llama sentencepiece ``.model`` so they
load with NO network inside the eval container. Example:

    python scripts/stage_reference_tokenizers.py --output-dir /data/reference-tokenizers
"""

from __future__ import annotations

import sys

from prism_challenge.evaluator.reference_tokenizers import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
