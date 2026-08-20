"""Tokenizer anti-cheat card: facts + soft flags for the agentic review.

A miner may bring any tokenizer (files or build_tokenizer hook); the card
ships in METRICS_JSON["tokenizer"]["card"] so the master-side LLM can tell
an honestly weak tokenizer (fine) from one engineered to game metrics
(cheat: answer-phrase tokens, memorizing compression, rewrite-y decode).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prismlib import tokenizer as tok_contract  # noqa: E402


class ByteTok:
    """Honest minimal byte-level tokenizer (weak but legitimate)."""

    vocab_size = 256

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return {"input_ids": list(text.encode("utf-8"))}

    def decode(self, ids):
        return bytes(int(i) & 0xFF for i in ids).decode("utf-8", "ignore")


class CheatTok:
    """Metric-gaming tokenizer: memorizes whole texts as single tokens and
    stuffs the vocab with multi-word answer phrases."""

    vocab_size = 512

    def __init__(self):
        self._memo = {}

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        self._memo[7] = text
        return {"input_ids": [7]}

    def decode(self, ids):
        if list(ids) == [7] and 7 in self._memo:
            return self._memo[7]
        return "the answer is Paris"  # single "token" = an answer phrase

    def convert_ids_to_tokens(self, ids):
        return ["the answer is Paris" for _ in ids]


def main():
    # Honest byte tokenizer: high tokens/byte, faithful roundtrip, single-
    # char vocab entries — zero flags. Weak is not cheat.
    c = tok_contract.card(ByteTok(), 256)
    assert c["flags"] == [], c
    assert c["probe_roundtrip_ok"] is True, c
    assert c["probe_tokens_per_byte"] >= 0.9, c
    assert c["vocab_multiword_frac"] == 0.0, c

    # Gaming tokenizer: whole-probe memorization (extreme compression) and
    # multi-word answer tokens — both flagged for the LLM reviewer.
    c = tok_contract.card(CheatTok(), 512)
    assert "extreme_compression" in c["flags"], c
    assert "multiword_tokens" in c["flags"], c
    assert c["probe_tokens_per_byte"] < tok_contract.CARD_MIN_TOKENS_PER_BYTE, c

    # validate() embeds the card in the cross-phase spec (METRICS_JSON path)
    # without touching the checked keys (source/id/vocab_size/fingerprint).
    spec = tok_contract.validate(ByteTok(), "hook")
    assert spec["card"]["flags"] == [], spec
    assert "card" not in tok_contract.CHECKED_SPEC_KEYS
    tok_contract.assert_matches(spec, {k: spec[k] for k in tok_contract.CHECKED_SPEC_KEYS})

    print("tokenizer card OK")


if __name__ == "__main__":
    main()
