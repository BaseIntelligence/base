from __future__ import annotations

from hashlib import blake2b

import torch


class HashTokenizer:
    def __init__(self, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        self.vocab_size = vocab_size

    def encode(self, text: str, length: int) -> list[int]:
        raw = text.encode("utf-8", errors="ignore")
        tokens: list[int] = []
        for index, byte in enumerate(raw):
            digest = blake2b(bytes([byte]) + index.to_bytes(4, "little"), digest_size=2)
            tokens.append(int.from_bytes(digest.digest(), "little") % self.vocab_size)
            if len(tokens) >= length:
                break
        if not tokens:
            tokens = [0]
        while len(tokens) < length:
            tokens.append((tokens[-1] + 17) % self.vocab_size)
        return tokens[:length]

    def batch(self, texts: list[str], length: int, device: torch.device) -> torch.Tensor:
        encoded = [self.encode(text, length) for text in texts]
        return torch.tensor(encoded, dtype=torch.long, device=device)
