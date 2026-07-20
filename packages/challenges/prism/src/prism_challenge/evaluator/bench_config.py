from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchConfig:
    train_steps: int = 16
    eval_steps: int = 8
    seeds: tuple[int, ...] = (13, 37)
    sequence_lengths: tuple[int, ...] = (128, 256, 512)
    batch_size: int = 2
    vocab_size: int = 4096
    max_tokens: int = 8192
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0


DEFAULT_BENCH_CONFIG = BenchConfig()
