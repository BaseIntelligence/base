from __future__ import annotations


def needle_text(length: int, key: int = 7) -> str:
    filler = " ".join(f"fact{i}=neutral" for i in range(max(2, length // 16)))
    return f"{filler} needle_key={key} answer={key} {filler}"


def copy_text(length: int) -> str:
    pattern = "A B C D " * max(2, length // 16)
    return f"copy task: {pattern} repeat: {pattern}"


def parentheses_text(length: int) -> str:
    unit = "( [ { } ] ) "
    return (unit * max(2, length // len(unit)))[: length * 2]


def modular_addition_text(limit: int = 64) -> str:
    return " ".join(f"{i}+{(i * 3) % 10}={(i + (i * 3) % 10) % 10}" for i in range(limit))


def pattern_text(length: int) -> str:
    return ("red blue green red blue green " * max(2, length // 16))[: length * 2]


def reasoning_corpus() -> list[str]:
    return [parentheses_text(128), modular_addition_text(), pattern_text(128)]
