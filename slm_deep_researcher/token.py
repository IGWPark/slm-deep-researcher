"""Approximate token counting and truncation helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ApproximateTokenizer:
    """Lightweight tokenizer that approximates 1 token per ~4 characters."""

    chunk_size: int = 4

    def encode(self, text: str) -> list[str]:
        if not text:
            return []
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def decode(self, tokens: list[str]) -> str:
        return "".join(tokens)


_TOKENIZER = ApproximateTokenizer()


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_TOKENIZER.encode(text))


def truncate_text(text: str, max_tokens: int) -> str:
    if not text:
        return text
    tokens = _TOKENIZER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = _TOKENIZER.decode(tokens[:max_tokens])
    return truncated + "\n\n[Truncated]"


def truncate_to_fit(
    system_msg: str,
    user_msg: str,
    max_input: int,
    max_output: int,
) -> tuple[str, str]:
    available = max_input - max_output - 100
    system_tokens = count_tokens(system_msg)
    remaining = available - system_tokens

    if remaining <= 0:
        raise ValueError(f"System message too large: {system_tokens} tokens")

    user_truncated = truncate_text(user_msg, remaining)
    return system_msg, user_truncated
