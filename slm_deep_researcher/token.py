"""Quick token counting and text truncation."""


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(text) // 4


def truncate_text(text: str, max_tokens: int) -> str:
    if not text:
        return text

    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    return text[:max_chars] + "\n\n[Truncated]"


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
