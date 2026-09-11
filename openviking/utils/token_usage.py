"""Read provider usage without masking absent counters or rejecting valid responses."""

from typing import Any

LLM_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def usage_value(raw: Any, field: str) -> Any:
    return raw.get(field) if isinstance(raw, dict) else getattr(raw, field, None)


def token_count(value: Any) -> int | None:
    """Reject missing/malformed counters, including negative and fractional values."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        count = int(value)
        if count < 0 or (not isinstance(value, str) and count != value):
            return None
        return count
    except (TypeError, ValueError, OverflowError):
        return None


def parse_llm_usage(raw: Any, *, include_details: bool = False) -> dict[str, Any]:
    # Keep absent/null/invalid primary fields detectable by the QA accumulator.
    usage = {}
    for key in LLM_FIELDS:
        value = usage_value(raw, key)
        if value is not None:
            usage[key] = value
    for output, detail, field in (
        ("cache_read_input_tokens", "prompt_tokens_details", "cached_tokens"),
        ("reasoning_tokens", "completion_tokens_details", "reasoning_tokens"),
    ):
        value = token_count(usage_value(usage_value(raw, detail), field))
        if not value:
            value = token_count(usage_value(raw, output))
        if value:
            usage[output] = value
    if include_details and raw:
        usage["prompt_tokens_details"] = usage_value(raw, "prompt_tokens_details")
    return usage
