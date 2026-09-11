"""Accumulate known generation usage without letting missing metadata stop a QA."""

from typing import Any

from loguru import logger

from openviking.utils.token_usage import LLM_FIELDS, token_count, usage_value


def record_usage_issue(
    usage: dict[str, Any],
    kind: str,
    fields: list[str],
    *,
    iteration: int | None = None,
    reason: str = "missing_or_invalid",
) -> None:
    flag = "llm_usage_complete" if kind == "llm" else "retrieval_embedding_usage_complete"
    usage[flag] = False
    issue = {"kind": kind, "fields": fields, "reason": reason}
    if iteration is not None:
        issue["iteration"] = iteration
    usage.setdefault("usage_issues", []).append(issue)
    logger.warning("Incomplete QA token usage: {}; missing values count as 0", issue)


def record_llm_usage(usage: dict[str, Any], raw: Any, *, iteration: int) -> None:
    counts = {key: token_count(usage_value(raw, key)) for key in LLM_FIELDS}
    missing = [key for key, value in counts.items() if value is None]
    if missing:
        record_usage_issue(usage, "llm", missing, iteration=iteration)
    # A missing total can still include the known input/output; unknown parts are 0.
    if counts["total_tokens"] is None:
        counts["total_tokens"] = (counts["prompt_tokens"] or 0) + (counts["completion_tokens"] or 0)
    for key, value in counts.items():
        usage[key] = usage.get(key, 0) + (value or 0)


def record_embedding_usage(usage: dict[str, Any], telemetry: Any) -> None:
    value = telemetry
    for field in ("summary", "tokens", "embedding"):
        value = usage_value(value, field)
    count = token_count(usage_value(value, "total"))
    if count is None:
        record_usage_issue(usage, "embedding", ["summary.tokens.embedding.total"])
    else:
        usage["retrieval_embedding_tokens"] = usage.get("retrieval_embedding_tokens", 0) + count
        if usage_value(value, "usage_missing"):
            record_usage_issue(
                usage, "embedding", ["provider.usage"],
                reason="embedding_provider_usage_incomplete",
            )
