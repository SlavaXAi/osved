"""Cache identity for detailed articles, independent of model/schema imports.

An evidence-policy change invalidates reuse; it never re-stamps historical
articles as if they had passed the new checks.
"""

import hashlib
import json

from app.research.policy import (
    FACT_EXTRACTION_PROMPT_VERSION,
    FACT_VALIDATION_POLICY_VERSION,
)

DETAIL_SCHEMA_VERSION = "detail_article_v1"
DETAIL_PROMPT_VERSION = (
    f"detail_article_ru_v5:{FACT_EXTRACTION_PROMPT_VERSION}:{FACT_VALIDATION_POLICY_VERSION}"
)
NEWS_READER_SCHEMA_VERSION = "news_reader_v1"
NEWS_READER_PROMPT_VERSION = (
    f"news_reader_ru_v2:{FACT_EXTRACTION_PROMPT_VERSION}:{FACT_VALIDATION_POLICY_VERSION}"
)
NEWS_READER_ARTICLE_PROMPT_VERSION = (
    f"news_reader_ru_v3:{FACT_EXTRACTION_PROMPT_VERSION}:{FACT_VALIDATION_POLICY_VERSION}"
)


def article_reference_fingerprint(reader_context: dict) -> str:
    """Identity of the previously shown text; operational retry state is excluded."""
    text = str(reader_context.get("expanded_reference_text") or "")
    material = (
        {
            "mode": "expanded",
            "article_id": reader_context.get("expanded_reference_article_id"),
            "text": text,
            "units": len(text.encode("utf-16-le")) // 2,
        }
        if text
        else {"mode": "baseline", "baseline": reader_context.get("context_fingerprint")}
    )
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def detail_policy_versions(depth: str = "legacy") -> tuple[str, str]:
    if depth == "legacy":
        return DETAIL_SCHEMA_VERSION, DETAIL_PROMPT_VERSION
    if depth == "expanded":
        return NEWS_READER_SCHEMA_VERSION, NEWS_READER_PROMPT_VERSION
    if depth == "article":
        return NEWS_READER_SCHEMA_VERSION, NEWS_READER_ARTICLE_PROMPT_VERSION
    raise ValueError(f"Unknown news reader depth: {depth}")


def current_detail_policy_sql(alias: str = "detailed_articles", *, depth: str = "legacy") -> str:
    # Callers supply only internal SQL identifiers, never external text.
    schema_version, prompt_version = detail_policy_versions(depth)
    return (
        f"{alias}.schema_version = '{schema_version}' AND "
        f"{alias}.prompt_version = '{prompt_version}'"
    )
