"""Cache identity for detailed articles, independent of model/schema imports.

An evidence-policy change invalidates reuse; it never re-stamps historical
articles as if they had passed the new checks.
"""

from app.research.policy import (
    FACT_EXTRACTION_PROMPT_VERSION,
    FACT_VALIDATION_POLICY_VERSION,
)

DETAIL_SCHEMA_VERSION = "detail_article_v1"
DETAIL_PROMPT_VERSION = (
    f"detail_article_ru_v5:{FACT_EXTRACTION_PROMPT_VERSION}:{FACT_VALIDATION_POLICY_VERSION}"
)


def current_detail_policy_sql(alias: str = "detailed_articles") -> str:
    # Callers supply only internal SQL identifiers, never external text.
    return (
        f"{alias}.schema_version = '{DETAIL_SCHEMA_VERSION}' AND "
        f"{alias}.prompt_version = '{DETAIL_PROMPT_VERSION}'"
    )
