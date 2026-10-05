"""Persisted evidence-policy identity, separate from JSON schema shape.

Bump the relevant version whenever extraction instructions or factual guards
change materially; old sheets must be re-extracted/revalidated, not re-stamped.
"""

FACT_EXTRACTION_PROMPT_VERSION = "fact_extraction_v7"
FACT_VALIDATION_POLICY_VERSION = "fact_validation_v7"


def fact_policy_metadata() -> dict[str, str]:
    return {
        "extraction_prompt_version": FACT_EXTRACTION_PROMPT_VERSION,
        "validation_policy_version": FACT_VALIDATION_POLICY_VERSION,
    }


def current_fact_policy_sql(alias: str = "event_fact_sheets") -> str:
    # Only internal SQL identifiers are passed as alias.
    return f"""CASE WHEN json_valid({alias}.coverage_json) THEN
        json_extract({alias}.coverage_json, '$.extraction_prompt_version')
            = '{FACT_EXTRACTION_PROMPT_VERSION}'
        AND json_extract({alias}.coverage_json, '$.validation_policy_version')
            = '{FACT_VALIDATION_POLICY_VERSION}' ELSE 0 END"""
