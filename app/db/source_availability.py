"""One source lifecycle predicate for selection and last-moment delivery checks."""


def available_source_sql(source="sources", item="source_items"):
    # Aliases are internal constants, never user input. Unknown dates fail closed
    # for managed YouTube channels with a future-only subscription boundary.
    return f"""(
        {source}.admission_mode != 'history'
        AND COALESCE((SELECT CASE value WHEN 'true' THEN 1 WHEN 'false' THEN 0 END
            FROM runtime_settings WHERE key='source_enabled:'||{source}.id),{source}.enabled)=1
        AND NOT EXISTS(SELECT 1 FROM youtube_channels lifecycle
            WHERE lifecycle.source_id={source}.id AND
              (lifecycle.removed_at IS NOT NULL OR
               julianday({item}.published_at) IS NULL OR
               julianday({item}.published_at)<julianday(lifecycle.not_before)))
    )"""
