"""One SQL definition of a finished card, shared by research and delivery."""


def finished_card_sql(alias: str = "cards") -> str:
    # Aliases are fixed identifiers supplied by application code, never user input.
    # Legacy object drafts without evaluation_stage remain compatible. Broken JSON
    # and discovery previews cannot consume the finished-publication queue.
    return f"""{alias}.validation_status = 'valid'
        AND CASE WHEN json_valid({alias}.structured_output_json) THEN
            json_type({alias}.structured_output_json) = 'object'
            AND COALESCE(json_extract({alias}.structured_output_json,
                '$.evaluation_stage'), '') NOT IN ('discovery', 'candidate_preview')
            ELSE 0 END"""
