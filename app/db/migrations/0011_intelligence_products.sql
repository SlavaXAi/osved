CREATE TABLE IF NOT EXISTS intelligence_products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (
        kind IN ('weekly_map', 'monthly_theses', 'weekly_personas', 'project_signals')
    ),
    generation_key TEXT NOT NULL UNIQUE,
    schema_version TEXT NOT NULL,
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    rendered_html TEXT NOT NULL,
    source_event_ids_json TEXT NOT NULL DEFAULT '[]',
    validation_status TEXT NOT NULL CHECK (
        validation_status IN ('valid', 'insufficient_data', 'invalid')
    ),
    validation_errors_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS intelligence_products_kind_period_idx
ON intelligence_products(kind, period_end DESC, created_at DESC);
