CREATE TABLE IF NOT EXISTS detailed_articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    generation_key TEXT NOT NULL UNIQUE,
    schema_version TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    provider_response_metadata_json TEXT NOT NULL DEFAULT '{}',
    fact_snapshot_json TEXT NOT NULL,
    document_ids_json TEXT NOT NULL DEFAULT '[]',
    structured_output_json TEXT NOT NULL,
    rendered_html TEXT NOT NULL,
    source_urls_json TEXT NOT NULL DEFAULT '[]',
    validation_status TEXT NOT NULL,
    validation_errors_json TEXT NOT NULL DEFAULT '[]',
    generation_cost_usd REAL NOT NULL DEFAULT 0 CHECK (generation_cost_usd >= 0),
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS detailed_articles_event_idx
ON detailed_articles(event_id, created_at DESC);

CREATE INDEX IF NOT EXISTS detailed_articles_cache_idx
ON detailed_articles(generation_key, validation_status, expires_at);

CREATE TABLE IF NOT EXISTS detail_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    delivery_id INTEGER NOT NULL UNIQUE REFERENCES deliveries(id) ON DELETE CASCADE,
    card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    generation_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (
        status IN (
            'queued', 'running', 'ready', 'sent',
            'insufficient_evidence', 'budget_blocked', 'failed'
        )
    ),
    stage TEXT NOT NULL DEFAULT 'queued' CHECK (
        stage IN ('queued', 'planning', 'search', 'facts', 'writing', 'delivery', 'complete')
    ),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    lease_until TEXT,
    placeholder_message_id INTEGER,
    placeholder_lease_until TEXT,
    keyboard_synced_status TEXT,
    article_id INTEGER REFERENCES detailed_articles(id) ON DELETE SET NULL,
    cache_hit INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0, 1)),
    last_error TEXT,
    requested_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS detail_requests_queue_idx
ON detail_requests(status, lease_until, requested_at);

CREATE INDEX IF NOT EXISTS detail_requests_generation_idx
ON detail_requests(generation_key, status);

CREATE INDEX IF NOT EXISTS detail_requests_user_date_idx
ON detail_requests(user_id, requested_at);

ALTER TABLE feedback_events
ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}';
