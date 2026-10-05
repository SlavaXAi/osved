CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    locator TEXT NOT NULL,
    role TEXT NOT NULL,
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    candidate_enabled INTEGER NOT NULL CHECK (candidate_enabled IN (0, 1)),
    priority TEXT NOT NULL,
    region TEXT NOT NULL,
    language TEXT NOT NULL,
    topics_json TEXT NOT NULL,
    poll_interval_minutes INTEGER NOT NULL,
    calibration_lookback_days INTEGER NOT NULL,
    production_max_age_days INTEGER NOT NULL,
    daily_candidate_cap INTEGER NOT NULL,
    last_cursor TEXT,
    last_success_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(kind, locator)
);

CREATE TABLE IF NOT EXISTS source_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES sources(id),
    external_id TEXT NOT NULL,
    source_url TEXT NOT NULL,
    original_text TEXT,
    original_language TEXT,
    published_at TEXT NOT NULL,
    discovered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    outbound_urls_json TEXT NOT NULL DEFAULT '[]',
    media_metadata_json TEXT,
    payload_hash TEXT NOT NULL,
    raw_payload_json TEXT,
    processing_status TEXT NOT NULL DEFAULT 'new',
    processing_error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_id, external_id),
    UNIQUE(source_id, payload_hash)
);

CREATE INDEX IF NOT EXISTS source_items_published_at_idx
ON source_items(published_at);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL DEFAULT 'candidate',
    event_type TEXT,
    headline TEXT,
    entities_json TEXT NOT NULL DEFAULT '{}',
    topics_json TEXT NOT NULL DEFAULT '[]',
    regions_json TEXT NOT NULL DEFAULT '[]',
    event_at TEXT,
    first_seen_at TEXT NOT NULL,
    latest_source_at TEXT NOT NULL,
    primary_source_item_id INTEGER REFERENCES source_items(id),
    score REAL,
    score_features_json TEXT,
    fingerprint TEXT,
    decision_reason TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS events_fingerprint_idx ON events(fingerprint);
CREATE INDEX IF NOT EXISTS events_status_score_idx ON events(status, score DESC);

CREATE TABLE IF NOT EXISTS event_source_items (
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    source_item_id INTEGER NOT NULL REFERENCES source_items(id) ON DELETE CASCADE,
    evidence_role TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(event_id, source_item_id)
);

CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id),
    card_version INTEGER NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT,
    generation_cost_usd REAL NOT NULL DEFAULT 0,
    structured_output_json TEXT NOT NULL,
    rendered_html TEXT NOT NULL,
    source_urls_json TEXT NOT NULL,
    primary_preview_url TEXT,
    validation_status TEXT NOT NULL,
    validation_errors_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_id, card_version)
);

CREATE TABLE IF NOT EXISTS deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER NOT NULL REFERENCES cards(id),
    chat_id INTEGER NOT NULL,
    telegram_message_id INTEGER,
    batch_id TEXT,
    keyboard_state TEXT NOT NULL DEFAULT 'reactions',
    delivery_status TEXT NOT NULL DEFAULT 'pending',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    delivered_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(card_id, chat_id)
);

CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    delivery_id INTEGER NOT NULL REFERENCES deliveries(id),
    event_id INTEGER NOT NULL REFERENCES events(id),
    card_id INTEGER NOT NULL REFERENCES cards(id),
    action TEXT NOT NULL,
    reaction TEXT,
    comment TEXT,
    previous_reaction TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS feedback_events_event_idx
ON feedback_events(event_id, created_at);

CREATE TABLE IF NOT EXISTS usage_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    operation TEXT NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 1,
    input_units INTEGER,
    output_units INTEGER,
    cost_usd REAL NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS usage_ledger_created_at_idx
ON usage_ledger(created_at);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    run_type TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    processed_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    error TEXT
);

CREATE INDEX IF NOT EXISTS runs_started_at_idx ON runs(started_at DESC);
