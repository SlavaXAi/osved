CREATE TABLE IF NOT EXISTS research_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    query TEXT NOT NULL,
    freshness TEXT,
    status TEXT NOT NULL,
    result_count INTEGER NOT NULL DEFAULT 0,
    results_json TEXT NOT NULL DEFAULT '[]',
    cost_usd REAL NOT NULL DEFAULT 0 CHECK (cost_usd >= 0),
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS research_queries_event_idx
ON research_queries(event_id, created_at DESC);

CREATE INDEX IF NOT EXISTS research_queries_expiry_idx
ON research_queries(expires_at);

CREATE TABLE IF NOT EXISTS research_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_url TEXT NOT NULL UNIQUE,
    url TEXT NOT NULL,
    title TEXT,
    publisher TEXT,
    source_class TEXT NOT NULL,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    content_text TEXT,
    content_hash TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS research_documents_expiry_idx
ON research_documents(expires_at);

CREATE TABLE IF NOT EXISTS event_research_documents (
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    research_document_id INTEGER NOT NULL REFERENCES research_documents(id) ON DELETE CASCADE,
    evidence_role TEXT NOT NULL,
    relevance REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(event_id, research_document_id)
);

CREATE INDEX IF NOT EXISTS event_research_documents_role_idx
ON event_research_documents(event_id, evidence_role);

CREATE TABLE IF NOT EXISTS event_fact_sheets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    fact_sheet_version INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    facts_json TEXT NOT NULL,
    coverage_json TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    validation_errors_json TEXT NOT NULL DEFAULT '[]',
    research_cost_usd REAL NOT NULL DEFAULT 0 CHECK (research_cost_usd >= 0),
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_id, fact_sheet_version)
);

CREATE INDEX IF NOT EXISTS event_fact_sheets_status_idx
ON event_fact_sheets(event_id, validation_status, fact_sheet_version DESC);

CREATE INDEX IF NOT EXISTS event_fact_sheets_expiry_idx
ON event_fact_sheets(expires_at);
