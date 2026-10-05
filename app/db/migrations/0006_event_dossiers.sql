CREATE TABLE IF NOT EXISTS event_dossiers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    dossier_version INTEGER NOT NULL,
    schema_version TEXT NOT NULL,
    fact_sheet_id INTEGER REFERENCES event_fact_sheets(id) ON DELETE SET NULL,
    decision TEXT NOT NULL CHECK (
        decision IN ('ready', 'needs_research', 'blocked_conflict')
    ),
    decision_reason TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    validation_status TEXT NOT NULL DEFAULT 'valid',
    validation_errors_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_id, dossier_version)
);

CREATE INDEX IF NOT EXISTS event_dossiers_event_idx
ON event_dossiers(event_id, dossier_version DESC);

CREATE INDEX IF NOT EXISTS event_dossiers_decision_idx
ON event_dossiers(decision, validation_status, created_at DESC);
