CREATE TABLE IF NOT EXISTS schedule_slots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL CHECK (job_type IN ('radar', 'research', 'delivery')),
    slot_at TEXT NOT NULL,
    mode TEXT NOT NULL CHECK (mode IN ('dry_run', 'live')),
    status TEXT NOT NULL CHECK (
        status IN ('planned', 'running', 'completed', 'failed', 'skipped')
    ),
    run_id TEXT REFERENCES runs(id) ON DELETE SET NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_type, slot_at)
);

CREATE INDEX IF NOT EXISTS schedule_slots_status_idx
ON schedule_slots(status, slot_at DESC);
