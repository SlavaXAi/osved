-- Read-only browser observations. Never store browser/session cookies here.
CREATE TABLE IF NOT EXISTS browser_source_tasks (
    id TEXT PRIMARY KEY,
    task_key TEXT NOT NULL UNIQUE,
    source_id TEXT NOT NULL,
    platform TEXT NOT NULL CHECK (platform = 'x'),
    task_kind TEXT NOT NULL CHECK (task_kind = 'visible_feed_snapshot'),
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('queued', 'claimed', 'staged', 'committed', 'completed', 'failed', 'expired')
    ),
    claim_token TEXT,
    lease_until REAL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    result_json TEXT,
    result_digest TEXT,
    error_code TEXT,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS browser_source_tasks_queue_idx
ON browser_source_tasks(status, created_at);
