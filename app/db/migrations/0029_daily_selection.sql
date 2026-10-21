CREATE TABLE daily_selection_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_key TEXT NOT NULL UNIQUE,
    owner_id INTEGER NOT NULL,
    stage TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    context_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK(json_valid(payload_json)),
    created_at TEXT NOT NULL
);
CREATE INDEX idx_daily_selection_owner_time ON daily_selection_runs(owner_id,created_at);
