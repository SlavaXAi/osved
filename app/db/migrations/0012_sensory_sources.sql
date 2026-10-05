ALTER TABLE sources ADD COLUMN adapter_id TEXT NOT NULL DEFAULT 'auto';
ALTER TABLE sources ADD COLUMN strategy TEXT NOT NULL DEFAULT 'feed';
ALTER TABLE sources ADD COLUMN admission_mode TEXT NOT NULL DEFAULT 'event_candidate';
UPDATE sources SET admission_mode = CASE
  WHEN role = 'own_history' THEN 'history'
  WHEN candidate_enabled = 0 THEN 'trigger_only'
  WHEN role = 'primary' THEN 'primary_evidence'
  ELSE 'event_candidate' END;
ALTER TABLE source_items ADD COLUMN canonical_url TEXT;
ALTER TABLE source_items ADD COLUMN content_type TEXT NOT NULL DEFAULT 'post';
ALTER TABLE source_items ADD COLUMN author_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE source_items ADD COLUMN engagement_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE source_items ADD COLUMN discovery_mode TEXT NOT NULL DEFAULT 'feed';
ALTER TABLE source_items ADD COLUMN last_observed_at TEXT;
ALTER TABLE source_items ADD COLUMN observation_count INTEGER NOT NULL DEFAULT 1;
CREATE TABLE source_runs (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES sources(id),
    adapter_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    status TEXT NOT NULL,
    cursor_before TEXT,
    cursor_after TEXT,
    items_seen INTEGER NOT NULL DEFAULT 0,
    items_inserted INTEGER NOT NULL DEFAULT 0,
    items_rejected INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    reserved_usd REAL NOT NULL DEFAULT 0,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    lease_until TEXT NOT NULL
);
CREATE INDEX source_runs_source_time ON source_runs(source_id, started_at DESC);
CREATE TABLE source_health (
    source_id TEXT PRIMARY KEY REFERENCES sources(id),
    status TEXT NOT NULL DEFAULT 'ok',
    consecutive_errors INTEGER NOT NULL DEFAULT 0,
    cooldown_until TEXT,
    last_attempt_at TEXT,
    last_productive_at TEXT,
    last_error TEXT
);
