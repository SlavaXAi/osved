CREATE TABLE IF NOT EXISTS biography_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    english_name TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    topic TEXT NOT NULL,
    era TEXT NOT NULL,
    region TEXT NOT NULL,
    life_arc TEXT NOT NULL,
    living INTEGER NOT NULL DEFAULT 0,
    familiar INTEGER NOT NULL DEFAULT 0,
    approved INTEGER NOT NULL DEFAULT 0,
    life_scale INTEGER NOT NULL,
    historical_impact INTEGER NOT NULL,
    exceptional_quality INTEGER NOT NULL,
    rationale TEXT NOT NULL,
    image_query TEXT NOT NULL,
    discovery_urls_json TEXT NOT NULL DEFAULT '[]',
    seed_order INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS biography_cycles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    local_date TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'prepare',
    status TEXT NOT NULL DEFAULT 'running',
    max_cost_usd REAL NOT NULL CHECK(max_cost_usd>0 AND max_cost_usd<=0.25),
    artifact_json TEXT NOT NULL DEFAULT '{}',
    last_error TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS biography_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle_id INTEGER NOT NULL REFERENCES biography_cycles(id),
    operation_key TEXT NOT NULL,
    provider TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'reserved',
    reservation_usd REAL NOT NULL,
    cost_usd REAL NOT NULL,
    receipt_json TEXT NOT NULL DEFAULT '{}',
    response_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(cycle_id, operation_key)
);
CREATE TABLE IF NOT EXISTS biography_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES biography_people(id),
    url TEXT NOT NULL,
    publisher TEXT NOT NULL,
    title TEXT NOT NULL,
    markdown TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(person_id, url)
);
CREATE TABLE IF NOT EXISTS biography_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id INTEGER NOT NULL REFERENCES biography_people(id),
    cycle_id INTEGER NOT NULL UNIQUE REFERENCES biography_cycles(id),
    status TEXT NOT NULL DEFAULT 'ready',
    payload_json TEXT NOT NULL,
    dossier_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS biography_deliveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    issue_id INTEGER NOT NULL REFERENCES biography_issues(id),
    owner_id INTEGER NOT NULL,
    slot_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    retry_at TEXT,
    telegram_message_id INTEGER,
    last_error TEXT,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    UNIQUE(owner_id, slot_key)
);
CREATE TABLE IF NOT EXISTS biography_feedback (
    delivery_id INTEGER NOT NULL REFERENCES biography_deliveries(id),
    owner_id INTEGER NOT NULL,
    hero INTEGER CHECK(hero IN (-1,1)),
    text INTEGER CHECK(text IN (-1,1)),
    reaction TEXT CHECK(reaction IN ('like','dislike','superlike','superdislike')),
    comment TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(delivery_id, owner_id)
);
CREATE INDEX IF NOT EXISTS biography_issue_status_idx ON biography_issues(status,id);
CREATE INDEX IF NOT EXISTS biography_usage_cycle_idx ON biography_usage(cycle_id,status);
CREATE INDEX IF NOT EXISTS biography_delivery_status_idx ON biography_deliveries(status);
CREATE UNIQUE INDEX IF NOT EXISTS biography_issue_delivery_idx ON biography_deliveries(issue_id)
WHERE status IN ('pending','sending','sent','unknown');
