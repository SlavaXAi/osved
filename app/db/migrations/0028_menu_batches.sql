CREATE TABLE menu_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN('news','telegram','twitter','youtube','people','institutions','bio')),
    request_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'queued',
    period_start TEXT NOT NULL,
    period_end TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    status_message_id INTEGER,
    header_status TEXT NOT NULL DEFAULT 'new',
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX menu_batches_owner_idx ON menu_batches(owner_id,kind,id DESC);
CREATE TABLE menu_batch_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL REFERENCES menu_batches(id),
    owner_id INTEGER NOT NULL,
    entity_key TEXT NOT NULL,
    item_kind TEXT NOT NULL CHECK(item_kind IN('news','video','bio')),
    entity_id INTEGER NOT NULL,
    delivery_id INTEGER,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    telegram_message_id INTEGER,
    lease_until REAL,
    available_at REAL NOT NULL DEFAULT 0,
    last_error TEXT
);
CREATE UNIQUE INDEX menu_batch_entity_claim ON menu_batch_items(owner_id,entity_key)
WHERE status NOT IN('failed','skipped');
CREATE INDEX menu_batch_items_queue_idx ON menu_batch_items(status,available_at,id);
CREATE TABLE menu_preparations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL UNIQUE REFERENCES menu_batches(id),
    owner_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'offered',
    limit_usd REAL NOT NULL CHECK(limit_usd>=0),
    cost_usd REAL NOT NULL DEFAULT 0,
    budget_json TEXT NOT NULL DEFAULT '{}',
    biography_cycle_id INTEGER REFERENCES biography_cycles(id),
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE menu_preparation_events (
    preparation_id INTEGER NOT NULL REFERENCES menu_preparations(id),
    event_id INTEGER NOT NULL REFERENCES events(id),
    status TEXT NOT NULL DEFAULT 'running',
    PRIMARY KEY(preparation_id,event_id)
);
