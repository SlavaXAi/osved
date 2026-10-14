ALTER TABLE detailed_articles ADD COLUMN depth TEXT NOT NULL DEFAULT 'legacy';
ALTER TABLE detailed_articles ADD COLUMN reader_context_json TEXT NOT NULL DEFAULT '{}';
ALTER TABLE detailed_articles ADD COLUMN context_fingerprint TEXT NOT NULL DEFAULT '';

CREATE TABLE detail_requests_reader (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    delivery_id INTEGER NOT NULL REFERENCES deliveries(id) ON DELETE CASCADE,
    card_id INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    chat_id INTEGER NOT NULL,
    user_id INTEGER NOT NULL,
    generation_key TEXT NOT NULL,
    depth TEXT NOT NULL DEFAULT 'legacy' CHECK(depth IN ('legacy','expanded','article')),
    reader_context_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'queued' CHECK(status IN (
        'queued','running','ready','sent','insufficient_evidence','budget_blocked','failed')),
    stage TEXT NOT NULL DEFAULT 'queued' CHECK(stage IN (
        'queued','planning','search','facts','writing','delivery','complete')),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count>=0),
    lease_until TEXT,
    placeholder_message_id INTEGER,
    placeholder_lease_until TEXT,
    keyboard_synced_status TEXT,
    article_id INTEGER REFERENCES detailed_articles(id) ON DELETE SET NULL,
    cache_hit INTEGER NOT NULL DEFAULT 0 CHECK(cache_hit IN(0,1)),
    last_error TEXT,
    requested_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    UNIQUE(delivery_id,depth)
);

INSERT INTO detail_requests_reader (
    id,delivery_id,card_id,event_id,chat_id,user_id,generation_key,status,stage,
    attempt_count,lease_until,placeholder_message_id,placeholder_lease_until,
    keyboard_synced_status,article_id,cache_hit,last_error,requested_at,started_at,
    completed_at,updated_at)
SELECT id,delivery_id,card_id,event_id,chat_id,user_id,generation_key,status,stage,
    attempt_count,lease_until,placeholder_message_id,placeholder_lease_until,
    keyboard_synced_status,article_id,cache_hit,last_error,requested_at,started_at,
    completed_at,updated_at FROM detail_requests;

DROP TABLE detail_requests;
ALTER TABLE detail_requests_reader RENAME TO detail_requests;
CREATE INDEX detail_requests_queue_idx ON detail_requests(status,lease_until,requested_at);
CREATE INDEX detail_requests_generation_idx ON detail_requests(generation_key,status);
CREATE INDEX detail_requests_user_date_idx ON detail_requests(user_id,requested_at);
CREATE INDEX detailed_articles_reader_cache_idx
    ON detailed_articles(event_id,depth,context_fingerprint,validation_status,expires_at);

CREATE TABLE news_panels (
    delivery_id INTEGER PRIMARY KEY REFERENCES deliveries(id) ON DELETE CASCADE,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    baseline_html TEXT NOT NULL,
    reader_context_json TEXT NOT NULL,
    view TEXT NOT NULL DEFAULT 'base' CHECK(view IN('base','expanded','article','loading','error')),
    displayed_request_id INTEGER REFERENCES detail_requests(id) ON DELETE SET NULL,
    active_request_id INTEGER REFERENCES detail_requests(id) ON DELETE SET NULL,
    previous_view TEXT NOT NULL DEFAULT 'base',
    page INTEGER NOT NULL DEFAULT 0,
    previous_page INTEGER NOT NULL DEFAULT 0,
    revision INTEGER NOT NULL DEFAULT 0,
    error_text TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chat_id,message_id)
);

CREATE TABLE news_result_feedback (
    article_id INTEGER NOT NULL REFERENCES detailed_articles(id) ON DELETE CASCADE,
    chat_id INTEGER NOT NULL,
    action TEXT NOT NULL CHECK(action IN('useful','not_worth_it')),
    feedback_event_id INTEGER REFERENCES feedback_events(id),
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(article_id,chat_id)
);

CREATE TABLE details_provider_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    depth TEXT NOT NULL CHECK(depth IN('expanded','article')),
    baseline_fingerprint TEXT NOT NULL,
    call_key TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN('preparing','completed','uncertain','rejected','failed')),
    usage_id INTEGER REFERENCES usage_ledger(id),
    reserved_usd REAL NOT NULL CHECK(reserved_usd>=0),
    response_json TEXT,
    error_json TEXT,
    retry_authorization TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX details_provider_calls_scope_idx
    ON details_provider_calls(event_id,depth,baseline_fingerprint,status);
