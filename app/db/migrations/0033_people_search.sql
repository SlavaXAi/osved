ALTER TABLE menu_batches ADD COLUMN persona_id TEXT;
CREATE INDEX menu_batches_persona_idx ON menu_batches(owner_id,persona_id,created_at);

CREATE TABLE people_search_documents (
    preparation_id INTEGER NOT NULL REFERENCES menu_preparations(id),
    document_id INTEGER NOT NULL REFERENCES research_documents(id),
    snapshot_json TEXT NOT NULL,
    PRIMARY KEY(preparation_id,document_id)
);
CREATE TABLE people_search_answers (
    preparation_id INTEGER PRIMARY KEY REFERENCES menu_preparations(id),
    status TEXT NOT NULL DEFAULT 'preparing',
    request_json TEXT NOT NULL,
    response_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
