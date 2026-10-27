ALTER TABLE x_reposts ADD COLUMN native_kind TEXT;

CREATE TABLE x_repost_outbox (
    repost_id INTEGER NOT NULL REFERENCES x_reposts(id),
    part_key TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('preparing','sent','uncertain')),
    telegram_message_id INTEGER,
    result_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(repost_id,part_key)
);
