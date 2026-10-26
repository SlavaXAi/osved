CREATE TABLE news_question_sessions (
    chat_id INTEGER NOT NULL,
    delivery_id INTEGER NOT NULL REFERENCES deliveries(id) ON DELETE CASCADE,
    prompt_message_id INTEGER,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(chat_id, delivery_id)
);

CREATE TABLE news_question_inputs (
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    delivery_id INTEGER NOT NULL REFERENCES deliveries(id) ON DELETE CASCADE,
    input_kind TEXT NOT NULL CHECK(input_kind IN ('text','voice')),
    question TEXT,
    status TEXT NOT NULL CHECK(status IN ('receiving','queued','failed','cancelled')),
    request_id INTEGER REFERENCES detail_requests(id),
    article_reference_fingerprint TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY(chat_id, message_id)
);
CREATE INDEX news_question_inputs_day_idx ON news_question_inputs(chat_id, created_at);
