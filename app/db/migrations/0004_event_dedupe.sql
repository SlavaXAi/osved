CREATE UNIQUE INDEX IF NOT EXISTS events_fingerprint_unique_idx
ON events(fingerprint)
WHERE fingerprint IS NOT NULL;
