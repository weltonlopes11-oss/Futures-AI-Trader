CREATE TABLE IF NOT EXISTS indicator_snapshots (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    symbol TEXT NOT NULL,

    timestamp TEXT NOT NULL,

    close REAL,

    snapshot_json TEXT NOT NULL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP

);