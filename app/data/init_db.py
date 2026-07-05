"""
init_db.py — creates app/data/tickets.db with the schema the TicketTool
(action_tool.py) and ReportTool (report_tool.py) will read/write.

Run once to (re)create the database:
    python init_db.py

Safe to re-run: uses CREATE TABLE IF NOT EXISTS, so it won't wipe existing data.
If you want a clean slate, delete tickets.db first.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "tickets.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id       TEXT PRIMARY KEY,      -- e.g. "TCK-0001"
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    category        TEXT NOT NULL CHECK (category IN
                        ('hardware', 'software', 'network', 'account', 'other')),
    priority        TEXT NOT NULL CHECK (priority IN
                        ('low', 'medium', 'high', 'critical')) DEFAULT 'medium',
    status          TEXT NOT NULL CHECK (status IN
                        ('open', 'in_progress', 'pending_user', 'resolved', 'closed'))
                        DEFAULT 'open',
    requester_name  TEXT NOT NULL,
    requester_email TEXT,
    assigned_to     TEXT,                  -- agent/human owner, nullable until triaged
    resolution_notes TEXT,
    created_at      TEXT NOT NULL,         -- ISO 8601
    updated_at      TEXT NOT NULL,
    resolved_at     TEXT
);

-- Audit trail: every status change gets a row here.
-- This is what report_tool.py will query for metrics (time-to-resolution, etc).
CREATE TABLE IF NOT EXISTS ticket_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id       TEXT NOT NULL REFERENCES tickets(ticket_id),
    old_status      TEXT,
    new_status      TEXT NOT NULL,
    changed_by      TEXT,                  -- "agent" or a human name/id
    note            TEXT,
    changed_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
CREATE INDEX IF NOT EXISTS idx_history_ticket_id ON ticket_history(ticket_id);
"""

SEED_TICKETS = [
    (
        "TCK-0001", "VPN not connecting", "User cannot connect to corporate VPN since this morning.",
        "network", "high", "open", "Jane Doe", "jane.doe@example.com", None, None,
    ),
    (
        "TCK-0002", "Laptop won't power on", "Laptop screen stays black, no LED indicators.",
        "hardware", "critical", "in_progress", "Mark Lee", "mark.lee@example.com", "IT-Team-A", None,
    ),
]


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript(SCHEMA)

    # Only seed if the table is empty, so re-running doesn't duplicate rows.
    cur.execute("SELECT COUNT(*) FROM tickets")
    if cur.fetchone()[0] == 0:
        now = datetime.utcnow().isoformat()
        for row in SEED_TICKETS:
            cur.execute(
                """INSERT INTO tickets
                   (ticket_id, title, description, category, priority, status,
                    requester_name, requester_email, assigned_to, resolution_notes,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row + (now, now),
            )
            cur.execute(
                """INSERT INTO ticket_history
                   (ticket_id, old_status, new_status, changed_by, note, changed_at)
                   VALUES (?, NULL, ?, 'system', 'Ticket created', ?)""",
                (row[0], row[5], now),
            )
        print(f"Seeded {len(SEED_TICKETS)} sample tickets.")

    conn.commit()
    conn.close()
    print(f"Database ready at {DB_PATH}")


if __name__ == "__main__":
    init_db()
