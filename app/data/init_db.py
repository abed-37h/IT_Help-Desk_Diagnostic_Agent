"""
init_db.py — creates app/data/tickets.db with the schema used by:
- TicketTool / action_tool.py
- ReportTool / report_tool.py

Safe to re-run:
- CREATE TABLE IF NOT EXISTS will not wipe existing data.
- Seed tickets are inserted only when the tickets table is empty.

"""

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


# Works whether this file is placed in:
# 1) project root: init_db.py
# 2) app/: app/init_db.py
# 3) app/data/: app/data/init_db.py
THIS_DIR = Path(__file__).resolve().parent

if THIS_DIR.name == "data":
    DATA_DIR = THIS_DIR
elif THIS_DIR.name == "app":
    DATA_DIR = THIS_DIR / "data"
else:
    DATA_DIR = THIS_DIR / "app" / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "tickets.db"


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id        TEXT PRIMARY KEY,
    user_id          TEXT NOT NULL,
    title            TEXT NOT NULL,
    description      TEXT NOT NULL,
    category         TEXT NOT NULL CHECK (
                        category IN ('Network', 'Account', 'OS', 'Application', 'unknown')
                    ),
    priority         TEXT NOT NULL CHECK (
                        priority IN ('low', 'medium', 'high', 'critical')
                    ) DEFAULT 'medium',
    status           TEXT NOT NULL CHECK (
                        status IN ('open', 'in_progress', 'pending_user', 'resolved', 'closed')
                    ) DEFAULT 'open',
    assigned_to      TEXT,
    resolution_notes TEXT,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    resolved_at      TEXT
);

CREATE TABLE IF NOT EXISTS ticket_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   TEXT NOT NULL,
    old_status  TEXT,
    new_status  TEXT NOT NULL,
    changed_by  TEXT NOT NULL DEFAULT 'system',
    note        TEXT,
    changed_at  TEXT NOT NULL,

    FOREIGN KEY (ticket_id)
        REFERENCES tickets(ticket_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tickets_status
    ON tickets(status);

CREATE INDEX IF NOT EXISTS idx_tickets_category
    ON tickets(category);

CREATE INDEX IF NOT EXISTS idx_tickets_user_id
    ON tickets(user_id);

CREATE INDEX IF NOT EXISTS idx_history_ticket_id
    ON ticket_history(ticket_id);

CREATE INDEX IF NOT EXISTS idx_history_changed_at
    ON ticket_history(changed_at);
"""


SEED_TICKETS = [
    {
        "user_id": "user_jane_doe",
        "title": "VPN not connecting",
        "description": "User cannot connect to corporate VPN since this morning.",
        "category": "Network",
        "priority": "high",
        "status": "open",
        "assigned_to": None,
    },
    {
        "user_id": "user_mark_lee",
        "title": "Laptop won't power on",
        "description": "Laptop screen stays black, no LED indicators.",
        "category": "OS",
        "priority": "critical",
        "status": "in_progress",
        "assigned_to": "IT-Team-A",
    },
]


VALID_CATEGORIES = {"Network", "Account", "OS", "Application", "unknown"}
VALID_PRIORITIES = {"low", "medium", "high", "critical"}
VALID_STATUSES = {"open", "in_progress", "pending_user", "resolved", "closed"}


def utc_now() -> str:
    """Return a timezone-aware UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    """Create a SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def get_next_ticket_id(cur: sqlite3.Cursor) -> str:
    """
    Generate the next sequential ticket_id.

    Example:
        TCK-0001, TCK-0002, TCK-0003, ...
    """
    cur.execute(
        """
        SELECT MAX(CAST(SUBSTR(ticket_id, 5) AS INTEGER))
        FROM tickets
        WHERE ticket_id LIKE 'TCK-%'
        """
    )

    max_num = cur.fetchone()[0]
    next_num = 1 if max_num is None else max_num + 1

    return f"TCK-{next_num:04d}"


def validate_ticket_fields(category: str, priority: str, status: str) -> None:
    """Validate fields before inserting into SQLite."""
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category: {category}. Use one of {sorted(VALID_CATEGORIES)}"
        )

    if priority not in VALID_PRIORITIES:
        raise ValueError(
            f"Invalid priority: {priority}. Use one of {sorted(VALID_PRIORITIES)}"
        )

    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status: {status}. Use one of {sorted(VALID_STATUSES)}"
        )


def create_ticket(
    conn: sqlite3.Connection,
    user_id: str,
    title: str,
    description: str,
    category: str,
    priority: str = "medium",
    status: str = "open",
    assigned_to: Optional[str] = None,
) -> str:
    """
    Insert a new ticket with an auto-assigned ticket_id.

    action_tool.py can call this function directly instead of duplicating
    ticket ID logic.
    """
    validate_ticket_fields(category, priority, status)

    now = utc_now()

    # Prevent two writers from generating the same ticket ID at the same time.
    conn.execute("BEGIN IMMEDIATE;")

    try:
        cur = conn.cursor()
        ticket_id = get_next_ticket_id(cur)

        cur.execute(
            """
            INSERT INTO tickets (
                ticket_id,
                user_id,
                title,
                description,
                category,
                priority,
                status,
                assigned_to,
                resolution_notes,
                created_at,
                updated_at,
                resolved_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL)
            """,
            (
                ticket_id,
                user_id,
                title,
                description,
                category,
                priority,
                status,
                assigned_to,
                now,
                now,
            ),
        )

        cur.execute(
            """
            INSERT INTO ticket_history (
                ticket_id,
                old_status,
                new_status,
                changed_by,
                note,
                changed_at
            )
            VALUES (?, NULL, ?, 'system', 'Ticket created', ?)
            """,
            (ticket_id, status, now),
        )

        conn.commit()
        return ticket_id

    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    """Create tables and seed starter data only if the tickets table is empty."""
    with connect() as conn:
        conn.executescript(SCHEMA)

        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM tickets")
        ticket_count = cur.fetchone()[0]

        if ticket_count == 0:
            for seed_ticket in SEED_TICKETS:
                ticket_id = create_ticket(conn, **seed_ticket)
                print(f"Seeded {ticket_id}")

    print(f"Database ready at {DB_PATH}")


if __name__ == "__main__":
    init_db()