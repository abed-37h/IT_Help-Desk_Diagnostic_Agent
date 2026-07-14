from __future__ import annotations

import json
from multiprocessing.dummy import connection
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LongTermMemoryStore:
    """
    Persistent long-term memory using SQLite.

    Each memory is stored using:
    - user_id
    - memory_key
    - memory_value
    - updated_at
    """

    def __init__(self, db_path: str | Path = "data/memory.db") -> None:
        self.db_path = Path(db_path)

        # Create the parent directory automatically when needed.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """
        Open a SQLite connection and always close it safely.

        Successful operations are committed automatically.
        Failed operations are rolled back.
        """
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row

        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_database(self) -> None:
        """Create the long-term memory table if it does not exist."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    memory_value TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, memory_key)
                )
                """
            )

    @staticmethod
    def _validate_user_id(user_id: str) -> str:
        """Validate and normalize a user identifier."""
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be a non-empty string.")
        return user_id.strip()

    @staticmethod
    def _validate_key(key: str) -> str:
        """Validate and normalize a memory key."""
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Memory key must be a non-empty string.")
        return key.strip()

    def save(
        self,
        user_id: str,
        key: str,
        value: Any,
    ) -> None:
        """Save or update one long-term memory value."""
        normalized_user_id = self._validate_user_id(user_id)
        normalized_key = self._validate_key(key)

        try:
            serialized_value = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("Long-term memory value must be JSON serializable.") from exc

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO long_term_memory (
                    user_id,
                    memory_key,
                    memory_value,
                    updated_at
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, memory_key)
                DO UPDATE SET
                    memory_value = excluded.memory_value,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_user_id,
                    normalized_key,
                    serialized_value,
                    utc_now(),
                ),
            )

    def get(
        self,
        user_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """Return one stored memory value."""
        normalized_user_id = self._validate_user_id(user_id)
        normalized_key = self._validate_key(key)

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT memory_value
                FROM long_term_memory
                WHERE user_id = ? AND memory_key = ?
                """,
                (normalized_user_id, normalized_key),
            ).fetchone()

        if row is None:
            return default

        return json.loads(row["memory_value"])

    def get_all(self, user_id: str) -> dict[str, Any]:
        """Return all long-term memory belonging to one user."""
        normalized_user_id = self._validate_user_id(user_id)

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT memory_key, memory_value
                FROM long_term_memory
                WHERE user_id = ?
                ORDER BY memory_key
                """,
                (normalized_user_id,),
            ).fetchall()

        return {
            row["memory_key"]: json.loads(row["memory_value"])
            for row in rows
        }

    def delete(self, user_id: str, key: str) -> None:
        """Delete one long-term memory entry."""
        normalized_user_id = self._validate_user_id(user_id)
        normalized_key = self._validate_key(key)

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM long_term_memory
                WHERE user_id = ? AND memory_key = ?
                """,
                (normalized_user_id, normalized_key),
            )

    def clear_user(self, user_id: str) -> None:
        """Delete all long-term memory for one user."""
        normalized_user_id = self._validate_user_id(user_id)

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM long_term_memory
                WHERE user_id = ?
                """,
                (normalized_user_id,),
            )