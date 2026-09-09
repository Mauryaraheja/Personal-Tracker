"""
SQLite-backed persistence for skill roadmaps and progress tracking.

Skills are now durable per role: the first time a role's roadmap is
generated, it gets saved here. Every visit after that loads the saved
skills instead of re-calling Groq/Tavily. This is deliberate -- an LLM
call isn't guaranteed to return the same skill_ids (or even the same
skills) twice, and a "progress tracker" only makes sense if the thing
being tracked stays fixed over time.

Two tables, not one:
- `skills` holds what a role's roadmap actually is (name, description).
- `tracker_items` holds the user's progress against each skill (status,
  notes, when it last changed).
They're kept separate so that skill info lives in exactly one place --
if it were duplicated into every tracker row, editing a skill's
description later would mean updating N rows instead of one.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .roadmap import get_skill_roadmap

# src/personaltracker/tracker.py -> parent -> parent -> parent = repo root
DB_PATH = Path(__file__).resolve().parent.parent.parent / "personaltracker.db"

VALID_STATUSES = {"not_started", "in_progress", "completed"}


@contextmanager
def _connect():
    """Opens a connection, commits on success, rolls back on failure, and
    always closes it afterwards.

    Worth being explicit about why this wrapper exists: sqlite3's own
    `with conn:` block manages the *transaction*, not the connection --
    it commits or rolls back, but never closes. So the obvious-looking
    `with sqlite3.connect(...) as conn:` leaks one open connection per
    call. Wrapping both concerns here keeps every call site to a single
    `with` and makes the cleanup impossible to forget.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us do dict(row) and row["column_name"]
    try:
        with conn:  # commit / rollback
            yield conn
    finally:
        conn.close()  # release the file handle


def init_db() -> None:
    """Creates both tables if they don't already exist. Safe to call every time."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS skills (
                role TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                why_it_matters TEXT,
                priority TEXT,
                level_required TEXT,
                source_url TEXT,
                PRIMARY KEY (role, skill_id)
            );

            CREATE TABLE IF NOT EXISTS tracker_items (
                id TEXT NOT NULL,
                role TEXT NOT NULL,
                skill_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'not_started',
                notes TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (role, skill_id),
                FOREIGN KEY (role, skill_id) REFERENCES skills (role, skill_id)
            );
            """
        )


def _role_has_saved_roadmap(conn: sqlite3.Connection, role: str) -> bool:
    row = conn.execute("SELECT 1 FROM skills WHERE role = ? LIMIT 1", (role,)).fetchone()
    return row is not None


def _save_roadmap(conn: sqlite3.Connection, role: str, skills: list[dict]) -> None:
    """Persists a freshly generated roadmap and creates one tracker row per skill."""
    now = datetime.now(timezone.utc).isoformat()
    for i, skill in enumerate(skills, start=1):
        conn.execute(
            """INSERT INTO skills
               (role, skill_id, name, description, why_it_matters, priority, level_required, source_url)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                role,
                skill["id"],
                skill.get("name"),
                skill.get("description"),
                skill.get("why_it_matters"),
                skill.get("priority"),
                skill.get("level_required"),
                skill.get("source_url"),
            ),
        )
        conn.execute(
            """INSERT INTO tracker_items (id, role, skill_id, status, notes, updated_at)
               VALUES (?, ?, ?, 'not_started', NULL, ?)""",
            (f"trk_{i:03d}", role, skill["id"], now),
        )


def get_or_create_roadmap(role: str) -> list[dict]:
    """
    Returns this role's skill list. Loads it from the database if it was
    generated before; otherwise generates it via get_skill_roadmap() (the
    existing Groq/Tavily pipeline) and saves it so future visits are instant.
    """
    init_db()
    with _connect() as conn:
        if _role_has_saved_roadmap(conn, role):
            rows = conn.execute(
                """SELECT skill_id AS id, name, description, why_it_matters,
                          priority, level_required, source_url
                   FROM skills WHERE role = ?""",
                (role,),
            ).fetchall()
            return [dict(row) for row in rows]

        skills = get_skill_roadmap(role)
        _save_roadmap(conn, role, skills)
        return skills


def get_tracker_items(role: str) -> list[dict]:
    """Returns progress rows for a role, each joined with its skill's name/description."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT t.id, t.skill_id, s.name, s.description, t.status, t.notes, t.updated_at
               FROM tracker_items t
               JOIN skills s ON s.role = t.role AND s.skill_id = t.skill_id
               WHERE t.role = ?
               ORDER BY t.id""",
            (role,),
        ).fetchall()
        return [dict(row) for row in rows]


def update_tracker_status(role: str, skill_id: str, status: str, notes: str | None = None) -> None:
    """Updates one skill's progress status (and optionally its notes)."""
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """UPDATE tracker_items
               SET status = ?, notes = COALESCE(?, notes), updated_at = ?
               WHERE role = ? AND skill_id = ?""",
            (status, notes, now, role, skill_id),
        )