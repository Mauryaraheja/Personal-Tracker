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

A third table, `role_aliases`, maps whatever the user typed onto the one
canonical role it means. "Gen AI", "gen ai" and "Generative AI Engineer"
should all reach the same roadmap, but only the LLM can tell us the last
one is the same job as the first two -- and asking it on every page load
would undo the whole point of caching roadmaps in the first place. So the
answer is looked up once and then remembered here.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .roadmap import build_roadmap, refine_role

# src/personaltracker/tracker.py -> parent -> parent -> parent = repo root
DB_PATH = Path(__file__).resolve().parent.parent.parent / "personaltracker.db"

VALID_STATUSES = {"not_started", "in_progress", "completed"}


def _normalize_role(role: str) -> str:
    """Collapse a typed role into the single key used to store it.

    The role string is half of the primary key in both tables, so
    "Gen AI", "gen ai" and " Gen  AI " have to resolve to the same row.
    Without this they don't: SQLite compares them as distinct strings, so
    each spelling silently generates its own roadmap (a wasted Groq +
    Tavily call) and its own progress rows -- and a user who capitalizes
    differently on their next visit appears to have lost everything they
    had tracked, with no error to explain why.

    Only the storage key is normalized. The role as the user actually
    typed it is still what gets sent to the LLM, since casing can carry
    real meaning in a job title.
    """
    normalized = " ".join(role.split()).lower()
    if not normalized:
        raise ValueError("role must not be empty")
    return normalized


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

            CREATE TABLE IF NOT EXISTS role_aliases (
                typed_role TEXT PRIMARY KEY,
                role_key TEXT NOT NULL,
                refined_title TEXT NOT NULL
            );
            """
        )


def _resolve_role_key(conn: sqlite3.Connection, role: str) -> tuple[str, str]:
    """Work out which canonical role the user's typed text refers to.

    Returns (role_key, refined_title). `role_key` is the database key;
    `refined_title` is the properly-spelled job title, used for searching.

    _normalize_role alone can't do this job. It fixes casing and spacing,
    so "Gen AI" and "gen ai" already agree -- but it has no way to know
    that "Gen AI" and "Generative AI Engineer" are the same job. Only the
    LLM knows that, via refine_role().

    The catch is that refine_role() is a Groq call, and calling it on
    every visit would defeat the point of caching roadmaps at all. So the
    answer gets written to role_aliases the first time and read from
    there forever after: a new spelling costs one call, a spelling we've
    seen before costs nothing.
    """
    typed = _normalize_role(role)

    row = conn.execute(
        "SELECT role_key, refined_title FROM role_aliases WHERE typed_role = ?",
        (typed,),
    ).fetchone()
    if row:
        return row["role_key"], row["refined_title"]

    refined_title = refine_role(role)
    role_key = _normalize_role(refined_title)
    conn.execute(
        """INSERT OR REPLACE INTO role_aliases (typed_role, role_key, refined_title)
           VALUES (?, ?, ?)""",
        (typed, role_key, refined_title),
    )
    return role_key, refined_title


def _lookup_role_key(conn: sqlite3.Connection, role: str) -> str:
    """The read-only half of _resolve_role_key -- never calls the LLM.

    Used by the tracker functions, which only ever run after a roadmap
    exists, so the alias is already recorded. Falling back to the typed
    form keeps roadmaps saved before role_aliases existed reachable.
    """
    row = conn.execute(
        "SELECT role_key FROM role_aliases WHERE typed_role = ?",
        (_normalize_role(role),),
    ).fetchone()
    return row["role_key"] if row else _normalize_role(role)


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
        # A spelling we've seen before resolves from role_aliases with no
        # API call at all; a new one costs a single refine_role().
        role_key, refined_title = _resolve_role_key(conn, role)

        if _role_has_saved_roadmap(conn, role_key):
            rows = conn.execute(
                """SELECT skill_id AS id, name, description, why_it_matters,
                          priority, level_required, source_url
                   FROM skills WHERE role = ?""",
                (role_key,),
            ).fetchall()
            return [dict(row) for row in rows]

        # Build from the refined title, not the raw text -- searching for
        # "Gen AI" returns generic listicles, "Generative AI Engineer"
        # returns real job requirements.
        skills = build_roadmap(refined_title)
        _save_roadmap(conn, role_key, skills)
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
            (_lookup_role_key(conn, role),),
        ).fetchall()
        return [dict(row) for row in rows]


def update_tracker_status(role: str, skill_id: str, status: str, notes: str | None = None) -> None:
    """Updates one skill's progress status (and optionally its notes)."""
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}, got {status!r}")

    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            """UPDATE tracker_items
               SET status = ?, notes = COALESCE(?, notes), updated_at = ?
               WHERE role = ? AND skill_id = ?""",
            (status, notes, now, _lookup_role_key(conn, role), skill_id),
        )
        # An UPDATE that matches nothing is not an error in SQL -- it just
        # does nothing. Silently accepting a write that never landed is
        # how a tracker ends up disagreeing with what the user sees.
        if cursor.rowcount == 0:
            raise KeyError(f"no tracked skill {skill_id!r} for role {role!r}")
