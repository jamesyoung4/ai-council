import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(os.environ.get("COUNCIL_DB_PATH", str(Path(__file__).parent / "council.db")))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL DEFAULT 'New conversation',
  personas_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS turns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  conversation_id INTEGER NOT NULL,
  user_question TEXT NOT NULL,
  attachments_json TEXT NOT NULL DEFAULT '[]',
  responses_json TEXT NOT NULL DEFAULT '{}',
  synthesis TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conversation_id, id);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        # Migration: add personas_json to pre-existing conversations tables
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
        if "personas_json" not in cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN personas_json TEXT NOT NULL DEFAULT '[]'"
            )


def list_conversations() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.title, c.updated_at,
                   (SELECT COUNT(*) FROM turns WHERE conversation_id = c.id) AS turn_count
            FROM conversations c
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
        return [dict(r) for r in rows]


def create_conversation(
    title: str = "New conversation",
    personas: list[dict] | None = None,
) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, personas_json) VALUES (?, ?)",
            (title, json.dumps(personas or [])),
        )
        return cur.lastrowid


def get_conversation(conv_id: int) -> dict | None:
    """Return the full conversation with all turns (attachments metadata only)."""
    with connect() as conn:
        conv = conn.execute(
            "SELECT * FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if not conv:
            return None
        turns = conn.execute(
            "SELECT * FROM turns WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
        return {
            "id": conv["id"],
            "title": conv["title"],
            "personas": json.loads(conv["personas_json"] or "[]"),
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "turn_count": len(turns),
            "turns": [
                {
                    "id": t["id"],
                    "user_question": t["user_question"],
                    "attachments": _strip_attachment_content(json.loads(t["attachments_json"])),
                    "responses": json.loads(t["responses_json"]),
                    "synthesis": t["synthesis"],
                    "created_at": t["created_at"],
                }
                for t in turns
            ],
        }


def get_personas(conv_id: int) -> list[dict]:
    with connect() as conn:
        row = conn.execute(
            "SELECT personas_json FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        return json.loads(row["personas_json"]) if row else []


def update_personas(conv_id: int, personas: list[dict]) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE conversations SET personas_json = ? WHERE id = ?",
            (json.dumps(personas), conv_id),
        )


def get_history(conv_id: int) -> list[dict]:
    """Return raw turns (with full attachment content) for prompt building."""
    with connect() as conn:
        turns = conn.execute(
            "SELECT * FROM turns WHERE conversation_id = ? ORDER BY id ASC",
            (conv_id,),
        ).fetchall()
        return [
            {
                "user_question": t["user_question"],
                "attachments": json.loads(t["attachments_json"]),
                "responses": json.loads(t["responses_json"]),
                "synthesis": t["synthesis"],
            }
            for t in turns
        ]


def turn_count(conv_id: int) -> int:
    with connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM turns WHERE conversation_id = ?", (conv_id,)
        ).fetchone()
        return row["n"]


def save_turn(
    conv_id: int,
    user_question: str,
    attachments: list[dict],
    responses: dict,
    synthesis: str,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO turns
              (conversation_id, user_question, attachments_json, responses_json, synthesis)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conv_id,
                user_question,
                json.dumps(attachments),
                json.dumps(responses),
                synthesis,
            ),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (conv_id,),
        )


def update_title(conv_id: int, title: str) -> None:
    with connect() as conn:
        conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conv_id))


def delete_conversation(conv_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))


def _strip_attachment_content(attachments: list[dict]) -> list[dict]:
    return [
        {"filename": a.get("filename", "?"), "char_count": len(a.get("content", ""))}
        for a in attachments
    ]
