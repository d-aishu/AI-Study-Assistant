"""
Lightweight SQLite persistence layer for user accounts and chat history.

Kept deliberately simple (no ORM) since this app has two small tables and
runs as a single-process Streamlit app. Swap for a real database later if
this needs to scale beyond a single instance.
"""
import os
import sqlite3

DB_PATH = os.getenv("APP_DB_PATH", "app_data.db")


def get_connection() -> sqlite3.Connection:
    """
    Open a new SQLite connection. check_same_thread=False because Streamlit
    can hand off work to background threads; each function below opens and
    closes its own short-lived connection rather than sharing one.
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist yet. Safe to call on every app start."""
    conn = get_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                document_name TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id, document_name)"
        )
        conn.commit()
    finally:
        conn.close()
