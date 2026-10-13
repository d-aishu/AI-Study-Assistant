"""
Persistent chat history: every Q&A turn is saved per user (and per document)
so it survives page refreshes, logouts, and restarts — not just the current
Streamlit session.
"""
import json
from typing import List, Dict, Optional

from src.db import get_connection


def save_chat_turn(user_id: int, document_name: Optional[str], question: str, answer: str, sources: List[Dict]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO chat_history (user_id, document_name, question, answer, sources_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, document_name, question, answer, json.dumps(sources or [])),
        )
        conn.commit()
    finally:
        conn.close()


def load_chat_history(user_id: int, document_name: Optional[str] = None, limit: int = 200) -> List[Dict]:
    """Loads a user's chat history, optionally scoped to one document, oldest first."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        if document_name:
            cur.execute(
                """
                SELECT question, answer, sources_json, document_name, created_at
                FROM chat_history
                WHERE user_id = ? AND document_name = ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (user_id, document_name, limit),
            )
        else:
            cur.execute(
                """
                SELECT question, answer, sources_json, document_name, created_at
                FROM chat_history
                WHERE user_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (user_id, limit),
            )
        rows = cur.fetchall()
        return [
            {
                "question": r["question"],
                "answer": r["answer"],
                "sources": json.loads(r["sources_json"] or "[]"),
                "document_name": r["document_name"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    finally:
        conn.close()


def clear_chat_history(user_id: int, document_name: Optional[str] = None) -> None:
    conn = get_connection()
    try:
        if document_name:
            conn.execute(
                "DELETE FROM chat_history WHERE user_id = ? AND document_name = ?",
                (user_id, document_name),
            )
        else:
            conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()


def list_documents_with_history(user_id: int) -> List[str]:
    """Distinct document names the user has chatted about, most recent first."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT document_name, MAX(created_at) as last_used
            FROM chat_history
            WHERE user_id = ? AND document_name IS NOT NULL
            GROUP BY document_name
            ORDER BY last_used DESC
            """,
            (user_id,),
        )
        return [r["document_name"] for r in cur.fetchall()]
    finally:
        conn.close()
