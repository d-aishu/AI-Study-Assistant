"""
User authentication: registration and login against the local SQLite
users table. Passwords are hashed with bcrypt — never stored in plain text.
"""
import re
from typing import Optional, TypedDict

import bcrypt

from src.db import get_connection

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,50}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthUser(TypedDict):
    id: int
    username: str
    email: str


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # Malformed hash (shouldn't happen, but fail closed rather than crash)
        return False


def validate_registration(username: str, email: str, password: str, confirm_password: str) -> Optional[str]:
    """Returns an error message if invalid, else None."""
    if not username or not email or not password:
        return "All fields are required."
    if not USERNAME_RE.match(username):
        return "Username must be 3-50 characters (letters, numbers, ., _, - only)."
    if not EMAIL_RE.match(email):
        return "Please enter a valid email address."
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if password != confirm_password:
        return "Passwords do not match."
    return None


def register_user(username: str, email: str, password: str) -> int:
    """Creates a new user. Raises ValueError if the username/email is taken."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM users WHERE username = ? OR email = ?",
            (username, email),
        )
        if cur.fetchone():
            raise ValueError("That username or email is already registered.")

        cur.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, hash_password(password)),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[AuthUser]:
    """Returns the user dict on success, or None on invalid credentials."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, username, email, password_hash FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            return None
        return {"id": row["id"], "username": row["username"], "email": row["email"]}
    finally:
        conn.close()
