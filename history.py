"""
היסטוריית שיחה — הזיכרון של הבוט, ב-SQLite.

מקור: טבלת conversations + save_message / get_conversation_history ב-database.py
של ai-business-bot. שלושה שינויים מכוונים:
1. בלי שם משתמש ובלי מקורות — הזיכרון צריך רק תפקיד ותוכן (צמצום מידע אישי).
2. שאלה ותשובה נשמרות יחד, בטרנזקציה אחת ורק אחרי תשובה מוצלחת. במקור
   השאלה נשמרה לפני קריאת ה-AI, וכשל השאיר שאלה יתומה בהיסטוריה.
3. ההיסטוריה נשמרת רק HISTORY_RETENTION_DAYS ימים — מחיקה יומית, וגם
   השאילתה עצמה מסננת לפי גיל כדי שבין מחיקה למחיקה לא ייכנס לפרומפט ישן.

כל ה-API של המודול הוא ארבע פונקציות. מעבר לאחסון אחר (למשל זיכרון בלבד)
= החלפת הקובץ הזה בלבד.
"""

import sqlite3
from contextlib import contextmanager

from config import DB_PATH, HISTORY_RETENTION_DAYS


@contextmanager
def _connect():
    """חיבור חדש לכל פעולה — כמו get_connection בפרויקט המקור.

    הפונקציות כאן נקראות מתוך asyncio.to_thread, כלומר מ-threads שונים.
    חיבור SQLite אינו thread-safe, ולכן לא משתפים חיבור בין קריאות.
    """
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        # אין natural key: זה יומן הודעות שרק מתווספות, ו-id הוא המפתח.
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                message     TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON conversations(user_id, id);
            CREATE INDEX IF NOT EXISTS idx_conversations_created
                ON conversations(created_at);
            """
        )


def save_exchange(user_id: str, user_text: str, assistant_text: str) -> None:
    """שמירת שאלה + תשובה באותה טרנזקציה — או ששתיהן נשמרות או אף אחת."""
    with _connect() as conn:
        conn.executemany(
            "INSERT INTO conversations (user_id, role, message) VALUES (?, ?, ?)",
            [(user_id, "user", user_text), (user_id, "assistant", assistant_text)],
        )


def get_history(user_id: str, limit: int) -> list[dict]:
    """ההודעות האחרונות של המשתמש, מהישנה לחדשה, בפורמט של OpenAI."""
    if limit <= 0:
        return []
    with _connect() as conn:
        rows = conn.execute(
            """SELECT role, message FROM conversations
               WHERE user_id = ? AND created_at >= datetime('now', ?)
               ORDER BY id DESC LIMIT ?""",
            (user_id, f"-{HISTORY_RETENTION_DAYS} days", limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["message"]} for r in reversed(rows)]


def purge_old_history() -> int:
    """מחיקת הודעות מעבר לתקופת השמירה. מחזיר כמה שורות נמחקו."""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM conversations WHERE created_at < datetime('now', ?)",
            (f"-{HISTORY_RETENTION_DAYS} days",),
        )
        return cur.rowcount
