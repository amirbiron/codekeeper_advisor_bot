"""
הגדרות הבוט — כל הערכים נקראים ממשתני סביבה פעם אחת, בעליית התהליך.

מקור: config.py בריפו ai-business-bot, אחרי הסרת כל מה שקשור לעסק, לפאנל,
ל-WhatsApp ול-multi-tenant.
"""

import math
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ─── טלגרם ───────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

# ─── OpenAI ──────────────────────────────────────────────────────────────────
# OPENAI_API_KEY (ואופציונלית OPENAI_BASE_URL) נקראים ישירות ע"י ה-SDK.
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini").strip()
# ברירת המחדל של ה-SDK היא 600 שניות — thread שתקוע כל הזמן הזה חוסם את
# הגולש. 45 שניות כמו בפרויקט המקור.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
LLM_MAX_COMPLETION_TOKENS = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "1500"))
# שני הפרמטרים הבאים לא נשלחים כשהם ריקים — לא כל מודל מקבל אותם, ומודל
# שדוחה פרמטר מחזיר 400 על כל הודעה. ריק = ברירת המחדל של המודל.
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE", "").strip()
OPENAI_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT", "").strip()

# ─── מסמך הידע ───────────────────────────────────────────────────────────────
KNOWLEDGE_FILE = Path(
    os.getenv("KNOWLEDGE_FILE", str(BASE_DIR / "knowledge" / "codekeeper.md"))
).resolve()

# ─── היסטוריית שיחה (זיכרון הבוט) ────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data"))).resolve()
DB_PATH = Path(os.getenv("DB_PATH", str(DATA_DIR / "history.db"))).resolve()
# כמה הודעות אחרונות נכנסות לפרומפט (זוג שאלה+תשובה = 2 הודעות)
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "10"))
# ההיסטוריה קיימת רק בשביל הזיכרון של הבוט — מעבר לטווח הזה היא נמחקת
HISTORY_RETENTION_DAYS = int(os.getenv("HISTORY_RETENTION_DAYS", "7"))

# ─── יצירת קשר ───────────────────────────────────────────────────────────────
# קישור קבוע שמוצג ככפתור כשאין תשובה במסמך או כשגולש מבקש לדבר עם המפתח.
# למשל https://t.me/<username>. ריק = בלי כפתור.
CONTACT_URL = os.getenv("CONTACT_URL", "").strip()

# ─── שאלות המשך ──────────────────────────────────────────────────────────────
FOLLOW_UP_ENABLED = os.getenv("FOLLOW_UP_ENABLED", "true").lower() in ("true", "1", "yes")

# ─── הגבלת קצב (פר משתמש) ────────────────────────────────────────────────────
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "50"))
RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "100"))


def validate_config() -> list[str]:
    """בדיקת תצורה בעלייה. רשימה ריקה = תקין.

    בבוט הזה ה-AI הוא כל הפיצ'ר, ולכן חוסר במפתח או במסמך הוא סיבה לא
    לעלות בכלל — עדיף כשל ברור בלוג הפריסה מבוט שעונה "תקלה" לכל גולש.
    """
    errors: list[str] = []
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN לא מוגדר")
    if not os.getenv("OPENAI_API_KEY", "").strip():
        errors.append("OPENAI_API_KEY לא מוגדר")
    if not KNOWLEDGE_FILE.is_file():
        errors.append(f"קובץ הידע לא נמצא: {KNOWLEDGE_FILE}")
    # Bot API מקבל בכפתור url רק HTTP או tg:// — כתובת אחרת הייתה נכשלת
    # בכל תשובה שמציגה את הכפתור, לא בעלייה.
    if CONTACT_URL and not CONTACT_URL.startswith(("https://", "http://", "tg://")):
        errors.append("CONTACT_URL חייב להתחיל ב-https:// או ב-tg://")
    # הטווח לפי ה-SDK: "between 0 and 2". float() מקבל גם nan ו-inf, ו-OpenAI
    # היה דוחה כל הודעה. מקור: openai/resources/chat/completions/completions.py
    if LLM_TEMPERATURE:
        try:
            temperature = float(LLM_TEMPERATURE)
        except ValueError:
            temperature = math.nan  # לא מספר — נדחה בבדיקה שבשורה הבאה
        if not (math.isfinite(temperature) and 0 <= temperature <= 2):
            errors.append("LLM_TEMPERATURE חייב להיות מספר בין 0 ל-2")
    # ערך שלילי יוצר ב-SQLite את המשנה '--1 days', שמחזיר NULL: ההיסטוריה לא
    # מוצגת וגם לא נמחקת לעולם — בניגוד למה שהבוט מבטיח לגולש ב-/start.
    if HISTORY_RETENTION_DAYS < 0:
        errors.append("HISTORY_RETENTION_DAYS לא יכול להיות שלילי")
    return errors
