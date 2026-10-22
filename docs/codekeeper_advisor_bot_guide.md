# מדריך מימוש: בוט "יועץ CodeKeeper" לטלגרם

::: abstract
בוט טלגרם שעונה לגולשים ששוקלים את CodeKeeper על השאלה אחת שמעניינת אותם: "זה יתאים לאופן שבו אני עובד?". הוא עונה מתוך מסמך ידע אחד, זוכר את רצף השיחה, ואין לו פאנל ניהול. הקוד חולץ מהריפו `ai-business-bot`, צומצם למה שהבוט הזה צריך, ותוקנו בו שבעה באגים שקיימים במקור. כל קובץ במדריך הוא קוד שרץ, וטסט מוודא שהוא זהה לקובץ בריפו: 38 טסטים עוברים, כל טסט הוכח כמסוגל להיכשל, והזרימה המלאה הורצה עם הספריות האמיתיות של OpenAI וטלגרם מול שרתים מקומיים.
:::

**נבדק מול:** python-telegram-bot 22.8, openai 3.36.1, Python 3.13 — אוקטובר 2026. הגרסאות נעוצות ב-`requirements.txt`.

---

## ההכרעה המרכזית: המסמך כולו בפרומפט, בלי RAG

בבוט העסקי ה-AI מקבל ידע בשיטת RAG: בסיס הידע מחולק לקטעים, כל קטע הופך לווקטור, ובכל שאלה נשלפים עשרת הקטעים הכי דומים לה ונכנסים לפרומפט. זה הכרחי כשיש מאות רשומות — אי אפשר לשלוח את כולן בכל הודעה.

לבוט הזה יש מסמך אחד. ה-README של Ck, למשל, שוקל 9KB. במצב כזה RAG לא רק מיותר — הוא מזיק. גולש שכותב "אני עובד מטאבלט עם סוכן AI, זה יעזור לי?" צריך תשובה שמחברת כמה סעיפים שונים במסמך: מה יש ב-Webapp, מה יש בבוט, איך הסוכן מתחבר. חיפוש דמיון יחזיר את הקטעים שהכי דומים לניסוח של השאלה, ולא בהכרח את אלה שעונים עליה. כשהמסמך כולו בפרומפט, המודל רואה את התמונה המלאה בכל שאלה.

**RAG פותר בעיה של כמות. לבוט הזה אין בעיה של כמות.**

ההכרעה הזו מורידה מהפרויקט כ-1,000 שורות (כל התיקייה `rag/`), שלוש תלויות (faiss-cpu, numpy, tiktoken), קריאת embeddings בכל שאלה ושתי טבלאות.

### ומה עם העלות של לשלוח את המסמך בכל הודעה?

OpenAI שומרת במטמון את תחילת הפרומפט ומחייבת עליה מחיר מוזל. לפי התיעוד, זה פועל אוטומטית במודלים הנתמכים, בתנאי אחד: התחילית צריכה להיות זהה לחלוטין בין קריאות ("Cache reuse requires the entire rendered prefix to match"). ההנחיה שם היא לשים קודם את ההוראות ואת חומר הרקע הקבוע, ורק אחריהם את מה שמשתנה.

לכן הפרומפט בבוט הזה סטטי לגמרי: הוראות ← מסמך הידע ← היסטוריית השיחה ← השאלה. אין בו שעה, אין בו שם הגולש. בבוט העסקי, לעומת זאת, מידע שעות הפעילות יושב בתוך הודעת ה-system וכולל את השעה הנוכחית ברמת דקה (`business_hours.get_hours_context_for_llm`). אם תכניסו לכאן משהו כזה לפני המסמך, המסמך כולו ייצא מהמטמון בכל דקה.

**דוגמה למספרים.** במודל `gpt-5.4-mini` (ברירת המחדל כאן), לפי עמוד המודל של OpenAI באוקטובר 2026: קלט רגיל עולה $0.75 למיליון טוקנים, וקלט מהמטמון $0.075. אם ההוראות והמסמך יחד הם, נניח, 5,000 טוקנים, החלק הזה עולה בכל שאלה כ-0.04 סנט, כלומר יותר מ-2,500 שאלות לדולר, לפני עלות התשובה עצמה.

::: tip
איך יודעים שהמטמון עובד בפועל: כל קריאה כותבת ללוג שורת `LLM diag` עם השדה `cached_tokens`. מההודעה השנייה והלאה המספר צריך להיות גדול מאפס. אם הוא נשאר 0, משהו בתחילית משתנה בין קריאות.
:::

### מתי כן לעבור ל-RAG

כשהידע גדל לעשרות מסמכים או למאות קילובייטים. חלון ההקשר של `gpt-5.4-mini` הוא 400 אלף טוקנים, כך שמסמך מוצר אחד רחוק מהגבול. כשתגיעו לשם, המדריך `docs/rag_extraction_guide.md` בריפו `ai-business-bot` מכסה את החילוץ המלא.

---

## מה קורה כשגולש שולח הודעה

1. טלגרם מעביר את ההודעה. רק צ'אט פרטי — בקבוצות הבוט שותק.
2. הגבלת קצב: 10 הודעות בדקה, 50 בשעה, 100 ביום לכל גולש. כל קריאה ל-AI עולה כסף.
3. מנעול לכל גולש: גולשים שונים נענים במקביל, אבל הודעות של אותו גולש נענות אחת-אחת ובסדר.
4. היסטוריה: עשר ההודעות האחרונות של הגולש מ-SQLite, רק מהשבוע האחרון.
5. קריאה ל-OpenAI: הוראות ומסמך ← היסטוריה ← השאלה.
6. פענוח התשובה: הטוקן `[CONTACT]` הופך לכפתור "לדבר עם המפתח", ובלוק `[שאלות_המשך: ...]` הופך לכפתורים.
7. ניקוי HTML לטלגרם, פיצול אם התשובה ארוכה מ-4096 תווים, שליחה.
8. שמירת השאלה והתשובה יחד, בטרנזקציה אחת.
9. שאלות ההמשך נשלחות בהודעה נפרדת. לחיצה על אחת מהן עוברת באותו צינור בדיוק, משלב 2.

---

## מבנה הפרויקט

```
codekeeper-advisor-bot/
├── main.py              ← עלייה, רישום handlers, הרצה
├── config.py            ← משתני סביבה + ולידציה
├── llm.py               ← פרומפט, קריאה ל-OpenAI, פענוח התשובה
├── history.py           ← זיכרון השיחה ב-SQLite
├── rate_limiter.py      ← הגבלת קצב פר גולש
├── telegram_format.py   ← סניטציית HTML, פיצול הודעות
├── handlers.py          ← /start, הודעות, שאלות המשך, שגיאות
├── knowledge/
│   └── codekeeper.md    ← מסמך הידע
├── tests/
├── docs/
│   └── codekeeper_advisor_bot_guide.md   ← המדריך הזה
├── requirements.txt
├── pytest.ini
├── .env.example
├── .gitignore
├── render.yaml          ← פריסה ל-Render: עובד רקע + דיסק קבוע
└── CLAUDE.md            ← הנחיות לסשנים של Claude Code
```

| קובץ | מאיפה בריפו `ai-business-bot` | מה השתנה |
|---|---|---|
| `config.py` | `config.py` | נשארו רק טלגרם, OpenAI, היסטוריה, קצב ויצירת קשר |
| `llm.py` | `llm.py`, `llm_client.py`, `openai_client.py`, `config.build_system_prompt` | בלי RAG, בלי בחירת ספק, בלי זיכרון לקוח, בלי סיכומי שיחה. פרמטרים עודכנו ל-SDK הנוכחי |
| `history.py` | `database.py` — טבלת `conversations` | בלי שם משתמש, שמירה אטומית של זוג, מחיקה אחרי 7 ימים |
| `rate_limiter.py` | `rate_limiter.py` | בלי tenant, בלי live chat, בלי רשימת חסומים |
| `telegram_format.py` | `llm.sanitize_telegram_html` | נוספו פיצול הודעות ו-fallback לטקסט נקי |
| `handlers.py` | `bot/handlers.py` | נשארו 4 handlers מתוך עשרות. לוג השגיאות תוקן |
| `main.py` | `bot/telegram_bot.py`, `main.py` | עיבוד מקבילי, השתקת לוג httpx, polling בלבד |

---

## 1. תלויות והגדרות

**`requirements.txt`**

```text
# גרסאות נעוצות — אלה הגרסאות שהקוד נבדק מולן (אוקטובר 2026).
# openai כבר בגרסה 3; שדרוג לא נעוץ עלול לשבור API בשקט.
python-telegram-bot[job-queue]==22.8
openai==3.36.1
python-dotenv==1.2.4
pytest==9.1.1
```

**`.env.example`**

```bash
# חובה
TELEGRAM_BOT_TOKEN=
OPENAI_API_KEY=

# קישור לכפתור "לדבר עם המפתח" (https:// או tg://). ריק = בלי כפתור
CONTACT_URL=https://t.me/your_username

# אופציונלי — ברירות המחדל בסוגריים
# OPENAI_MODEL=gpt-5.4-mini
# LLM_MAX_COMPLETION_TOKENS=1500
# LLM_TIMEOUT_SECONDS=45
# LLM_TEMPERATURE=            (ריק = לא נשלח)
# OPENAI_REASONING_EFFORT=    (ריק = לא נשלח)
# KNOWLEDGE_FILE=knowledge/codekeeper.md
# DATA_DIR=data
# CONTEXT_WINDOW_SIZE=10
# HISTORY_RETENTION_DAYS=7
# FOLLOW_UP_ENABLED=true
# RATE_LIMIT_PER_MINUTE=10
# RATE_LIMIT_PER_HOUR=50
# RATE_LIMIT_PER_DAY=100
```

**`pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
```

שני דברים לא נכנסים לגיט: `.env` עם הטוקן והמפתח, ו-`data/` עם היסטוריית השיחות של הגולשים.

**`.gitignore`**

```gitignore
# סודות — הטוקן ומפתח OpenAI
.env

# היסטוריית השיחות (SQLite) — נתוני גולשים, לא קוד
data/

# סביבות ותוצרים
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
```

---

## 2. `config.py`

שתי החלטות כאן חורגות מהמקור. **הראשונה:** `LLM_TEMPERATURE` ו-`OPENAI_REASONING_EFFORT` לא נשלחים כשהם ריקים. `gpt-5.4-mini` הוא מודל reasoning, ותיעוד המודל לא מציין אם הוא מקבל `temperature`. מודל שדוחה פרמטר מחזיר שגיאה על כל הודעה, אז ברירת המחדל הבטוחה היא לא לשלוח אותו. אותו כלל כבר קיים במקור, בענף של Claude ב-`llm_client.py`.

**השנייה:** הבוט לא עולה בכלל בלי טוקן, בלי מפתח OpenAI או בלי מסמך ידע. בבוט העסקי ה-AI הוא פיצ'ר אחד מתוך רבים, ולכן כשל שלו לא מפיל את השרת. כאן ה-AI הוא כל הפיצ'ר. עדיף שגיאה ברורה בלוג הפריסה מבוט שעונה "תקלה" לכל גולש.

**`config.py`**

```python
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
```

---

## 3. `llm.py` — הלב

### הפרומפט מסביר, לא רק מורה

הפרומפט אומר למודל מי הגולשים, מה הם מחפשים, ולמה כנות חשובה יותר ממכירה. מודל שמבין את ההקשר מטפל טוב יותר במקרים שאף כלל לא צפה. דוגמה: גולש כותב "אני עובד רק עם VS Code מקומי, בלי AI". מודל שקיבל רק את הכלל "ענה מהמסמך" ימנה פיצ'רים. מודל שהבין שהמטרה היא החלטה מושכלת יגיד בכנות אם יש כאן ערך בשבילו.

### הטוקן `[CONTACT]` — המודל מסמן, הקוד מוסיף את הקישור

זה אותו מנגנון כמו `HANDOFF_MARKER` במקור. כשאין תשובה במסמך, המודל פותח את התשובה במחרוזת קבועה, והקוד בודק `startswith` בלבד. אין חיפוש של ניסוחים כמו "אין לי מידע" — בבוט העסקי ניסו את זה, וזה סימן גם תשובות תמימות כבקשה להעברה.

הקישור עצמו לא עובר דרך המודל. המודל לא יודע את הכתובת, ומודל שמתבקש לכתוב קישור עלול להמציא אחד. הכפתור נבנה בקוד מתוך `CONTACT_URL`.

### פרמטרים שהשתנו מאז הבוט העסקי

| מה | במקור | כאן | מקור |
|---|---|---|---|
| תקרת אורך | `max_tokens` | `max_completion_tokens` | ב-SDK: "now deprecated in favor of `max_completion_tokens`" |
| תפקיד ההוראות | `system` | `developer` | ב-SDK: "With o1 models and newer, `developer` messages replace the previous `system` messages" |
| מודל ברירת מחדל | `gpt-4.1-mini` | `gpt-5.4-mini` | ב-SDK: ל-`gpt-4.1-mini` נקבע תאריך כיבוי, 2027-01-14 |
| `temperature` | תמיד 0.3 | רק כשהוגדר | ראו `config.py` למעלה |

### ולידציה של התשובה

ב-SDK השדה `content` מוגדר `Optional`, ולצידו יש שדה `refusal` נפרד לסירוב. הקוד בודק את זה במפורש ולא סומך על `try/except` שיתפוס `AttributeError`. תקלה מסומנת `ok=False`: הגולש מקבל "נסו שוב בעוד רגע", וההודעה לא נשמרת בזיכרון.

**זה שונה מהמקור.** שם תקלת API הופכת ל-`FALLBACK_RESPONSE` ונספרת כ"הבוט לא ידע לענות". כשהשרת של OpenAI נופל, הגולש מתבקש "לנסח את השאלה אחרת", וההודעה הזו נשמרת בהיסטוריה.

### לוג אבחון — מספרים בלבד

שורת `LLM diag` רושמת מודל, סיבת סיום, טוקנים (כולל `cached_tokens`) ואורך. בלי תוכן: במקור נרשמו 80 התווים האחרונים של התשובה, והתשובה יכולה לצטט את מה שהגולש כתב.

**`llm.py`**

```python
"""
שכבת ה-AI — בניית הפרומפט, קריאה ל-OpenAI וניקוי התשובה.

מקור: llm.py + llm_client.py + openai_client.py + config.build_system_prompt
בריפו ai-business-bot. מה הוסר: RAG (המסמך כולו נכנס לפרומפט), בחירת ספק
פר-tenant, זיכרון לקוח, סיכומי שיחה, ציון מקור וזיהוי כוונות.
"""

import logging
import re
import threading
from dataclasses import dataclass, field

import openai

from config import (
    FOLLOW_UP_ENABLED,
    KNOWLEDGE_FILE,
    LLM_MAX_COMPLETION_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
)

logger = logging.getLogger(__name__)

# טוקן שהמודל שם בתחילת התשובה כשאין לו תשובה במסמך או כשהגולש מבקש לדבר
# עם המפתח. זה אותו מנגנון כמו HANDOFF_MARKER בפרויקט המקור: זיהוי
# דטרמיניסטי של startswith, בלי חיפוש ניסוחים ("אין לי מידע" וכו') שמייצר
# זיהויים שגויים.
CONTACT_MARKER = "[CONTACT]"

FALLBACK_ERROR = "מצטער, הייתה תקלה זמנית. נסו לשלוח את השאלה שוב בעוד רגע 🙏"
# כשהמודל שלח רק את הטוקן בלי טקסט
CONTACT_DEFAULT_TEXT = "על זה כדאי לשאול ישירות את המפתח 👇"


# ─── הפרומפט ─────────────────────────────────────────────────────────────────

_FORMATTING_RULES = (
    "חוק ברזל: השתמש אך ורק בתגי HTML של טלגרם. אסור בהחלט להשתמש בתחביר Markdown "
    "(כוכביות, קווים תחתונים, סולמיות).\n"
    "תגים מותרים:\n"
    "- תג b (פתיחה וסגירה) — לכותרות ולשמות של פיצ'רים\n"
    "- תג i (פתיחה וסגירה) — להערות משניות, הבהרות ותנאים\n"
    "- תג code (פתיחה וסגירה) — לשמות של פקודות, קבצים וכלים\n"
    "- רשימות עם תבליטים (•) כשיש כמה פריטים\n"
    "- רווח ברור בין פסקאות ונושאים שונים\n"
    "דוגמה נכונה: <b>שמירת קבצים</b> — <i>כולל היסטוריית גרסאות</i>\n"
    "דוגמה שגויה: **שמירת קבצים** — *כולל היסטוריית גרסאות*\n"
    "אם תשתמש בכוכביות (*) או בקווים תחתונים (_) להדגשה — התשובה תיחשב שגויה."
)

_FOLLOW_UP_RULE = (
    "\n── שאלות המשך ──\n"
    "בסוף כל תשובה (חוץ מתשובה שנפתחת ב-" + CONTACT_MARKER + "), הוסף בשורה נפרדת "
    "2-3 שאלות המשך קצרות (עד 6 מילים) שהגולש עשוי לרצות לשאול, בפורמט הזה בדיוק:\n"
    "[שאלות_המשך: שאלה ראשונה | שאלה שנייה | שאלה שלישית]\n"
    "הצע רק שאלות שהתשובה עליהן מופיעה במסמך הידע, ואל תציע שאלה שכבר נענתה בשיחה.\n"
)


def build_system_prompt(knowledge: str, follow_up_enabled: bool = FOLLOW_UP_ENABLED) -> str:
    """הפרומפט כולו סטטי — הוראות ואז מסמך הידע.

    חשוב לעלות: OpenAI שומרת במטמון את תחילת הפרומפט ומחייבת עליה מחיר מוזל,
    אבל רק כשהתחילית זהה לחלוטין בין קריאות. לכן אסור להכניס לכאן שום דבר
    שמשתנה (שעה, שם הגולש) — כל דבר דינמי בא אחרי, בהודעות השיחה.
    """
    follow_up_rule = _FOLLOW_UP_RULE if follow_up_enabled else ""
    return f"""אתה היועץ הדיגיטלי של CodeKeeper (בקיצור Ck), ואתה משוחח עם גולשים בטלגרם.

── מי פונה אליך ולמה ──
הפונים אליך הם מפתחים ואנשים שעובדים עם קוד ועם סוכני AI, ששמעו על CodeKeeper ושוקלים אם להשתמש בו. רובם עוד לא יודעים בדיוק מה המוצר עושה, ואין להם סבלנות לקרוא תיעוד ארוך. הם באים עם שאלה אחת: "האם זה יעזור לי, בדרך שבה אני עובד?".

── המטרה שלך ──
לעזור לגולש להגיע להחלטה מושכלת — לא למכור לו בכל מחיר.
- כשהגולש מתאר את אופן העבודה שלו, חבר את התיאור לפיצ'רים ספציפיים מהמסמך, והסבר איך כל פיצ'ר פוגש צורך שהוא תיאר. דוגמה קונקרטית מהעולם שלו משכנעת יותר מרשימת פיצ'רים.
- כשחסר לך מידע כדי לענות אם המוצר מתאים, שאל שאלה אחת ממוקדת על אופן העבודה שלו (למשל: עם אילו כלים הוא עובד, מאיזה מכשיר, לבד או בצוות). שאלה אחת בכל פעם, לא שאלון.
- כש-Ck לא מתאים לצורך שהגולש תיאר, או שהמסמך לא מזכיר יכולת שהוא מחפש, אמור את זה בכנות. גולש שקיבל "לא" כן חוזר כשהצורך שלו משתנה; גולש שקיבל הבטחה שלא התממשה לא חוזר.

── מקור הידע: מסמך אחד בלבד ──
בסוף ההודעה הזו מופיע מסמך הידע של CodeKeeper. הוא המקור היחיד שלך לעובדות על המוצר: יכולות, מחירים, מגבלות, התקנה. אל תשלים מידע על Ck מהידע הכללי שלך על מוצרים דומים, גם כשההשלמה נראית סבירה — גולש שיסתמך על יכולת שלא קיימת ייפגע, והאמון במוצר ייפגע איתו. ידע כללי מותר רק כדי להסביר מושגים (למשל מה זה MCP), לא כדי לטעון טענות על Ck.

── כשאין תשובה במסמך, או כשהגולש רוצה לדבר עם המפתח ──
אם התשובה לא נמצאת במסמך, או שהגולש מבקש לדבר עם אדם או עם המפתח, פתח את התשובה בדיוק במחרוזת {CONTACT_MARKER} (כולל הסוגריים המרובעים, בשורה משלה). אחריה כתוב משפט או שניים: מה כן ידוע מהמסמך אם יש משהו רלוונטי, ושאפשר לפנות ישירות למפתח. אל תכתוב קישור או שם משתמש — כפתור יצירת קשר מצורף להודעה אוטומטית. אל תכתוב את {CONTACT_MARKER} בשום מצב אחר.

── עיצוב טקסט (חובה!) ──
{_FORMATTING_RULES}
{follow_up_rule}
── סגנון ──
- דבר תמיד על עצמך בלשון זכר ("אני יכול", "אשמח לעזור"), לעולם לא בלשון נקבה.
- ענה באותה שפה שבה הגולש כותב.
- הגולשים קוראים בטלפון: עד שלוש פסקאות קצרות, אלא אם ביקשו פירוט.
- אם שואלים אם אתה בוט — אתה העוזר הדיגיטלי של CodeKeeper, שעונה מתוך התיעוד של המוצר.
- שאלות שאינן קשורות ל-CodeKeeper או לאופן העבודה של הגולש — הפנה בעדינות חזרה לנושא.

── מסמך הידע ──
{knowledge}
── סוף מסמך הידע ──"""


_system_prompt: str | None = None


def init_llm() -> None:
    """טעינת מסמך הידע ובניית הפרומפט — פעם אחת, בעלייה.

    קובץ חסר או ריק זורק חריגה: בוט בלי מסמך ידע לא יכול לענות על כלום.
    """
    global _system_prompt
    knowledge = KNOWLEDGE_FILE.read_text(encoding="utf-8").strip()
    if not knowledge:
        raise RuntimeError(f"קובץ הידע ריק: {KNOWLEDGE_FILE}")
    _system_prompt = build_system_prompt(knowledge)
    logger.info("Knowledge loaded: %d chars", len(knowledge))


def build_messages(system_prompt: str, history: list[dict], user_text: str) -> list[dict]:
    """הסדר קבוע: הוראות + מסמך (סטטי, נשמר במטמון) ← היסטוריה ← השאלה."""
    # role=developer ולא system: לפי ה-SDK, "With o1 models and newer, `developer`
    # messages replace the previous `system` messages".
    # מקור: openai/types/chat/chat_completion_developer_message_param.py
    messages = [{"role": "developer", "content": system_prompt}]
    messages.extend({"role": m["role"], "content": m["content"]} for m in history)
    messages.append({"role": "user", "content": user_text})
    return messages


# ─── client ──────────────────────────────────────────────────────────────────

_client: openai.OpenAI | None = None
_client_lock = threading.Lock()


def get_openai_client() -> openai.OpenAI:
    """singleton עצל עם double-checked locking — כמו openai_client.py במקור.

    ה-SDK קורא בעצמו את OPENAI_API_KEY ואת OPENAI_BASE_URL מהסביבה.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = openai.OpenAI(timeout=LLM_TIMEOUT_SECONDS)
    return _client


# ─── פענוח התשובה ────────────────────────────────────────────────────────────

# תבנית ראשית: [שאלות_המשך: שאלה1 | שאלה2 | שאלה3]
# תבנית חלופית: אותו דבר בלי סוגריים מרובעים (מודלים לפעמים משמיטים אותם)
_FOLLOW_UP_PATTERN = re.compile(r"\[שאלות[_ ]המשך:\s*(.+?)\]")
_FOLLOW_UP_PATTERN_ALT = re.compile(r"שאלות[_ ]המשך:\s*(.+?)(?:\n|$)")


def extract_follow_up_questions(text: str) -> list[str]:
    """חילוץ עד 3 שאלות המשך מתשובת המודל. אין התאמה = רשימה ריקה."""
    match = _FOLLOW_UP_PATTERN.search(text) or _FOLLOW_UP_PATTERN_ALT.search(text)
    if not match:
        return []
    questions = [q.strip() for q in match.group(1).split("|") if q.strip()]
    return questions[:3]


def strip_follow_up_questions(text: str) -> str:
    """הסרת בלוק שאלות ההמשך (כולל שורות ריקות לפניו) לפני שליחה לגולש."""
    text = re.sub(r"\n*\[שאלות[_ ]המשך:\s*.*?\]", "", text)
    text = re.sub(r"\n*שאלות[_ ]המשך:\s*.+?(?:\n|$)", "\n", text)
    return text.strip()


def split_contact_marker(text: str) -> tuple[bool, str]:
    """(האם המודל ביקש כפתור יצירת קשר, הטקסט בלי הטוקן).

    הזיהוי — רק בתחילת התשובה. ההסרה — מכל מקום, כי הטוקן הוא אות פנימי
    בין המודל לקוד ואסור שיגיע לגולש גם כשהמודל שם אותו במקום הלא נכון.
    """
    needs_contact = text.lstrip().startswith(CONTACT_MARKER)
    return needs_contact, text.replace(CONTACT_MARKER, "").strip()


def trim_to_last_sentence(text: str) -> str:
    """חיתוך תשובה שנעצרה בתקרת הטוקנים לגבול בטוח (שורה או משפט שלמים).

    מקור: llm._trim_to_last_sentence — תיקון לבאג חוזר שבו רשימה נקטעה באמצע
    מילה. שורה אחרונה שלא מסתיימת בסימן פיסוק נחתכת; בשורה בודדת חותכים
    לסוף המשפט האחרון.
    """
    text = (text or "").rstrip()
    if not text:
        return ""
    lines = text.split("\n")
    if len(lines) > 1:
        last = lines[-1].rstrip()
        if last and last[-1] not in ".!?:;":
            return "\n".join(lines[:-1]).rstrip() + "\n…"
        return text + "\n…"
    for ch in (".", "!", "?"):
        idx = text.rfind(ch)
        if idx >= len(text) // 3:  # לפחות שליש מהטקסט נשאר
            return text[: idx + 1] + " …"
    return text + "…"


# ─── הפקת תשובה ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Answer:
    text: str                      # HTML של טלגרם, לפני סניטציה
    ok: bool                       # False = תקלה; text הוא הודעת התקלה ולא נשמר בהיסטוריה
    needs_contact: bool = False    # להציג כפתור יצירת קשר
    follow_up_questions: list[str] = field(default_factory=list)


def generate_answer(history: list[dict], user_text: str) -> Answer:
    """קריאה אחת ל-OpenAI. סינכרוני — הקורא מריץ אותו ב-asyncio.to_thread."""
    if _system_prompt is None:
        raise RuntimeError("init_llm() was not called")

    kwargs = {}
    # שני הפרמטרים נשלחים רק כשהוגדרו במפורש (ראה config.py)
    if LLM_TEMPERATURE:
        kwargs["temperature"] = float(LLM_TEMPERATURE)
    if OPENAI_REASONING_EFFORT:
        kwargs["reasoning_effort"] = OPENAI_REASONING_EFFORT

    try:
        # max_completion_tokens ולא max_tokens: ב-SDK מסומן max_tokens כ-"now
        # deprecated in favor of max_completion_tokens".
        # מקור: openai/resources/chat/completions/completions.py, create()
        response = get_openai_client().chat.completions.create(
            model=OPENAI_MODEL,
            messages=build_messages(_system_prompt, history, user_text),
            max_completion_tokens=LLM_MAX_COMPLETION_TOKENS,
            **kwargs,
        )
    except openai.OpenAIError as e:
        # מחלקת הבסיס של כל שגיאות ה-SDK (APIError, timeout, חיבור, אימות).
        # הודעת השגיאה מגיעה מ-OpenAI ולא מכילה את תוכן השיחה.
        logger.error("OpenAI call failed: %s: %s", type(e).__name__, e)
        return Answer(text=FALLBACK_ERROR, ok=False)

    # ולידציה של התשובה — content יכול להיות None (למשל refusal)
    choice = response.choices[0] if response.choices else None
    content = choice.message.content if choice is not None else None
    finish_reason = choice.finish_reason if choice is not None else None
    if not isinstance(content, str) or not content.strip():
        logger.error("OpenAI returned no text (finish_reason=%s)", finish_reason)
        return Answer(text=FALLBACK_ERROR, ok=False)

    # לוג אבחון — מספרים בלבד. בלי תוכן התשובה: היא יכולה לצטט את הגולש.
    usage = response.usage
    details = usage.prompt_tokens_details if usage is not None else None
    cached_tokens = details.cached_tokens if details is not None and details.cached_tokens is not None else 0
    logger.info(
        "LLM diag: model=%s finish_reason=%s prompt_tokens=%s cached_tokens=%s "
        "completion_tokens=%s chars=%d",
        response.model, finish_reason,
        usage.prompt_tokens if usage is not None else -1, cached_tokens,
        usage.completion_tokens if usage is not None else -1, len(content),
    )

    text = content.strip()
    if finish_reason == "length":
        logger.warning("Answer hit max_completion_tokens=%d — trimming", LLM_MAX_COMPLETION_TOKENS)
        text = trim_to_last_sentence(text)

    follow_ups: list[str] = []
    if FOLLOW_UP_ENABLED:
        follow_ups = extract_follow_up_questions(text)
        text = strip_follow_up_questions(text)

    needs_contact, text = split_contact_marker(text)
    if needs_contact:
        # אין טעם להציע שאלות המשך כשאין למודל מידע לענות
        follow_ups = []
        text = text or CONTACT_DEFAULT_TEXT
    if not text:
        return Answer(text=FALLBACK_ERROR, ok=False)
    return Answer(text=text, ok=True, needs_contact=needs_contact, follow_up_questions=follow_ups)
```

---

## 4. `history.py` — הזיכרון של הבוט

הטבלה זהה כמעט לטבלת `conversations` במקור, עם שלושה שינויים:

**בלי שם הגולש.** הזיכרון צריך רק תפקיד ותוכן. שם שלא נשמר לא יכול לדלוף.

**שאלה ותשובה נשמרות יחד, רק אחרי הצלחה.** במקור השאלה נשמרת לפני הקריאה ל-AI, והתשובה אחריה. כאן שתיהן נכנסות באותה טרנזקציה — או ששתיהן נשמרות, או אף אחת. הטסט `test_exchange_is_atomic` מכשיל את השורה השנייה בכוונה ובודק שגם הראשונה בוטלה.

**7 ימים ולא יותר.** ההיסטוריה קיימת רק בשביל הזיכרון של הבוט, אז אין סיבה להחזיק אותה מעבר לזה. משימה יומית מוחקת הודעות ישנות, והשאילתה עצמה מסננת לפי גיל. בלי הסינון, הודעה בת 7 ימים ושעה הייתה נכנסת לפרומפט עד המחיקה הבאה.

כל הממשק של המודול הוא ארבע פונקציות. אם תרצו בעתיד זיכרון ב-RAM בלבד, מחליפים את הקובץ הזה ולא נוגעים בשום דבר אחר.

**`history.py`**

```python
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
```

---

## 5. `rate_limiter.py`

כמעט כמו המקור: שלושה חלונות זזים בזיכרון, ומחיקת המשתמש הכי ישן כשעוברים 10,000 משתמשים. המונים מתאפסים ב-restart — מקובל לבוט קטן.

נקודה אחת עדינה, כי הבוט כאן מעבד עדכונים במקביל: בין הבדיקה לרישום אין `await`. ב-asyncio, רק `await` מוסר את השליטה לקוד אחר, ולכן שתי הודעות לא יכולות לעבור יחד את אותה בדיקה. הטסט `test_guard_holds_under_concurrent_updates` שולח בבת אחת עדכון אחד יותר מהמגבלה לדקה, ובודק שבדיוק המגבלה עוברת. הוספת `await asyncio.sleep(0)` בין השתיים מכשילה אותו.

**`rate_limiter.py`**

```python
"""
הגבלת קצב פר משתמש, בזיכרון — שלושה חלונות זזים (דקה / שעה / יום).

מקור: rate_limiter.py בריפו ai-business-bot, בלי מפתח tenant, בלי live chat
ובלי רשימת חסומים. כל קריאה ל-AI עולה כסף, ולכן כל נתיב שמגיע ל-AI (הודעה
וגם לחיצה על שאלת המשך) עובר בדיקה + רישום.

המונים מתאפסים ב-restart — מקובל לבוט קטן: חלון ההתעללות מוגבל ממילא.
"""

import bisect
import logging
import time
from collections import OrderedDict, deque
from functools import wraps

from telegram import Update
from telegram.ext import ContextTypes

from config import RATE_LIMIT_PER_DAY, RATE_LIMIT_PER_HOUR, RATE_LIMIT_PER_MINUTE

logger = logging.getLogger(__name__)

# OrderedDict כ-LRU: מעל התקרה נמחקים המשתמשים שלא נראו הכי הרבה זמן
_MAX_TRACKED_USERS = 10_000
_user_timestamps: OrderedDict[str, deque[float]] = OrderedDict()

# (גודל חלון בשניות, מקסימום הודעות, הודעה לגולש)
_WINDOWS = [
    (60, RATE_LIMIT_PER_MINUTE, "קצב ההודעות מהיר מדי. המתינו כחצי דקה ונסו שוב"),
    (3600, RATE_LIMIT_PER_HOUR, "הגעתם למגבלת ההודעות לשעה הקרובה. אפשר להמשיך בתום השעה"),
    (86400, RATE_LIMIT_PER_DAY, "הגעתם למכסת ההודעות היומית. אפשר להמשיך מחר"),
]


def _bucket(user_id: str) -> deque[float]:
    if user_id in _user_timestamps:
        _user_timestamps.move_to_end(user_id)
    else:
        _user_timestamps[user_id] = deque()
        # eviction גם ב-check ולא רק ב-record — אחרת משתמשים חסומים מנפחים את המילון
        while len(_user_timestamps) > _MAX_TRACKED_USERS:
            _user_timestamps.popitem(last=False)
    return _user_timestamps[user_id]


def check_rate_limit(user_id: str) -> str | None:
    """הודעת חסימה אם המשתמש חרג, אחרת None. לא רושם — זה תפקיד record_message."""
    # שעון מונוטוני ולא שעון המערכת: שעון מערכת שזז קדימה היה מאפס את המכסה,
    # ושזז אחורה היה שובר את הסדר העולה שה-bisect שלמטה מניח.
    now = time.monotonic()
    timestamps = _bucket(user_id)
    while timestamps and timestamps[0] < now - 86400:
        timestamps.popleft()
    ts_list = list(timestamps)  # ממוין, כי תמיד מוסיפים זמן עולה
    for window_seconds, max_messages, message in _WINDOWS:
        count = len(ts_list) - bisect.bisect_left(ts_list, now - window_seconds)
        if count >= max_messages:
            # user_id של טלגרם הוא מזהה פנימי שלנו, לא PII כמו שם או טלפון
            logger.info("Rate limit hit for user %s: %d msgs in %ds", user_id, count, window_seconds)
            return message
    return None


def record_message(user_id: str) -> None:
    _bucket(user_id).append(time.monotonic())


def rate_limit_guard(handler):
    """דקורטור ל-handler של הודעות.

    check ו-record רצים ברצף בלי await ביניהם, ולכן גם כשטלגרם מעבד כמה
    עדכונים במקביל (concurrent_updates) אין חלון שבו שתי הודעות עוברות
    יחד את אותה בדיקה — ב-asyncio רק await מוסר את השליטה.
    """

    @wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if user is None:
            return await handler(update, context)
        user_id = str(user.id)
        limit_msg = check_rate_limit(user_id)
        if limit_msg is not None:
            if update.effective_message:
                await update.effective_message.reply_text(limit_msg)
            return
        record_message(user_id)
        return await handler(update, context)

    return wrapper
```

---

## 6. `telegram_format.py`

`sanitize_telegram_html` עובר כמו שהוא. נוספו שניים שסוגרים חורים במקור:

**fallback שבאמת נקי.** כשטלגרם דוחה HTML, המקור שולח את אותו טקסט בלי `parse_mode`, אבל הטקסט כבר עבר escape. הגולש רואה `<b>` ו-`&amp;` כטקסט. כאן ה-fallback מסיר תגים ומחזיר תווים אמיתיים.

**פיצול מעל 4096 תווים.** זו התקרה של טלגרם (`MessageLimit.MAX_TEXT_LENGTH` בספרייה). בנתיב התשובות במקור אין פיצול, ותשובה ארוכה נכשלת כולה. הפיצול כאן חותך בגבול פסקה, ואם אין כזה בגבול שורה, ואם אין כזה ברווח.

**`telegram_format.py`**

```python
"""
הכנת טקסט לשליחה בטלגרם — סניטציית HTML, גרסת טקסט רגיל, ופיצול הודעות ארוכות.

מקור: llm.sanitize_telegram_html בריפו ai-business-bot (כמו שהוא).
שני חדשים, שסוגרים חורים שקיימים במקור:
- html_to_plain: במקור, כשטלגרם דחה HTML, נשלח אותו טקסט בלי parse_mode —
  והגולש ראה תגים ו-&amp; כטקסט. כאן ה-fallback הוא טקסט נקי באמת.
- split_message: טלגרם דוחה הודעה מעל 4096 תווים. במקור אין פיצול, ותשובה
  ארוכה נכשלת כולה.
"""

import html as _html
import re

# Bot API: עד 4096 תווים להודעה.
# מקור: telegram/constants.py, MessageLimit.MAX_TEXT_LENGTH
TELEGRAM_MAX_LENGTH = 4096

# תגי HTML שטלגרם תומך בהם — רק אותם משחזרים אחרי ה-escape
_TELEGRAM_HTML_TAGS = {"b", "i", "u", "s", "code", "pre"}

_ESCAPED_TAG_RE = re.compile(
    r"&lt;(/?)(" + "|".join(_TELEGRAM_HTML_TAGS) + r")(\s[^&]*?)?&gt;"
)
_TAG_RE = re.compile(r"</?(?:" + "|".join(_TELEGRAM_HTML_TAGS) + r")>")


def sanitize_telegram_html(text: str) -> str:
    """פלט LLM ← HTML בטוח לטלגרם.

    קודם מבריח את כל התווים המיוחדים (&, <, >), ואז משחזר רק תגים שטלגרם
    תומך בהם. תג עם מאפיינים (כמו class) נמחק יחד עם תג הסגירה שלו, כדי
    שלא יישאר HTML שבור.
    """
    escaped = _html.escape(text, quote=False)
    orphan_counts: dict[str, int] = {}

    def _restore_or_strip(m: re.Match) -> str:
        slash, tag, attrs = m.group(1), m.group(2), m.group(3)
        if not slash and attrs:
            orphan_counts[tag] = orphan_counts.get(tag, 0) + 1
            return ""
        if slash and orphan_counts.get(tag, 0) > 0:
            orphan_counts[tag] -= 1
            return ""
        return f"<{slash}{tag}>"

    return _ESCAPED_TAG_RE.sub(_restore_or_strip, escaped)


def html_to_plain(sanitized: str) -> str:
    """ההפך מ-sanitize: בלי תגים, עם תווים אמיתיים במקום &lt; / &amp;."""
    return _html.unescape(_TAG_RE.sub("", sanitized))


def split_message(text: str, limit: int = TELEGRAM_MAX_LENGTH) -> list[str]:
    """פיצול לחלקים של עד limit תווים — בגבול פסקה, אחר כך שורה, אחר כך רווח.

    התגים שהמודל מתבקש להשתמש בהם (b/i/code) הם בתוך שורה, ולכן חיתוך
    בגבול פסקה או שורה לא שובר אותם. אם בכל זאת נשבר תג, טלגרם דוחה את
    החלק והשולח נופל לטקסט נקי (ראה handlers._send_text).
    """
    chunks: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        window = rest[:limit]
        cut = limit  # אין מפריד בחצי השני של החלון — חיתוך קשיח
        for sep in ("\n\n", "\n", " "):
            idx = window.rfind(sep)
            # מפריד בחצי הראשון היה מייצר חלקים קטנים מדי — מנסים את הבא
            if idx >= limit // 2:
                cut = idx
                break
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        chunks.append(rest)
    return chunks
```

---

## 7. `handlers.py`

### צינור אחד

גם הודעה שהגולש מקליד וגם לחיצה על שאלת המשך עוברות דרך `_answer_and_reply` בלבד — אותו עיקרון כמו `_handle_rag_query` במקור. שני נתיבים שמשכפלים לוגיקה מתפצלים בשקט ברגע שמעדכנים רק אחד מהם.

### מנעול לכל גולש

הבוט עונה לכמה גולשים במקביל (ראו `main.py`). בלי מנעול, גולש ששולח שתי הודעות ברצף מהיר היה מקבל שתי קריאות ל-AI שרצות יחד. שתיהן היו קוראות את אותה היסטוריה, ואף אחת לא הייתה רואה את השנייה. בנוסף, התשובות היו מגיעות בסדר שבו OpenAI סיים, לא בסדר שבו נשאלו. המנעול יושב ב-`context.user_data`, ש-python-telegram-bot מחזיק בנפרד לכל משתמש.

### לוג שגיאות בלי מידע אישי

במקור, `error_handler` כותב ללוג את כל אובייקט ה-`update`. בדקתי על python-telegram-bot 22.8: המחרוזת הזו כוללת את שם הגולש, את שם המשתמש שלו ואת טקסט ההודעה המלא. כאן נרשמים רק מזהה העדכון וה-traceback.

### שאלות המשך

`callback_data` של כפתור מוגבל ל-64 בתים, אז טקסט השאלה נשמר ב-`bot_data` והכפתור נושא רק מפתח. כפתורים שלא נלחצו נמחקים אחרי שעה. הרישום במגבלת הקצב קורה רק אחרי שווידאנו שהשאלה עוד קיימת — לחיצה על כפתור שפג לא אוכלת מכסה.

**`handlers.py`**

```python
"""
Handlers של טלגרם — /start, הודעת טקסט, לחיצה על שאלת המשך, ושגיאות.

מקור: bot/handlers.py בריפו ai-business-bot (_typing_indicator,
_reply_html_safe, _build_follow_up_keyboard, follow_up_callback,
error_handler). כל נתיב שמגיע ל-AI עובר דרך _answer_and_reply בלבד — אותו
עיקרון כמו _handle_rag_query במקור: צינור אחד, בלי שכפול.
"""

import asyncio
import logging
import sqlite3
import time
from contextlib import asynccontextmanager

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import ContextTypes

from config import CONTACT_URL, CONTEXT_WINDOW_SIZE, HISTORY_RETENTION_DAYS
from history import get_history, save_exchange
from llm import generate_answer
from rate_limiter import check_rate_limit, rate_limit_guard, record_message
from telegram_format import html_to_plain, sanitize_telegram_html, split_message

logger = logging.getLogger(__name__)

WELCOME_TEXT = (
    "היי! 👋 אני היועץ של <b>CodeKeeper</b>.\n\n"
    "ספרו לי איך אתם עובדים — עם אילו כלים, מאיזה מכשיר, מה מעצבן אתכם היום — "
    "ואגיד לכם בכנות אם ואיך Ck יכול לעזור.\n\n"
    f"<i>כדי שאזכור את ההקשר, השיחה נשמרת אצלי {HISTORY_RETENTION_DAYS} ימים ואז נמחקת.</i>"
)


# ─── שליחה ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _typing_indicator(bot, chat_id: int, interval: float = 4.0):
    """חיווי "מקליד..." כל interval שניות עד שהבלוק מסתיים (החיווי פג אחרי ~5 שניות)."""
    stop = asyncio.Event()

    async def _loop():
        while not stop.is_set():
            try:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except TelegramError as e:
                logger.debug("typing indicator failed for chat %s: %s", chat_id, e)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                # לא כשל: עברו interval שניות והבלוק עוד רץ — זה המסלול הרגיל לחיווי הבא
                pass

    task = asyncio.create_task(_loop())
    try:
        yield
    finally:
        stop.set()
        await task


async def _send_text(bot, chat_id: int, text: str, reply_markup=None) -> None:
    """שליחת תשובת AI: סניטציה ← פיצול ← HTML, ו-fallback לטקסט נקי.

    המקלדת מוצמדת רק לחלק האחרון, כך שהכפתור יופיע מתחת לסוף התשובה.
    """
    chunks = split_message(sanitize_telegram_html(text))
    for i, chunk in enumerate(chunks):
        markup = reply_markup if i == len(chunks) - 1 else None
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML, reply_markup=markup)
        except BadRequest as e:
            # HTML שטלגרם לא הצליח לפרסר (למשל תג שנחתך בפיצול)
            logger.warning("HTML rejected by Telegram (%s) — sending plain text", e)
            await bot.send_message(chat_id=chat_id, text=html_to_plain(chunk), reply_markup=markup)


def _contact_keyboard() -> InlineKeyboardMarkup | None:
    if not CONTACT_URL:
        return None
    return InlineKeyboardMarkup([[InlineKeyboardButton("💬 לדבר עם המפתח", url=CONTACT_URL)]])


# ─── שאלות המשך ──────────────────────────────────────────────────────────────

# callback_data מוגבל ל-64 בתים (telegram/constants.py, MAX_CALLBACK_DATA),
# ולכן טקסט השאלה נשמר ב-bot_data והכפתור נושא רק מפתח.
FOLLOW_UP_CB_PREFIX = "followup_"
_FOLLOW_UP_TTL_SECONDS = 3600


def _cleanup_stale_follow_ups(bot_data: dict) -> None:
    """מחיקת שאלות שלא נלחצו תוך שעה — אחרת bot_data גדל בלי סוף."""
    now = int(time.time())
    stale_keys = []
    for key in bot_data:
        if not key.startswith(FOLLOW_UP_CB_PREFIX):
            continue
        # מבנה המפתח: followup_{user_id}_{timestamp}_{index}
        try:
            if now - int(key.split("_")[-2]) > _FOLLOW_UP_TTL_SECONDS:
                stale_keys.append(key)
        except (ValueError, IndexError):
            # מפתח עם הקידומת שלא בנוי לפי המבנה — לא נוצר כאן, ולכן לא שלנו למחוק
            continue
    for key in stale_keys:
        bot_data.pop(key, None)


def _build_follow_up_keyboard(questions: list[str], bot_data: dict, user_id: str) -> InlineKeyboardMarkup:
    _cleanup_stale_follow_ups(bot_data)
    now = int(time.time())
    buttons = []
    for i, q in enumerate(questions):
        cb_id = f"{FOLLOW_UP_CB_PREFIX}{user_id}_{now}_{i}"
        bot_data[cb_id] = q
        buttons.append([InlineKeyboardButton(f"💡 {q}", callback_data=cb_id)])
    return InlineKeyboardMarkup(buttons)


# ─── הצינור היחיד ל-AI ───────────────────────────────────────────────────────


async def _answer_and_reply(context: ContextTypes.DEFAULT_TYPE, *, user_id: str, chat_id: int, text: str) -> None:
    """היסטוריה ← AI ← שליחה ← שמירה ← שאלות המשך.

    הבוט מעבד עדכונים של משתמשים שונים במקביל (concurrent_updates), אבל
    הודעות של אותו משתמש עוברות אחת-אחת דרך מנעול ב-user_data. בלי המנעול,
    שתי הודעות מהירות היו קוראות את אותה היסטוריה, והתשובות היו מגיעות
    בסדר אקראי.
    """
    lock = context.user_data.setdefault("answer_lock", asyncio.Lock())
    async with lock:
        async with _typing_indicator(context.bot, chat_id):
            try:
                history = await asyncio.to_thread(get_history, user_id, CONTEXT_WINDOW_SIZE)
            except sqlite3.Error:
                # תשובה בלי זיכרון עדיפה על אי-תשובה
                logger.exception("Failed to load history for user %s", user_id)
                history = []
            answer = await asyncio.to_thread(generate_answer, history, text)

        reply_markup = _contact_keyboard() if answer.needs_contact else None
        await _send_text(context.bot, chat_id, answer.text, reply_markup=reply_markup)
        if not answer.ok:
            return  # הודעת תקלה לא נכנסת לזיכרון של הבוט

        try:
            await asyncio.to_thread(save_exchange, user_id, text, answer.text)
        except sqlite3.Error:
            logger.exception("Failed to save history for user %s", user_id)

        if answer.follow_up_questions:
            keyboard = _build_follow_up_keyboard(answer.follow_up_questions, context.bot_data, user_id)
            await context.bot.send_message(
                chat_id=chat_id,
                text="💡 <b>אולי תרצו גם לשאול:</b>",
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
            )


# ─── Handlers ────────────────────────────────────────────────────────────────


@rate_limit_guard
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(WELCOME_TEXT, parse_mode=ParseMode.HTML)


@rate_limit_guard
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await _answer_and_reply(
        context,
        user_id=str(update.effective_user.id),
        chat_id=update.effective_chat.id,
        text=update.message.text,
    )


async def follow_up_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """לחיצה על שאלת המשך — נשלחת לצינור כאילו הגולש הקליד אותה.

    בלי דקורטור: הבדיקה והרישום ידניים, כי הרישום חייב לבוא רק אחרי שווידאנו
    שהשאלה עוד קיימת — לחיצה על כפתור שפג לא צריכה לאכול מכסה.
    """
    query = update.callback_query
    await query.answer()
    user_id = str(update.effective_user.id)

    limit_msg = check_rate_limit(user_id)
    if limit_msg is not None:
        await query.edit_message_text(limit_msg)
        return

    question = context.bot_data.pop(query.data, None)
    if not question:
        await query.edit_message_text("⏳ השאלה כבר לא זמינה. אפשר לשאול אותי ישירות!")
        return

    record_message(user_id)
    # הכפתורים נעלמים ובמקומם נשארת השאלה שנבחרה (טקסט רגיל, בלי parse_mode)
    await query.edit_message_text(f"💡 {question}")
    await _answer_and_reply(context, user_id=user_id, chat_id=update.effective_chat.id, text=question)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """לוג + הודעה כללית לגולש.

    במקור נרשם ללוג אובייקט ה-update כולו — שכולל את שם הגולש, את שם המשתמש
    שלו ואת טקסט ההודעה. כאן נרשמים רק מזהה העדכון וה-traceback.
    """
    update_id = update.update_id if isinstance(update, Update) else None
    logger.error("Update %s caused an error", update_id, exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("מצטער, משהו השתבש. נסו שוב בעוד רגע 🙏")
        except TelegramError:
            logger.exception("Failed to send error reply for update %s", update_id)
```

---

## 8. `main.py`

**טוקן הבוט נכתב ללוג במקור.** python-telegram-bot שולח בקשות דרך httpx, ו-httpx כותב כל בקשה ברמת INFO עם ה-URL המלא, שכולל את הטוקן. בבוט העסקי `logging.basicConfig(level=logging.INFO)` בלי שום השתקה. הרצתי את אותה תצורה מול שרת טלגרם מקומי, והטוקן הופיע בשורה הראשונה של הלוג. כאן ה-logger של httpx מושתק ל-WARNING.

**עיבוד במקביל.** ברירת המחדל של python-telegram-bot היא עדכון אחד בכל פעם (`SimpleUpdateProcessor(max_concurrent_updates=1)`), והבוט העסקי לא משנה אותה. המשמעות: גולש אחד שמחכה כמה שניות לתשובת AI מעכב את כל השאר. כאן `concurrent_updates(True)` מאפשר עד 256 במקביל, והסדר נשמר לכל גולש בזכות המנעול.

**בלי `drop_pending_updates`.** במקור הודעות שהגיעו בזמן deploy נזרקות. כאן גולש שכתב בדיוק באותו רגע מקבל תשובה כשהבוט עולה.

**הודעה ערוכה לא מגיעה ל-handlers.** `CommandHandler` מאזין כברירת מחדל גם להודעות ערוכות ("listens to messages as well as edited messages", בקוד הספרייה), ו-`filters.TEXT` בודק את `effective_message`, שכולל גם אותן. שני ה-handlers קוראים את `update.message`, שהוא `None` בעריכה. בהרצה מול שרת טלגרם מקומי: גולש שערך הודעה קיבל "משהו השתבש", ובלוג נרשם `AttributeError`. אותו רישום קיים במקור: `message_handler` ב-`bot/handlers.py` קורא `update.message.text` ונרשם עם `filters.TEXT & ~filters.COMMAND`. כאן שני ה-handlers נרשמים עם `filters.UpdateType.MESSAGE`, ועריכה לא נענית: תשובה שנייה לאותה שאלה הייתה עולה עוד קריאת AI.

**`main.py`**

```python
"""
נקודת הכניסה — ולידציה, אתחול, רישום handlers והרצה ב-polling.

מקור: bot/telegram_bot.py + main.py בריפו ai-business-bot.
"""

import asyncio
import logging
import re
import sqlite3
import sys

from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import TELEGRAM_BOT_TOKEN, validate_config
from handlers import (
    FOLLOW_UP_CB_PREFIX,
    error_handler,
    follow_up_callback,
    message_handler,
    start_command,
)
from history import init_db, purge_old_history
from llm import init_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
# python-telegram-bot שולח בקשות דרך httpx, ו-httpx כותב כל בקשה ברמת INFO
# עם ה-URL המלא: https://api.telegram.org/bot<TOKEN>/getUpdates. בלי השורה
# הזו הטוקן של הבוט נכתב ללוג בכל מחזור polling.
# מקור: httpx/_client.py (logger = logging.getLogger("httpx"), 'HTTP Request: %s %s ...')
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def _purge_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        deleted = await asyncio.to_thread(purge_old_history)
        if deleted:
            logger.info("History purge: removed %d messages", deleted)
    except sqlite3.Error:
        logger.exception("History purge failed")


async def _post_init(application: Application) -> None:
    # job_queue קיים רק עם python-telegram-bot[job-queue] (אחרת None + אזהרה)
    if application.job_queue is None:
        logger.error("JobQueue unavailable — history purge is disabled")
        return
    application.job_queue.run_repeating(_purge_job, interval=24 * 3600, first=60, name="history_purge")


def build_application(token: str, base_url: str | None = None) -> Application:
    """base_url — רק לבדיקות מול שרת טלגרם מקומי."""
    builder = (
        ApplicationBuilder()
        .token(token)
        # ברירת המחדל היא עדכון אחד בכל פעם: גולש אחד שמחכה ל-AI היה מעכב
        # את כל השאר. 256 במקביל; הסדר נשמר פר משתמש (ראה handlers._answer_and_reply).
        .concurrent_updates(True)
        .post_init(_post_init)
    )
    if base_url:
        builder = builder.base_url(base_url)
    app = builder.build()

    # יועץ אישי — לא עונה בקבוצות. ורק הודעות חדשות: CommandHandler ו-filters.TEXT
    # מקבלים גם הודעה ערוכה (שניהם בודקים את effective_message), ושני ה-handlers
    # קוראים את update.message — שהוא None בעריכה, ולכן עריכה הייתה מפילה אותם.
    # מקור: telegram/ext/_handlers/commandhandler.py ("listens to messages as well
    # as edited messages"), telegram/ext/filters.py (UpdateType.MESSAGE: update.message is not None)
    private = filters.ChatType.PRIVATE & filters.UpdateType.MESSAGE
    app.add_handler(CommandHandler(["start", "help"], start_command, filters=private))
    app.add_handler(CallbackQueryHandler(follow_up_callback, pattern=rf"^{re.escape(FOLLOW_UP_CB_PREFIX)}"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & private, message_handler))
    app.add_error_handler(error_handler)
    return app


def main() -> None:
    errors = validate_config()
    if errors:
        for err in errors:
            logger.error("Config: %s", err)
        sys.exit(1)

    init_db()
    init_llm()
    app = build_application(TELEGRAM_BOT_TOKEN)
    logger.info("Starting bot (polling)")
    # בלי drop_pending_updates: גולש שכתב בזמן deploy מקבל תשובה כשהבוט עולה.
    app.run_polling()


if __name__ == "__main__":
    main()
```

---

## 9. טסטים

38 טסטים, בלי רשת: OpenAI מוחלף ב-client מזויף שמחזיר אובייקטי `ChatCompletion` אמיתיים של ה-SDK, וטלגרם מוחלף ב-`FakeBot`. כל טסט מקבל DB זמני משלו.

**כל טסט הוכח כמסוגל להיכשל.** שברתי את הקוד בכוונה בכל נקודה רגישה וווידאתי שהטסט המתאים נופל:

| השבירה | הטסט שנפל |
|---|---|
| הסרת המנעול פר גולש | `test_same_user_messages_are_answered_in_order` |
| `await` בין בדיקת קצב לרישום | `test_guard_holds_under_concurrent_updates` |
| לוג של ה-`update` המלא (כמו במקור) | `test_error_handler_does_not_log_user_content` |
| fallback של HTML כמו במקור | `test_rejected_html_falls_back_to_clean_plain_text` |
| שליחה בלי פיצול | `test_long_answer_is_split_and_button_goes_on_last_chunk` |
| שמירת שאלה ותשובה בשתי טרנזקציות | `test_exchange_is_atomic` |
| שאילתת היסטוריה בלי סינון גיל | `test_old_messages_are_hidden_and_purged` |
| `max_tokens` במקום `max_completion_tokens` | `test_request_uses_max_completion_tokens_and_omits_unset_params` |
| `temperature` שנשלח תמיד | אותו טסט |
| זנב התשובה בלוג (כמו במקור) | `test_diag_log_has_numbers_but_no_content` |
| הטוקן לא מוסר מאמצע התשובה | `test_contact_marker_detected_only_at_start_but_removed_everywhere` |
| ניתוב בלי `filters.UpdateType.MESSAGE` (כמו במקור) | `test_edited_message_reaches_no_handler` |
| תו RLM בהערה בעברית | `test_no_bidi_controls_in_source` |
| שינוי בקובץ קוד בלי אותו שינוי במדריך | `test_every_file_in_the_guide_matches_the_repo` |

לטסט ה-BiDi יש גם הוכחה שלא תוכננה: בכתיבה הראשונה שלו, ה-escapes נשמרו בקובץ כתווים עצמם, והטסט נפל על עצמו.

הרצה:

```bash
python -m pytest -q
```

::: details tests/conftest.py + tests/fakes.py

**`tests/conftest.py`**

```python
"""fixtures משותפים. אין כאן קריאות רשת — OpenAI וטלגרם מוחלפים בכפילים."""

import pytest

import history
import rate_limiter
from fakes import FakeBot


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """DB זמני לכל טסט — לעולם לא נוגעים בקובץ אמיתי."""
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    history.init_db()
    return tmp_path / "history.db"


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    rate_limiter._user_timestamps.clear()
    yield
    rate_limiter._user_timestamps.clear()


@pytest.fixture
def fake_bot():
    return FakeBot()
```

**`tests/fakes.py`**

```python
"""כפילים לבדיקות — מודול נפרד, כי לא מייבאים מ-conftest."""

from types import SimpleNamespace


class FakeBot:
    """כפיל של telegram.Bot — רושם כל קריאה, ויכול לדחות HTML כמו טלגרם."""

    def __init__(self, reject_html_containing: str | None = None):
        self.sent: list[dict] = []
        self.reject_html_containing = reject_html_containing

    async def send_message(self, **kwargs):
        from telegram.error import BadRequest

        if (
            self.reject_html_containing
            and kwargs.get("parse_mode") == "HTML"
            and self.reject_html_containing in kwargs["text"]
        ):
            raise BadRequest("Can't parse entities")
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=len(self.sent))

    async def send_chat_action(self, **kwargs):
        return True
```

:::

::: details tests/test_llm.py

**`tests/test_llm.py`**

```python
from types import SimpleNamespace

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion

import llm


# ─── פענוח ───────────────────────────────────────────────────────────────────


def test_extract_and_strip_follow_ups():
    text = "תשובה.\n\n[שאלות_המשך: יש גיבוי? | עובד בנייד? | כמה זה עולה?]"
    assert llm.extract_follow_up_questions(text) == ["יש גיבוי?", "עובד בנייד?", "כמה זה עולה?"]
    assert llm.strip_follow_up_questions(text) == "תשובה."


def test_follow_ups_without_brackets_are_parsed_too():
    text = "תשובה.\nשאלות המשך: א | ב"
    assert llm.extract_follow_up_questions(text) == ["א", "ב"]
    assert llm.strip_follow_up_questions(text) == "תשובה."


def test_contact_marker_detected_only_at_start_but_removed_everywhere():
    assert llm.split_contact_marker("[CONTACT]\nאין לי מידע") == (True, "אין לי מידע")
    assert llm.split_contact_marker("טקסט [CONTACT] באמצע") == (False, "טקסט  באמצע")


def test_trim_to_last_sentence():
    assert llm.trim_to_last_sentence("שורה שלמה.\nשורה שנקט") == "שורה שלמה.\n…"
    assert llm.trim_to_last_sentence("משפט ראשון. משפט שני שנק") == "משפט ראשון. …"


# ─── פרומפט ──────────────────────────────────────────────────────────────────


def test_system_prompt_is_static_and_keeps_knowledge_verbatim():
    knowledge = "מסמך עם {סוגריים} ו-%s"
    first = llm.build_system_prompt(knowledge)
    assert first == llm.build_system_prompt(knowledge)  # תנאי להתאמה במטמון של OpenAI
    assert knowledge in first
    assert first.rstrip().endswith("── סוף מסמך הידע ──")


def test_follow_up_rule_toggles():
    assert "שאלות_המשך" in llm.build_system_prompt("x", follow_up_enabled=True)
    assert "שאלות_המשך" not in llm.build_system_prompt("x", follow_up_enabled=False)


def test_build_messages_order():
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    msgs = llm.build_messages("PROMPT", history, "q2")
    assert [m["role"] for m in msgs] == ["developer", "user", "assistant", "user"]
    assert msgs[0]["content"] == "PROMPT" and msgs[-1]["content"] == "q2"


# ─── generate_answer מול כפיל של ה-client ───────────────────────────────────


def _completion(content, finish_reason="stop", cached=0):
    # אובייקט אמיתי של ה-SDK, לא namespace — כך שגישה לשדה לא קיים נכשלת
    return ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-5.4-mini",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 2000, "completion_tokens": 20, "total_tokens": 2020,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    })


class StubClient:
    def __init__(self, response=None, error=None):
        self.calls: list[dict] = []
        self._response, self._error = response, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


@pytest.fixture
def use_client(monkeypatch):
    monkeypatch.setattr(llm, "_system_prompt", "PROMPT")

    def _install(client):
        monkeypatch.setattr(llm, "_client", client)
        return client

    return _install


def test_request_uses_max_completion_tokens_and_omits_unset_params(use_client, monkeypatch):
    monkeypatch.setattr(llm, "LLM_TEMPERATURE", "")
    monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", "")
    client = use_client(StubClient(_completion("תשובה.")))
    llm.generate_answer([], "שאלה")
    kwargs = client.calls[0]
    assert "max_completion_tokens" in kwargs and "max_tokens" not in kwargs
    assert "temperature" not in kwargs and "reasoning_effort" not in kwargs


def test_optional_params_are_sent_when_configured(use_client, monkeypatch):
    monkeypatch.setattr(llm, "LLM_TEMPERATURE", "0.3")
    monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", "low")
    client = use_client(StubClient(_completion("תשובה.")))
    llm.generate_answer([], "שאלה")
    assert client.calls[0]["temperature"] == 0.3
    assert client.calls[0]["reasoning_effort"] == "low"


def test_answer_with_follow_ups(use_client, monkeypatch):
    monkeypatch.setattr(llm, "FOLLOW_UP_ENABLED", True)
    use_client(StubClient(_completion("כן, זה מתאים.\n[שאלות_המשך: א | ב]")))
    answer = llm.generate_answer([], "שאלה")
    assert answer == llm.Answer(text="כן, זה מתאים.", ok=True, follow_up_questions=["א", "ב"])


def test_contact_answer_has_no_follow_ups(use_client, monkeypatch):
    monkeypatch.setattr(llm, "FOLLOW_UP_ENABLED", True)
    use_client(StubClient(_completion("[CONTACT]\nאין לי מידע על זה.\n[שאלות_המשך: א]")))
    answer = llm.generate_answer([], "שאלה")
    assert answer.needs_contact and answer.ok
    assert answer.text == "אין לי מידע על זה." and answer.follow_up_questions == []


def test_bare_contact_marker_gets_default_text(use_client):
    use_client(StubClient(_completion("[CONTACT]")))
    answer = llm.generate_answer([], "שאלה")
    assert answer.needs_contact and answer.text == llm.CONTACT_DEFAULT_TEXT


def test_api_error_returns_fallback_not_raise(use_client):
    error = openai.APITimeoutError(request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"))
    use_client(StubClient(error=error))
    assert llm.generate_answer([], "שאלה") == llm.Answer(text=llm.FALLBACK_ERROR, ok=False)


def test_empty_content_returns_fallback(use_client):
    use_client(StubClient(_completion(None)))
    assert llm.generate_answer([], "שאלה").ok is False


def test_length_finish_reason_trims(use_client):
    use_client(StubClient(_completion("שורה שלמה.\nשורה שנקט", finish_reason="length")))
    assert llm.generate_answer([], "שאלה").text == "שורה שלמה.\n…"


def test_diag_log_has_numbers_but_no_content(use_client, caplog):
    caplog.set_level("INFO")
    use_client(StubClient(_completion("התשובה הסודית של הגולש.", cached=1920)))
    llm.generate_answer([], "שאלה")
    assert "cached_tokens=1920" in caplog.text
    assert "הסודית" not in caplog.text


def test_init_llm_rejects_empty_knowledge(tmp_path, monkeypatch):
    empty = tmp_path / "k.md"
    empty.write_text("  \n", encoding="utf-8")
    monkeypatch.setattr(llm, "KNOWLEDGE_FILE", empty)
    with pytest.raises(RuntimeError):
        llm.init_llm()
```

:::

::: details tests/test_handlers.py

**`tests/test_handlers.py`**

```python
import asyncio
import datetime as dtm
import time
from types import SimpleNamespace

from telegram import Chat, Message, MessageEntity, Update, User

import handlers
import llm
from fakes import FakeBot


def _context(bot):
    return SimpleNamespace(bot=bot, user_data={}, bot_data={})


def test_same_user_messages_are_answered_in_order(tmp_db, monkeypatch, fake_bot):
    """שתי הודעות מהירות של אותו גולש: השנייה חייבת לראות את הראשונה בהיסטוריה."""
    seen_history_lengths = []

    def fake_generate(history, text):
        seen_history_lengths.append(len(history))
        time.sleep(0.05)  # קריאת AI איטית — רצה ב-thread
        return llm.Answer(text=f"תשובה ל-{text}", ok=True)

    monkeypatch.setattr(handlers, "generate_answer", fake_generate)
    context = _context(fake_bot)

    async def main():
        await asyncio.gather(
            handlers._answer_and_reply(context, user_id="u1", chat_id=1, text="ראשונה"),
            handlers._answer_and_reply(context, user_id="u1", chat_id=1, text="שנייה"),
        )

    asyncio.run(main())
    assert seen_history_lengths == [0, 2]
    assert [m["text"] for m in fake_bot.sent] == ["תשובה ל-ראשונה", "תשובה ל-שנייה"]


def test_failed_answer_is_not_saved(tmp_db, monkeypatch, fake_bot):
    monkeypatch.setattr(handlers, "generate_answer", lambda h, t: llm.Answer(text=llm.FALLBACK_ERROR, ok=False))
    asyncio.run(handlers._answer_and_reply(_context(fake_bot), user_id="u1", chat_id=1, text="שאלה"))
    assert fake_bot.sent[0]["text"] == llm.FALLBACK_ERROR
    assert handlers.get_history("u1", 10) == []


def test_contact_button_and_follow_ups(tmp_db, monkeypatch, fake_bot):
    monkeypatch.setattr(handlers, "CONTACT_URL", "https://t.me/someone")
    monkeypatch.setattr(handlers, "generate_answer",
                        lambda h, t: llm.Answer(text="אין לי מידע", ok=True, needs_contact=True))
    context = _context(fake_bot)
    asyncio.run(handlers._answer_and_reply(context, user_id="u1", chat_id=1, text="שאלה"))
    button = fake_bot.sent[0]["reply_markup"].inline_keyboard[0][0]
    assert button.url == "https://t.me/someone"

    monkeypatch.setattr(handlers, "generate_answer",
                        lambda h, t: llm.Answer(text="כן", ok=True, follow_up_questions=["א?"]))
    asyncio.run(handlers._answer_and_reply(context, user_id="u1", chat_id=1, text="שאלה"))
    cb = fake_bot.sent[-1]["reply_markup"].inline_keyboard[0][0].callback_data
    assert context.bot_data[cb] == "א?" and len(cb.encode()) <= 64


def test_rejected_html_falls_back_to_clean_plain_text():
    bot = FakeBot(reject_html_containing="<b>")
    asyncio.run(handlers._send_text(bot, 1, "<b>כותרת</b> & עוד"))
    assert bot.sent == [{"chat_id": 1, "text": "כותרת & עוד", "reply_markup": None}]


def test_long_answer_is_split_and_button_goes_on_last_chunk(fake_bot):
    text = "\n\n".join(["פסקה " + "מילה " * 200] * 6)  # ~6000 תווים
    asyncio.run(handlers._send_text(fake_bot, 1, text, reply_markup="KB"))
    assert len(fake_bot.sent) == 2
    assert all(len(m["text"]) <= 4096 for m in fake_bot.sent)
    assert [m["reply_markup"] for m in fake_bot.sent] == [None, "KB"]


def test_error_handler_does_not_log_user_content(caplog, fake_bot):
    message = Message(
        message_id=5,
        date=dtm.datetime.now(dtm.timezone.utc),
        chat=Chat(id=1, type="private"),
        from_user=User(id=1, first_name="Dana", is_bot=False, username="dana_x"),
        text="הטלפון שלי 0501234567",
    )
    message.set_bot(fake_bot)
    update = Update(update_id=9, message=message)
    caplog.set_level("ERROR")
    asyncio.run(handlers.error_handler(update, SimpleNamespace(error=RuntimeError("boom"))))
    assert "Update 9 caused an error" in caplog.text
    for secret in ("Dana", "dana_x", "0501234567"):
        assert secret not in caplog.text
    assert fake_bot.sent  # הגולש קיבל הודעת תקלה


def test_edited_message_reaches_no_handler():
    """הודעה ערוכה לא מגיעה לאף handler: שניהם קוראים update.message, שהוא None בעריכה.

    עובר דרך הניתוב האמיתי של build_application — כך האפליקציה בוחרת handler לכל עדכון.
    """
    import main  # בתוך הטסט: מודול הכניסה מגדיר logging כבר בייבוא

    app = main.build_application("123:TEST")
    routes = [h for group in app.handlers.values() for h in group]
    bot = SimpleNamespace(username="advisor_bot")  # CommandHandler משווה את הפקודה לשם הבוט

    for text, handler in (("שאלה", handlers.message_handler), ("/start", handlers.start_command)):
        entities = [MessageEntity(type=MessageEntity.BOT_COMMAND, offset=0, length=6)] if text == "/start" else None
        message = Message(
            message_id=5,
            date=dtm.datetime.now(dtm.timezone.utc),
            chat=Chat(id=1, type="private"),
            from_user=User(id=1, first_name="Dana", is_bot=False),
            text=text,
            entities=entities,
        )
        message.set_bot(bot)
        new = Update(update_id=1, message=message)
        edited = Update(update_id=2, edited_message=message)
        assert [h.callback for h in routes if h.check_update(new)] == [handler]
        assert [h.callback for h in routes if h.check_update(edited)] == []
```

:::

::: details tests/test_no_bidi.py + tests/test_guide_sync.py

**`tests/test_no_bidi.py`**

```python
"""אין תווי BiDi בקבצי הקוד.

הם נכנסים להערות בעברית בהעתקה מדפדפן או מצ'אט, לא נראים בשום עורך, ומפילים
לינטרים (bandit B613, ruff PLE2502). מקור הכלל: amir-bug-patterns,
BY-STACK/hebrew-source.md H1.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ה-escapes מכוונים: התווים עצמם בקובץ הזה היו מפילים את הבדיקה על עצמה
BIDI = ("\u200e", "\u200f", "\u202a", "\u202b", "\u202c",
        "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069")

# סביבות וירטואליות ותיקיות נסתרות (.git, .venv) אינן קוד שלנו
_SKIP_DIRS = {"venv", "env", "__pycache__"}


def _source_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*.py")
        if not any(part.startswith(".") or part in _SKIP_DIRS for part in path.relative_to(ROOT).parts[:-1])
    ]


def test_no_bidi_controls_in_source():
    files = _source_files()
    assert files  # סריקה שלא מצאה קבצים הייתה עוברת על כל דבר
    for path in files:
        text = path.read_text(encoding="utf-8")
        for char in BIDI:
            assert char not in text, f"{path.relative_to(ROOT)}: תו BiDi {char!r}"
```

**`tests/test_guide_sync.py`**

```python
"""המדריך ב-docs/ מכיל עותק מלא של קבצי הפרויקט — והטסט שומר ששני העותקים זהים.

הטענה המרכזית של המדריך היא שכל קובץ בו הוא קוד שרץ ונבדק. בלי הטסט, שינוי
בקוד בלי אותו שינוי במדריך (או להפך) היה הופך את הטענה לשקרית בשקט.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "codekeeper_advisor_bot_guide.md"

# כותרת קובץ במדריך: שורה שכולה **`path`**, ואחריה בלוק קוד
_HEADER = re.compile(r"^\*\*`([^`]+)`\*\*$")


def _guide_files() -> dict[str, str]:
    lines = GUIDE.read_text(encoding="utf-8").splitlines()
    files: dict[str, str] = {}
    i = 0
    while i < len(lines):
        match = _HEADER.match(lines[i])
        if not match:
            i += 1
            continue
        start = next(j for j in range(i + 1, len(lines)) if lines[j].startswith("```")) + 1
        end = next(j for j in range(start, len(lines)) if lines[j] == "```")
        path = match.group(1)
        assert path not in files, f"{path} מופיע פעמיים במדריך"
        files[path] = "\n".join(lines[start:end]) + "\n"
        i = end + 1
    return files


def test_every_file_in_the_guide_matches_the_repo():
    files = _guide_files()
    assert files  # פענוח שלא מצא אף קובץ היה עובר על כל דבר
    for path, body in files.items():
        assert (ROOT / path).read_text(encoding="utf-8") == body, f"{path}: הקוד במדריך שונה מהקובץ בריפו"
```

:::

::: details tests/test_history.py

**`tests/test_history.py`**

```python
import sqlite3

import pytest

import history


def _insert_old(db_path, user_id, days_ago):
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO conversations (user_id, role, message, created_at) "
        "VALUES (?, 'user', 'ישן', datetime('now', ?))",
        (user_id, f"-{days_ago} days"),
    )
    conn.commit()
    conn.close()


def test_history_is_chronological_and_limited(tmp_db):
    for i in range(3):
        history.save_exchange("u1", f"q{i}", f"a{i}")
    history.save_exchange("u2", "other", "other")
    assert history.get_history("u1", 4) == [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
    ]
    assert history.get_history("u1", 0) == []


def test_exchange_is_atomic(tmp_db):
    # השורה השנייה נכשלת ב-bind — הראשונה חייבת להתבטל איתה
    with pytest.raises(sqlite3.Error):
        history.save_exchange("u1", "שאלה", {"not": "a string"})
    assert history.get_history("u1", 10) == []


def test_old_messages_are_hidden_and_purged(tmp_db, monkeypatch):
    monkeypatch.setattr(history, "HISTORY_RETENTION_DAYS", 7)
    _insert_old(tmp_db, "u1", days_ago=10)
    history.save_exchange("u1", "חדש", "תשובה")
    assert [m["content"] for m in history.get_history("u1", 10)] == ["חדש", "תשובה"]
    assert history.purge_old_history() == 1
    assert history.purge_old_history() == 0
```

:::

::: details tests/test_rate_limiter.py

**`tests/test_rate_limiter.py`**

```python
import asyncio
from types import SimpleNamespace

import rate_limiter


class _Msg:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


def _update(user_id=1):
    return SimpleNamespace(effective_user=SimpleNamespace(id=user_id), effective_message=_Msg())


def test_minute_limit_blocks_the_eleventh_message():
    for _ in range(rate_limiter.RATE_LIMIT_PER_MINUTE):
        assert rate_limiter.check_rate_limit("u") is None
        rate_limiter.record_message("u")
    assert rate_limiter.check_rate_limit("u") is not None
    assert rate_limiter.check_rate_limit("other") is None


def test_guard_holds_under_concurrent_updates():
    """בדיקת מקביליות רצה במקביל: עדכון אחד יותר מהמגבלה, בבת אחת ← בדיוק המגבלה עוברת."""
    ran = []

    @rate_limiter.rate_limit_guard
    async def handler(update, context):
        await asyncio.sleep(0.01)  # ה-handler משחרר את ה-loop, כמו קריאה ל-AI
        ran.append(1)

    async def main():
        await asyncio.gather(*(handler(_update(), None) for _ in range(rate_limiter.RATE_LIMIT_PER_MINUTE + 1)))

    asyncio.run(main())
    assert len(ran) == rate_limiter.RATE_LIMIT_PER_MINUTE
```

:::

::: details tests/test_telegram_format.py

**`tests/test_telegram_format.py`**

```python
from telegram_format import html_to_plain, sanitize_telegram_html, split_message


def test_sanitize_keeps_telegram_tags_and_escapes_the_rest():
    out = sanitize_telegram_html("<b>שלום</b> <script>x</script> a < b & c")
    assert out == "<b>שלום</b> &lt;script&gt;x&lt;/script&gt; a &lt; b &amp; c"


def test_sanitize_drops_tag_with_attributes_and_its_closing_tag():
    assert sanitize_telegram_html('<b class="x">טקסט</b>') == "טקסט"


def test_html_to_plain_restores_real_characters():
    sanitized = sanitize_telegram_html("<b>א</b> < ב & ג")
    assert html_to_plain(sanitized) == "א < ב & ג"


def test_split_message_respects_limit_and_keeps_all_content():
    paragraphs = [f"פסקה {i} " + "מילה " * 30 for i in range(40)]
    text = "\n\n".join(paragraphs)
    chunks = split_message(text, limit=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_split_message_prefers_paragraph_boundary():
    text = "א" * 300 + "\n\n" + "ב" * 300
    assert split_message(text, limit=500) == ["א" * 300, "ב" * 300]


def test_split_message_hard_cuts_a_single_huge_word():
    chunks = split_message("x" * 1200, limit=500)
    assert [len(c) for c in chunks] == [500, 500, 200]


def test_short_message_is_one_chunk():
    assert split_message("  שלום  ") == ["שלום"]
```

:::

---

## 10. הפעלה

1. יוצרים בוט ב-BotFather ומקבלים טוקן.
2. שמים את מסמך הידע ב-`knowledge/codekeeper.md`.
3. ממלאים `.env` (ראו `.env.example`), ומריצים:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

**עדכון מסמך הידע** = commit + deploy. המסמך נטען פעם אחת בעלייה.

**בשרת:** polling לא צריך כתובת ציבורית ולא פורט פתוח, אז מספיק שירות שרץ ברקע. SQLite צריך דיסק קבוע — בלעדיו ההיסטוריה מתאפסת בכל deploy. עם תקופת שמירה של 7 ימים זה מחיר שאפשר לחיות איתו, אבל כדאי לדעת שזה קורה.

::: warning
שני מופעים של הבוט עם אותו טוקן במצב polling מתנגשים — הספרייה זורקת `telegram.error.Conflict` ("a long poll or webhook conflicts with another one"). בזמן deploy, ודאו שהמופע הישן נעצר לפני שהחדש עולה.
:::

**ב-Render:** `render.yaml` מגדיר את הבוט כעובד רקע עם דיסק קבוע, והדיסק פותר את שתי הבעיות שלמעלה. ה-SQLite נשמר בין deploys, ו-Render עוצר את המופע הישן לפני שהחדש עולה, כי שירות עם דיסק לא מקבל zero-downtime deploy. בלי דיסק, המופע הישן מקבל `SIGTERM` רק 60 שניות אחרי שהחדש עלה, וכל הזמן הזה שניהם עושים polling. ב-`SIGTERM` הבוט נעצר נקי, כי `run_polling` מאזין לו כברירת מחדל. ביצירת ה-Blueprint, Render מבקש בדשבורד את הטוקן, את מפתח OpenAI ואת `CONTACT_URL`.

**`render.yaml`**

```yaml
# Render Blueprint. מקור לכל השדות: https://render.com/docs/blueprint-spec
services:
  # עובד רקע ולא web: polling לא צריך כתובת ציבורית ולא פורט פתוח
  - type: worker
    name: codekeeper-advisor-bot
    runtime: python
    # התוכנית הקטנה ביותר לעובד רקע (0.5 CPU / 512MB). לעובד רקע אין free,
    # ודיסק קבוע זמין רק בתוכנית בתשלום.
    plan: 0.5c-512mb
    buildCommand: pip install -r requirements.txt
    startCommand: python main.py

    # SQLite של היסטוריית השיחות — בלי דיסק היא מתאפסת בכל deploy.
    # ועוד דבר שהדיסק פותר: עם דיסק, Render עוצר את המופע הישן לפני שהחדש
    # עולה. בלעדיו (zero-downtime) שני המופעים עושים polling במקביל כ-60
    # שניות, וטלגרם דוחה את שניהם ב-Conflict.
    # מקור: https://render.com/docs/disks, https://render.com/docs/deploys#zero-downtime-deploys
    disk:
      name: advisor-data
      mountPath: /var/data
      sizeGB: 1

    envVars:
      # הגרסה שהקוד נבדק מולה. בלי זה Render בוחר לשירות חדש את 3.14.
      - key: PYTHON_VERSION
        value: 3.13.16
      # חייב להיות זהה ל-mountPath שלמעלה, אחרת ה-DB נכתב לדיסק הזמני
      - key: DATA_DIR
        value: /var/data
      # Render מבקש את שלושת אלה בדשבורד, ביצירת ה-Blueprint בלבד
      - key: TELEGRAM_BOT_TOKEN
        sync: false
      - key: OPENAI_API_KEY
        sync: false
      # ריק = בלי כפתור "לדבר עם המפתח"
      - key: CONTACT_URL
        sync: false
```

---

## 11. מה הושמט מהבוט העסקי, ולמה

| הושמט | למה |
|---|---|
| RAG (`rag/`, טבלאות KB) | מסמך אחד נכנס כולו לפרומפט — ראו ההכרעה בתחילת המדריך |
| זיהוי כוונות בקריאת AI נפרדת | מודל שרואה את המסמך המלא מטפל לבד בברכות ובשאלות כלליות. קריאה נוספת = עוד עלות ועוד השהיה |
| העברה לבעל העסק (`_notify_owner`) | הוחלפה בקישור קבוע. שום דבר מהשיחה לא מועבר הלאה, והגולש מחליט בעצמו אם לפנות. זה גם מייתר באג במקור: `_notify_owner` מחזיר `False` כשההתראה נכשלת, ו-`_create_request_and_notify_owner` מתעלם מזה |
| סיכומי שיחה | נכנסים לפעולה רק אחרי 10 הודעות. שיחת ייעוץ לפני רכישה כמעט לא מגיעה לשם |
| ציון מקור (`מקור: ...`) | יש מקור אחד |
| מסך הסכמה, `/forget` | הבוט שומר רק את תוכן השיחה, ל-7 ימים, ומודיע על זה ב-`/start` |
| תורים, שעות פעילות, live chat, חופשה, הפניות, WhatsApp, multi-tenant, פאנל | לא רלוונטיים ליועץ |

---

## 12. מה אומת ומה לא

**אומת:**
- 38 טסטים עוברים, ו-14 שבירות מכוונות של הקוד נתפסו כל אחת בטסט שלה. לפני כל שבירה ריצת הבסיס עברה, והכשל אחריה היה ב-assert ולא בקריסה.
- הרצה מקצה לקצה עם python-telegram-bot ו-openai האמיתיים, מול שרת Bot API מקומי ושרת OpenAI מקומי: הודעה ← חיווי הקלדה ← תשובה ב-HTML ← כפתורי שאלות המשך ← לחיצה ← תשובה עם כפתור יצירת קשר. הבקשה ל-OpenAI יצאה עם `max_completion_tokens` ובלי `temperature`, ההוראות בתפקיד `developer`, ובקריאה השנייה ההיסטוריה כללה את הזוג הראשון. הודעה בקבוצה לא הגיעה ל-AI. הודעה ערוכה, גם טקסט וגם `/start`, לא הגיעה לאף handler — לפני התיקון שתיהן הפילו `AttributeError`, והגולש קיבל "משהו השתבש".
- הלוג של אותה הרצה: בלי הטוקן, בלי שם הגולש ובלי טקסט ההודעה. אותה הרצה בלי השורה שמשתיקה את `httpx`: הטוקן הופיע ב-17 שורות.
- ולידציית ההגדרות: הרצה בלי טוקן, בלי מפתח, בלי מסמך ועם `CONTACT_URL` שגוי — ארבע שגיאות ברורות ויציאה עם קוד 1.

**לא אומת:**
- **מול ה-API האמיתי של OpenAI**, כי לא היה מפתח בסביבת הבדיקה. שם המודל `gpt-5.4-mini`, `max_completion_tokens` וויתור על `temperature` לקוחים מקוד ה-SDK ומעמוד המודל. בהרצה הראשונה, שימו לב לשגיאות 400 בלוג.
- **אחוזי פגיעה אמיתיים במטמון.** השרת המקומי החזיר מספר קבוע. בפועל בודקים את `cached_tokens` בלוג.
- **מול טלגרם האמיתי**, כולל איך HTML בעברית נראה באפליקציה.

---

## 13. הרחבות אפשריות

- **`/reset`** — "שיחה חדשה", מחיקת ההיסטוריה של הגולש. פונקציה אחת ב-`history.py` ו-handler אחד.
- **webhook במקום polling** — `run_bot` ב-`bot/telegram_bot.py` של הריפו המקורי מראה את שני המצבים. polling פשוט יותר ומספיק לבוט בהיקף הזה.
- **טעינת המסמך מ-Ck** במקום מהריפו, עם פקודת `/reload` — עדכון בלי deploy, במחיר תלות בזמינות של Ck בעלייה.
