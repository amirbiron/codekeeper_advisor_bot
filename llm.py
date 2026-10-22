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
