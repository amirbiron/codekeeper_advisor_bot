"""
הכנת טקסט לשליחה בטלגרם — סניטציית HTML, גרסת טקסט רגיל, פיצול הודעות ארוכות
וכיוון שורות.

מקור: llm.sanitize_telegram_html בריפו ai-business-bot (כמו שהוא).
שני חדשים, שסוגרים חורים שקיימים במקור:
- html_to_plain: במקור, כשטלגרם דחה HTML, נשלח אותו טקסט בלי parse_mode —
  והגולש ראה תגים ו-&amp; כטקסט. כאן ה-fallback הוא טקסט נקי באמת.
- split_message: טלגרם דוחה הודעה מעל 4096 תווים. במקור אין פיצול, ותשובה
  ארוכה נכשלת כולה.
ובנוסף rtl_lines: שורה עברית שמתחילה במילה באנגלית מוצגת בטלגרם משמאל לימין.
"""

import html as _html
import re
import unicodedata

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


# RIGHT-TO-LEFT MARK: תו בלתי נראה מסוג R — "אות" מימין לשמאל שאינה מוצגת
RLM = "\u200f"

# שורות בתוך <pre> הן קוד: תו נסתר שם נכנס למה שהגולש מעתיק
_PRE_TAG_RE = re.compile(r"</?pre>")
_STRONG = ("L", "R", "AL")


def _starts_ltr_in_rtl_line(visible: str) -> bool:
    """האות החזקה הראשונה משמאל לימין, ויש בשורה גם אות מימין לשמאל."""
    classes = [unicodedata.bidirectional(ch) for ch in visible]
    first_strong = next((c for c in classes if c in _STRONG), None)
    return first_strong == "L" and any(c in ("R", "AL") for c in classes)


def rtl_lines(text: str) -> str:
    """RLM בתחילת כל שורה עברית שהאות הראשונה בה באנגלית.

    טלגרם קובע את הכיוון של כל שורה לפי האות החזקה הראשונה בה — כל שורה היא
    פסקה, והכיוון שלה נקבע לפי התו הראשון מסוג L, R או AL. שורה כמו
    "CodeKeeper עוזר למפתחים" מוצגת לכן משמאל לימין, והסדר בה מתבלבל.
    תבליט, אימוג'י ומספר אינם אותיות חזקות, ולכן "• CodeKeeper ..." נשברת גם היא.
    מקור: https://www.unicode.org/reports/tr9/#P2 (כללים P1-P3)

    הבדיקה על הטקסט הגלוי, בלי תגים — <b> לא מוצג, ולכן אינו קובע כיוון.
    שורה שמתחילה כבר ב-RLM לא מקבלת עוד אחד, ושורות בתוך <pre> לא משתנות.
    """
    out = []
    in_pre = False
    for line in text.split("\n"):
        starts_in_code = in_pre or line.lstrip().startswith("<pre>")
        for tag in _PRE_TAG_RE.findall(line):
            in_pre = tag == "<pre>"
        if not starts_in_code and _starts_ltr_in_rtl_line(html_to_plain(line)):
            line = RLM + line
        out.append(line)
    return "\n".join(out)


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
