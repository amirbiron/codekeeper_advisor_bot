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
