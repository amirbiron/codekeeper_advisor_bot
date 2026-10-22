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
    "היי! אני כאן כדי לעזור לך לבדוק בכנות אם CodeKeeper באמת מתאים לאיך שאתה עובד עם סוכני AI, "
    "או שאולי פתרון פשוט יותר יספיק לך. כדי שלא אשפוך עליך סתם מידע, בוא נתחיל מאיך אתה עובד היום: "
    "באילו סוכנים אתה משתמש (Claude Code, צ'אט בדפדפן), "
    "והאם קורה לך הרבה שהם שוכחים החלטות ארכיטקטורה ישנות וחוזרים על טעויות?\n\n"
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


def _user_lock(context: ContextTypes.DEFAULT_TYPE) -> asyncio.Lock:
    """מנעול לכל גולש — כל מה שקורא או כותב את ההיסטוריה שלו עובר דרכו, אחד-אחד."""
    return context.user_data.setdefault("answer_lock", asyncio.Lock())


async def _answer_and_reply(context: ContextTypes.DEFAULT_TYPE, *, user_id: str, chat_id: int, text: str) -> None:
    """היסטוריה ← AI ← שליחה ← שמירה ← שאלות המשך.

    הבוט מעבד עדכונים של משתמשים שונים במקביל (concurrent_updates), אבל
    הודעות של אותו משתמש עוברות אחת-אחת דרך מנעול ב-user_data. בלי המנעול,
    שתי הודעות מהירות היו קוראות את אותה היסטוריה, והתשובות היו מגיעות
    בסדר אקראי.
    """
    async with _user_lock(context):
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
    """הפתיחה נשמרת בהיסטוריה כמו כל זוג שאלה-תשובה.

    היא שואלת את הגולש שאלות, ובלעדיה בהיסטוריה התשובה הראשונה שלו ("כן, כל
    הזמן") הייתה מגיעה ל-AI בלי השאלה שהיא עונה עליה. תחת אותו מנעול כמו
    _answer_and_reply, כדי שהודעה שנשלחת מיד אחרי /start תראה את הפתיחה.
    """
    user_id = str(update.effective_user.id)
    async with _user_lock(context):
        await update.message.reply_text(WELCOME_TEXT, parse_mode=ParseMode.HTML)
        try:
            await asyncio.to_thread(save_exchange, user_id, update.message.text, WELCOME_TEXT)
        except sqlite3.Error:
            logger.exception("Failed to save welcome for user %s", user_id)


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
