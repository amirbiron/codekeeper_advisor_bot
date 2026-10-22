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
