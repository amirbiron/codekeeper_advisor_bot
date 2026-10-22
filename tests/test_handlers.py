import asyncio
import datetime as dtm
import time
from types import SimpleNamespace

from telegram import Chat, Message, MessageEntity, Update, User

import handlers
import llm
from fakes import FakeBot
from telegram_format import RLM


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


def test_start_saves_the_welcome_so_the_first_reply_has_its_question(tmp_db, fake_bot):
    """הפתיחה שואלת את הגולש שאלות — בלעדיה בהיסטוריה, התשובה שלו מגיעה ל-AI בלי השאלה."""
    message = Message(
        message_id=1,
        date=dtm.datetime.now(dtm.timezone.utc),
        chat=Chat(id=1, type="private"),
        from_user=User(id=1, first_name="Dana", is_bot=False),
        text="/start",
    )
    message.set_bot(fake_bot)
    asyncio.run(handlers.start_command(Update(update_id=1, message=message), _context(fake_bot)))
    assert fake_bot.sent[0]["text"] == handlers.WELCOME_TEXT
    assert handlers.get_history("1", 10) == [
        {"role": "user", "content": "/start"},
        {"role": "assistant", "content": handlers.WELCOME_TEXT},
    ]


class _SlowBot(FakeBot):
    async def send_message(self, **kwargs):
        await asyncio.sleep(0.05)  # שליחה איטית: בלי המנעול, ההודעה הבאה קוראת היסטוריה לפני שהפתיחה נשמרה
        return await super().send_message(**kwargs)


def test_message_right_after_start_sees_the_welcome(tmp_db, monkeypatch):
    """בדיקת מקביליות רצה במקביל: /start והודעה מיד אחריו — ה-AI חייב לראות את הפתיחה."""
    seen_history = []

    def fake_generate(history, text):
        seen_history.append([m["content"] for m in history])
        return llm.Answer(text="תשובה", ok=True)

    monkeypatch.setattr(handlers, "generate_answer", fake_generate)
    bot = _SlowBot()
    context = _context(bot)
    message = Message(
        message_id=1,
        date=dtm.datetime.now(dtm.timezone.utc),
        chat=Chat(id=1, type="private"),
        from_user=User(id=1, first_name="Dana", is_bot=False),
        text="/start",
    )
    message.set_bot(bot)

    async def main():
        await asyncio.gather(
            handlers.start_command(Update(update_id=1, message=message), context),
            handlers._answer_and_reply(context, user_id="1", chat_id=1, text="כן, כל הזמן"),
        )

    asyncio.run(main())
    assert seen_history == [["/start", handlers.WELCOME_TEXT]]


def test_english_first_lines_reach_telegram_rtl_but_history_stays_clean(tmp_db, monkeypatch, fake_bot):
    """טלגרם מציג שורה עברית שמתחילה באנגלית משמאל לימין — RLM מתקן, רק ביציאה."""
    answer_text = "<b>CodeKeeper</b> עוזר למפתחים.\nהמסמך נגיש מכל סוכן."
    monkeypatch.setattr(handlers, "generate_answer", lambda h, t: llm.Answer(
        text=answer_text, ok=True, follow_up_questions=["Claude Code מתאים לי?"]))
    asyncio.run(handlers._answer_and_reply(_context(fake_bot), user_id="u1", chat_id=1, text="שאלה"))
    first, second = fake_bot.sent[0]["text"].split("\n")
    assert first.startswith(RLM) and not second.startswith(RLM)
    assert fake_bot.sent[-1]["reply_markup"].inline_keyboard[0][0].text.startswith(RLM)
    assert handlers.get_history("u1", 10)[1]["content"] == answer_text  # הזיכרון בלי סימנים נסתרים


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
