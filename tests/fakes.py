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
