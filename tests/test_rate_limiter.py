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
    """בדיקת מקביליות רצה במקביל: 15 עדכונים בבת אחת ← בדיוק 10 עוברים."""
    ran = []

    @rate_limiter.rate_limit_guard
    async def handler(update, context):
        await asyncio.sleep(0.01)  # ה-handler משחרר את ה-loop, כמו קריאה ל-AI
        ran.append(1)

    async def main():
        await asyncio.gather(*(handler(_update(), None) for _ in range(15)))

    asyncio.run(main())
    assert len(ran) == rate_limiter.RATE_LIMIT_PER_MINUTE
