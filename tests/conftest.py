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
