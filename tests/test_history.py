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
