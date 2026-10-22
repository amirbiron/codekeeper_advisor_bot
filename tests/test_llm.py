from types import SimpleNamespace

import httpx2
import openai
import pytest
from openai.types.chat import ChatCompletion

import llm


# ─── פענוח ───────────────────────────────────────────────────────────────────


def test_extract_and_strip_follow_ups():
    text = "תשובה.\n\n[שאלות_המשך: יש גיבוי? | עובד בנייד? | כמה זה עולה?]"
    assert llm.extract_follow_up_questions(text) == ["יש גיבוי?", "עובד בנייד?", "כמה זה עולה?"]
    assert llm.strip_follow_up_questions(text) == "תשובה."


def test_follow_ups_without_brackets_are_parsed_too():
    text = "תשובה.\nשאלות המשך: א | ב"
    assert llm.extract_follow_up_questions(text) == ["א", "ב"]
    assert llm.strip_follow_up_questions(text) == "תשובה."


def test_contact_marker_detected_only_at_start_but_removed_everywhere():
    assert llm.split_contact_marker("[CONTACT]\nאין לי מידע") == (True, "אין לי מידע")
    assert llm.split_contact_marker("טקסט [CONTACT] באמצע") == (False, "טקסט  באמצע")


def test_trim_to_last_sentence():
    assert llm.trim_to_last_sentence("שורה שלמה.\nשורה שנקט") == "שורה שלמה.\n…"
    assert llm.trim_to_last_sentence("משפט ראשון. משפט שני שנק") == "משפט ראשון. …"


# ─── פרומפט ──────────────────────────────────────────────────────────────────


def test_system_prompt_is_static_and_keeps_knowledge_verbatim():
    knowledge = "מסמך עם {סוגריים} ו-%s"
    first = llm.build_system_prompt(knowledge)
    assert first == llm.build_system_prompt(knowledge)  # תנאי להתאמה במטמון של OpenAI
    assert knowledge in first
    assert first.rstrip().endswith("── סוף מסמך הידע ──")


def test_follow_up_rule_toggles():
    assert "שאלות_המשך" in llm.build_system_prompt("x", follow_up_enabled=True)
    assert "שאלות_המשך" not in llm.build_system_prompt("x", follow_up_enabled=False)


def test_build_messages_order():
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    msgs = llm.build_messages("PROMPT", history, "q2")
    assert [m["role"] for m in msgs] == ["developer", "user", "assistant", "user"]
    assert msgs[0]["content"] == "PROMPT" and msgs[-1]["content"] == "q2"


# ─── generate_answer מול כפיל של ה-client ───────────────────────────────────


def _completion(content, finish_reason="stop", cached=0):
    # אובייקט אמיתי של ה-SDK, לא namespace — כך שגישה לשדה לא קיים נכשלת
    return ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-5.4-mini",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 2000, "completion_tokens": 20, "total_tokens": 2020,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    })


class StubClient:
    def __init__(self, response=None, error=None):
        self.calls: list[dict] = []
        self._response, self._error = response, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return self._response


@pytest.fixture
def use_client(monkeypatch):
    monkeypatch.setattr(llm, "_system_prompt", "PROMPT")

    def _install(client):
        monkeypatch.setattr(llm, "_client", client)
        return client

    return _install


def test_request_uses_max_completion_tokens_and_omits_unset_params(use_client, monkeypatch):
    monkeypatch.setattr(llm, "LLM_TEMPERATURE", "")
    monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", "")
    client = use_client(StubClient(_completion("תשובה.")))
    llm.generate_answer([], "שאלה")
    kwargs = client.calls[0]
    assert "max_completion_tokens" in kwargs and "max_tokens" not in kwargs
    assert "temperature" not in kwargs and "reasoning_effort" not in kwargs


def test_optional_params_are_sent_when_configured(use_client, monkeypatch):
    monkeypatch.setattr(llm, "LLM_TEMPERATURE", "0.3")
    monkeypatch.setattr(llm, "OPENAI_REASONING_EFFORT", "low")
    client = use_client(StubClient(_completion("תשובה.")))
    llm.generate_answer([], "שאלה")
    assert client.calls[0]["temperature"] == 0.3
    assert client.calls[0]["reasoning_effort"] == "low"


def test_answer_with_follow_ups(use_client, monkeypatch):
    monkeypatch.setattr(llm, "FOLLOW_UP_ENABLED", True)
    use_client(StubClient(_completion("כן, זה מתאים.\n[שאלות_המשך: א | ב]")))
    answer = llm.generate_answer([], "שאלה")
    assert answer == llm.Answer(text="כן, זה מתאים.", ok=True, follow_up_questions=["א", "ב"])


def test_contact_answer_has_no_follow_ups(use_client, monkeypatch):
    monkeypatch.setattr(llm, "FOLLOW_UP_ENABLED", True)
    use_client(StubClient(_completion("[CONTACT]\nאין לי מידע על זה.\n[שאלות_המשך: א]")))
    answer = llm.generate_answer([], "שאלה")
    assert answer.needs_contact and answer.ok
    assert answer.text == "אין לי מידע על זה." and answer.follow_up_questions == []


def test_bare_contact_marker_gets_default_text(use_client):
    use_client(StubClient(_completion("[CONTACT]")))
    answer = llm.generate_answer([], "שאלה")
    assert answer.needs_contact and answer.text == llm.CONTACT_DEFAULT_TEXT


def test_api_error_returns_fallback_not_raise(use_client):
    error = openai.APITimeoutError(request=httpx2.Request("POST", "https://api.openai.com/v1/chat/completions"))
    use_client(StubClient(error=error))
    assert llm.generate_answer([], "שאלה") == llm.Answer(text=llm.FALLBACK_ERROR, ok=False)


def test_empty_content_returns_fallback(use_client):
    use_client(StubClient(_completion(None)))
    assert llm.generate_answer([], "שאלה").ok is False


def test_length_finish_reason_trims(use_client):
    use_client(StubClient(_completion("שורה שלמה.\nשורה שנקט", finish_reason="length")))
    assert llm.generate_answer([], "שאלה").text == "שורה שלמה.\n…"


def test_diag_log_has_numbers_but_no_content(use_client, caplog):
    caplog.set_level("INFO")
    use_client(StubClient(_completion("התשובה הסודית של הגולש.", cached=1920)))
    llm.generate_answer([], "שאלה")
    assert "cached_tokens=1920" in caplog.text
    assert "הסודית" not in caplog.text


def test_init_llm_rejects_empty_knowledge(tmp_path, monkeypatch):
    empty = tmp_path / "k.md"
    empty.write_text("  \n", encoding="utf-8")
    monkeypatch.setattr(llm, "KNOWLEDGE_FILE", empty)
    with pytest.raises(RuntimeError):
        llm.init_llm()
