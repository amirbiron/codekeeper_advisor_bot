from telegram_format import html_to_plain, sanitize_telegram_html, split_message


def test_sanitize_keeps_telegram_tags_and_escapes_the_rest():
    out = sanitize_telegram_html("<b>שלום</b> <script>x</script> a < b & c")
    assert out == "<b>שלום</b> &lt;script&gt;x&lt;/script&gt; a &lt; b &amp; c"


def test_sanitize_drops_tag_with_attributes_and_its_closing_tag():
    assert sanitize_telegram_html('<b class="x">טקסט</b>') == "טקסט"


def test_html_to_plain_restores_real_characters():
    sanitized = sanitize_telegram_html("<b>א</b> < ב & ג")
    assert html_to_plain(sanitized) == "א < ב & ג"


def test_split_message_respects_limit_and_keeps_all_content():
    paragraphs = [f"פסקה {i} " + "מילה " * 30 for i in range(40)]
    text = "\n\n".join(paragraphs)
    chunks = split_message(text, limit=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")


def test_split_message_prefers_paragraph_boundary():
    text = "א" * 300 + "\n\n" + "ב" * 300
    assert split_message(text, limit=500) == ["א" * 300, "ב" * 300]


def test_split_message_hard_cuts_a_single_huge_word():
    chunks = split_message("x" * 1200, limit=500)
    assert [len(c) for c in chunks] == [500, 500, 200]


def test_short_message_is_one_chunk():
    assert split_message("  שלום  ") == ["שלום"]
