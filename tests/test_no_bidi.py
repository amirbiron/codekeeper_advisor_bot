"""אין תווי BiDi בקבצי הקוד.

הם נכנסים להערות בעברית בהעתקה מדפדפן או מצ'אט, לא נראים בשום עורך, ומפילים
לינטרים (bandit B613, ruff PLE2502). מקור הכלל: amir-bug-patterns,
BY-STACK/hebrew-source.md H1.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ה-escapes מכוונים: התווים עצמם בקובץ הזה היו מפילים את הבדיקה על עצמה
BIDI = ("\u200e", "\u200f", "\u202a", "\u202b", "\u202c",
        "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069")

# סביבות וירטואליות ותיקיות נסתרות (.git, .venv) אינן קוד שלנו
_SKIP_DIRS = {"venv", "env", "__pycache__"}


def _source_files() -> list[Path]:
    return [
        path for path in ROOT.rglob("*.py")
        if not any(part.startswith(".") or part in _SKIP_DIRS for part in path.relative_to(ROOT).parts[:-1])
    ]


def test_no_bidi_controls_in_source():
    files = _source_files()
    assert files  # סריקה שלא מצאה קבצים הייתה עוברת על כל דבר
    for path in files:
        text = path.read_text(encoding="utf-8")
        for char in BIDI:
            assert char not in text, f"{path.relative_to(ROOT)}: תו BiDi {char!r}"
