"""המדריך ב-docs/ מכיל עותק מלא של קבצי הפרויקט — והטסט שומר ששני העותקים זהים.

הטענה המרכזית של המדריך היא שכל קובץ בו הוא קוד שרץ ונבדק. בלי הטסט, שינוי
בקוד בלי אותו שינוי במדריך (או להפך) היה הופך את הטענה לשקרית בשקט.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE = ROOT / "docs" / "codekeeper_advisor_bot_guide.md"

# כותרת קובץ במדריך: שורה שכולה **`path`**, ואחריה בלוק קוד
_HEADER = re.compile(r"^\*\*`([^`]+)`\*\*$")


def _guide_files() -> dict[str, str]:
    lines = GUIDE.read_text(encoding="utf-8").splitlines()
    files: dict[str, str] = {}
    i = 0
    while i < len(lines):
        match = _HEADER.match(lines[i])
        if not match:
            i += 1
            continue
        start = next(j for j in range(i + 1, len(lines)) if lines[j].startswith("```")) + 1
        end = next(j for j in range(start, len(lines)) if lines[j] == "```")
        path = match.group(1)
        assert path not in files, f"{path} מופיע פעמיים במדריך"
        files[path] = "\n".join(lines[start:end]) + "\n"
        i = end + 1
    return files


def test_every_file_in_the_guide_matches_the_repo():
    files = _guide_files()
    assert files  # פענוח שלא מצא אף קובץ היה עובר על כל דבר
    for path, body in files.items():
        assert (ROOT / path).read_text(encoding="utf-8") == body, f"{path}: הקוד במדריך שונה מהקובץ בריפו"
