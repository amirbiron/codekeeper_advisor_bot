# codekeeper_advisor_bot

בוט טלגרם — "היועץ של CodeKeeper". גולש שמתלבט אם Ck מתאים לו מספר איך הוא עובד, והבוט עונה בכנות אם ואיך Ck יעזור לו. התשובות מגיעות ממסמך ידע אחד, `knowledge/codekeeper.md`, שנכנס כולו לפרומפט של OpenAI, והבוט זוכר את רצף השיחה שבוע אחד.

המדריך המלא — כל קובץ, וההסבר לכל החלטה: [`docs/codekeeper_advisor_bot_guide.md`](docs/codekeeper_advisor_bot_guide.md).

## הרצה

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # למלא TELEGRAM_BOT_TOKEN ו-OPENAI_API_KEY
python main.py
```

טסטים, בלי רשת:

```bash
python -m pytest -q
```

## עדכון מסמך הידע

עורכים את `knowledge/codekeeper.md`, ואז commit ו-deploy. המסמך נטען פעם אחת, בעליית הבוט.
