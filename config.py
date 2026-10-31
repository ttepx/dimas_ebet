"""
Налаштування бота. Секрети беруться з .env або змінних середовища.
"""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _key(name: str):
    """Повертає ключ або None, якщо він порожній чи є заглушкою."""
    value = (os.getenv(name) or "").strip()
    if not value or "ТВІЙ" in value.upper() or value.lower().startswith("your"):
        return None
    return value


BOT_TOKEN = _key("BOT_TOKEN")

GROQ_API_KEY = _key("GROQ_API_KEY")
OPENROUTER_API_KEY = _key("OPENROUTER_API_KEY")
GEMINI_API_KEY = _key("GEMINI_API_KEY")

# У списку навмисно залишені лише провайдери та моделі з безкоштовним режимом.
PROVIDER_ORDER = os.getenv("PROVIDER_ORDER", "groq,openrouter,gemini")

MODELS = {
    "groq": "openai/gpt-oss-20b",
    "openrouter": "openrouter/free",
    "gemini": "gemini-2.5-flash",
}
