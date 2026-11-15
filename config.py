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
MISTRAL_API_KEY = _key("MISTRAL_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
WEBHOOK_BASE_URL = (
    os.getenv("WEBHOOK_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL") or ""
).strip().rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()

# Free providers run first; Mistral is an optional paid fallback.
_provider_order = [
    provider.strip()
    for provider in os.getenv("PROVIDER_ORDER", "groq,openrouter,gemini").split(",")
    if provider.strip()
]
if MISTRAL_API_KEY and "mistral" not in _provider_order:
    _provider_order.append("mistral")
PROVIDER_ORDER = ",".join(_provider_order)

MODELS = {
    "groq": "openai/gpt-oss-20b",
    "openrouter": "openrouter/free",
    "gemini": "gemini-2.5-flash",
    "mistral": "mistral-small-latest",
}
