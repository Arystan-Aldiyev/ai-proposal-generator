"""Центральная конфигурация. Пути и модели в одном месте."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows-консоль по умолчанию cp1252 — принудительно UTF-8, иначе print()
# с кириллицей падает с UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Данные (датасет агентства лежит в корне проекта) ---
DATA_DIR = ROOT
PROJECTS_METADATA = DATA_DIR / "projects_metadata.txt"
RAG_CARDS = DATA_DIR / "rag_cards.md"
PORTFOLIO = DATA_DIR / "portfolio.md"

# --- Хранилища ---
STORAGE = ROOT / "storage"
OUTPUT_DIR = STORAGE / "output"
# RAG-индекс: эмбеддинги 6 карточек + метаданные в одном JSON.
# Векторная БД для 6 документов избыточна — храним вектор-индекс файлом,
# поиск делаем brute-force косинусом (см. app/rag/retriever.py).
INDEX_PATH = STORAGE / "rag_index.json"
# Отдельный индекс для Portfolio — те же эмбеддинги, но другой корпус
# (28 проектов с публичными метками вместо 6 КП-карточек для Estimator).
PORTFOLIO_INDEX_PATH = STORAGE / "portfolio_index.json"

# --- Шаблоны / статика ---
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"
WEB_DIR = ROOT / "web"
PROMPTS_DIR = ROOT / "app" / "prompts"

# --- LLM ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Fallback провайдер (опционально). Активируется автоматически при сбоях
# OpenAI если ключ задан. Без ключа — fallback no-op, ошибка пробрасывается.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MODEL_FALLBACK = os.getenv("MODEL_FALLBACK", "claude-sonnet-5-5")

# --- Telegram-бот (третий клиент MCP) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
# URL, по которому в публичном Telegram будут видны ссылки на PDF.
# Локально http://localhost:8000, в продакшене — https://your-domain.com.
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")

# --- Доступ к веб-интерфейсу ---
# Если APP_PASSWORD задан, все страницы и API требуют HTTP Basic.
# Обязательно для любого запуска не на своём компьютере.
APP_USER = os.getenv("APP_USER", "admin")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
MODEL_EXTRACTOR = os.getenv("MODEL_EXTRACTOR", "gpt-4o-mini")
# Estimator поднят до gpt-4o: на mini варианты получались идентичными по
# составу этапов (различались только числами дней) — модель не дифференцировала
# Эконом и Корпоративный структурно. С gpt-4o дифференциация чёткая.
MODEL_ESTIMATOR = os.getenv("MODEL_ESTIMATOR", "gpt-4o")
MODEL_RISK = os.getenv("MODEL_RISK", "gpt-4o-mini")
MODEL_WRITER = os.getenv("MODEL_WRITER", "gpt-4o")
MODEL_REVISER = os.getenv("MODEL_REVISER", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# --- Temperature override для temperature-A/B (evals/temperature.py) ---
# Если задано, applies к Writer (где тон-чувствительность максимальна).
# None = использовать дефолт в call_structured (0.3).
WRITER_TEMPERATURE = None

# --- Абляция для экспериментов (evals/run.py --config no-rag) ---
# False — узел retrieve не передаёт аналоги в Estimator и Writer.
USE_RAG = True

# Цены OpenAI, USD за 1M токенов (input, output), по openai.com/api/pricing.
# Нужны только для подсчёта стоимости прогона в evals.
PRICES_PER_1M = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}

# --- Observability (LangSmith) ---
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
# project и endpoint поддерживают кавычки в .env (dotenv их обычно сам срежет,
# но на всякий случай чистим). EU-инстанс требует явного endpoint.
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "cns-proposal").strip("\"'")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "").strip("\"'")
# Если LangSmith включён — пробрасываем настройки в окружение, чтобы
# @traceable и wrap_openai их подхватили (LangSmith читает оба имени —
# LANGSMITH_* и LANGCHAIN_*).
if LANGSMITH_TRACING:
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if LANGSMITH_API_KEY:
        os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
        os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
    os.environ.setdefault("LANGCHAIN_PROJECT", LANGSMITH_PROJECT)
    if LANGSMITH_ENDPOINT:
        os.environ.setdefault("LANGSMITH_ENDPOINT", LANGSMITH_ENDPOINT)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", LANGSMITH_ENDPOINT)

# --- Реквизиты исполнителя (footer КП, премиум-формат) ---
# Персональные данные и реквизиты не храним в коде — только в .env.
AGENCY = {
    "name": "CNS Agency",
    "logo_text": "CNS",  # плейсхолдер лого — одна переменная для замены
    "manager": os.getenv("AGENCY_MANAGER", "Менеджер проекта"),
    "phone": os.getenv("AGENCY_PHONE", ""),
    "email": os.getenv("AGENCY_EMAIL", ""),
    "legal": os.getenv("AGENCY_LEGAL", ""),
    "year": 2026,
}

# Тема оформления КП. Сейчас реализуем только премиум; вторую заложили переменной.
THEME = "premium"  # premium | invoice (invoice пока не реализован)

STORAGE.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
