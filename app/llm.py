"""Обёртка над OpenAI с fallback на Anthropic.

Две формы вызова:
  - call_structured: strict structured output по Pydantic-схеме
                     (для extractor / estimator / risk / writer)
  - call_json:       JSON-объект без strict-схемы
                     (для reviser — ProposalState глубже strict-лимита)

Fallback-стратегия: если основной OpenAI-вызов падает с любой transient
ошибкой (rate limit, connection error, 5xx, internal server error) — и
при этом задан ANTHROPIC_API_KEY — пробуем тот же запрос на Claude (MODEL_FALLBACK)
с тем же JSON-форматом. Без ключа fallback no-op, ошибка пробрасывается.

Это паттерн готов в коде, активируется добавлением одной env переменной.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable, Type, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel

from app import config
from app.metrics import add_usage

T = TypeVar("T", bound=BaseModel)

log = logging.getLogger(__name__)

# Ошибки OpenAI, на которые имеет смысл переключаться на fallback.
# RateLimit и 5xx — точно transient. APIConnectionError — сетевые проблемы.
# AuthenticationError и BadRequest — НЕ включаем (это наша проблема, не провайдера).
FALLBACK_TRIGGERS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY не задан. Скопируй .env.example в .env и впиши ключ.")
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    # LangSmith-обёртка: при включённом LANGSMITH_TRACING все запросы и
    # ответы OpenAI автоматически летят в дашборд. Без env — no-op.
    if config.LANGSMITH_TRACING:
        try:
            from langsmith.wrappers import wrap_openai

            client = wrap_openai(client)
        except Exception:  # noqa: BLE001
            # не валим приложение из-за проблем с обсервабилити
            pass
    return client


@lru_cache(maxsize=1)
def _anthropic_client():
    """Anthropic client опционально. None если ключ не задан или SDK нет."""
    if not config.ANTHROPIC_API_KEY:
        return None
    try:
        from anthropic import Anthropic

        return Anthropic(api_key=config.ANTHROPIC_API_KEY)
    except ImportError:
        return None


def _with_fallback(primary: Callable, fallback: Callable | None):
    """Запускает primary; при transient-ошибке OpenAI пробует fallback.

    Если fallback недоступен (нет ключа Anthropic) — primary-ошибка
    пробрасывается без перехвата.
    """
    try:
        return primary()
    except FALLBACK_TRIGGERS as exc:
        if fallback is None:
            raise
        log.warning(
            "OpenAI transient error (%s), переключаюсь на Anthropic fallback...",
            type(exc).__name__,
        )
        return fallback()


def call_structured(
    model: str,
    system: str,
    user: str,
    schema: Type[T],
    temperature: float = 0.3,
) -> T:
    """Один вызов LLM со строгим structured output по Pydantic-схеме.

    temperature по умолчанию 0.3 (стандарт для structured generation —
    низкая стохастика для надёжного парсинга при сохранении живости текста).
    Переопределяется в evals/temperature.py для A/B по температуре.
    """

    def primary():
        completion = _client().beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
            temperature=temperature,
        )
        if completion.usage:
            add_usage(model, completion.usage.prompt_tokens, completion.usage.completion_tokens)
        message = completion.choices[0].message
        if message.refusal:
            raise RuntimeError(f"Модель отказалась отвечать: {message.refusal}")
        if message.parsed is None:
            raise RuntimeError("Пустой ответ модели при structured output.")
        return message.parsed

    anthropic = _anthropic_client()
    fallback = None
    if anthropic is not None:

        def fallback():
            # Structured outputs Anthropic: SDK передаёт схему Pydantic-модели
            # и сам валидирует ответ. temperature не передаём — текущие модели
            # Claude не принимают параметры сэмплинга.
            resp = anthropic.messages.parse(
                model=config.MODEL_FALLBACK,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
            )
            if resp.stop_reason == "refusal":
                raise RuntimeError("Anthropic fallback: модель отказалась отвечать")
            if resp.parsed_output is None:
                raise RuntimeError("Anthropic fallback: пустой ответ")
            return resp.parsed_output

    return _with_fallback(primary, fallback)


def call_json(model: str, system: str, user: str, temperature: float = 0.2) -> str:
    """Вызов LLM в режиме JSON-объекта (без strict-схемы).

    Нужен для ревизора: он возвращает целиком ProposalState, чьё дерево
    глубже лимита strict structured output. Валидируем ответ Pydantic-ом
    на своей стороне.
    """

    def primary():
        completion = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        if completion.usage:
            add_usage(model, completion.usage.prompt_tokens, completion.usage.completion_tokens)
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("Пустой ответ модели (JSON mode).")
        return content

    anthropic = _anthropic_client()
    fallback = None
    if anthropic is not None:

        def fallback():
            resp = anthropic.messages.create(
                model=config.MODEL_FALLBACK,
                max_tokens=16000,
                system=system
                + "\n\nReturn a single valid JSON object. No prose, no markdown fences.",
                messages=[{"role": "user", "content": user}],
            )
            if resp.stop_reason == "refusal":
                raise RuntimeError("Anthropic fallback: модель отказалась отвечать")
            text = "".join(b.text for b in resp.content if b.type == "text")
            start = text.find("{")
            end = text.rfind("}") + 1
            if start < 0 or end <= start:
                raise RuntimeError("Anthropic fallback: JSON не найден в ответе")
            return text[start:end]

    return _with_fallback(primary, fallback)


def embed(texts: list[str]) -> list[list[float]]:
    """Векторизация списка текстов через OpenAI embeddings.

    Fallback на Anthropic не применяется — у Anthropic нет embeddings API.
    На production для отказоустойчивости embeddings берут Voyage / Cohere
    как secondary provider; для прототипа этого не делаем.
    """
    resp = _client().embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    if resp.usage:
        add_usage(config.EMBEDDING_MODEL, resp.usage.prompt_tokens, 0)
    return [item.embedding for item in resp.data]
