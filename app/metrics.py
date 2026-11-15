"""Счётчики одного прогона: токены по моделям и срабатывания guardrails.

Перед прогоном вызывается start_run(), после — get_run_stats().
Хранится в contextvar, потому что LangGraph выполняет узлы в своих потоках,
а контекст туда копируется (ссылка на один и тот же словарь).
"""

from __future__ import annotations

from contextvars import ContextVar

from app import config

_RUN_STATS: ContextVar[dict | None] = ContextVar("run_stats", default=None)


def start_run() -> dict:
    stats = {
        "tokens": {},
        "llm_calls": 0,
        "estimator_retries": 0,
        "nda_portfolio_removed": 0,
        "nda_text_fixed": 0,
    }
    _RUN_STATS.set(stats)
    return stats


def get_run_stats() -> dict:
    return _RUN_STATS.get() or {}


def add_usage(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    stats = _RUN_STATS.get()
    if stats is None:
        return
    t = stats["tokens"].setdefault(model, {"prompt": 0, "completion": 0})
    t["prompt"] += prompt_tokens
    t["completion"] += completion_tokens
    stats["llm_calls"] += 1


def count(key: str, n: int = 1) -> None:
    stats = _RUN_STATS.get()
    if stats is not None:
        stats[key] = stats.get(key, 0) + n


def cost_usd(tokens: dict) -> float:
    """Стоимость по ценам из config.PRICES_PER_1M (USD за 1M токенов)."""
    total = 0.0
    for model, t in tokens.items():
        price_in, price_out = config.PRICES_PER_1M.get(model, (0.0, 0.0))
        total += t["prompt"] / 1e6 * price_in + t["completion"] / 1e6 * price_out
    return round(total, 5)
