"""Multi-agent цепочка: Spec Extractor → Estimator → Risk Analyst → Writer."""

from __future__ import annotations

from functools import lru_cache

from app import config


@lru_cache(maxsize=8)
def load_prompt(name: str) -> str:
    """Загружает системный промпт агента из app/prompts/<name>.md."""
    path = config.PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")
