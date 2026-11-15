"""Агент 1 — Spec Extractor. Сырой бриф/ТЗ → структурированный Spec."""

from __future__ import annotations

from app import config
from app.agents import load_prompt
from app.llm import call_structured
from app.schemas import Spec


def extract_spec(raw_input: str) -> Spec:
    system = load_prompt("extractor")
    user = (
        "Вот что прислал клиент. Определи режим ввода и извлеки требования.\n\n"
        "=== ВХОД КЛИЕНТА ===\n"
        f"{raw_input.strip()}"
    )
    return call_structured(config.MODEL_EXTRACTOR, system, user, Spec)
