"""Агент 2 — Estimator. Spec + RAG-аналоги → смета (Estimation).

Дополнительно: программный QA-фильтр. После получения Estimation от LLM
проверяем что варианты структурно различаются (разные названия этапов,
не просто разные числа дней). Если варианты слишком похожи — даём
Estimator-у retry с явным указанием что переделать. Это страховка от
ошибки модели, которую сложно поймать промптом одним.
"""

from __future__ import annotations

import logging

from app import config
from app.agents import load_prompt
from app.llm import call_structured
from app.metrics import count
from app.rag.dataset import get_summary_patterns
from app.rag.retriever import Analogue
from app.schemas import Estimation, Spec

log = logging.getLogger(__name__)

MAX_QA_RETRIES = 2  # 1 первая + 2 retry = max 3 LLM-вызова


def _normalize_name(s: str) -> str:
    """Нормализуем для сравнения: lowercase + убрать пунктуацию и
    типовые «хвосты» (плюсы/тире/слова-связки)."""
    s = s.lower().strip()
    for ch in ".,:;()/-+&":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def _variants_indistinguishable(est: Estimation) -> str | None:
    """Возвращает причину если варианты слишком похожи, иначе None.

    «Слишком похожи» — это копи-паст одного варианта в другой. Критерий:
    - НЕТ ни одного этапа уникального хотя бы в одном варианте, И
    - НЕТ разницы в количестве этапов между вариантами.

    Названия типа «Аналитика» / «Тестирование» — типовые в любом проекте,
    их совпадение — не баг. Различие должно быть В СОСТАВЕ работ:
    например, отдельный «Backend + CMS» только в Корпоративном.
    """
    if len(est.variants) < 2:
        return None
    for i, v1 in enumerate(est.variants):
        for v2 in est.variants[i + 1 :]:
            n1 = {_normalize_name(s.name) for s in v1.stages}
            n2 = {_normalize_name(s.name) for s in v2.stages}
            if not n1 or not n2:
                continue
            unique_to_v1 = n1 - n2
            unique_to_v2 = n2 - n1
            same_length = len(n1) == len(n2)
            # Бракуем только если у обоих вариантов НЕТ уникальных
            # этапов И количество этапов одинаково — это копи-паст.
            if same_length and not unique_to_v1 and not unique_to_v2:
                return (
                    f'варианты "{v1.name}" и "{v2.name}" имеют идентичный '
                    f"набор этапов (тот же состав, то же количество). У них "
                    f"ДОЛЖЕН быть хотя бы один уникальный этап в каждом — "
                    f"например, у Корпоративного отдельный 'Backend + CMS' "
                    f"или 'Миграция данных', которого нет в Экономе, а у "
                    f"Эконома — 'Сборка на конструкторе' вместо 'Backend'."
                )
    return None


def estimate(spec: Spec, analogues: list[Analogue]) -> Estimation:
    system = load_prompt("estimator")
    analogues_text = "\n\n".join(a.as_context() for a in analogues) or "(аналоги не найдены)"
    base_user = (
        "Составь смету по требованиям клиента, опираясь на аналоги.\n\n"
        "=== ТРЕБОВАНИЯ КЛИЕНТА (Spec) ===\n"
        f"{spec.model_dump_json(indent=2)}\n\n"
        "=== ПОХОЖИЕ ПРОШЛЫЕ ПРОЕКТЫ (RAG) ===\n"
        f"{analogues_text}\n\n"
        "=== СВОДНЫЕ ПАТТЕРНЫ АГЕНТСТВА ===\n"
        f"{get_summary_patterns()}"
    )

    estimation = call_structured(config.MODEL_ESTIMATOR, system, base_user, Estimation)
    for attempt in range(MAX_QA_RETRIES):
        reason = _variants_indistinguishable(estimation)
        if reason is None:
            return estimation
        log.warning("estimator QA-retry %d: %s", attempt + 1, reason)
        count("estimator_retries")
        retry_user = (
            base_user
            + "\n\n!!! ПРЕДЫДУЩАЯ ПОПЫТКА ОТКЛОНЕНА QA-ПРОВЕРКОЙ:\n"
            + reason
            + "\n\nПерегенерируй СМЕТУ с СОВЕРШЕННО разным набором этапов "
            + "у каждого варианта (разные названия, разное количество, разный "
            + "стек). Не дублируй этапы между вариантами."
        )
        estimation = call_structured(config.MODEL_ESTIMATOR, system, retry_user, Estimation)
    # после MAX_QA_RETRIES попыток возвращаем что есть — лучше чем ничего
    return estimation
