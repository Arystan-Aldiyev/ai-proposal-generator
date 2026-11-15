"""Baseline: «просто попросить LLM составить КП».

Один вызов gpt-4o без поиска по архиву агентства, без конвейера агентов и
без программных проверок. Модель возвращает тот же ProposalState, что и
основная система, поэтому к результату применяются те же ассерты и тот же
LLM-судья (evals/run.py --config baseline).
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from app import state as state_store
from app.llm import call_json
from app.pipeline import RunResult
from app.schemas import ProposalState

BASELINE_MODEL = "gpt-4o"

BASELINE_SYSTEM = """Ты менеджер digital-агентства в Казахстане (сайты, чат-боты, \
автоматизация для бизнеса). По брифу клиента подготовь коммерческое предложение.

Нужно: разобрать требования клиента; предложить 2–3 варианта реализации с ценой \
в тенге без НДС и этапами со сроками в рабочих днях; состав команды; условия \
оплаты этапами; регулярные расходы (только если они реально есть, например у \
ботов и интеграций); риски и решения, которые надо согласовать до старта; что \
входит и что не входит в стоимость; гарантии; следующий шаг для клиента.

Ответ — один JSON-объект строго по JSON-схеме из сообщения пользователя, \
все поля обязательны. Язык значений — русский."""


def run_baseline(brief: str, run_id: str | None = None) -> RunResult:
    run_id = run_id or state_store.new_run_id()
    schema = json.dumps(ProposalState.model_json_schema(), ensure_ascii=False)
    user = f"=== JSON-СХЕМА ОТВЕТА ===\n{schema}\n\n=== БРИФ КЛИЕНТА ===\n{brief}"

    # JSON mode не гарантирует схему — при ошибке валидации одна повторная попытка
    for attempt in range(2):
        raw = call_json(BASELINE_MODEL, BASELINE_SYSTEM, user, temperature=0.3)
        try:
            state = ProposalState.model_validate_json(raw)
            break
        except ValidationError:
            if attempt == 1:
                raise

    state_store.save_state(run_id, state)
    return RunResult(
        run_id=run_id,
        output_dir=state_store.run_dir(run_id),
        state=state,
    )
