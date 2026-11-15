"""Агент-ревизор. Применяет текстовую правку к готовому КП за один проход.

Не перезапускает цепочку (extractor/RAG/estimator/...). Берёт текущее
состояние + инструкцию и возвращает обновлённое состояние целиком.
"""

from __future__ import annotations

from app import config
from app.agents import load_prompt
from app.agents.writer import _enforce_portfolio_nda
from app.llm import call_json
from app.rag.portfolio import load_portfolio
from app.schemas import ProposalState


def revise(state: ProposalState, instruction: str) -> ProposalState:
    system = load_prompt("reviser")
    user = (
        "Текущее КП (JSON):\n"
        f"{state.model_dump_json(indent=2)}\n\n"
        "Инструкция правки от менеджера:\n"
        f"{instruction.strip()}\n\n"
        "Верни полный JSON состояния с применённой правкой."
    )
    raw = call_json(config.MODEL_REVISER, system, user)
    new_state = ProposalState.model_validate_json(raw)
    # NDA-страховка: если менеджер попросил «добавь имя X в портфолио», а
    # X не в нашем portfolio.md — Reviser-агент мог выдумать. Фильтруем.
    new_state.proposal = _enforce_portfolio_nda(new_state.proposal, load_portfolio())
    return new_state
