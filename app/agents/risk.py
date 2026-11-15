"""Агент 3 — Risk Analyst. Spec + Estimation → риски и архитектурные решения."""

from __future__ import annotations

from app import config
from app.agents import load_prompt
from app.llm import call_structured
from app.schemas import Estimation, Risks, Spec


def analyze_risks(spec: Spec, estimation: Estimation) -> Risks:
    system = load_prompt("risk")
    user = (
        "Найди архитектурные решения для обсуждения до старта и риски проекта.\n\n"
        "=== ТРЕБОВАНИЯ (Spec) ===\n"
        f"{spec.model_dump_json(indent=2)}\n\n"
        "=== СМЕТА (Estimation) ===\n"
        f"{estimation.model_dump_json(indent=2)}"
    )
    return call_structured(config.MODEL_RISK, system, user, Risks)
