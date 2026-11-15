"""Фабрики объектов для тестов: без вызовов LLM и без закрытых данных."""

from __future__ import annotations

import pytest

from app.rag.retriever import Analogue
from app.schemas import (
    CostItem,
    Estimation,
    PaymentStage,
    Proposal,
    ProposalState,
    Risks,
    Spec,
    Stage,
    TeamRole,
    Variant,
)


def make_spec(scope: list[str] | None = None, budget_kzt: int | None = None) -> Spec:
    return Spec(
        input_mode="brief",
        client_name=None,
        client_label="Автосервис",
        client_industry="Автоуслуги",
        project_type="Сайт с онлайн-записью",
        scope=["сайт", "онлайн-запись", "прайс"] if scope is None else scope,
        sub_projects=[],
        deadline=None,
        budget_kzt=budget_kzt,
        budget_signal=None,
        special_requirements=[],
        assumptions=[],
        unclear_points=[],
    )


def make_variant(name: str, stage_names: list[str], amounts: list[int], days: list[int]) -> Variant:
    return Variant(
        name=name,
        summary="",
        cost_items=[
            CostItem(item=f"Статья {i}", amount_kzt=a, sub_project=None)
            for i, a in enumerate(amounts)
        ],
        stages=[
            Stage(name=n, description="", work_days=d)
            for n, d in zip(stage_names, days, strict=True)
        ],
        total_kzt=sum(amounts),
        total_our_tax_kzt=0,
        duration="",
        work_days=sum(days),
    )


def make_estimation(variants: list[Variant]) -> Estimation:
    return Estimation(
        variants=variants,
        recommended_variant=variants[0].name,
        recommendation_reason="",
        team=[TeamRole(role="PM", responsibility="")],
        payment_schedule=[
            PaymentStage(name="Предоплата", share="50%", moment="старт"),
            PaymentStage(name="Сдача", share="50%", moment="запуск"),
        ],
        recurring_costs=[],
        analogues_used=[],
        assumptions=[],
    )


def make_proposal(**overrides) -> Proposal:
    data = dict(
        subtitle="Сайт автосервиса",
        intro="Мы поняли задачу.",
        architecture_intro="",
        stages_intro="",
        pricing_intro="",
        team_intro="",
        included=["Дизайн"],
        not_included=["Контент", "Фото"],
        guarantees=["3 месяца"],
        portfolio=[],
        next_step="Созвон.",
    )
    data.update(overrides)
    return Proposal(**data)


def make_analogue(score: float) -> Analogue:
    return Analogue(
        project_id="project-a", title="Проект A", card_text="", metadata_block="", score=score
    )


@pytest.fixture
def two_variants() -> Estimation:
    return make_estimation(
        [
            make_variant(
                "Эконом", ["Аналитика", "Сборка на конструкторе"], [300_000, 200_000], [3, 7]
            ),
            make_variant(
                "Корпоративный",
                ["Аналитика", "Backend + CMS", "Тестирование"],
                [900_000, 600_000],
                [5, 20, 5],
            ),
        ]
    )


@pytest.fixture
def proposal_state(two_variants) -> ProposalState:
    return ProposalState(
        spec=make_spec(),
        estimation=two_variants,
        risks=Risks(architectural_decisions=[], risks=[], caveats=[]),
        proposal=make_proposal(),
    )
