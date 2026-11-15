"""Pydantic-схемы. Каждый агент возвращает строго один из этих объектов
через OpenAI structured output. Между агентами передаём model_dump().

Замечание по structured output: OpenAI strict-режим требует, чтобы все поля
были обязательными. Поэтому необязательные значения объявлены как Optional[...]
(union с null), а списки агент обязан вернуть всегда (пустыми, если данных нет).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────
# 1. Spec Extractor → Spec
# ─────────────────────────────────────────────────────────────────────────
class Spec(BaseModel):
    """Структурированные требования, извлечённые из брифа или ТЗ."""

    input_mode: Literal["brief", "spec"] = Field(
        description='"brief" — короткий свободный текст (WhatsApp); '
        '"spec" — длинный структурированный документ / ТЗ'
    )
    client_name: Optional[str] = Field(
        description="Реальное название компании клиента, если оно названо. Иначе null."
    )
    client_label: str = Field(
        description="Короткий читаемый ярлык клиента для обложки и шапки КП. "
        "Если есть название компании — оно. Если нет — описательный ярлык из "
        'сути брифа: "Магазин цветов", "Стоматология", "Юридическая фирма". '
        'Никогда не "не указано" и не пустая строка.'
    )
    client_industry: Optional[str] = Field(description="Отрасль клиента")
    project_type: str = Field(description="Тип проекта одной фразой")
    scope: list[str] = Field(description="Состав работ — пунктами, что нужно сделать")
    sub_projects: list[str] = Field(
        description='Подпроекты для раздельной сметы (режим "spec"). '
        "Например два сайта. Для брифа — пустой список."
    )
    deadline: Optional[str] = Field(description="Дедлайн/сроки, если названы")
    budget_kzt: Optional[int] = Field(description="Бюджет в тенге, если названа цифра")
    budget_signal: Optional[str] = Field(
        description="Ценовой сигнал словами, если цифры нет "
        '(например "не миллионы", "ограниченный бюджет")'
    )
    special_requirements: list[str] = Field(
        description="Особые требования: интеграции, языки, mobile-first и т.п."
    )
    assumptions: list[str] = Field(
        description='Разумные допущения, достроенные агентом (режим "brief"). '
        "Для ТЗ — обычно пусто."
    )
    unclear_points: list[str] = Field(description="Что не указано и требует уточнения у клиента")


# ─────────────────────────────────────────────────────────────────────────
# 2. Estimator → Estimation
# ─────────────────────────────────────────────────────────────────────────
class CostItem(BaseModel):
    item: str = Field(description="Статья работ")
    amount_kzt: int = Field(description="Стоимость статьи в тенге без НДС")
    sub_project: Optional[str] = Field(
        description="К какому подпроекту относится (режим ТЗ с раздельной сметой). "
        "Если сметы по подпроектам нет — null."
    )


class Stage(BaseModel):
    name: str = Field(description="Название этапа")
    description: str = Field(description="Что делается на этапе")
    work_days: int = Field(description="Рабочие дни")


class Variant(BaseModel):
    name: str = Field(
        description='Название варианта: "Эконом" / "Корпоративный" / "Вариант А" и т.п.'
    )
    summary: str = Field(description="Чем этот вариант отличается, одна-две фразы")
    cost_items: list[CostItem] = Field(description="Разбивка стоимости по статьям")
    stages: list[Stage] = Field(
        description="Этапы проекта со сроками для ЭТОГО варианта. "
        "У каждого варианта свой набор этапов."
    )
    total_kzt: int = Field(description="Итого в тенге без НДС (ИП упрощёнка)")
    total_our_tax_kzt: int = Field(description="Итого в режиме ОУР ≈ total_kzt × 1.1")
    duration: str = Field(description="Срок словами, пересчитывается из work_days")
    work_days: int = Field(description="Итого рабочих дней (= сумма work_days этапов)")


class TeamRole(BaseModel):
    role: str = Field(description="Роль")
    responsibility: str = Field(description="Зона ответственности")


class PaymentStage(BaseModel):
    name: str = Field(description="Название этапа оплаты")
    share: str = Field(description='Доля, например "30%"')
    moment: str = Field(description="Момент оплаты")


class RecurringCost(BaseModel):
    item: str = Field(description="Статья регулярного расхода")
    amount: str = Field(description='Сумма с периодом, например "~9 900 ₸/мес"')
    note: Optional[str] = Field(description="Пояснение, если нужно")


class Estimation(BaseModel):
    """Оценка сроков и стоимости на основе RAG-аналогов."""

    variants: list[Variant] = Field(
        description="Варианты реализации. Обычно 2 (эконом/корпоратив или А/Б)."
    )
    recommended_variant: str = Field(description="Название рекомендуемого варианта")
    recommendation_reason: str = Field(description="Почему рекомендуем именно его")
    team: list[TeamRole] = Field(description="Состав команды (роли без ставок)")
    payment_schedule: list[PaymentStage] = Field(
        description="Условия оплаты этапами. Правило: итог < 1 млн → 50/50; "
        "иначе 30/25/25/20 (или 30/40/30 для среднего)."
    )
    recurring_costs: list[RecurringCost] = Field(
        description="Регулярные расходы — ТОЛЬКО для ботов/CRM/интеграций. "
        "Для обычного сайта — пустой список."
    )
    analogues_used: list[str] = Field(description="project_id аналогов, на которые опирались")
    assumptions: list[str] = Field(
        description="Допущения оценки. Если прямых аналогов нет — честно написать здесь."
    )


# ─────────────────────────────────────────────────────────────────────────
# 3. Risk Analyst → Risks
# ─────────────────────────────────────────────────────────────────────────
class ArchDecision(BaseModel):
    title: str = Field(description="Архитектурное решение / что важно проговорить")
    description: str = Field(description="Объяснение и рекомендация")


class Risk(BaseModel):
    title: str = Field(description="Риск")
    impact: str = Field(description="Чем грозит")
    mitigation: str = Field(description="Как снимаем риск")


class Risks(BaseModel):
    architectural_decisions: list[ArchDecision] = Field(
        description="2–3 решения, которые надо утвердить с клиентом до старта"
    )
    risks: list[Risk] = Field(description="3–5 конкретных рисков с митигацией")
    caveats: list[str] = Field(description="Короткие оговорки/примечания")


# ─────────────────────────────────────────────────────────────────────────
# 4. Writer → Proposal (прозаические секции; цифры берутся из Estimation)
# ─────────────────────────────────────────────────────────────────────────
class PortfolioItem(BaseModel):
    title: str = Field(description="Название проекта из портфолио")
    blurb: str = Field(description="Краткое описание, 1–2 предложения")


class Proposal(BaseModel):
    """Тексты разделов КП. Числа Writer НЕ пересчитывает."""

    subtitle: str = Field(
        description='Подзаголовок КП на обложке, например "Редизайн сайта стоматологии"'
    )
    intro: str = Field(
        description='Раздел "CNS Agency / мы поняли вашу задачу" — кто мы + что поняли из брифа'
    )
    architecture_intro: str = Field(description="Вводный абзац к разделу архитектурных решений")
    stages_intro: str = Field(description="Вводный абзац к разделу этапов и сроков")
    pricing_intro: str = Field(description="Вводный абзац к разделу стоимости")
    team_intro: str = Field(description="Вводный абзац к разделу команды")
    included: list[str] = Field(description="Что входит в стоимость — пунктами")
    not_included: list[str] = Field(description='Блок "что НЕ входит" — пунктами')
    guarantees: list[str] = Field(description="Гарантии и поддержка — пунктами")
    portfolio: list[PortfolioItem] = Field(description="2–3 релевантных проекта из RAG-аналогов")
    next_step: str = Field(description="Понятный следующий шаг для клиента")


# ─────────────────────────────────────────────────────────────────────────
# 5. Состояние КП — объединяет все артефакты цепочки.
#    Это редактируемый источник истины: из него рендерятся превью и PDF.
# ─────────────────────────────────────────────────────────────────────────
class ProposalState(BaseModel):
    spec: Spec
    estimation: Estimation
    risks: Risks
    proposal: Proposal
