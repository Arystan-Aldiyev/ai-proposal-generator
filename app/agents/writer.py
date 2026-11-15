"""Агент 4 — Writer. Все артефакты → прозаические секции КП (Proposal)."""

from __future__ import annotations

import re

from app import config
from app.agents import load_prompt
from app.llm import call_structured
from app.metrics import count
from app.rag.dataset import find_nda_terms, get_writer_memo
from app.rag.portfolio import PortfolioProject, load_portfolio, search_portfolio
from app.rag.retriever import Analogue
from app.schemas import Estimation, Proposal, Risks, Spec


def _portfolio_query(spec: Spec) -> str:
    """Текстовый запрос для семантического поиска portfolio — собирает суть
    задачи клиента (тип + индустрия + scope), чтобы embedding попал в ту же
    нишу что и в карточках портфолио.
    """
    parts = [
        spec.project_type,
        spec.client_industry or "",
        spec.client_label,
        " ".join(spec.scope),
        " ".join(spec.special_requirements),
    ]
    return ". ".join(p for p in parts if p)


def _format_portfolio_candidates(items: list[PortfolioProject]) -> str:
    """Текстовый блок для промпта Writer-а — список разрешённых проектов
    с заголовками и описаниями.

    Для звёздных карточек (с description_long/metrics) Writer получает
    богатый контекст и может сжать его до 1-2 фраз сам. Для коротких
    карточек — только blurb_short или вообще «без деталей под NDA».
    """
    if not items:
        return "(подходящих проектов в портфолио агентства нет — раздел portfolio оставь пустым списком)"
    blocks: list[str] = []
    for p in items:
        parts: list[str] = [f"=== {p.show_title} ==="]
        if p.show_blurb:
            parts.append(f"blurb: {p.show_blurb}")
        else:
            parts.append("blurb: (детали услуги под NDA — оставь blurb пустым в КП)")
        if p.description_long and p.service_disclosure != "none":
            # Writer должен сжать длинное описание до 1-2 фраз для секции
            # portfolio в КП. Полный текст даём как материал, не для копирования.
            parts.append(f"context (сожми до 1-2 фраз, не копируй дословно):\n{p.description_long}")
        if p.metrics and p.service_disclosure != "none":
            parts.append(f"metrics: {p.metrics}")
        if p.note_for_agent:
            parts.append(f"NDA правило: {p.note_for_agent}")
        blocks.append("\n".join(parts))
    return "\n\n".join(blocks)


CAUTIOUS_HINT = """

## Особый режим: «слабый ввод» (cautious)
Бриф/ТЗ распознан как ненадёжный: либо в нём слишком мало данных, либо
в архиве агентства нет похожих проектов, либо бюджет клиента сильно ниже
сметы.
Это означает следующие правила тона:

1. В `intro` явно обозначь, что часть деталей мы достроили допущениями и
   уточним их на стартовом созвоне до подписания договора. Не претендуй
   на полную картину.
2. В `pricing_intro` и `stages_intro` подчеркни, что итоговая цена и срок
   будут зафиксированы после уточнения требований. Сейчас — оценочный
   диапазон.
3. В `next_step` сделай шагом именно созвон/встречу для уточнения, а не
   подписание договора.
4. Не убирай уверенности в команде и портфолио — но категоричных обещаний
   по конкретике («сделаем именно это, именно так, именно за столько») —
   избегай.
5. Тон остаётся профессиональным, без заискивания и без обильных
   извинений за неполноту. Один-два уверенных абзаца про «уточним и
   зафиксируем» лучше, чем три неуверенных.
"""

BUDGET_HINT = """

## Бюджет клиента сильно ниже сметы
Не оправдывайся и не извиняйся за цену, не доказывай, что бюджета мало.
В `pricing_intro` одной-двумя фразами: в названную сумму укладывается
только стартовая часть работ, полный объём стоит как в смете, а состав
первого этапа под бюджет предложим на созвоне. Цифры сметы не меняй и не
обещай уложиться в бюджет клиента.
"""

NDA_REPLACEMENT = "один из наших клиентов"


def _enforce_portfolio_nda(proposal: Proposal, allowed: list[PortfolioProject]) -> Proposal:
    """Программная гарантия NDA: оставляем в Proposal.portfolio только те
    item-ы, чьё title совпадает с public_label из portfolio.md.

    Дополнительно: если для проекта service_disclosure=none, занулям blurb
    (LLM могла придумать описание услуги, даже если ей разрешили только имя).

    Это страховка от LLM-галлюцинации — даже если writer.md промпт
    проигнорирован, имя клиента-под-NDA в финальный КП не уйдёт.
    """
    by_label = {p.public_label.strip().lower(): p for p in allowed if p.public_label}
    filtered: list = []
    for item in proposal.portfolio:
        key = (item.title or "").strip().lower()
        match = by_label.get(key)
        if match is None:
            # title не совпадает ни с одним public_label — выкидываем
            continue
        # service_disclosure=none → blurb должен быть пустым
        if match.service_disclosure == "none":
            item.blurb = ""
        filtered.append(item)
    count("nda_portfolio_removed", len(proposal.portfolio) - len(filtered))
    proposal.portfolio = filtered
    return proposal


def _enforce_text_nda(proposal: Proposal) -> Proposal:
    """Вторая программная проверка NDA: имена клиентов под NDA не должны
    появиться в свободном тексте КП (intro, pricing_intro и т.д.).

    Writer видит внутренние карточки аналогов с реальными именами, поэтому
    фильтра только по portfolio недостаточно. Найденное имя заменяем на
    нейтральную формулировку.
    """
    found = find_nda_terms(proposal.model_dump_json())
    if not found:
        return proposal

    def clean(value):
        if isinstance(value, str):
            for term in sorted(found, key=len, reverse=True):  # сначала длинные имена
                value = re.sub(re.escape(term), NDA_REPLACEMENT, value, flags=re.IGNORECASE)
            return value
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value

    count("nda_text_fixed", len(found))
    return Proposal.model_validate(clean(proposal.model_dump()))


def write_proposal(
    spec: Spec,
    estimation: Estimation,
    risks: Risks,
    analogues: list[Analogue],
    cautious: bool = False,
    reasons: list[str] | None = None,
) -> Proposal:
    system = (
        load_prompt("writer") + "\n\n## Памятка по тону (из базы агентства)\n" + get_writer_memo()
    )
    if cautious:
        system += CAUTIOUS_HINT
    if reasons and "budget_conflict" in reasons:
        system += BUDGET_HINT

    # RAG-аналоги — только как контекст для тона и стиля КП.
    # Имена клиентов из них в Portfolio НЕ переносятся (NDA-фильтр).
    analogues_context = (
        "\n\n".join(f"### Аналог КП: {a.title} [{a.project_id}]\n{a.card_text}" for a in analogues)
        or "(аналоги не найдены)"
    )

    # Portfolio агентства — отдельный публичный справочник. Подбор
    # семантический (cosine similarity между эмбеддингом брифа и
    # эмбеддингами карточек портфолио). Это лучше tag-based: на нишевых
    # брифах теги слепнут, а embeddings ловят семантику ниши.
    portfolio_all = load_portfolio()
    matches = search_portfolio(_portfolio_query(spec), k=3)
    portfolio_candidates = [project for project, _score in matches]
    portfolio_text = _format_portfolio_candidates(portfolio_candidates)

    user = (
        "Напиши текстовые разделы КП. Цифры не трогай — они уже посчитаны.\n\n"
        "=== ТРЕБОВАНИЯ (Spec) ===\n"
        f"{spec.model_dump_json(indent=2)}\n\n"
        "=== СМЕТА (Estimation) ===\n"
        f"{estimation.model_dump_json(indent=2)}\n\n"
        "=== РИСКИ (Risks) ===\n"
        f"{risks.model_dump_json(indent=2)}\n\n"
        "=== РЕФЕРЕНСНЫЕ КП АГЕНТСТВА (только для тона и формулировок) ===\n"
        "Это образцы хороших КП агентства. Бери из них стилистику, тон, "
        "структуру разделов и фирменные приёмы. НИКОГДА не используй "
        "имена клиентов из этих карточек в Portfolio финального КП.\n\n"
        f"{analogues_context}\n\n"
        "=== ПОРТФОЛИО АГЕНТСТВА (единственный источник для раздела portfolio) ===\n"
        "Это публично разрешённые проекты агентства, УЖЕ отобранные по\n"
        "релевантности задаче. Включи в Proposal.portfolio ВСЕ проекты из\n"
        "списка ниже (обычно 2-3). Используй ровно те title и blurb, что\n"
        "указаны — ничего не выдумывай, не достраивай имена клиентов. Если\n"
        "blurb пустой — оставь его пустой строкой в КП. НЕ ИЗМЕНЯЙ порядок и\n"
        "НЕ ВЫБИРАЙ из списка — все указанные проекты должны попасть в КП.\n\n"
        f"{portfolio_text}"
    )
    # temperature берётся из config.WRITER_TEMPERATURE если задана (evals/temperature.py),
    # иначе дефолт 0.3 внутри call_structured.
    kwargs = {}
    if config.WRITER_TEMPERATURE is not None:
        kwargs["temperature"] = config.WRITER_TEMPERATURE
    proposal = call_structured(config.MODEL_WRITER, system, user, Proposal, **kwargs)
    # NDA-страховка после LLM: даже если writer.md проигнорирован — имя
    # клиента-под-NDA не попадёт в финальный КП.
    proposal = _enforce_portfolio_nda(proposal, portfolio_all)
    return _enforce_text_nda(proposal)
