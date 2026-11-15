"""MCP-сервер CNS Proposal Generator.

Экспонирует базу знаний агентства и весь пайплайн генерации КП через
Model Context Protocol (stdio transport). Любой MCP-клиент (Claude
Desktop, Claude Code, Cursor) получает доступ к тем же функциям, что и
веб-UI на http://localhost:8000.

Tools:
  1) search_analogues(query, k)        — поиск проектов-аналогов
  2) get_project_card(project_id)      — карточка проекта
  3) estimate_from_brief(brief)        — быстрая смета (без полного КП)
  4) generate_proposal(brief)          — ПОЛНОЕ КП через LangGraph
  5) revise_proposal(run_id, ...)      — правки словами через ревизора
  6) get_proposal(run_id)              — текст КП + URL на PDF

Запуск:        python -m app.mcp_server
Регистрация:   mcp.json в корне репо для Claude Desktop / Code
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import FastMCP

from app import state as state_store
from app.agents.estimator import estimate
from app.agents.extractor import extract_spec
from app.agents.reviser import revise
from app.pipeline import run_pipeline
from app.rag.dataset import parse_rag_cards
from app.rag.retriever import search

mcp = FastMCP("cns-proposal")

# URL базы веб-сервера — для генерации ссылок на PDF/превью.
# Если веб-сервер не запущен, get_proposal вернёт ссылку без проверки.
WEB_BASE_URL = "http://localhost:8000"


@mcp.tool()
def search_analogues(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Найти проекты-аналоги в архиве CNS Agency по семантической близости.

    Args:
        query: Описание задачи на естественном языке (русский).
               Например: "Telegram-бот для приёма заказов в магазине".
        k: Сколько аналогов вернуть (1–6, по умолчанию 3).

    Returns:
        Список аналогов с project_id, заголовком карточки, косинусной
        близостью score и полным текстом карточки.
    """
    k = max(1, min(int(k), 6))
    analogues = search(query, k=k)
    return [
        {
            "project_id": a.project_id,
            "title": a.title,
            "score": round(a.score, 3),
            "card": a.card_text,
            "metadata": a.metadata_block,
        }
        for a in analogues
    ]


@mcp.tool()
def get_project_card(project_id: str) -> dict[str, Any]:
    """Достать полную карточку конкретного проекта по project_id.

    Args:
        project_id: Идентификатор проекта (например, "project-b",
                    "project-f", "project-a").

    Returns:
        Карточка с title, card_text и metadata_block. Если id не найден —
        поле "error" в ответе.
    """
    cards = parse_rag_cards()
    for c in cards:
        if c.project_id == project_id:
            return {
                "project_id": c.project_id,
                "title": c.title,
                "card": c.body,
                "metadata": c.metadata_block,
            }
    return {"error": f"project_id '{project_id}' не найден в архиве"}


@mcp.tool()
def estimate_from_brief(brief: str) -> dict[str, Any]:
    """Быстрая оценка проекта по тексту брифа: Spec + RAG + Estimator.

    Прогоняет первые три шага полного пайплайна (без Risk Analyst и Writer).
    Возвращает структурированную смету в JSON. Удобно для уточняющих
    итераций в чате до запуска полной генерации КП.

    Args:
        brief: Свободный текст брифа клиента (русский).

    Returns:
        Структура с распознанным spec, найденными аналогами и сметой
        (варианты + рекомендация + payment_schedule).
    """
    brief = (brief or "").strip()
    if not brief:
        return {"error": "Пустой бриф"}

    spec = extract_spec(brief)
    analogues = search(
        ". ".join(
            p for p in [spec.project_type, spec.client_industry or "", " ".join(spec.scope)] if p
        ),
        k=3,
    )
    estimation = estimate(spec, analogues)

    return {
        "spec": json.loads(spec.model_dump_json()),
        "analogues": [
            {"project_id": a.project_id, "title": a.title, "score": round(a.score, 3)}
            for a in analogues
        ],
        "estimation": json.loads(estimation.model_dump_json()),
    }


@mcp.tool()
def generate_proposal(brief: str) -> dict[str, Any]:
    """Сгенерировать полное КП по брифу клиента: запускает всю цепочку
    LangGraph (extract → retrieve → estimate → risk → triage → write).

    Это самый «тяжёлый» tool — занимает 30–60 секунд (5 LLM-вызовов).
    Возвращает run_id, краткое summary и ссылки на превью/PDF, которые
    можно открыть в браузере (веб-UI должен быть запущен).

    Args:
        brief: Свободный текст брифа клиента (русский). Может быть как
               короткой WhatsApp-фразой, так и полным ТЗ.

    Returns:
        run_id, client_label, needs_clarification (флаг неполного брифа),
        краткое summary КП, рекомендуемый вариант с ценой и сроком, URLs
        для редактируемого превью и скачивания PDF.
    """
    brief = (brief or "").strip()
    if not brief:
        return {"error": "Пустой бриф"}

    result = run_pipeline(brief)
    state = result.state
    est = state.estimation
    rec = next(
        (v for v in est.variants if v.name == est.recommended_variant),
        est.variants[0] if est.variants else None,
    )

    return {
        "run_id": result.run_id,
        "client_label": state.spec.client_label,
        "project_type": state.spec.project_type,
        "needs_clarification": result.needs_clarification,
        "analogues": result.analogues,
        "recommended_variant": {
            "name": rec.name if rec else "",
            "total_kzt": rec.total_kzt if rec else 0,
            "duration": rec.duration if rec else "",
            "work_days": rec.work_days if rec else 0,
        }
        if rec
        else None,
        "variants": [
            {"name": v.name, "total_kzt": v.total_kzt, "duration": v.duration} for v in est.variants
        ],
        "preview_url": f"{WEB_BASE_URL}/editor/{result.run_id}",
        "pdf_url": f"{WEB_BASE_URL}/pdf/{result.run_id}",
        "next_step": state.proposal.next_step,
    }


@mcp.tool()
def revise_proposal(run_id: str, instruction: str) -> dict[str, Any]:
    """Применить правку словами к существующему КП через агента-ревизора.

    Использует тот же агент, что и поле «Правка словами» в веб-UI. Все
    NDA-гарантии portfolio-фильтра применяются.

    Args:
        run_id: Идентификатор КП (получен из generate_proposal).
        instruction: Инструкция правки. Примеры: «подними цену Эконом до
                     950 000», «убери блок про CRM», «тон суше».

    Returns:
        ok=True если применено успешно, иначе error.
    """
    run_dir = state_store.run_dir(run_id)
    if not (run_dir / "state.json").exists():
        return {"error": f"КП с run_id '{run_id}' не найдено"}
    instruction = (instruction or "").strip()
    if not instruction:
        return {"error": "Пустая инструкция правки"}

    state = state_store.load_state(run_id)
    try:
        updated = revise(state, instruction)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Ошибка ревизора: {exc}"}

    state_store.snapshot(run_id)
    state_store.save_state(run_id, updated)
    return {
        "ok": True,
        "preview_url": f"{WEB_BASE_URL}/editor/{run_id}",
        "pdf_url": f"{WEB_BASE_URL}/pdf/{run_id}",
    }


@mcp.tool()
def get_proposal(run_id: str) -> dict[str, Any]:
    """Получить текущее содержимое КП по run_id: тексты разделов, варианты
    стоимости, портфолио и ссылку на PDF.

    Args:
        run_id: Идентификатор КП (получен из generate_proposal).

    Returns:
        Текстовые разделы proposal (subtitle/intro/included/not_included/
        guarantees/portfolio/next_step) + ключевые цифры estimation +
        URL на скачивание PDF.
    """
    run_dir = state_store.run_dir(run_id)
    if not (run_dir / "state.json").exists():
        return {"error": f"КП с run_id '{run_id}' не найдено"}

    state = state_store.load_state(run_id)
    prop = state.proposal
    est = state.estimation

    return {
        "run_id": run_id,
        "client_label": state.spec.client_label,
        "subtitle": prop.subtitle,
        "intro": prop.intro,
        "variants": [
            {
                "name": v.name,
                "summary": v.summary,
                "total_kzt": v.total_kzt,
                "duration": v.duration,
                "work_days": v.work_days,
            }
            for v in est.variants
        ],
        "recommended": est.recommended_variant,
        "recommendation_reason": est.recommendation_reason,
        "included": prop.included,
        "not_included": prop.not_included,
        "guarantees": prop.guarantees,
        "portfolio": [{"title": p.title, "blurb": p.blurb} for p in prop.portfolio],
        "next_step": prop.next_step,
        "pdf_url": f"{WEB_BASE_URL}/pdf/{run_id}",
        "preview_url": f"{WEB_BASE_URL}/editor/{run_id}",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
