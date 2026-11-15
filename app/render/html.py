"""Сборка контекста и рендер HTML-документа КП (Jinja2)."""

from __future__ import annotations

import base64
from datetime import datetime
from functools import lru_cache

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app import config
from app.rag.retriever import Analogue
from app.schemas import Estimation, Proposal, Risks, Spec, Variant

_RU_MONTHS = [
    "Январь",
    "Февраль",
    "Март",
    "Апрель",
    "Май",
    "Июнь",
    "Июль",
    "Август",
    "Сентябрь",
    "Октябрь",
    "Ноябрь",
    "Декабрь",
]


def _kzt(value: int | None) -> str:
    """3900000 -> '3 900 000 ₸'."""
    if value is None:
        return "—"
    return f"{value:,}".replace(",", " ") + " ₸"


@lru_cache(maxsize=1)
def _font_face_css() -> str:
    """@font-face с Arimo, зашитым в base64.

    Arimo (Apache 2.0) метрически совместим с Arial и имеет полную кириллицу.
    Шрифт встраивается в документ, чтобы PDF рендерился одинаково на любой
    машине — headless Chromium на Windows иначе уводит кириллицу в Times.
    """
    fonts_dir = config.STATIC_DIR / "fonts"
    faces = []
    for fname, weight in (("Arimo-Regular.ttf", 400), ("Arimo-Bold.ttf", 700)):
        b64 = base64.b64encode((fonts_dir / fname).read_bytes()).decode()
        faces.append(
            "@font-face{font-family:'Arimo';font-style:normal;"
            f"font-weight:{weight};"
            f"src:url(data:font/ttf;base64,{b64}) format('truetype');}}"
        )
    return "\n".join(faces)


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(config.TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml", "j2"]),
    )
    env.filters["kzt"] = _kzt
    return env


def _variant_view(variant: Variant) -> dict:
    """Готовит смету варианта к рендеру: плоская таблица или группировка
    по подпроектам (требование ТЗ — раздельная смета по каждому продукту)."""
    grouped = any(ci.sub_project for ci in variant.cost_items)
    if not grouped:
        return {"variant": variant, "grouped": False, "groups": []}

    order: list[str] = []
    buckets: dict[str, list] = {}
    # idx — исходный индекс статьи в variant.cost_items (нужен для data-field
    # при инлайн-редактировании, группировка его перемешивает).
    for idx, ci in enumerate(variant.cost_items):
        key = ci.sub_project or "Общие работы"
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append({"idx": idx, "ci": ci})

    groups = [
        {
            "name": name,
            "items": buckets[name],
            "subtotal": sum(x["ci"].amount_kzt for x in buckets[name]),
        }
        for name in order
    ]
    return {"variant": variant, "grouped": True, "groups": groups}


def build_context(
    spec: Spec,
    estimation: Estimation,
    risks: Risks,
    proposal: Proposal,
    analogues: list[Analogue] | None = None,
    run_id: str = "",
) -> dict:
    now = datetime.now()
    label = (spec.client_label or "").strip()
    return {
        "agency": config.AGENCY,
        "theme": config.THEME,
        "css": _font_face_css()
        + "\n"
        + (config.STATIC_DIR / "proposal.css").read_text(encoding="utf-8"),
        "date_label": f"{_RU_MONTHS[now.month - 1]} {now.year}",
        "client_label": label,
        "header_label": f"КП для {label}" if label else "Коммерческое предложение",
        "run_id": run_id,
        "spec": spec,
        "estimation": estimation,
        "risks": risks,
        "proposal": proposal,
        "variants": [_variant_view(v) for v in estimation.variants],
        "analogues": analogues or [],
    }


def context_from_state(state, run_id: str = "") -> dict:
    """Контекст рендера из ProposalState (для превью/PDF после правок)."""
    return build_context(state.spec, state.estimation, state.risks, state.proposal, run_id=run_id)


def render_html(context: dict) -> str:
    """HTML-документ КП для экспорта в PDF."""
    return _env().get_template("proposal.html.j2").render(**context)


def render_editor(context: dict) -> str:
    """Редактируемое HTML-превью КП (inline-правки + debug-панель)."""
    return _env().get_template("editor.html.j2").render(**context)
