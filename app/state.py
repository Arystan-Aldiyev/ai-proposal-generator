"""Управление состоянием КП: сохранение, версии, точечные правки.

ProposalState — единственный источник истины. Из него рендерятся и
превью, и PDF. Перед каждой мутацией текущее состояние кладётся в
history/ — это даёт откат «на одну правку назад».

Инвариант: итог варианта всегда равен сумме статей сметы. Поэтому после
любой мутации вызывается recompute_totals().
"""

from __future__ import annotations

import math
import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path

from app import config
from app.schemas import ProposalState

# run_id попадает в путь на диске и в URL (/editor/<run_id>), поэтому только
# буквы, цифры, «-» и «_»: никаких точек и слэшей.
RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")


def new_run_id() -> str:
    """Время для сортировки + случайный хвост: id нельзя угадать, а два
    запуска в одну секунду не перезапишут друг друга."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(4)}"


def run_dir(run_id: str) -> Path:
    return config.OUTPUT_DIR / run_id


def _state_path(run_id: str) -> Path:
    return run_dir(run_id) / "state.json"


def _history_dir(run_id: str) -> Path:
    return run_dir(run_id) / "history"


# ─────────────────────────────────────────────────────────────────────────
# Итоги
# ─────────────────────────────────────────────────────────────────────────
def _round_50k(value: float) -> int:
    return int(round(value / 50_000) * 50_000)


def _plural_weeks(n: int) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return "недель"
    d = n % 10
    if d == 1:
        return "неделя"
    if 2 <= d <= 4:
        return "недели"
    return "недель"


def _format_weeks(days: int) -> str:
    """Срок словами из рабочих дней: ceil(days/5)–ceil(days/4) недель."""
    if days <= 0:
        return "—"
    w_min = math.ceil(days / 5)
    w_max = math.ceil(days / 4)
    if w_min == w_max:
        return f"{w_min} {_plural_weeks(w_min)}"
    return f"{w_min}–{w_max} {_plural_weeks(w_max)}"


def recompute_totals(state: ProposalState) -> None:
    """Пересчёт производных полей вариантов.

    Инварианты (та же логика, что у статей сметы):
    - total_kzt = сумма статей сметы; ОУР ≈ ×1.1, округление до 50 000 ₸;
    - work_days = сумма рабочих дней этапов;
    - duration  = срок словами, пересчитанный из work_days.
    """
    for variant in state.estimation.variants:
        variant.total_kzt = sum(ci.amount_kzt for ci in variant.cost_items)
        variant.total_our_tax_kzt = _round_50k(variant.total_kzt * 1.1)
        variant.work_days = sum(s.work_days for s in variant.stages)
        variant.duration = _format_weeks(variant.work_days)


# ─────────────────────────────────────────────────────────────────────────
# Чтение / запись
# ─────────────────────────────────────────────────────────────────────────
def save_state(run_id: str, state: ProposalState) -> None:
    """Пишет state.json + отдельные артефакты (для отладки и evals)."""
    recompute_totals(state)
    d = run_dir(run_id)
    d.mkdir(parents=True, exist_ok=True)
    _state_path(run_id).write_text(state.model_dump_json(indent=2), encoding="utf-8")
    for name, obj in (
        ("spec", state.spec),
        ("estimation", state.estimation),
        ("risks", state.risks),
        ("proposal", state.proposal),
    ):
        (d / f"{name}.json").write_text(obj.model_dump_json(indent=2), encoding="utf-8")


def load_state(run_id: str) -> ProposalState:
    path = _state_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Состояние не найдено: {run_id}")
    return ProposalState.model_validate_json(path.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────
# Версии / откат
# ─────────────────────────────────────────────────────────────────────────
def snapshot(run_id: str) -> None:
    """Кладёт текущее state.json в history/ — вызывать ПЕРЕД мутацией."""
    current = _state_path(run_id)
    if not current.exists():
        return
    hist = _history_dir(run_id)
    hist.mkdir(parents=True, exist_ok=True)
    n = len(list(hist.glob("*.json"))) + 1
    shutil.copy(current, hist / f"{n:03d}.json")


def revert(run_id: str) -> bool:
    """Откат на одну правку назад. False — если откатывать нечего."""
    hist = _history_dir(run_id)
    snaps = sorted(hist.glob("*.json")) if hist.exists() else []
    if not snaps:
        return False
    shutil.copy(snaps[-1], _state_path(run_id))
    snaps[-1].unlink()
    return True


def has_history(run_id: str) -> bool:
    hist = _history_dir(run_id)
    return hist.exists() and bool(list(hist.glob("*.json")))


# ─────────────────────────────────────────────────────────────────────────
# Точечная правка по dotted-path (инлайн-редактирование в превью)
# ─────────────────────────────────────────────────────────────────────────
def _set_by_path(data: dict, path: str, value: str) -> None:
    """Устанавливает значение по пути вида "estimation.variants.0.name"."""
    parts = path.split(".")
    node = data
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    key: int | str = int(last) if last.isdigit() else last
    current = node[key]

    # типобезопасная подстановка: число остаётся числом
    if isinstance(current, bool):
        node[key] = str(value).strip().lower() in ("true", "1", "да")
    elif isinstance(current, int):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        node[key] = int(digits) if digits else 0
    else:
        node[key] = str(value).strip()


def apply_field_patch(state: ProposalState, field_path: str, value: str) -> ProposalState:
    """Применяет одну точечную правку и возвращает новое состояние."""
    data = state.model_dump()
    _set_by_path(data, field_path, value)
    new_state = ProposalState.model_validate(data)
    recompute_totals(new_state)
    return new_state
