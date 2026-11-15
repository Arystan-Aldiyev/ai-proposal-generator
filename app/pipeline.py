"""Оркестратор multi-agent цепочки на LangGraph.

Граф: extract → retrieve → estimate → risk → triage →
      (conditional) → write_standard | write_cautious → END

Условная логика (triage → write_*), причины «слабого ввода»:
  low_info        — в брифе 1–2 пункта работ (len(spec.scope) <= 2)
  no_analogue     — лучший аналог из архива похож слабо (score < 0.375)
  budget_conflict — самый дешёвый вариант дороже бюджета клиента в 2+ раза
  есть хоть одна причина → write_cautious (Writer добавляет оговорки и
                           приглашение к уточнению), иначе → write_standard.
  Пороги подобраны на dev-наборе (evals/cases.py), test в подборе не участвует.

Это не косметика: ветка влияет на тон КП. См. app/agents/writer.py.

SSE-стрим этапов выдаётся через LangGraph custom stream channel
(get_stream_writer). Финальное событие "done" эмитим вручную после
сохранения состояния — в нём run_id и список аналогов для UI.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app import config
from app import state as state_store
from app.agents.estimator import estimate
from app.agents.extractor import extract_spec
from app.agents.risk import analyze_risks
from app.agents.writer import write_proposal
from app.rag.retriever import Analogue, search
from app.schemas import Estimation, Proposal, ProposalState, Risks, Spec

# Шаги цепочки — для прогресс-стрима в UI (порядок узлов графа).
STEPS = [
    ("extract", "Извлекаю требования"),
    ("retrieve", "Ищу похожие проекты"),
    ("estimate", "Считаю смету"),
    ("risk", "Анализирую риски"),
    ("triage", "Оцениваю полноту брифа"),
    ("write", "Пишу текст КП"),
]

# Пороги «слабого ввода». Подобраны на dev-наборе: старое правило
# unclear_points >= 3 срабатывало почти всегда (Extractor обычно
# возвращает ровно 3 неясных пункта), поэтому от него отказались.
WEAK_MIN_SCOPE = 2
WEAK_RAG_SCORE = 0.375
BUDGET_CONFLICT_RATIO = 2.0


@dataclass
class RunResult:
    run_id: str
    output_dir: Path
    state: ProposalState
    analogues: list[str] = field(default_factory=list)
    needs_clarification: bool = False
    triage_reasons: list[str] = field(default_factory=list)
    top_score: float = 0.0

    @property
    def spec(self):
        return self.state.spec

    @property
    def estimation(self):
        return self.state.estimation

    @property
    def risks(self):
        return self.state.risks

    @property
    def proposal(self):
        return self.state.proposal


class GraphState(TypedDict, total=False):
    """Совокупное состояние графа. Каждая нода возвращает частичное обновление."""

    raw_input: str
    spec: Spec
    analogues: list[Analogue]
    estimation: Estimation
    risks: Risks
    proposal: Proposal
    needs_clarification: bool
    triage_reasons: list[str]


def _emit(event: dict) -> None:
    """Прокидываем событие прогресса в custom-канал LangGraph (для SSE).

    Если граф вызвали без stream — writer вернёт no-op, тихо игнорим.
    """
    try:
        writer = get_stream_writer()
        writer(event)
    except Exception:
        pass


def _rag_query(spec: Spec) -> str:
    parts = [spec.project_type, spec.client_industry or "", " ".join(spec.scope)]
    return ". ".join(p for p in parts if p)


@traceable(name="extract")
def node_extract(state: GraphState) -> dict:
    _emit({"step": "extract", "status": "start"})
    spec = extract_spec(state["raw_input"])
    return {"spec": spec}


@traceable(name="retrieve")
def node_retrieve(state: GraphState) -> dict:
    _emit({"step": "retrieve", "status": "start"})
    if not config.USE_RAG:
        return {"analogues": []}
    analogues = search(_rag_query(state["spec"]), k=3)
    return {"analogues": analogues}


@traceable(name="estimate")
def node_estimate(state: GraphState) -> dict:
    _emit({"step": "estimate", "status": "start"})
    estimation = estimate(state["spec"], state["analogues"])
    return {"estimation": estimation}


@traceable(name="risk")
def node_risk(state: GraphState) -> dict:
    _emit({"step": "risk", "status": "start"})
    risks = analyze_risks(state["spec"], state["estimation"])
    return {"risks": risks}


@traceable(name="triage")
def node_triage(state: GraphState) -> dict:
    """Оцениваем «надёжность» брифа: достаточно ли данных и аналогов."""
    _emit({"step": "triage", "status": "start"})
    spec = state["spec"]
    analogues = state["analogues"]
    est = state["estimation"]
    top_score = max((a.score for a in analogues), default=0.0)
    cheapest = min((v.total_kzt for v in est.variants), default=0)
    reasons = []
    if len(spec.scope) <= WEAK_MIN_SCOPE:
        reasons.append("low_info")
    if analogues and top_score < WEAK_RAG_SCORE:
        reasons.append("no_analogue")
    if spec.budget_kzt and cheapest > BUDGET_CONFLICT_RATIO * spec.budget_kzt:
        reasons.append("budget_conflict")
    return {"needs_clarification": bool(reasons), "triage_reasons": reasons}


def route_after_triage(state: GraphState) -> str:
    """Conditional edge: ветка зависит от needs_clarification."""
    return "write_cautious" if state.get("needs_clarification") else "write_standard"


@traceable(name="write_standard")
def node_write_standard(state: GraphState) -> dict:
    _emit({"step": "write", "status": "start", "mode": "standard"})
    proposal = write_proposal(
        state["spec"], state["estimation"], state["risks"], state["analogues"]
    )
    return {"proposal": proposal}


@traceable(name="write_cautious")
def node_write_cautious(state: GraphState) -> dict:
    _emit({"step": "write", "status": "start", "mode": "cautious"})
    proposal = write_proposal(
        state["spec"],
        state["estimation"],
        state["risks"],
        state["analogues"],
        cautious=True,
        reasons=state.get("triage_reasons", []),
    )
    return {"proposal": proposal}


def _build_graph(with_hitl: bool = True):
    """Сборка графа.

    with_hitl=True — добавляем interrupt_after=["extract"] и checkpointer.
      Граф приостанавливается после извлечения Spec, чтобы менеджер мог
      проверить/поправить распознанный Spec до запуска тяжёлых LLM-вызовов.
    with_hitl=False — линейный режим (для evals, чтобы не блокировать
      автоматический прогон).
    """
    g = StateGraph(GraphState)
    g.add_node("extract", node_extract)
    g.add_node("retrieve", node_retrieve)
    g.add_node("estimate", node_estimate)
    g.add_node("risk", node_risk)
    g.add_node("triage", node_triage)
    g.add_node("write_standard", node_write_standard)
    g.add_node("write_cautious", node_write_cautious)

    g.add_edge(START, "extract")
    g.add_edge("extract", "retrieve")
    g.add_edge("retrieve", "estimate")
    g.add_edge("estimate", "risk")
    g.add_edge("risk", "triage")
    g.add_conditional_edges(
        "triage",
        route_after_triage,
        {"write_standard": "write_standard", "write_cautious": "write_cautious"},
    )
    g.add_edge("write_standard", END)
    g.add_edge("write_cautious", END)

    if with_hitl:
        return g.compile(
            checkpointer=InMemorySaver(),
            interrupt_after=["extract"],
        )
    return g.compile()


# Два графа: с HITL для UI и без для автоматических прогонов (evals).
# Checkpointer в HITL-варианте in-memory — для прототипа достаточно;
# на production SqliteSaver/PostgresSaver чтобы state переживал рестарт.
graph_hitl = _build_graph(with_hitl=True)
graph = _build_graph(with_hitl=False)


def _save_artifacts(run_id: str, final_state: dict, analogues: list[Analogue]) -> ProposalState:
    state = ProposalState(
        spec=final_state["spec"],
        estimation=final_state["estimation"],
        risks=final_state["risks"],
        proposal=final_state["proposal"],
    )
    state_store.save_state(run_id, state)

    rag_path = state_store.run_dir(run_id) / "rag.json"
    rag_path.write_text(
        json.dumps(
            [
                {
                    "project_id": a.project_id,
                    "title": re.sub(r"^КАРТОЧКА\s*\d+\s*[—-]\s*", "", a.title).strip(),
                    "score": round(a.score, 3),
                }
                for a in analogues
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return state


def run_pipeline_stream(raw_input: str, run_id: str | None = None) -> Iterator[dict]:
    """Линейный прогон графа без HITL (для evals и MCP).

    Yield-ит события прогресса из custom-канала LangGraph по мере того,
    как ноды входят в работу, и финальное «done» с run_id, аналогами и
    флагом needs_clarification.
    """
    raw_input = (raw_input or "").strip()
    if not raw_input:
        raise ValueError("Пустой ввод — нечего обрабатывать.")

    run_id = run_id or state_store.new_run_id()

    final_state: dict = {}
    for mode, chunk in graph.stream(
        {"raw_input": raw_input},
        stream_mode=["custom", "values"],
        config={"run_name": "run_pipeline", "tags": ["cns-proposal"]},
    ):
        if mode == "custom":
            yield chunk
        elif mode == "values":
            final_state = chunk

    analogues: list[Analogue] = final_state["analogues"]
    state = _save_artifacts(run_id, final_state, analogues)

    yield {
        "step": "done",
        "run_id": run_id,
        "analogues": [a.project_id for a in analogues],
        "client_label": state.spec.client_label,
        "needs_clarification": final_state.get("needs_clarification", False),
        "triage_reasons": final_state.get("triage_reasons", []),
        "top_score": max((a.score for a in analogues), default=0.0),
    }


# ─────────────────────────────────────────────────────────────────────────
# HITL-режим для UI: двухфазный пайплайн через interrupt_after=["extract"]
# Фаза 1: extract → пауза с распознанным Spec
# Фаза 2: (после правок Spec пользователем) retrieve → ... → write
# ─────────────────────────────────────────────────────────────────────────
def _hitl_config(run_id: str) -> dict:
    """Config для checkpointed-графа: thread_id фиксирует state между фазами."""
    return {
        "configurable": {"thread_id": run_id},
        "run_name": "run_pipeline_hitl",
        "tags": ["cns-proposal", "hitl"],
    }


def start_with_hitl(raw_input: str) -> Iterator[dict]:
    """Фаза 1: запускаем граф до interrupt_after=["extract"].

    Yield-ит: события прогресса extract → финальный «spec_ready» с
    распознанным Spec для проверки/правок пользователем. Граф
    приостановлен в checkpointer — продолжить через continue_after_spec.
    """
    raw_input = (raw_input or "").strip()
    if not raw_input:
        raise ValueError("Пустой ввод — нечего обрабатывать.")

    run_id = state_store.new_run_id()
    config = _hitl_config(run_id)

    final_state: dict = {}
    for mode, chunk in graph_hitl.stream(
        {"raw_input": raw_input},
        stream_mode=["custom", "values"],
        config=config,
    ):
        if mode == "custom":
            yield chunk
        elif mode == "values":
            final_state = chunk

    spec = final_state["spec"]
    yield {
        "step": "spec_ready",
        "run_id": run_id,
        "spec": spec.model_dump(),
    }


def continue_after_spec(run_id: str, spec_patch: dict | None = None) -> Iterator[dict]:
    """Фаза 2: продолжаем граф после подтверждения/правки Spec.

    spec_patch — словарь с полями Spec, которые нужно перезаписать.
    None или {} — продолжаем без изменений.
    """
    config = _hitl_config(run_id)

    # Применяем правки к Spec, если есть
    if spec_patch:
        current = graph_hitl.get_state(config)
        spec: Spec = current.values["spec"]
        spec_dict = spec.model_dump()
        # принимаем только поля, реально существующие в Spec — игнорируем
        # лишний мусор из UI чтобы не сломать схему
        for key, value in spec_patch.items():
            if key in spec_dict:
                spec_dict[key] = value
        updated_spec = Spec.model_validate(spec_dict)
        graph_hitl.update_state(config, {"spec": updated_spec})

    final_state: dict = {}
    for mode, chunk in graph_hitl.stream(
        None,  # продолжаем с сохранённого state, без нового input
        stream_mode=["custom", "values"],
        config=config,
    ):
        if mode == "custom":
            yield chunk
        elif mode == "values":
            final_state = chunk

    analogues: list[Analogue] = final_state["analogues"]
    state = _save_artifacts(run_id, final_state, analogues)

    yield {
        "step": "done",
        "run_id": run_id,
        "analogues": [a.project_id for a in analogues],
        "client_label": state.spec.client_label,
        "needs_clarification": final_state.get("needs_clarification", False),
        "triage_reasons": final_state.get("triage_reasons", []),
        "top_score": max((a.score for a in analogues), default=0.0),
    }


def run_pipeline(raw_input: str, run_id: str | None = None) -> RunResult:
    """Синхронный прогон без стрима — для evals и скриптов."""
    done: dict = {}
    for event in run_pipeline_stream(raw_input, run_id=run_id):
        if event.get("step") == "done":
            done = event
    return RunResult(
        run_id=done["run_id"],
        output_dir=state_store.run_dir(done["run_id"]),
        state=state_store.load_state(done["run_id"]),
        analogues=done.get("analogues", []),
        needs_clarification=done.get("needs_clarification", False),
        triage_reasons=done.get("triage_reasons", []),
        top_score=done.get("top_score", 0.0),
    )
