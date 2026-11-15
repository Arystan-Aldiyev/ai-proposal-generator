"""Прогон эксперимента: набор (dev/test) × конфигурация × повторы.

    python -m evals.run --split test --config final --repeats 3

Конфигурации:
    final     — итоговая система: extractor/risk gpt-4o-mini, estimator/writer gpt-4o
    all-mini  — та же система, все агенты на gpt-4o-mini (оптимизация стоимости)
    no-rag    — итоговая система без поиска аналогов (абляция RAG)
    baseline  — один запрос к gpt-4o без данных агентства (evals/baseline.py)

Для каждого прогона считаются: 10 ассертов, оценка LLM-судьи, задержка,
токены и стоимость генерации, решение triage, утечки имён под NDA.
Результаты дописываются в evals/results/<split>-<config>.json после каждого
прогона, поэтому прерванный запуск продолжается той же командой.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from app import config
from app.metrics import cost_usd, start_run
from app.pipeline import RunResult, run_pipeline
from app.rag.dataset import find_nda_terms
from evals.baseline import run_baseline
from evals.cases import CASES, Case
from evals.judge import JUDGE_MODEL, judge_verdict
from evals.test_cases import TEST_CASES

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evals" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SPLITS = {"dev": CASES, "test": TEST_CASES}

# модели: (extractor, estimator, risk, writer)
CONFIGS = {
    "final": {"models": ("gpt-4o-mini", "gpt-4o", "gpt-4o-mini", "gpt-4o"), "use_rag": True},
    "all-mini": {"models": ("gpt-4o-mini",) * 4, "use_rag": True},
    "no-rag": {"models": ("gpt-4o-mini", "gpt-4o", "gpt-4o-mini", "gpt-4o"), "use_rag": False},
    "baseline": {"models": ("gpt-4o",) * 4, "use_rag": False},
}

ASSERT_PASS_THRESHOLD = 8  # из 10 — минимум, чтобы кейс считался «пройденным»


def _has(text: str, *needles: str) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _assert_checks(case: Case, r: RunResult) -> list[tuple[str, bool]]:
    """10 программных проверок результата."""
    spec, est, prop = r.spec, r.estimation, r.proposal
    totals = [v.total_kzt for v in est.variants]
    full_text = " ".join(
        [spec.project_type, " ".join(spec.scope), " ".join(spec.special_requirements)]
    )

    checks: list[tuple[str, bool]] = []

    # 1 — тип проекта распознан осмысленно
    site_markers = ("сайт", "редизайн", "лендинг", "корпоратив", "каталог", "запись", "брон")
    bot_markers = ("бот", "whatsapp", "ватсап", "telegram", "телеграм")
    erp_markers = ("erp", "odoo", "учёт", "учет", "договор", "поставк")
    if case.kind == "bot":
        c1 = _has(full_text, *bot_markers)
    elif case.kind == "spec":
        c1 = spec.input_mode == "spec"
    elif case.kind == "erp":
        c1 = _has(full_text, *erp_markers)
    elif case.kind == "other":
        c1 = bool(spec.project_type.strip())
    else:
        c1 = _has(full_text, *site_markers)
    checks.append(("Тип проекта определён верно", c1))

    # 2 — найден хотя бы один ожидаемый аналог (если ожиданий нет — не проверяем)
    found = set(r.analogues) | set(est.analogues_used)
    c2 = not case.expected_analogues or bool(found & set(case.expected_analogues))
    checks.append(("Найден ожидаемый аналог", c2))

    # 3 — рекомендуемый вариант в ценовом коридоре, самый дорогой не выше планки
    rec = next((v for v in est.variants if v.name == est.recommended_variant), est.variants[0])
    lo, hi = case.price_range_kzt
    c3 = lo <= rec.total_kzt <= hi and max(totals) <= case.max_price_kzt
    checks.append(("Цена в коридоре", c3))

    # 4 — обязательные секции КП присутствуют
    c4 = bool(
        prop.intro
        and prop.subtitle
        and est.team
        and all(v.stages for v in est.variants)
        and prop.included
        and prop.guarantees
    )
    checks.append(("Все обязательные секции КП присутствуют", c4))

    # 5 — ≥2 варианта стоимости
    checks.append(("Есть 2–3 варианта", len(est.variants) >= 2))

    # 6 — блок «что НЕ входит» заполнен
    checks.append(('Есть блок "что НЕ входит"', len(prop.not_included) >= 2))

    # 7 — условия оплаты этапами указаны
    checks.append(("Условия оплаты указаны", len(est.payment_schedule) >= 2))

    # 8 — регулярные расходы соответствуют типу проекта
    if case.recurring_expected:
        c8 = len(est.recurring_costs) >= 1
    else:
        c8 = len(est.recurring_costs) == 0
    checks.append(("Регулярные расходы соответствуют типу проекта", c8))

    # 9 — режим ввода распознан / смета разбита по подпроектам ТЗ
    if case.kind == "spec":
        c9 = any(ci.sub_project for v in est.variants for ci in v.cost_items)
    else:
        c9 = spec.input_mode == "brief"
    checks.append(("Режим ввода распознан", c9))

    # 10 — нет явно выдуманных цифр
    c10 = len(totals) > 0 and all(100_000 <= t <= 20_000_000 for t in totals)
    checks.append(("Нет явно выдуманных цифр", c10))

    return checks


def _client_facing_text(r: RunResult) -> str:
    """Всё, что уходит клиенту в КП: тексты, смета и риски.
    Список аналогов и допущения видит только менеджер — их не берём."""
    est = r.estimation.model_dump(exclude={"analogues_used", "assumptions"})
    return " ".join(
        [
            r.proposal.model_dump_json(),
            json.dumps(est, ensure_ascii=False),
            r.risks.model_dump_json(),
        ]
    )


def _set_config(name: str) -> None:
    cfg = CONFIGS[name]
    (config.MODEL_EXTRACTOR, config.MODEL_ESTIMATOR, config.MODEL_RISK, config.MODEL_WRITER) = cfg[
        "models"
    ]
    config.USE_RAG = cfg["use_rag"]


def run_one(case: Case, idx: int, rep: int, split: str, cfg_name: str, judge_model: str) -> dict:
    run_id = f"{split}-{cfg_name}-{idx:02d}-r{rep}-{time.strftime('%Y%m%d%H%M%S')}"
    stats = start_run()
    t0 = time.time()
    if cfg_name == "baseline":
        r = run_baseline(case.brief, run_id)
    else:
        r = run_pipeline(case.brief, run_id)
    latency = round(time.time() - t0, 1)
    gen_stats = copy.deepcopy(stats)  # до судьи: стоимость самой генерации

    checks = _assert_checks(case, r)
    leaks = find_nda_terms(_client_facing_text(r))
    if leaks:
        print(f"  NDA-утечка в {run_id}: {leaks}")
    verdict = judge_verdict(case, r, model=judge_model)
    judge_tokens = {
        m: {k: t[k] - gen_stats["tokens"].get(m, {}).get(k, 0) for k in ("prompt", "completion")}
        for m, t in stats["tokens"].items()
    }
    judge = {
        "factual_accuracy": verdict.factual_accuracy,
        "tone": verdict.tone,
        "readability": verdict.readability,
        "completeness": verdict.completeness,
    }
    rec = next(
        (v for v in r.estimation.variants if v.name == r.estimation.recommended_variant),
        r.estimation.variants[0],
    )
    return {
        "case": case.name,
        "idx": idx,
        "rep": rep,
        "run_id": run_id,
        "latency_s": latency,
        "tokens": gen_stats["tokens"],
        "cost_usd": cost_usd(gen_stats["tokens"]),
        "judge_tokens": {m: t for m, t in judge_tokens.items() if t["prompt"] or t["completion"]},
        "llm_calls": gen_stats["llm_calls"],
        "estimator_retries": gen_stats["estimator_retries"],
        "nda_portfolio_removed": gen_stats["nda_portfolio_removed"],
        "nda_text_fixed": gen_stats["nda_text_fixed"],
        "asserts": [ok for _, ok in checks],
        "asserts_passed": sum(ok for _, ok in checks),
        "retrieval_ok": checks[1][1],
        "price_ok": checks[2][1],
        "recommended_kzt": rec.total_kzt,
        "expected_cautious": case.expected_cautious,
        "cautious": None if cfg_name == "baseline" else r.needs_clarification,
        "triage_reasons": r.triage_reasons,
        "top_score": round(r.top_score, 3),
        "analogues": r.analogues,
        # в отчёт пишем только число: сами имена под NDA публиковать нельзя
        "nda_leaks": len(leaks),
        "judge": judge,
        "judge_total": sum(judge.values()),
        "judge_explanation": verdict.explanation,
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _big_model_tokens(run: dict) -> int:
    """Токены «больших» моделей (gpt-4o и т.п.) — у них отдельный дневной лимит."""
    total = 0
    for tokens in (run["tokens"], run.get("judge_tokens", {})):
        for model, t in tokens.items():
            if not any(s in model for s in ("mini", "nano", "embedding")):
                total += t["prompt"] + t["completion"]
    return total


def run_config(
    split: str, cfg_name: str, repeats: int, judge_model: str, budget: int, spent: int
) -> int:
    """Прогоняет одну конфигурацию; возвращает потраченные токены больших моделей.

    budget — лимит этих токенов на весь запуск (0 — без лимита). Новый прогон
    не начинается, если до лимита осталось меньше ~25 тыс. (один прогон).
    """
    _set_config(cfg_name)
    cases = SPLITS[split]
    cases_file = Path(__file__).parent / ("test_cases.py" if split == "test" else "cases.py")
    out = RESULTS_DIR / f"{split}-{cfg_name}.json"
    if out.exists():
        report = json.loads(out.read_text(encoding="utf-8"))
        if report["cases_hash"] != _file_hash(cases_file):
            raise SystemExit(
                f"{cases_file.name} изменился после начала прогонов — удалите {out.name}"
            )
    else:
        report = {
            "split": split,
            "config": cfg_name,
            "models": dict(
                zip(
                    ("extractor", "estimator", "risk", "writer"),
                    CONFIGS[cfg_name]["models"],
                    strict=False,
                )
            ),
            "use_rag": CONFIGS[cfg_name]["use_rag"],
            "judge_model": judge_model,
            "cases_hash": _file_hash(cases_file),
            "runs": [],
        }
    done = {(run["idx"], run["rep"]) for run in report["runs"]}

    for rep in range(repeats):
        for idx, case in enumerate(cases, start=1):
            if (idx, rep) in done:
                continue
            if budget and spent + 25_000 > budget:
                print(
                    f"Дневной бюджет исчерпан: {spent} из {budget} токенов. Продолжить завтра той же командой."
                )
                return spent
            print(f"[{split}/{cfg_name}] повтор {rep + 1}/{repeats} · {case.name}")
            try:
                run = run_one(case, idx, rep, split, cfg_name, judge_model)
            except Exception as exc:  # noqa: BLE001
                # ошибку не сохраняем — при следующем запуске прогон повторится
                print(f"  ОШИБКА: {exc}")
                continue
            spent += _big_model_tokens(run)
            print(
                f"  ассерты {run['asserts_passed']}/10 · судья {run['judge_total']}/40 · "
                f"{run['latency_s']} с · ${run['cost_usd']} · cautious={run['cautious']}"
                + (f" · NDA-утечек: {run['nda_leaks']}" if run["nda_leaks"] else "")
            )
            report["runs"].append(run)
            out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    runs = report["runs"]
    if runs:
        n = len(runs)
        print(f"\nИтого {split}/{cfg_name}: {n} прогонов")
        print(f"  ассерты:   {sum(r['asserts_passed'] for r in runs) / n:.2f} из 10")
        print(f"  судья:     {sum(r['judge_total'] for r in runs) / n:.2f} из 40")
        print(f"  задержка:  {sum(r['latency_s'] for r in runs) / n:.1f} с")
        print(f"  стоимость: ${sum(r['cost_usd'] for r in runs) / n:.4f} за КП")
        print(f"Отчёт: {out.relative_to(ROOT)}\n")
    return spent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--config", choices=[*CONFIGS, "all"], default="final")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument(
        "--budget",
        type=int,
        default=0,
        help="лимит токенов gpt-4o и других больших моделей на запуск (0 — без лимита)",
    )
    args = parser.parse_args()

    # all — конфигурации по очереди, сначала самые важные для сравнения
    names = ["final", "baseline", "all-mini", "no-rag"] if args.config == "all" else [args.config]
    spent = 0
    for name in names:
        spent = run_config(args.split, name, args.repeats, args.judge_model, args.budget, spent)
        if args.budget and spent + 25_000 > args.budget:
            break
    print(f"Потрачено токенов больших моделей за запуск: {spent}")


if __name__ == "__main__":
    main()
