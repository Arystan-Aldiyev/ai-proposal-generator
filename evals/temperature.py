"""A/B по температуре Writer — экспериментальное обоснование выбора T.

Прогоняет 4 кейса (один из каждой широкой группы) × 3 температуры:
T=0.0 (детерминизм), T=0.3 (текущая), T=0.7 (творчество).
Считает LLM-judge по 4 осям. Сохраняет отчёт в evals/results/temperature-*.json.

Запуск:  python -m evals.temperature
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from app import config
from app.pipeline import run_pipeline
from evals.cases import CASES, Case
from evals.judge import judge_proposal
from evals.run import _assert_checks

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evals" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 4 кейса для эксперимента — по одному из каждого типа (быстрее чем 10)
EXP_CASES_IDX = [0, 1, 4, 5]  # стоматология, бот, лендинг, ERP
TEMPERATURES = [0.0, 0.3, 0.7]


def _run_one(case: Case, temperature: float) -> dict:
    config.WRITER_TEMPERATURE = temperature
    t0 = time.time()
    r = run_pipeline(case.brief)
    elapsed = round(time.time() - t0, 1)
    checks = _assert_checks(case, r)
    asserts_passed = sum(1 for _, ok in checks if ok)
    try:
        judge = judge_proposal(case, r)
        judge_total = sum(judge.values())
    except Exception as exc:  # noqa: BLE001
        judge = {"error": str(exc)}
        judge_total = 0
    return {
        "case": case.name,
        "temperature": temperature,
        "run_id": r.run_id,
        "elapsed_s": elapsed,
        "asserts_passed": asserts_passed,
        "judge_scores": judge,
        "judge_total": judge_total,
    }


def main() -> None:
    cases = [CASES[i] for i in EXP_CASES_IDX]
    results: list[dict] = []

    for case in cases:
        for t in TEMPERATURES:
            print(f"\n— {case.name[:40]} | T={t} —")
            r = _run_one(case, t)
            print(
                f"  asserts={r['asserts_passed']}/10  judge={r['judge_total']}/40  elapsed={r['elapsed_s']}s"
            )
            if isinstance(r["judge_scores"], dict) and "factual_accuracy" in r["judge_scores"]:
                js = r["judge_scores"]
                print(
                    f"    factual={js['factual_accuracy']} tone={js['tone']} read={js['readability']} compl={js['completeness']}"
                )
            results.append(r)

    # сводим по температурам
    summary: dict[float, dict] = {}
    for t in TEMPERATURES:
        rows = [r for r in results if r["temperature"] == t]
        summary[t] = {
            "n_cases": len(rows),
            "asserts_total": sum(r["asserts_passed"] for r in rows),
            "asserts_max": 10 * len(rows),
            "judge_total": sum(r["judge_total"] for r in rows),
            "judge_max": 40 * len(rows),
            "avg_tone": round(
                sum(
                    r["judge_scores"].get("tone", 0)
                    for r in rows
                    if isinstance(r["judge_scores"], dict)
                )
                / max(len(rows), 1),
                2,
            ),
            "avg_readability": round(
                sum(
                    r["judge_scores"].get("readability", 0)
                    for r in rows
                    if isinstance(r["judge_scores"], dict)
                )
                / max(len(rows), 1),
                2,
            ),
            "avg_factual": round(
                sum(
                    r["judge_scores"].get("factual_accuracy", 0)
                    for r in rows
                    if isinstance(r["judge_scores"], dict)
                )
                / max(len(rows), 1),
                2,
            ),
            "avg_elapsed_s": round(sum(r["elapsed_s"] for r in rows) / max(len(rows), 1), 1),
        }

    print(f"\n{'═' * 72}\nИТОГИ по температурам Writer:")
    print(
        f"  {'T':<6}{'asserts':<14}{'judge':<14}{'avg_tone':<10}{'avg_read':<10}{'avg_fact':<10}{'avg_t':<8}"
    )
    for t, s in summary.items():
        print(
            f"  {t:<6}{s['asserts_total']}/{s['asserts_max']:<11}"
            f"{s['judge_total']}/{s['judge_max']:<11}"
            f"{s['avg_tone']:<10}{s['avg_readability']:<10}{s['avg_factual']:<10}{s['avg_elapsed_s']:<8}"
        )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"temperature-{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "experiment": "writer_temperature",
                "model_writer": config.MODEL_WRITER,
                "temperatures": TEMPERATURES,
                "cases": [c.name for c in cases],
                "summary": {str(k): v for k, v in summary.items()},
                "details": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nОтчёт: {out.relative_to(ROOT)}")
    # сбрасываем override чтобы не повлиять на последующие запуски
    config.WRITER_TEMPERATURE = None


if __name__ == "__main__":
    main()
