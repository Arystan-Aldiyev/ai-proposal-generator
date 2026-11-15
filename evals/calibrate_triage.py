"""Подбор порогов triage на dev-наборе (test не используется).

Для каждого dev-брифа несколько раз прогоняем только extract → retrieve
(дёшево: gpt-4o-mini + эмбеддинги) и смотрим два сигнала:
  - top_score — близость лучшего аналога из архива;
  - scope     — сколько пунктов работ извлечено из брифа.
Затем перебираем порог похожести и считаем, как часто правило
«нет похожих проектов» срабатывает правильно.

Запуск: python -m evals.calibrate_triage
"""

from __future__ import annotations

import json

import numpy as np

from app.agents.extractor import extract_spec
from app.pipeline import WEAK_MIN_SCOPE, _rag_query
from app.rag.retriever import search
from evals.cases import CASES
from evals.run import RESULTS_DIR

REPEATS = 3
THRESHOLDS = [0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45]  # выбран 0.375 (app/pipeline.py)


def main() -> None:
    rows = []
    for rep in range(REPEATS):
        for i, case in enumerate(CASES, start=1):
            spec = extract_spec(case.brief)
            top = search(_rag_query(spec), k=1)[0].score
            rows.append(
                {
                    "idx": i,
                    "rep": rep,
                    "top_score": round(top, 3),
                    "scope": len(spec.scope),
                    "off_domain": case.kind == "other",
                    "expected_cautious": case.expected_cautious,
                }
            )
            print(f"{case.name[:45]:<47} top={top:.3f} scope={len(spec.scope)}")

    # правило no_analogue должно срабатывать только на брифах вне профиля
    sweep = []
    for t in THRESHOLDS:
        acc = np.mean([(r["top_score"] < t) == r["off_domain"] for r in rows])
        false_alarm = np.mean([r["top_score"] < t for r in rows if not r["off_domain"]])
        recall = np.mean([r["top_score"] < t for r in rows if r["off_domain"]])
        sweep.append(
            {
                "threshold": t,
                "accuracy": round(float(acc), 3),
                "recall_off_domain": round(float(recall), 3),
                "false_alarm": round(float(false_alarm), 3),
            }
        )
        print(
            f"порог {t:.3f}: точность {acc:.2f}, срабатывает вне профиля {recall:.2f}, ложные {false_alarm:.2f}"
        )

    low_info = np.mean([(r["scope"] <= WEAK_MIN_SCOPE) == (r["idx"] == 9) for r in rows])
    print(f"\nправило low_info (scope <= {WEAK_MIN_SCOPE}): верно в {low_info:.0%} прогонов")

    (RESULTS_DIR / "triage_calibration.json").write_text(
        json.dumps(
            {"rows": rows, "sweep": sweep, "low_info_accuracy": round(float(low_info), 3)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
