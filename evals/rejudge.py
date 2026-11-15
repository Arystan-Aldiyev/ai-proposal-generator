"""Перепроверка оценок другим судьёй.

Основной судья (gpt-4o) — та же модель, что пишет тексты КП, и может
завышать «свои» тексты. Здесь сохранённые КП из test-прогонов заново
оцениваются другой моделью, без повторной генерации. Сравниваем, совпадает
ли порядок конфигураций и насколько коррелируют оценки.

Запуск: python -m evals.rejudge --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from app import state as state_store
from app.pipeline import RunResult
from evals.judge import judge_verdict
from evals.run import RESULTS_DIR
from evals.test_cases import TEST_CASES

CONFIGS = ["baseline", "no-rag", "all-mini", "final"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    out = RESULTS_DIR / f"rejudge-{args.model}.json"
    scores = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}

    for cfg in CONFIGS:
        path = RESULTS_DIR / f"test-{cfg}.json"
        if not path.exists():
            continue
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            if run["run_id"] in scores:
                continue
            case = TEST_CASES[run["idx"] - 1]
            state = state_store.load_state(run["run_id"])
            r = RunResult(
                run_id=run["run_id"], output_dir=state_store.run_dir(run["run_id"]), state=state
            )
            try:
                v = judge_verdict(case, r, model=args.model)
            except Exception as exc:  # noqa: BLE001
                print(f"  ОШИБКА {run['run_id']}: {exc}")
                continue
            total = v.factual_accuracy + v.tone + v.readability + v.completeness
            scores[run["run_id"]] = {
                "config": cfg,
                "main_judge": run["judge_total"],
                "second_judge": total,
            }
            print(f"{run['run_id']}: {run['judge_total']} → {total}")
            out.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    main_j = np.array([s["main_judge"] for s in scores.values()])
    second_j = np.array([s["second_judge"] for s in scores.values()])
    if len(scores) > 2:
        corr = float(np.corrcoef(main_j, second_j)[0, 1])
        print(f"\nКорреляция оценок двух судей (Пирсон): {corr:.2f} на {len(scores)} КП")
    for cfg in CONFIGS:
        rows = [s for s in scores.values() if s["config"] == cfg]
        if rows:
            print(
                f"{cfg:<10} gpt-4o: {np.mean([s['main_judge'] for s in rows]):.2f}   "
                f"{args.model}: {np.mean([s['second_judge'] for s in rows]):.2f}"
            )


if __name__ == "__main__":
    main()
