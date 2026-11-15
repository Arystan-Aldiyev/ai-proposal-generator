"""Проверка разделения dev/test на утечки и дубликаты.

1. Точные дубликаты: одинаковый текст брифа после нормализации.
2. Смысловые близнецы: косинусная близость эмбеддингов каждого test-брифа
   со всеми dev-брифами и с карточками базы знаний (storage/rag_index.json).
   Близость выше NEAR_DUP считаем подозрением на утечку.

Запуск: python -m evals.check_split
"""

from __future__ import annotations

import json
import re

import numpy as np

from app import config
from app.llm import embed
from evals.cases import CASES
from evals.run import RESULTS_DIR
from evals.test_cases import TEST_CASES

NEAR_DUP = 0.90


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _unit(vectors: list[list[float]]) -> np.ndarray:
    m = np.array(vectors, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def main() -> None:
    dev_texts = [c.brief for c in CASES]
    test_texts = [c.brief for c in TEST_CASES]

    exact = sorted({_norm(t) for t in dev_texts} & {_norm(t) for t in test_texts})

    kb = json.loads(config.INDEX_PATH.read_text(encoding="utf-8"))
    dev_m = _unit(embed(dev_texts))
    test_m = _unit(embed(test_texts))
    kb_m = _unit([row["embedding"] for row in kb])

    sim_dev = test_m @ dev_m.T
    sim_kb = test_m @ kb_m.T
    sim_test = test_m @ test_m.T
    np.fill_diagonal(sim_test, 0)

    rows = []
    print(f"{'test-кейс':<62}{'max dev':>9}{'max KB':>9}{'max test':>10}")
    for i, case in enumerate(TEST_CASES):
        row = {
            "case": case.name,
            "max_dev": round(float(sim_dev[i].max()), 3),
            "closest_dev": CASES[int(sim_dev[i].argmax())].name,
            "max_kb": round(float(sim_kb[i].max()), 3),
            "closest_kb": kb[int(sim_kb[i].argmax())]["project_id"],
            "max_test": round(float(sim_test[i].max()), 3),
        }
        rows.append(row)
        print(f"{case.name[:60]:<62}{row['max_dev']:>9}{row['max_kb']:>9}{row['max_test']:>10}")

    flagged = [r["case"] for r in rows if max(r["max_dev"], r["max_kb"], r["max_test"]) > NEAR_DUP]
    print(f"\nТочных дубликатов dev/test: {len(exact)}")
    print(f"Пар с близостью > {NEAR_DUP}: {len(flagged)} {flagged if flagged else ''}")

    out = RESULTS_DIR / "split_check.json"
    out.write_text(
        json.dumps(
            {
                "near_dup_threshold": NEAR_DUP,
                "exact_duplicates": exact,
                "flagged": flagged,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Отчёт: {out}")


if __name__ == "__main__":
    main()
