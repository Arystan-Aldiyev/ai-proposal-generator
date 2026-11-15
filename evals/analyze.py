"""Сводка результатов test-прогонов: таблицы и графики для отчёта.

Читает evals/results/test-<config>.json, считает по каждой конфигурации
средние по кейсам (сначала усредняем повторы внутри кейса), 95% bootstrap-
интервал по кейсам и разброс между повторами. Сохраняет
evals/results/summary.json и графики в evals/figures/.

Запуск: python -m evals.analyze
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evals.run import RESULTS_DIR
from evals.test_cases import TEST_CASES

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

CONFIG_ORDER = ["baseline", "no-rag", "all-mini", "final"]
LABELS = {
    "baseline": "Single-prompt\nbaseline",
    "no-rag": "No RAG",
    "all-mini": "All on\ngpt-4o-mini",
    "final": "Final\nsystem",
}
COLORS = {"baseline": "#2a78d6", "no-rag": "#eb6834", "all-mini": "#1baf7a", "final": "#eda100"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
JUDGE_AXES = ["factual_accuracy", "tone", "readability", "completeness"]
# проверка №2 «Найден ожидаемый аналог»: у baseline нет RAG, для него она невыполнима,
# поэтому при сравнении с baseline считаем 9 общих проверок без неё
RETRIEVAL_CHECK = 1
CAUTION_FILE = RESULTS_DIR / "caution-gpt-4.1-mini.json"


def _per_case(runs: list[dict], key) -> np.ndarray:
    """Среднее значение метрики по повторам для каждого кейса."""
    values: dict[int, list[float]] = {}
    for r in runs:
        v = key(r)
        if v is not None:
            values.setdefault(r["idx"], []).append(float(v))
    return np.array([np.mean(values[i]) for i in sorted(values)])


def _ci(x: np.ndarray, n: int = 2000) -> tuple[float, float]:
    """95% bootstrap-интервал для среднего по кейсам."""
    rng = np.random.default_rng(0)  # свой seed на каждый расчёт: CI не зависит от порядка
    means = [rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _repeat_std(runs: list[dict], key) -> float:
    """Средний по кейсам разброс (std) между повторами — устойчивость."""
    values: dict[int, list[float]] = {}
    for r in runs:
        values.setdefault(r["idx"], []).append(float(key(r)))
    stds = [np.std(v) for v in values.values() if len(v) > 1]
    return float(np.mean(stds)) if stds else 0.0


def _shared_checks(r: dict) -> int:
    """Число пройденных проверок без проверки аналогов (9 общих для всех конфигураций)."""
    return sum(ok for i, ok in enumerate(r["asserts"]) if i != RETRIEVAL_CHECK)


def summarize(runs: list[dict], is_pipeline: bool) -> dict:
    judge = _per_case(runs, lambda r: r["judge_total"])
    asserts = _per_case(runs, lambda r: r["asserts_passed"])
    price = _per_case(runs, lambda r: r["price_ok"])
    shared = _per_case(runs, _shared_checks)
    out = {
        "n_runs": len(runs),
        "n_cases": len(judge),
        "judge_mean": round(float(judge.mean()), 2),
        "judge_ci": [round(v, 2) for v in _ci(judge)],
        "judge_repeat_std": round(_repeat_std(runs, lambda r: r["judge_total"]), 2),
        "judge_axes": {
            a: round(float(_per_case(runs, lambda r, a=a: r["judge"][a]).mean()), 2)
            for a in JUDGE_AXES
        },
        "asserts_mean": round(float(asserts.mean()), 2),
        "asserts_ci": [round(v, 2) for v in _ci(asserts)],
        "cases_passed_8of10": int((asserts >= 8).sum()),
        "shared_checks_mean": round(float(shared.mean()), 2),
        "shared_checks_ci": [round(v, 2) for v in _ci(shared)],
        "price_ok_rate": round(float(price.mean()), 3),
        "retrieval_ok_rate": round(float(_per_case(runs, lambda r: r["retrieval_ok"]).mean()), 3),
        "latency_mean_s": round(float(np.mean([r["latency_s"] for r in runs])), 1),
        "latency_p90_s": round(float(np.percentile([r["latency_s"] for r in runs], 90)), 1),
        "cost_mean_usd": round(float(np.mean([r["cost_usd"] for r in runs])), 4),
        "tokens_mean": round(
            float(
                np.mean(
                    [sum(t["prompt"] + t["completion"] for t in r["tokens"].values()) for r in runs]
                )
            )
        ),
        "nda_leaks": int(sum(r["nda_leaks"] for r in runs)),
        "nda_text_fixed": int(sum(r["nda_text_fixed"] for r in runs)),
        "nda_portfolio_removed": int(sum(r["nda_portfolio_removed"] for r in runs)),
        "estimator_retries": int(sum(r["estimator_retries"] for r in runs)),
    }
    if is_pipeline:
        correct = [r["cautious"] == r["expected_cautious"] for r in runs]
        hard = [r for r in runs if r["expected_cautious"]]
        easy = [r for r in runs if not r["expected_cautious"]]
        out["triage_accuracy"] = round(float(np.mean(correct)), 3)
        out["triage_recall_hard"] = (
            round(float(np.mean([r["cautious"] for r in hard])), 3) if hard else None
        )
        out["triage_false_alarm"] = (
            round(float(np.mean([r["cautious"] for r in easy])), 3) if easy else None
        )
    return out


def paired(a: list[dict], b: list[dict], key) -> dict:
    """Попарное сравнение двух конфигураций по одним и тем же кейсам (b минус a)."""
    diff = _per_case(b, key) - _per_case(a, key)
    return {
        "mean_diff": round(float(diff.mean()), 2),
        "diff_ci": [round(v, 2) for v in _ci(diff)],
        "wins": int((diff > 0.5).sum()),
        "ties": int((np.abs(diff) <= 0.5).sum()),
        "losses": int((diff < -0.5).sum()),
    }


def caution_summary(verdicts: dict) -> dict:
    """Осторожность по оценке судьи: доля КП с явной неопределённостью на сложных
    кейсах и на остальных, по кейсам (повторы усредняются), и совпадение с triage."""
    out = {}
    for cfg in CONFIG_ORDER:
        rows = [v for v in verdicts.values() if v["config"] == cfg]
        if not rows:
            continue
        per_case: dict[int, list[bool]] = {}
        for v in rows:
            per_case.setdefault(v["idx"], []).append(v["explicit_uncertainty"])
        expected = {v["idx"]: v["expected_cautious"] for v in rows}
        hard = [np.mean(per_case[i]) for i in per_case if expected[i]]
        easy = [np.mean(per_case[i]) for i in per_case if not expected[i]]
        s = {
            "hard_rate": round(float(np.mean(hard)), 3),
            "others_rate": round(float(np.mean(easy)), 3),
            "hard_by_case": {
                f"T{i:02d}": [bool(x) for x in per_case[i]] for i in sorted(per_case) if expected[i]
            },
        }
        with_triage = [v for v in rows if v["triage_cautious"] is not None]
        if with_triage:
            s["agrees_with_triage"] = round(
                float(
                    np.mean(
                        [v["triage_cautious"] == v["explicit_uncertainty"] for v in with_triage]
                    )
                ),
                3,
            )
        out[cfg] = s
    return out


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_quality(summary: dict, configs: list[str]) -> None:
    """Оценка судьи и доля цен в коридоре — две отдельные шкалы, два графика."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    panels = [
        ("judge_mean", "judge_ci", "LLM judge score, out of 40", 40),
        ("price_ok_rate", None, "Price within the manager's range, share of cases", 1),
    ]
    for ax, (key, ci_key, title, top) in zip(axes, panels, strict=False):
        x = np.arange(len(configs))
        vals = [summary[c][key] for c in configs]
        ax.bar(
            x, vals, width=0.6, color=[COLORS[c] for c in configs], edgecolor="white", linewidth=2
        )
        label_y = list(vals)
        if ci_key:
            lo = [summary[c][key] - summary[c][ci_key][0] for c in configs]
            hi = [summary[c][ci_key][1] - summary[c][key] for c in configs]
            ax.errorbar(x, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, capsize=4, linewidth=1)
            label_y = [summary[c][ci_key][1] for c in configs]  # подпись над интервалом
        for xi, v, y in zip(x, vals, label_y, strict=False):
            ax.text(
                xi,
                y + top * 0.03,
                f"{v:.1f}" if top > 1 else f"{v:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=INK,
            )
        ax.set_xticks(x, [LABELS[c] for c in configs])
        ax.set_ylim(0, top * 1.15)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quality.png", dpi=200)
    plt.close(fig)


def plot_cost_quality(summary: dict, configs: list[str]) -> None:
    """Компромисс стоимость/качество (оптимизация)."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for c in configs:
        s = summary[c]
        ax.scatter(
            s["cost_mean_usd"] * 100,
            s["judge_mean"],
            s=80,
            color=COLORS[c],
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
        ax.annotate(
            f"{LABELS[c].replace(chr(10), ' ')}\n{s['latency_mean_s']:.0f} s",
            (s["cost_mean_usd"] * 100, s["judge_mean"]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8,
            color=INK2,
        )
    ax.set_xlabel(
        "Generation cost per proposal, US cents (OpenAI list prices)", fontsize=9, color=INK2
    )
    ax.set_ylabel("LLM judge score, out of 40", fontsize=9, color=INK2)
    ax.set_title("Cost vs. quality", fontsize=10, color=INK, loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_quality.png", dpi=200)
    plt.close(fig)


def plot_per_case(results: dict, configs: list[str]) -> None:
    """Оценка судьи по каждому test-кейсу — где система ошибается."""
    names = [c.name.split(".")[0] for c in TEST_CASES]
    matrix = np.array([_per_case(results[c]["runs"], lambda r: r["judge_total"]) for c in configs])
    fig, ax = plt.subplots(figsize=(10, 2.8))
    im = ax.imshow(matrix, cmap="Blues", vmin=10, vmax=40, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if v > 30 else INK,
            )
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_yticks(range(len(configs)), [LABELS[c].replace("\n", " ") for c in configs], fontsize=8)
    ax.set_title(
        "LLM judge score per test case (mean over repeats, out of 40)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_case.png", dpi=200)
    plt.close(fig)


def main() -> None:
    results = {}
    for c in CONFIG_ORDER:
        path = RESULTS_DIR / f"test-{c}.json"
        if path.exists():
            results[c] = json.loads(path.read_text(encoding="utf-8"))
    configs = [c for c in CONFIG_ORDER if c in results]
    summary = {c: summarize(results[c]["runs"], c != "baseline") for c in configs}

    comparisons = {}
    if "final" in results:
        for other in configs:
            if other != "final":
                a, b = results[other]["runs"], results["final"]["runs"]
                comparisons[f"final_vs_{other}"] = {
                    "judge": paired(a, b, lambda r: r["judge_total"]),
                    "shared_checks": paired(a, b, _shared_checks),
                    "price_ok": paired(a, b, lambda r: r["price_ok"]),
                }

    caution = {}
    if CAUTION_FILE.exists():
        caution = caution_summary(json.loads(CAUTION_FILE.read_text(encoding="utf-8"))["verdicts"])

    for c in configs:
        s = summary[c]
        print(
            f"\n{c}: судья {s['judge_mean']} {s['judge_ci']} · ассерты {s['asserts_mean']} · "
            f"9 общих {s['shared_checks_mean']} {s['shared_checks_ci']} · "
            f"цена в коридоре {s['price_ok_rate']:.0%} · {s['latency_mean_s']} с · ${s['cost_mean_usd']} · "
            f"NDA-утечек {s['nda_leaks']}"
            + (f" · triage {s['triage_accuracy']:.0%}" if "triage_accuracy" in s else "")
        )
    for k, v in comparisons.items():
        print(f"{k}: {v}")
    for k, v in caution.items():
        print(f"осторожность {k}: {v}")

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(
            {"configs": summary, "comparisons": comparisons, "caution": caution},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if configs:
        plot_quality(summary, configs)
        plot_cost_quality(summary, configs)
        plot_per_case(results, configs)
        print(f"\nГрафики: {FIG_DIR}")


if __name__ == "__main__":
    main()
