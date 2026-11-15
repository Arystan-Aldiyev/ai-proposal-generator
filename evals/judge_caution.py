"""Осторожность КП, измеренная одинаково для всех конфигураций.

В test-прогонах осторожность пайплайна — это решение triage, а у baseline
triage нет (в результатах cautious = None), то есть у него она не измерялась.
Здесь сохранённые КП всех конфигураций получают один вопрос LLM-судье: явно ли
КП обозначает неопределённость и предлагает уточнить требования вместо твёрдой
цены. Судья видит только то, что уходит клиенту, без брифа и без ожиданий
кейса. Новой генерации нет.

Запуск: python -m evals.judge_caution --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json

from pydantic import BaseModel, Field

from app import state as state_store
from app.llm import _client
from app.pipeline import RunResult
from evals.run import RESULTS_DIR, _client_facing_text

CONFIGS = ["baseline", "no-rag", "all-mini", "final"]

QUESTION = (
    "КП явно обозначает неопределённость и предлагает уточнить требования, "
    "а не даёт твёрдую цену?"
)

SYSTEM = f"""Ты проверяешь коммерческое предложение веб-агентства клиенту.
Ответь на один вопрос: {QUESTION}

Да — КП прямо говорит клиенту, что данных недостаточно или объём не определён,
называет цены ориентировочными и предлагает уточнить требования (вопросы,
созвон) до того, как цена будет зафиксирована.
Нет — КП подаёт цены и сроки как готовое предложение. Дежурная фраза «детали
обсудим на созвоне» или обычный раздел «что не входит» сами по себе не дают «да».

Сначала приведи одну-две короткие цитаты из КП, на которых основан ответ."""


class CautionVerdict(BaseModel):
    evidence: str = Field(description="1–2 короткие цитаты из КП, на которых основан ответ")
    explicit_uncertainty: bool


def judge_caution(r: RunResult, model: str) -> CautionVerdict:
    completion = _client().beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "=== КП (всё, что видит клиент) ===\n" + _client_facing_text(r),
            },
        ],
        response_format=CautionVerdict,
        temperature=0,
    )
    message = completion.choices[0].message
    if message.parsed is None:
        raise RuntimeError(message.refusal or "Судья вернул пустой результат")
    return message.parsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args()

    out = RESULTS_DIR / f"caution-{args.model}.json"
    data = (
        json.loads(out.read_text(encoding="utf-8"))
        if out.exists()
        else {"model": args.model, "question": QUESTION, "verdicts": {}}
    )
    verdicts = data["verdicts"]

    for cfg in CONFIGS:
        path = RESULTS_DIR / f"test-{cfg}.json"
        if not path.exists():
            continue
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            if run["run_id"] in verdicts:
                continue
            state = state_store.load_state(run["run_id"])
            r = RunResult(
                run_id=run["run_id"], output_dir=state_store.run_dir(run["run_id"]), state=state
            )
            try:
                v = judge_caution(r, args.model)
            except Exception as exc:  # noqa: BLE001
                print(f"  ОШИБКА {run['run_id']}: {exc}")
                continue
            verdicts[run["run_id"]] = {
                "config": cfg,
                "idx": run["idx"],
                "expected_cautious": run["expected_cautious"],
                "triage_cautious": run["cautious"],
                "explicit_uncertainty": v.explicit_uncertainty,
                "evidence": v.evidence,
            }
            print(f"{run['run_id']}: {'да' if v.explicit_uncertainty else 'нет'}")
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    for cfg in CONFIGS:
        rows = [v for v in verdicts.values() if v["config"] == cfg]
        if not rows:
            continue
        hard = [v["explicit_uncertainty"] for v in rows if v["expected_cautious"]]
        easy = [v["explicit_uncertainty"] for v in rows if not v["expected_cautious"]]
        line = f"{cfg:<10} осторожно на сложных: {sum(hard)}/{len(hard)}, на остальных: {sum(easy)}/{len(easy)}"
        with_triage = [v for v in rows if v["triage_cautious"] is not None]
        if with_triage:
            agree = sum(v["triage_cautious"] == v["explicit_uncertainty"] for v in with_triage)
            line += f", совпадает с triage: {agree}/{len(with_triage)}"
        print(line)


if __name__ == "__main__":
    main()
