"""LLM-as-judge: вторая метрика поверх assert-проверок.

Оценивает готовое КП по 4 осям 1..10:
  - factual_accuracy:  цифры/сроки/состав работ не выглядят выдуманными,
                       не противоречат брифу
  - tone:              соответствует tone-of-voice CNS (профессионально,
                       без штампов «команда специалистов», «готовы помочь»)
  - readability:       структура понятна, абзацы короткие, нет воды
  - completeness:      покрыты все ключевые секции (intro, варианты,
                       команда, оплата, риски, что НЕ входит)

По умолчанию судья — gpt-4o (evals/rejudge.py перепроверяет другим судьёй) — это критерий 2 (вторая метрика) и одновременно
сильный фильтр против регрессий, которых assert не ловит (например,
скучный тон или хвалебная вода).
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from app.llm import _client
from app.metrics import add_usage
from app.pipeline import RunResult
from evals.cases import Case

JUDGE_MODEL = "gpt-4o"


class JudgeVerdict(BaseModel):
    factual_accuracy: int = Field(ge=1, le=10)
    tone: int = Field(ge=1, le=10)
    readability: int = Field(ge=1, le=10)
    completeness: int = Field(ge=1, le=10)
    explanation: str = Field(
        description="2–4 фразы: что хорошо, что плохо, какой балл занижен и за что"
    )


JUDGE_SYSTEM = """Ты независимый судья качества коммерческих предложений CNS Agency.

Перед тобой клиентский бриф и сгенерированное КП. Оцени КП по 4 осям,
каждой ставь целое от 1 до 10:

1) factual_accuracy — насколько цифры (цена, сроки, рабочие дни, расходы)
   и состав работ внутри КП согласованы между собой и не противоречат
   брифу. Выдуманные интеграции, скоуп «из воздуха», нереалистичные
   обещания — это минус. Если бриф мутный, а КП всё равно делает
   уверенные категоричные заявления — это минус.
   Смета дороже бюджета клиента — НЕ ошибка, если объём работ это
   оправдывает и КП честно об этом говорит. Минус — если цифры подогнаны
   под нереалистичный бюджет или КП обещает уложиться в невозможное.

2) tone — соответствие профессиональному тону агентства:
   - НЕТ клише «команда специалистов», «готовы помочь», «высокое качество»
   - НЕТ заискивания, восклицательных, «уверены что вам понравится»
   - ДА — конкретика, спокойный экспертный тон
   - ДА — фразы как в эталоне CNS: «зафиксируем», «возьмём в работу»

3) readability — структура и подача:
   - короткие абзацы, нет воды
   - в pricing_intro / stages_intro нет дублирующей таблицы текстом
   - каждый раздел читается за 10 секунд
   - заголовки не повторяются 1-в-1 в первой строке абзаца

4) completeness — покрытие обязательных секций:
   - subtitle, intro, architecture_intro, stages_intro, pricing_intro,
     team_intro, included, not_included, guarantees, portfolio, next_step
   - если что-то заметно пустое или формальное — минус

Прокомментируй своим explanation одним абзацем 2–4 фразы. Будь жёстким:
10 — это эталон уровня лучших КП агентства. 7–8 — нормальное КП.
5–6 — есть критика. Ниже 5 — серьёзные проблемы."""


def judge_verdict(case: Case, r: RunResult, model: str = JUDGE_MODEL) -> JudgeVerdict:
    """Оценки по 4 осям и пояснение судьи. Бросает RuntimeError если LLM отказался."""
    payload = {
        "spec": json.loads(r.spec.model_dump_json()),
        "estimation": json.loads(r.estimation.model_dump_json()),
        "proposal": json.loads(r.proposal.model_dump_json()),
    }
    # notes — ожидаемое поведение в кейсе (например, «не подгонять смету под
    # бюджет»): без них судья считал честную смету выше бюджета ошибкой
    notes = f"=== ЧТО ОЖИДАЕТСЯ ОТ КП В ЭТОМ КЕЙСЕ ===\n{case.notes}\n\n" if case.notes else ""
    user = (
        "Оцени КП по 4 осям. Верни оценки и краткое объяснение.\n\n"
        "=== БРИФ ===\n"
        f"{case.brief}\n\n"
        f"{notes}"
        "=== ИЗВЛЕЧЁННЫЙ SPEC ===\n"
        f"{json.dumps(payload['spec'], ensure_ascii=False, indent=2)}\n\n"
        "=== СМЕТА ===\n"
        f"{json.dumps(payload['estimation'], ensure_ascii=False, indent=2)}\n\n"
        "=== ТЕКСТЫ КП ===\n"
        f"{json.dumps(payload['proposal'], ensure_ascii=False, indent=2)}"
    )

    completion = _client().beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format=JudgeVerdict,
        temperature=0,
    )
    if completion.usage:
        add_usage(model, completion.usage.prompt_tokens, completion.usage.completion_tokens)
    message = completion.choices[0].message
    if message.refusal:
        raise RuntimeError(f"Судья отказался: {message.refusal}")
    if message.parsed is None:
        raise RuntimeError("Судья вернул пустой результат")
    return message.parsed


def judge_proposal(case: Case, r: RunResult, model: str = JUDGE_MODEL) -> dict[str, int]:
    """Только 4 оценки — для evals/temperature.py."""
    v = judge_verdict(case, r, model)
    return {
        "factual_accuracy": v.factual_accuracy,
        "tone": v.tone,
        "readability": v.readability,
        "completeness": v.completeness,
    }
