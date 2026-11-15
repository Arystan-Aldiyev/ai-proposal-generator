"""Рендер образца КП из фиксированных данных — БЕЗ вызова OpenAI.

Нужен для проверки вёрстки и стиля PDF без вызова API.
Данные имитируют выход цепочки агентов для брифа стоматологии (eval-тест 1).

Запуск:  python -m scripts.sample_render
"""

from __future__ import annotations

from app import config
from app.render.html import build_context, render_html
from app.render.pdf import html_to_pdf
from app.schemas import (
    ArchDecision,
    CostItem,
    Estimation,
    PaymentStage,
    PortfolioItem,
    Proposal,
    ProposalState,
    Risk,
    Risks,
    Spec,
    Stage,
    TeamRole,
    Variant,
)
from app.state import recompute_totals

spec = Spec(
    input_mode="brief",
    client_name=None,
    client_label="Стоматология (сеть из 2 филиалов)",
    client_industry="Медицина / стоматологические услуги",
    project_type="Редизайн сайта стоматологии с онлайн-записью",
    scope=[
        "Редизайн сайта, mobile-first",
        "Модуль онлайн-записи к врачу",
        "Страницы услуг и врачей по двум филиалам",
        "Интеграция заявок с мессенджерами",
    ],
    sub_projects=[],
    deadline=None,
    budget_kzt=None,
    budget_signal="не миллионы — бюджет ограничен",
    special_requirements=["Mobile-first", "Онлайн-запись", "Два филиала"],
    assumptions=[
        "Бюджет не назван; по фразе «не миллионы» исходим из ограниченного бюджета",
        "Контент (тексты, фото врачей) предоставляет клиент",
    ],
    unclear_points=[
        "Какая CRM/телефония используется сейчас",
        "Нужен ли личный кабинет пациента",
    ],
)

econom = Variant(
    name="Эконом",
    summary="Сайт на Tilda + WordPress с модулем заявок. Быстрый запуск, ограниченный бюджет.",
    cost_items=[
        CostItem(
            item="UX/UI дизайн, mobile-first, дизайн-система", amount_kzt=200_000, sub_project=None
        ),
        CostItem(
            item="Вёрстка и сборка страниц (услуги, врачи, филиалы)",
            amount_kzt=250_000,
            sub_project=None,
        ),
        CostItem(
            item="Модуль онлайн-записи (форма + расписание, без ЛК)",
            amount_kzt=200_000,
            sub_project=None,
        ),
        CostItem(
            item="Интеграция заявок с WhatsApp/Telegram, тестирование",
            amount_kzt=100_000,
            sub_project=None,
        ),
    ],
    stages=[
        Stage(
            name="Аналитика и подготовка",
            description="Аудит текущего сайта, точки потери заявок, структура",
            work_days=3,
        ),
        Stage(
            name="Дизайн UI/UX",
            description="Дизайн-система, макеты ПК и мобильной версии",
            work_days=7,
        ),
        Stage(name="Разработка", description="Вёрстка страниц, модуль онлайн-записи", work_days=8),
        Stage(
            name="Интеграции и тестирование",
            description="Заявки в мессенджеры, кросс-браузерность, запуск",
            work_days=4,
        ),
    ],
    total_kzt=750_000,
    total_our_tax_kzt=850_000,
    duration="4–5 недель",
    work_days=22,
)

corporate = Variant(
    name="Корпоративный",
    summary="Сайт на Next.js с полноценным модулем записи и интеграцией CRM. Запас на рост.",
    cost_items=[
        CostItem(item="UX/UI дизайн, CJM, mobile-first", amount_kzt=350_000, sub_project=None),
        CostItem(
            item="Frontend Next.js (SSR/SEO), каталог услуг и врачей",
            amount_kzt=700_000,
            sub_project=None,
        ),
        CostItem(
            item="Модуль онлайн-записи: расписание, выбор врача и филиала",
            amount_kzt=500_000,
            sub_project=None,
        ),
        CostItem(
            item="Интеграция CRM + аналитика, тестирование, запуск",
            amount_kzt=350_000,
            sub_project=None,
        ),
    ],
    stages=[
        Stage(
            name="Аналитика и подготовка",
            description="Аудит, CJM, архитектура, дизайн-система",
            work_days=8,
        ),
        Stage(
            name="Дизайн UI/UX",
            description="Макеты ПК и мобильной версии, согласование",
            work_days=12,
        ),
        Stage(
            name="Разработка ядра", description="Frontend Next.js, базовые страницы", work_days=14
        ),
        Stage(
            name="Модуль записи и интеграции",
            description="Онлайн-запись, CRM, аналитика",
            work_days=8,
        ),
        Stage(
            name="Тестирование и запуск",
            description="QA, кросс-браузерность, обучение",
            work_days=4,
        ),
    ],
    total_kzt=1_900_000,
    total_our_tax_kzt=2_100_000,
    duration="8–10 недель",
    work_days=46,
)

estimation = Estimation(
    variants=[econom, corporate],
    recommended_variant="Эконом",
    recommendation_reason="Клиент обозначил ограниченный бюджет; экономный вариант "
    "закрывает онлайн-запись и mobile-first без переплаты.",
    team=[
        TeamRole(role="Project Manager", responsibility="Ведение проекта, коммуникация, сроки"),
        TeamRole(
            role="UX/UI дизайнер", responsibility="Дизайн-система, макеты, мобильная адаптация"
        ),
        TeamRole(
            role="Frontend-разработчик", responsibility="Вёрстка, модуль онлайн-записи, интеграции"
        ),
        TeamRole(role="QA", responsibility="Тестирование форм, кросс-браузерность, мобильный UX"),
    ],
    payment_schedule=[
        PaymentStage(
            name="Предоплата", share="30%", moment="После подписания договора, до начала работ"
        ),
        PaymentStage(name="Дизайн утверждён", share="40%", moment="После согласования макетов"),
        PaymentStage(name="Запуск", share="30%", moment="После публикации и передачи доступов"),
    ],
    recurring_costs=[],
    analogues_used=["project-f", "project-e"],
    assumptions=[
        "Прямого аналога «стоматология» в базе нет; опора — project-f "
        "(сервисный сайт с онлайн-бронированием). Оценка ориентировочная.",
    ],
)

risks = Risks(
    architectural_decisions=[
        ArchDecision(
            title="Модуль онлайн-записи: своя форма vs внешний сервис",
            description="Рекомендуем простой собственный модуль (расписание + заявка) "
            "без личного кабинета на старте. Это дешевле и быстрее; ЛК пациента "
            "можно добавить отдельным этапом, когда поток записей вырастет.",
        ),
        ArchDecision(
            title="Два филиала — единая структура",
            description="Услуги и врачи привязываются к филиалу через фильтр, "
            "а не дублируются. Одна система контента, проще поддерживать.",
        ),
    ],
    risks=[
        Risk(
            title="Запись «уходит в никуда»",
            impact="Заявка с сайта не доходит до администратора — потерянный пациент",
            mitigation="Дублируем заявку в WhatsApp/Telegram администратора + email, "
            "проверяем доставку на этапе тестирования",
        ),
        Risk(
            title="Контент по врачам не готов к запуску",
            impact="Сайт запускается с пустыми карточками врачей",
            mitigation="Шаблон карточки и список нужных материалов выдаём в начале "
            "проекта, наполнение идёт параллельно разработке",
        ),
        Risk(
            title="Мобильная вёрстка старого сайта тянется в новый",
            impact="Повторяется проблема «криво на телефоне»",
            mitigation="Дизайн и вёрстка ведутся mobile-first, проверка на реальных "
            "устройствах в составе QA-этапа",
        ),
    ],
    caveats=[
        "Контент (тексты, фотографии врачей и кабинетов) предоставляет клиент",
        "Лицензии и тарифы сторонних сервисов (CRM, телефония) оплачиваются отдельно",
    ],
)

proposal = Proposal(
    subtitle="Редизайн сайта стоматологии с онлайн-записью",
    intro="CNS Agency — веб-студия полного цикла. Делаем сайты, которые приносят "
    "заявки, а не просто красиво выглядят.\n"
    "Мы поняли задачу так: текущий сайт на WordPress плохо работает на телефоне, "
    "и поток заявок почти остановился. Нужен современный сайт с удобной онлайн-"
    "записью, по которому пациент с телефона за минуту запишется в нужный филиал. "
    "Спасибо за рекомендацию — мы ценим, что пришли по совету.",
    architecture_intro="Прежде чем называть цифры, проговорим пару решений, которые "
    "влияют на результат и стоимость.",
    stages_intro="Работаем в четыре этапа. Ниже — рекомендуемый вариант «Эконом» "
    "(22 рабочих дня). Сроки по «Корпоративному» варианту выше за счёт более "
    "глубокой разработки.",
    pricing_intro="Два варианта под разный бюджет. Оба закрывают главное — "
    "mobile-first и онлайн-запись. Разница в технологии и запасе на рост.",
    team_intro="Небольшая команда под объём проекта. Без лишних ролей — это "
    "напрямую влияет на стоимость.",
    included=[
        "Полный цикл: аналитика, дизайн, разработка, тестирование, запуск",
        "Mobile-first вёрстка, проверка на реальных устройствах",
        "Модуль онлайн-записи к врачу с выбором филиала",
        "Дублирование заявок в мессенджеры администратора",
        "Базовая SEO-разметка и подключение аналитики",
        "Обучение администратора работе с заявками",
        "Передача исходного кода и доступов клиенту",
    ],
    not_included=[
        "Написание текстов и профессиональная фотосъёмка врачей и кабинетов",
        "Личный кабинет пациента с историей визитов",
        "Интеграция с медицинскими информационными системами (МИС)",
        "Платное продвижение и контекстная реклама",
    ],
    guarantees=[
        "1 месяц активной поддержки после запуска — приоритетное исправление ошибок",
        "12 месяцев гарантии на дефекты кода: ошибка в нашем коде — исправляем бесплатно",
        "Дальнейшее развитие — по отдельному договору сопровождения по желанию",
    ],
    portfolio=[
        PortfolioItem(
            title="Редизайн сервисного сайта с онлайн-бронированием",
            blurb="Редизайн сервисного сайта с модулем бронирования: календарь "
            "занятости и заявка без звонка. Продажа через аудит точек потери клиентов.",
        ),
        PortfolioItem(
            title="Корпоративный сайт международной школы + AI-чатбот",
            blurb="Лендинг на конструкторе с быстрым запуском и мультиязычным "
            "контентом, опциональный AI-бот квалификации обращений.",
        ),
    ],
    next_step="Выбрать вариант «Эконом» или «Корпоративный» → договор → "
    "предоплата 30%, и стартуем.",
)


def main() -> None:
    # пересчёт производных полей (итоги, work_days, duration) — как в рантайме
    state = ProposalState(spec=spec, estimation=estimation, risks=risks, proposal=proposal)
    recompute_totals(state)
    context = build_context(state.spec, state.estimation, state.risks, state.proposal)
    html = render_html(context)

    out_dir = config.OUTPUT_DIR / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "proposal.html").write_text(html, encoding="utf-8")
    pdf_path = out_dir / "proposal.pdf"
    html_to_pdf(html, pdf_path)
    print(f"HTML: {out_dir / 'proposal.html'}")
    print(f"PDF:  {pdf_path}")


if __name__ == "__main__":
    main()
