"""Vision-агент: PDF/изображение с ТЗ → распознанный текст брифа.

Логика:
  - PDF: сначала пробуем извлечь текст pypdf-ом через PyMuPDF. Если в
    документе есть нормальный текстовый слой (≥200 символов) — отдаём как
    есть. Если документ — скан (текстового слоя нет или он мусорный) —
    конвертируем первые страницы в PNG и зовём GPT-4o vision.
  - PNG/JPG: всегда vision.

Vision используется только когда без него нельзя — у текстового PDF
вытащить текст детерминированно быстрее и точнее. Это и есть «осмысленная
мультимодальность», а не vision для галочки.
"""

from __future__ import annotations

import base64
from typing import Literal

import fitz  # PyMuPDF

from app.llm import _client

MAX_PAGES_FOR_VISION = 4  # больше — слишком дорого и обычно избыточно
TEXT_LAYER_THRESHOLD = 200  # символов; меньше — считаем сканом
VISION_MODEL = "gpt-4o"  # gpt-4o-mini хуже распознаёт кириллицу на сканах

VISION_SYSTEM = """Ты парсер визуальных материалов от клиентов российских и
казахстанских веб-агентств. На вход приходит фото или скан, и материал
может быть одним из двух типов:

ТИП 1 — ТЕКСТОВАЯ СТРАНИЦА: ТЗ, бриф, договор, прайс, страница документа.
ТИП 2 — СХЕМА / ДИАГРАММА: блок-схема процесса, mindmap, BPMN, фото доски
        с маркером, скриншот Miro/Whimsical/draw.io, эскиз архитектуры
        сайта, user flow, ER-диаграмма, таблица процессов.

Сначала определи тип. Дальше работа разная.

═══ ДЛЯ ТИПА 1 (текст) ═══
Извлеки весь читаемый текст на русском/казахском/английском максимально
точно, в естественном порядке чтения.
- Не переписывай и не «улучшай» — это извлечение, не пересказ.
- Сохраняй структуру: заголовки, списки, цифры, цены, сроки.
- Таблицу представь Markdown-таблицей.
- Нечитаемое помечай как [нечитаемо].
- Без комментариев и выводов от себя.

═══ ДЛЯ ТИПА 2 (схема / диаграмма) ═══
Опиши структуру связно, чтобы по описанию можно было восстановить смысл
без оригинала. Формат:

[ТИП СХЕМЫ]: блок-схема процесса / mindmap / архитектура / user flow / …

УЗЛЫ:
- Узел 1 (что обозначает, какой текст внутри)
- Узел 2 (…)
…

СВЯЗИ И ПОТОК:
- Узел 1 → Узел 2 (стрелка, подпись если есть, "да"/"нет" если ветвление)
- …
(если есть циклы/условия — отметь словами).

ПОДПИСИ И ПОМЕТКИ НА ПОЛЯХ: всё, что написано рядом со стрелками, узлами
или отдельно (например, цифры/комментарии маркером на фото доски).

ВЫВОД (одна-две фразы): какой процесс или система здесь нарисована, своими
словами. Это единственное место, где можно «понять» картинку — нужно для
дальнейшей работы агентов оценки.

═══ ОБЩЕЕ ═══
Возвращай только результат, без префиксов вроде «Вот текст:» или «Это
схема, на которой…». Сразу с дела."""


def _extract_text_pdf(data: bytes) -> str:
    """Извлекаем текстовый слой из PDF, если он есть."""
    doc = fitz.open(stream=data, filetype="pdf")
    text_parts: list[str] = []
    for page in doc:
        text_parts.append(page.get_text("text"))
    doc.close()
    return "\n\n".join(t.strip() for t in text_parts if t.strip())


def _render_pdf_pages(data: bytes, max_pages: int = MAX_PAGES_FOR_VISION) -> list[bytes]:
    """Превращаем первые N страниц PDF в PNG-изображения."""
    doc = fitz.open(stream=data, filetype="pdf")
    images: list[bytes] = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        # 2x масштаб — иначе GPT-4o хуже видит мелкий шрифт
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        images.append(pix.tobytes("png"))
    doc.close()
    return images


def _vision_extract(images: list[bytes]) -> str:
    """GPT-4o vision: список PNG → склеенный текст."""
    user_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Извлеки текст из {len(images)} страниц(ы) ТЗ. "
                "Соблюдай порядок страниц. Между страницами вставь разделитель '— страница N —'."
            ),
        }
    ]
    for png in images:
        b64 = base64.b64encode(png).decode("ascii")
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )

    completion = _client().chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": VISION_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )
    text = completion.choices[0].message.content
    return (text or "").strip()


def extract_brief_from_file(
    filename: str, data: bytes
) -> tuple[str, Literal["pdf_text", "pdf_vision", "image_vision"]]:
    """Принимает имя файла и его байты, возвращает (текст, способ извлечения).

    Способ извлечения возвращаем для логов и debug-панели: пользователь
    видит, был ли использован vision или хватило текстового слоя.
    """
    name = filename.lower()
    if name.endswith(".pdf"):
        text = _extract_text_pdf(data)
        if len(text) >= TEXT_LAYER_THRESHOLD:
            return text, "pdf_text"
        images = _render_pdf_pages(data)
        return _vision_extract(images), "pdf_vision"

    if name.endswith((".png", ".jpg", ".jpeg", ".webp")):
        return _vision_extract([data]), "image_vision"

    raise ValueError(f"Неподдерживаемый формат файла: {filename}")
