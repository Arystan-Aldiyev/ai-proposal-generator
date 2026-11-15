"""Справочник Portfolio агентства — отдельно от RAG.

RAG-датасет (rag_cards.md + projects_metadata.txt) содержит ТОЛЬКО
проекты с настоящими полными КП и точными цифрами. Используется
Estimator-ом для оценки.

Portfolio (portfolio.md) — публичный справочник всех проектов агентства
с NDA-метками. Используется Writer-ом для секции «релевантные проекты»
в финальном КП клиенту. Цифры тут НЕ нужны — мы их клиенту не
показываем; нужны имя/обезличенный ярлык + краткое и развёрнутое
описание услуги.

Формат portfolio.md — Markdown:
  ## N. Заголовок
  - **name_disclosure:** full | name_only | none
  - **service_disclosure:** full | anonymized | none
  - **public_label:** ...
  - **blurb_short:** одна-две фразы
  - **description_long:** (опционально, многострочный, для звёздных кейсов)

      многострочный текст...

  - **metrics:** (опционально, цифры результата)
  - **stack:** (опционально, технологии)
  - **note_for_agent:** (опционально, особые правила NDA)
  - **tags:** comma, separated, list
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app import config

_HEADER_RE = re.compile(r"^##\s+\d+\.\s+(.+?)\s*$", re.MULTILINE)
_INLINE_FIELD_RE = re.compile(r"^-\s+\*\*([^:*]+?):\*\*\s*(.+)$")
_EMPTY_FIELD_RE = re.compile(r"^-\s+\*\*([^:*]+?):\*\*\s*$")


@dataclass
class PortfolioProject:
    """Проект агентства в публичном виде (с учётом NDA-меток).

    Все поля кроме disclosure/public_label могут быть пустыми. paranoid
    фильтр в Writer сравнивает по public_label — это единственный
    источник истины для «что разрешено показать клиенту».
    """

    public_label: str
    name_disclosure: str = "none"  # full | name_only | none
    service_disclosure: str = "none"  # full | anonymized | none
    blurb_short: str = ""
    description_long: str = ""
    metrics: str = ""
    stack: str = ""
    note_for_agent: str = ""
    tags: list[str] = field(default_factory=list)
    raw_title: str = ""  # заголовок карточки в portfolio.md, для логов

    @property
    def project_id(self) -> str:
        """Стабильный slug из public_label — для логов и сравнений."""
        return re.sub(r"[^\w]+", "-", self.public_label.lower()).strip("-")

    @property
    def can_show(self) -> bool:
        """Можно ли вообще включать в Portfolio КП."""
        if self.name_disclosure == "none" and self.service_disclosure == "none":
            return False
        return bool(self.public_label.strip())

    @property
    def show_title(self) -> str:
        return self.public_label

    @property
    def show_blurb(self) -> str:
        """Краткое описание для Portfolio КП (с учётом NDA)."""
        if self.service_disclosure == "none":
            return ""
        return self.blurb_short

    @property
    def show_details(self) -> str:
        """Развёрнутый текст для Writer-контекста (description_long если есть).

        Используется чтобы дать Writer-агенту богатый материал для секции
        portfolio. Сама секция в КП короткая — Writer сжимает до 1-2 фраз.
        """
        if self.service_disclosure == "none":
            return ""
        return self.description_long or self.blurb_short


def _parse_block(block: str) -> PortfolioProject | None:
    """Парсит один блок (между `---`) — заголовок + ключи."""
    title_match = _HEADER_RE.search(block)
    if not title_match:
        return None
    raw_title = title_match.group(1).strip()

    fields: dict[str, str] = {}
    current_key: str | None = None
    current_value: list[str] = []

    for line in block.splitlines():
        # на строке заголовка ничего не делаем
        if line.startswith("## "):
            continue

        # inline-поле: `- **key:** value`
        m_inline = _INLINE_FIELD_RE.match(line)
        if m_inline:
            if current_key is not None:
                fields[current_key] = "\n".join(current_value).strip()
                current_value = []
            key = _normalize_key(m_inline.group(1))
            fields[key] = m_inline.group(2).strip()
            current_key = None
            continue

        # пустое поле — multi-line value далее: `- **key:**`
        m_empty = _EMPTY_FIELD_RE.match(line)
        if m_empty:
            if current_key is not None:
                fields[current_key] = "\n".join(current_value).strip()
                current_value = []
            current_key = _normalize_key(m_empty.group(1))
            continue

        # продолжение multi-line значения
        if current_key is not None:
            current_value.append(line)

    if current_key is not None:
        fields[current_key] = "\n".join(current_value).strip()

    public_label = fields.get("public_label", "").strip()
    if not public_label:
        return None

    tags = [t.strip().lower() for t in fields.get("tags", "").split(",") if t.strip()]
    blurb = fields.get("blurb_short", "")
    # Markdown-метка «*(пусто)*» интерпретируем как пустое значение
    if "*(пусто" in blurb or "*(детали не раскрываются" in blurb:
        blurb = ""

    project = PortfolioProject(
        public_label=public_label,
        name_disclosure=fields.get("name_disclosure", "none").lower(),
        service_disclosure=fields.get("service_disclosure", "none").lower(),
        blurb_short=blurb,
        description_long=fields.get("description_long", ""),
        metrics=fields.get("metrics", ""),
        stack=fields.get("stack", ""),
        note_for_agent=fields.get("note_for_agent", ""),
        tags=tags,
        raw_title=raw_title,
    )
    return project if project.can_show else None


def _normalize_key(key: str) -> str:
    """Нормализует имя поля: убирает скобки/пробелы вокруг ключа.

    Примеры: `metrics (на момент сбора)` → `metrics`,
             `description_long` → `description_long`.
    """
    key = re.sub(r"\(.*?\)", "", key)
    return key.strip().lower()


def load_portfolio(path: Path | None = None) -> list[PortfolioProject]:
    """Парсит portfolio.md, возвращает проекты разрешённые к показу."""
    path = path or config.PORTFOLIO
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    projects: list[PortfolioProject] = []
    # блоки разделены строкой `---`; первый блок — шапка файла без `## N.`
    for chunk in text.split("\n---\n"):
        project = _parse_block(chunk)
        if project:
            projects.append(project)
    return projects


# ─────────────────────────────────────────────────────────────────────────
# Семантический поиск по портфолио — тот же RAG-подход, что для cards,
# но отдельный индекс под публичные проекты с NDA-фильтрацией.
# ─────────────────────────────────────────────────────────────────────────
def _project_search_doc(p: PortfolioProject) -> str:
    """Текст для эмбеддинга проекта. Стек тегами в конце усиливает сигнал."""
    parts = [p.public_label, p.blurb_short, p.description_long]
    if p.tags:
        parts.append("Теги: " + ", ".join(p.tags))
    return ". ".join(s.strip() for s in parts if s and s.strip())


def build_portfolio_index() -> int:
    """Считает эмбеддинги для всех show-able проектов portfolio.md и
    сохраняет вектор-индекс файлом. Тот же подход что и для rag_cards."""
    import json as _json

    # импорт здесь чтобы не было циклов между config и portfolio
    from app.llm import embed

    projects = load_portfolio()
    if not projects:
        return 0
    documents = [_project_search_doc(p) for p in projects]
    embeddings = embed(documents)
    index = [
        {
            "public_label": p.public_label,
            "project_id": p.project_id,
            "document": documents[i],
            "embedding": embeddings[i],
        }
        for i, p in enumerate(projects)
    ]
    config.PORTFOLIO_INDEX_PATH.write_text(_json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return len(index)


def search_portfolio(query: str, k: int = 3) -> list[tuple[PortfolioProject, float]]:
    """Семантический поиск релевантных портфолио-проектов под запрос.

    Возвращает [(project, score), ...] — top-k по cosine similarity.
    Если индекс не построен — строится автоматически (lazy).
    """
    import json as _json

    import numpy as np

    from app.llm import embed

    if not config.PORTFOLIO_INDEX_PATH.exists():
        build_portfolio_index()

    index = _json.loads(config.PORTFOLIO_INDEX_PATH.read_text(encoding="utf-8"))
    if not index:
        return []

    # сопоставление по public_label (project_id может быть пустым в старом
    # формате если slug сгенерировался иначе)
    by_label = {p.public_label: p for p in load_portfolio()}

    matrix = np.array([row["embedding"] for row in index], dtype=np.float32)
    q = np.array(embed([query])[0], dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    sims = (matrix @ q) / np.where(norms == 0, 1e-9, norms)
    top = np.argsort(-sims)[:k]

    results: list[tuple[PortfolioProject, float]] = []
    for i in top:
        label = index[i]["public_label"]
        project = by_label.get(label)
        if project:
            results.append((project, float(sims[i])))
    return results


# ─────────────────────────────────────────────────────────────────────────
# Правила ранжирования по типу клиента (из portfolio.md, секция «Логика
# ранжирования»). Используется select_relevant для подбора 4 кандидатов
# под бриф, из которых Writer выбирает 2-3 в финальный КП.
# ─────────────────────────────────────────────────────────────────────────
CLIENT_TYPE_PRIORITY_TAGS: dict[str, list[str]] = {
    "legal": ["b2b", "legal", "crm", "complex-integration", "multilingual", "premium"],
    "education": ["education", "platform", "b2c", "multilingual", "ai"],
    "industrial": ["industrial", "b2b", "complex-filters", "catalog", "oil-gas"],
    "fmcg_ecommerce": ["e-commerce", "b2c", "automation", "fmcg", "wellness"],
    "logistics": ["logistics", "real-time", "b2b", "complex-integration"],
    "bot": ["bot", "lead-qualification", "automation", "ai"],
    "default": ["b2b", "platform", "premium"],
}


def select_relevant(
    portfolio: list[PortfolioProject], target_tags: list[str], k: int = 4
) -> list[PortfolioProject]:
    """Подбирает top-k релевантных проектов по совпадению тегов.

    Если ни один проект не пересекается по тегам — возвращаем первые k
    (Writer сам решит, какие из них включать в финальный КП).
    """
    target = {t.lower() for t in target_tags}
    scored: list[tuple[int, PortfolioProject]] = []
    for p in portfolio:
        overlap = len(target & set(p.tags))
        scored.append((overlap, p))
    scored.sort(key=lambda x: -x[0])
    if scored and scored[0][0] > 0:
        return [p for score, p in scored[:k] if score > 0]
    return [p for _, p in scored[:k]]
