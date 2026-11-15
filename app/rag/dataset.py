"""Парсинг датасета агентства под его реальный формат.

- projects_metadata.txt — блоки [PROJECT_ID: ...], разделённые строками `---`.
- rag_cards.md — карточки, разделённые линиями из символа `═`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app import config


@dataclass
class Disclosure:
    """NDA-метки карточки: что можно показывать в Portfolio КП клиенту.

    Отделяет внутренний слой (RAG для оценки — здесь NDA не критично) от
    внешнего слоя (Portfolio в финальном КП — здесь имена клиента и
    детали услуги под NDA-фильтром).
    """

    name_disclosure: str = "none"  # full | name_only | none
    service_disclosure: str = "none"  # full | anonymized | none
    anonymized_label: str = ""
    anonymized_blurb: str = ""


def parse_disclosure(metadata_block: str) -> Disclosure:
    """Извлекает NDA-метки из текстового блока проекта."""
    fields: dict[str, str] = {}
    for line in metadata_block.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in (
            "name_disclosure",
            "service_disclosure",
            "anonymized_label",
            "anonymized_blurb",
        ):
            fields[key] = value.strip()
    return Disclosure(
        name_disclosure=fields.get("name_disclosure", "none").lower(),
        service_disclosure=fields.get("service_disclosure", "none").lower(),
        anonymized_label=fields.get("anonymized_label", ""),
        anonymized_blurb=fields.get("anonymized_blurb", ""),
    )


@dataclass
class Card:
    """Одна RAG-карточка проекта (корпус для эмбеддинга/поиска)."""

    title: str
    body: str
    project_id: str | None = None
    metadata_block: str = ""  # привязанный блок из projects_metadata.txt
    disclosure: Disclosure = field(default_factory=Disclosure)


# ─────────────────────────────────────────────────────────────────────────
# projects_metadata.txt
# ─────────────────────────────────────────────────────────────────────────
_PROJECT_ID_RE = re.compile(r"\[PROJECT_ID:\s*([^\]]+)\]")


def parse_projects_metadata(path: Path | None = None) -> dict[str, str]:
    """Возвращает {project_id: текст блока}. Порядок вставки сохраняется."""
    path = path or config.PROJECTS_METADATA
    text = path.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for chunk in text.split("\n---\n"):
        chunk = chunk.strip()
        m = _PROJECT_ID_RE.search(chunk)
        if m:
            blocks[m.group(1).strip()] = chunk
    return blocks


def get_summary_patterns(path: Path | None = None) -> str:
    """Блок 'СВОДНЫЕ ПАТТЕРНЫ' из projects_metadata.txt — общий контекст."""
    path = path or config.PROJECTS_METADATA
    text = path.read_text(encoding="utf-8")
    for chunk in text.split("\n---\n"):
        if "СВОДНЫЕ ПАТТЕРНЫ" in chunk:
            return chunk.strip()
    return ""


_NDA_HEADING_RE = re.compile(r"^## \d+\..*\(([^()—]+?)\s*—\s*обезличенно", re.MULTILINE)


def get_nda_terms() -> list[str]:
    """Имена клиентов, которые нельзя показывать клиенту.

    Берутся из самих данных, чтобы не дублировать их в коде:
    - projects_metadata.txt: client_name у проектов с name_disclosure: none
      (плюс сокращение в скобках после имени);
    - portfolio.md: заголовки вида «... (Имя — обезличенно)».
    """
    terms: set[str] = set()
    for block in parse_projects_metadata().values():
        if parse_disclosure(block).name_disclosure != "none":
            continue
        for line in block.splitlines():
            if not line.lower().startswith("client_name:"):
                continue
            value = line.split(":", 1)[1].strip()
            if value.startswith("("):
                continue  # вместо имени описание — нечего искать
            main, _, inner = value.partition("(")
            terms.add(main.strip())
            if inner:
                terms.add(inner.rstrip(")").split(".")[0].strip())
    if config.PORTFOLIO.exists():
        for m in _NDA_HEADING_RE.finditer(config.PORTFOLIO.read_text(encoding="utf-8")):
            terms.add(m.group(1).strip())
    return sorted(t for t in terms if len(t) >= 3)


def find_nda_terms(text: str) -> list[str]:
    """Какие NDA-имена встречаются в тексте (без учёта регистра)."""
    low = text.lower()
    return [t for t in get_nda_terms() if t.lower() in low]


# ─────────────────────────────────────────────────────────────────────────
# rag_cards.md
# ─────────────────────────────────────────────────────────────────────────
_SEP_RE = re.compile(r"^═+\s*$", re.MULTILINE)


def _split_cards(path: Path) -> tuple[list[tuple[str, str]], str]:
    """Возвращает (карточки [(title, body)], памятка-writer)."""
    text = path.read_text(encoding="utf-8")
    parts = [p.strip() for p in _SEP_RE.split(text)]
    # parts[0] — интро файла; дальше чередуются title, body
    pairs: list[tuple[str, str]] = []
    memo = ""
    for i in range(1, len(parts), 2):
        title = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if not title:
            continue
        if "ПАМЯТКА" in title.upper():
            memo = body
        else:
            pairs.append((title, body))
    return pairs, memo


def parse_rag_cards(
    cards_path: Path | None = None, metadata_path: Path | None = None
) -> list[Card]:
    """RAG-карточки проектов с привязкой к метаданным по порядку следования."""
    cards_path = cards_path or config.RAG_CARDS
    pairs, _ = _split_cards(cards_path)
    metadata = parse_projects_metadata(metadata_path)
    project_ids = list(metadata.keys())

    cards: list[Card] = []
    for idx, (title, body) in enumerate(pairs):
        pid = project_ids[idx] if idx < len(project_ids) else None
        meta_block = metadata.get(pid, "") if pid else ""
        cards.append(
            Card(
                title=title,
                body=body,
                project_id=pid,
                metadata_block=meta_block,
                disclosure=parse_disclosure(meta_block),
            )
        )
    return cards


def get_writer_memo(cards_path: Path | None = None) -> str:
    """Памятка по тону КП — идёт напрямую в промпт агента-Writer."""
    cards_path = cards_path or config.RAG_CARDS
    _, memo = _split_cards(cards_path)
    return memo
