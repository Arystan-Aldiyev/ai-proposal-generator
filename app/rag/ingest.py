"""Индексация датасета агентства: эмбеддинги карточек → JSON-индекс.

Запуск:  python -m app.rag.ingest
"""

from __future__ import annotations

import json

from app import config
from app.llm import embed
from app.rag.dataset import parse_rag_cards


def ingest() -> int:
    """Считает эмбеддинги rag-карточек и сохраняет вектор-индекс файлом."""
    cards = parse_rag_cards()
    if not cards:
        raise RuntimeError("rag_cards.md не дал ни одной карточки — проверь формат.")

    documents = [f"{c.title}\n\n{c.body}" for c in cards]
    embeddings = embed(documents)

    index = [
        {
            "project_id": c.project_id or f"card-{i}",
            "title": c.title,
            "document": documents[i],
            "metadata_block": c.metadata_block,
            "embedding": embeddings[i],
        }
        for i, c in enumerate(cards)
    ]
    config.INDEX_PATH.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return len(index)


if __name__ == "__main__":
    n = ingest()
    print(f"Проиндексировано карточек: {n}")
    print(f"Индекс: {config.INDEX_PATH}")
