"""Семантический поиск аналогов: brute-force косинус по вектор-индексу.

Архив = 6 проектов, поэтому полный перебор по эмбеддингам — это и есть
семантический векторный поиск, без накладных расходов на векторную БД.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from app import config
from app.llm import embed


@dataclass
class Analogue:
    """Найденный проект-аналог: карточка для поиска + жёсткие метаданные.

    Содержит NDA-метки disclosure, по которым Writer формирует Portfolio
    в КП клиенту. Estimator может опираться на полные данные (внутренний
    слой), Writer обязан их фильтровать (внешний слой).
    """

    project_id: str
    title: str
    card_text: str
    metadata_block: str
    score: float  # косинусная близость, 0..1
    name_disclosure: str = "none"
    service_disclosure: str = "none"
    anonymized_label: str = ""
    anonymized_blurb: str = ""

    def as_context(self) -> str:
        """Текстовый блок для передачи в промпт Estimator/Writer."""
        return (
            f"### Аналог: {self.title} [{self.project_id}]\n"
            f"{self.card_text}\n\n"
            f"--- Метаданные проекта ---\n{self.metadata_block}"
        )

    def portfolio_label(self) -> str | None:
        """Имя для Portfolio КП с учётом NDA. None — не показывать в portfolio."""
        if self.name_disclosure == "full":
            # имя клиента можно — берём из метаданных (client_name)
            return self._client_name_from_metadata() or self.title
        if self.name_disclosure == "name_only":
            return self._client_name_from_metadata() or self.title
        if self.anonymized_label:
            # имя нельзя, но есть ярлык типа «Частная школа в Астане»
            return self.anonymized_label
        return None  # ни имени, ни ярлыка — в portfolio не идёт

    def portfolio_blurb(self) -> str:
        """Описание услуги для Portfolio КП с учётом NDA."""
        if self.service_disclosure == "full":
            # детальное описание услуги можно — берём короткий резюме карточки
            return self._service_summary_from_metadata() or ""
        if self.service_disclosure == "anonymized" and self.anonymized_blurb:
            return self.anonymized_blurb
        return ""

    def can_show_in_portfolio(self) -> bool:
        """True если этот проект разрешён к показу в Portfolio КП."""
        if self.name_disclosure == "none" and self.service_disclosure == "none":
            return False
        # либо имя, либо услуга (или обе) — должны быть разрешены/анонимизированы
        return bool(self.portfolio_label())

    def _client_name_from_metadata(self) -> str:
        for line in self.metadata_block.splitlines():
            if line.strip().lower().startswith("client_name:"):
                return line.split(":", 1)[1].strip()
        return ""

    def _service_summary_from_metadata(self) -> str:
        """Берём scope_summary или project_type как короткое описание услуги."""
        for line in self.metadata_block.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith("scope_summary:"):
                return line.split(":", 1)[1].strip()
        for line in self.metadata_block.splitlines():
            if line.strip().lower().startswith("project_type:"):
                return line.split(":", 1)[1].strip()
        return ""


def _load_index() -> list[dict]:
    if not config.INDEX_PATH.exists():
        raise RuntimeError("RAG-индекс не найден. Сначала выполни: python -m app.rag.ingest")
    return json.loads(config.INDEX_PATH.read_text(encoding="utf-8"))


def search(query: str, k: int = 3) -> list[Analogue]:
    """top-k проектов-аналогов под текстовый запрос (косинусная близость)."""
    index = _load_index()

    matrix = np.array([row["embedding"] for row in index], dtype=np.float32)
    q = np.array(embed([query])[0], dtype=np.float32)

    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    sims = (matrix @ q) / np.where(norms == 0, 1e-9, norms)

    top = np.argsort(-sims)[:k]
    return [
        Analogue(
            project_id=index[i]["project_id"],
            title=index[i]["title"],
            card_text=index[i]["document"],
            metadata_block=index[i]["metadata_block"],
            score=float(sims[i]),
        )
        for i in top
    ]
