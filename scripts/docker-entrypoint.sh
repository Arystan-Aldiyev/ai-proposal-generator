#!/bin/sh
# Docker entrypoint: гарантируем, что RAG-индекс построен перед стартом
# uvicorn. Индекс лежит в /app/storage/rag_index.json (volume), при первом
# запуске контейнера его нет — строим. Дальше каждый старт быстрый.

set -e

INDEX_PATH="/app/storage/rag_index.json"

if [ ! -f "$INDEX_PATH" ]; then
    echo "RAG-индекс не найден — собираю (~10 секунд, 6 OpenAI embedding calls)..."
    uv run python -m app.rag.ingest
fi

exec "$@"
