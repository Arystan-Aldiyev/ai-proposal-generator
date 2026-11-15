# syntax=docker/dockerfile:1.7
# Proposal Generator — single-stage образ под FastAPI + Playwright Chromium.
# На production стоило бы multi-stage, но для прототипа простота важнее.

FROM python:3.12-slim

# системные зависимости Chromium (headless): шрифты, certs, libnss/libasound,
# минимальный набор для playwright install --with-deps. Без них Chromium
# падает с symbol lookup error на старте.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    fonts-liberation \
    fonts-noto-color-emoji \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv — менеджер пакетов; ставится одной командой, лёгкий
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

WORKDIR /app

# зависимости отдельным слоем — лучше кэшируется при изменении только кода
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

# Chromium для Playwright — большой слой (~150 MB), отдельный для кэша
RUN uv run playwright install chromium

# остальной код
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY templates/ ./templates/
COPY static/ ./static/
COPY web/ ./web/
COPY skills/ ./skills/
COPY rag_cards.md projects_metadata.txt portfolio.md ./

# директория для генерируемых артефактов (run_id, PDF). В compose монтируется
# volume чтобы файлы переживали restart контейнера.
RUN mkdir -p /app/storage/output

EXPOSE 8000

# entrypoint строит RAG-индекс при первом запуске, дальше — обычный старт
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# uvicorn без --reload (в проде reload не нужен); один воркер хватает.
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
