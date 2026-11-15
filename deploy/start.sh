#!/bin/bash
# Запуск приложения через docker compose.
# Использует .env из текущей директории.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
    echo "❌ .env не найден. Скопируй .env.example в .env и заполни."
    exit 1
fi

echo "→ docker compose up --build -d"
docker compose up --build -d

echo ""
echo "→ Логи: docker compose logs -f"
echo "→ Статус: docker compose ps"
echo "→ Остановить: docker compose down"
echo ""
echo "Приложение должно быть доступно на http://$(curl -fsS ifconfig.me 2>/dev/null || echo 'YOUR_IP'):8000"
