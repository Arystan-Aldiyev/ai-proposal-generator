#!/bin/bash
# Setup-скрипт для свежей Ubuntu 22.04+ виртуалки.
# Ставит Docker + Compose, клонирует репозиторий, готовит окружение.
#
# Использование (на сервере):
#   curl -fsSL https://raw.githubusercontent.com/Arystan-Aldiyev/ai-proposal-generator/main/deploy/setup.sh | bash
#   ИЛИ
#   git clone https://github.com/Arystan-Aldiyev/ai-proposal-generator.git
#   cd ai-proposal-generator
#   bash deploy/setup.sh

set -euo pipefail

REPO_URL="https://github.com/Arystan-Aldiyev/ai-proposal-generator.git"
APP_DIR="${APP_DIR:-/opt/cns-proposal}"

echo "═══ CNS Proposal Generator — server setup ═══"

# 1. Обновление системы
echo "→ apt update + базовые пакеты"
apt-get update -qq
apt-get install -y -qq curl ca-certificates git ufw

# 2. Docker
if ! command -v docker &> /dev/null; then
    echo "→ Устанавливаю Docker"
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
else
    echo "→ Docker уже установлен: $(docker --version)"
fi

# 3. Клонирование репозитория
if [ ! -d "$APP_DIR" ]; then
    echo "→ Клонирую репозиторий в $APP_DIR"
    git clone "$REPO_URL" "$APP_DIR"
else
    echo "→ Репозиторий уже есть, делаю git pull"
    cd "$APP_DIR" && git pull
fi

cd "$APP_DIR"

# 4. .env — пользователь должен заполнить вручную
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  ⚠️  ОТРЕДАКТИРУЙ .env:"
    echo "       nano $APP_DIR/.env"
    echo ""
    echo "  Минимально нужно:"
    echo "  - OPENAI_API_KEY=sk-..."
    echo "  - LANGSMITH_API_KEY=lsv2_pt_... (опционально)"
    echo "  - TELEGRAM_BOT_TOKEN=... (опционально)"
    echo "  - APP_PASSWORD=... (обязательно: без него интерфейс открыт всем)"
    echo "  - PUBLIC_BASE_URL=https://your-domain.com"
    echo ""
    echo "  После редактирования запусти:"
    echo "       bash $APP_DIR/deploy/start.sh"
    echo "═══════════════════════════════════════════════════════════"
else
    echo "→ .env уже существует"
fi

# 5. Firewall. Порт 8000 наружу не открываем: приложение слушает только
# localhost, снаружи доступ через reverse proxy с HTTPS (deploy/README.md).
echo "→ Настраиваю UFW: разрешаю SSH (22), HTTP (80) и HTTPS (443)"
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw --force enable

echo ""
echo "Setup завершён. Дальше: отредактируй .env и запусти deploy/start.sh"
