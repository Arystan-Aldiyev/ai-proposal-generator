# Deployment

Универсальная инструкция деплоя на любой Linux-сервер (Ubuntu 22.04+).
Проверено на DigitalOcean Droplet и Hetzner Cloud.

## Серверные требования

| Ресурс | Минимум | Рекомендуется |
|---|---|---|
| RAM | 1 GB | **2 GB** (Chromium PDF-рендер ест 500 MB) |
| vCPU | 1 | 1 |
| Disk | 25 GB SSD | 25 GB SSD |
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |
| Регион | — | Frankfurt / Amsterdam (ближе всего к KZ) |

**Цены (~май 2026):**

| Провайдер | Конфиг | $/мес |
|---|---|---|
| Hetzner Cloud | CX22 (2 vCPU / 4 GB / 40 GB) | ~$5 |
| DigitalOcean | s-1vcpu-2gb (1 vCPU / 2 GB / 50 GB) | $12 |
| AWS Lightsail | 2 GB / 1 vCPU | $7 |

## Подготовка сервера

### Hetzner Cloud
1. Зарегистрируйся на hetzner.com/cloud
2. Создай проект → **Add Server**
3. Image: Ubuntu 22.04, Type: **CX22**, Location: Falkenstein/Nuremberg/Helsinki
4. **SSH key**: добавь свой публичный ключ (`~/.ssh/id_ed25519.pub`) до создания
5. Запиши IP сервера

### DigitalOcean
1. Зарегистрируйся на digitalocean.com
2. **Create → Droplet**
3. Image: Ubuntu 22.04, Plan: **Basic / Regular / s-1vcpu-2gb**
4. Datacenter: Frankfurt 1
5. Authentication: SSH key (добавь свой)
6. Запиши IP

## Установка

Подключись к серверу по SSH:
```bash
ssh root@YOUR_SERVER_IP
```

Запусти setup-скрипт:
```bash
curl -fsSL https://raw.githubusercontent.com/Arystan-Aldiyev/ai-proposal-generator/main/deploy/setup.sh | bash
```

Скрипт:
1. Обновит apt-кеш
2. Установит Docker
3. Клонирует репозиторий в `/opt/cns-proposal`
4. Создаст пустой `.env` из `.env.example`
5. Откроет UFW порты 22, 80 и 443 (порт 8000 наружу не открывается)

Отредактируй `.env`:
```bash
nano /opt/cns-proposal/.env
```

Минимально нужно:
```
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=lsv2_pt_... # опционально, для дашборда
LANGSMITH_TRACING=true        # если задали API_KEY
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com  # для EU
APP_PASSWORD=...               # обязательно: HTTP Basic на весь интерфейс
PUBLIC_BASE_URL=https://proposal.your-domain.com
TELEGRAM_BOT_TOKEN=...        # опционально, для бота
```

Запусти:
```bash
cd /opt/cns-proposal
bash deploy/start.sh
```

## Проверка

```bash
docker compose ps              # должны быть Up healthy
docker compose logs -f         # смотрим живые логи
curl http://localhost:8000/health   # {"ok": true}
```

С внешнего адреса (через reverse proxy, см. ниже):
```bash
curl -u admin:ПАРОЛЬ https://proposal.your-domain.com/
```

## Обновление

```bash
cd /opt/cns-proposal
git pull
docker compose up --build -d
```

## HTTPS через reverse proxy (обязательно для доступа снаружи)

Приложение слушает только `127.0.0.1:8000`. Наружу его отдаёт reverse proxy
с HTTPS; пароль из `APP_PASSWORD` защищает интерфейс и ключ OpenAI от
посторонних.

Добавь Caddy как reverse proxy. Создай `Caddyfile`:
```
proposal.your-domain.com {
    reverse_proxy localhost:8000
}
```

Запусти Caddy в Docker:
```bash
docker run -d --name caddy --network host \
    -v ./Caddyfile:/etc/caddy/Caddyfile \
    -v caddy_data:/data caddy:2
```

Caddy сам получит Let's Encrypt сертификат.

## Telegram-бот

Telegram-бот живёт отдельным контейнером. После настройки `TELEGRAM_BOT_TOKEN`
в `.env`:
```bash
docker compose --profile bot up -d
```

Бот подключается к API локально на `http://proposal-generator:8000`.

## Troubleshooting

**Контейнер постоянно перезапускается:**
```bash
docker compose logs --tail=50 proposal-generator
```
Чаще всего — пустой `OPENAI_API_KEY` или сбой Playwright.

**Out of memory при рендере PDF:**
- Минимум 2 GB RAM. На 1 GB Chromium падает.
- Подключи swap: `fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile`

**LangSmith не пишет трейсы:**
- Проверь регион: `LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com` для EU аккаунтов.
