"""Telegram-бот — третий клиент того же бэкенда.

Принимает от менеджера:
- текстовый бриф → генерирует КП
- голосовое сообщение → Whisper → бриф → КП
- документ (PDF/изображение) → vision → бриф → КП

После генерации возвращает summary + ссылки на превью и PDF.
Поддерживает «правки словами» через reply на сообщение с run_id.

Запуск (локально):  uv run python -m bot.telegram_bot
Требует:            TELEGRAM_BOT_TOKEN в .env
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message

from app import config
from app import state as state_store
from app.agents.audio import is_audio_filename, transcribe_audio
from app.agents.reviser import revise
from app.agents.vision import extract_brief_from_file
from app.pipeline import run_pipeline

log = logging.getLogger("cns.bot")
logging.basicConfig(level=logging.INFO)

router = Router()

# Память «текущий run_id для каждого пользователя» — для правок словами.
# В прод-боте использовали бы Redis; для прототипа dict хватит.
_user_last_run: dict[int, str] = {}


def _format_summary(run_id: str, state) -> str:
    """Краткое summary КП для Telegram — что распозналось и какие варианты."""
    est = state.estimation
    lines = [
        f"<b>КП готово</b> · <code>{run_id}</code>",
        f"<b>Клиент:</b> {state.spec.client_label}",
        f"<b>Тип:</b> {state.spec.project_type}",
        "",
        "<b>Варианты:</b>",
    ]
    for v in est.variants:
        rec = "★" if v.name == est.recommended_variant else " "
        lines.append(f"{rec} <b>{v.name}</b> — {v.total_kzt:,} ₸ · {v.duration}")
    lines.extend(
        [
            "",
            f"<i>★ Рекомендация: {est.recommended_variant}</i>",
            f"<i>{est.recommendation_reason}</i>",
            "",
            "<i>📎 PDF отправлю следующим сообщением.</i>",
            "<i>✏️ Чтобы внести правку — ответь на это сообщение текстом правки.</i>",
        ]
    )
    return "\n".join(lines)


async def _render_pdf_bytes(run_id: str) -> bytes:
    """Запускает scripts.render_pdf как subprocess, возвращает PDF-байты.

    Subprocess нужен потому, что sync Playwright API на Windows не
    запускается из event-loop треда (та же причина что в app/main.py).
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "scripts.render_pdf",
        run_id,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"PDF render failed (code {proc.returncode}): "
            f"{stderr.decode(errors='replace')[-400:]}"
        )
    pdf_path = Path("storage/output") / run_id / "proposal.pdf"
    return pdf_path.read_bytes()


def _pdf_filename(state) -> str:
    """Имя файла PDF для клиента: КП-{client_short}.pdf."""
    label = state.spec.client_label or "Клиент"
    first = label.strip().split()[0]
    # Telegram любит ASCII в filename — оставляем как есть, Telegram сам
    # справится с кириллицей в современных клиентах.
    return f"KP-{first}.pdf"


async def _generate_and_reply(message: Message, brief: str) -> None:
    """Запускает пайплайн, отдаёт сводку + PDF файл в чат."""
    status_msg = await message.answer("⚙️ Запускаю агентов… 30–60 сек")
    try:
        result = await asyncio.to_thread(run_pipeline, brief)
    except Exception as exc:  # noqa: BLE001
        log.exception("pipeline error")
        await status_msg.edit_text(f"❌ Ошибка: {exc}")
        return

    _user_last_run[message.from_user.id] = result.run_id
    state = result.state
    await status_msg.edit_text(_format_summary(result.run_id, state), parse_mode="HTML")

    # рендерим PDF и шлём отдельным сообщением
    await message.answer("📄 Готовлю PDF…")
    try:
        pdf_bytes = await _render_pdf_bytes(result.run_id)
        filename = _pdf_filename(state)
        await message.answer_document(
            BufferedInputFile(pdf_bytes, filename=filename),
            caption="<i>Это финальный PDF. Чтобы внести правку — ответь на это сообщение.</i>",
            parse_mode="HTML",
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("pdf render error")
        await message.answer(
            f"⚠️ PDF не отрендерился: <code>{exc}</code>\n"
            f"Превью доступно: {config.PUBLIC_BASE_URL}/editor/{result.run_id}",
            parse_mode="HTML",
        )


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        "<b>CNS Proposal Bot</b>\n\n"
        "Я делаю коммерческие предложения по брифу клиента за 30–60 секунд.\n\n"
        "Можно прислать:\n"
        "• Текст брифа\n"
        "• Голосовое сообщение\n"
        "• Документ (PDF) или фото ТЗ\n\n"
        "После генерации — отвечай на сообщение чтобы попросить правку словами.",
        parse_mode="HTML",
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await cmd_start(message)


@router.message(F.voice | F.audio)
async def handle_voice(message: Message, bot: Bot) -> None:
    """Голосовое → Whisper → бриф → пайплайн."""
    voice = message.voice or message.audio
    file = await bot.get_file(voice.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    data = buf.getvalue()

    status = await message.answer("🎤 Распознаю голосовое…")
    try:
        # Telegram присылает voice как .oga (Opus), audio.transcribe_audio
        # автоматически определит формат по расширению. Подсовываем .oga.
        text = await asyncio.to_thread(transcribe_audio, "voice.oga", data)
    except Exception as exc:  # noqa: BLE001
        await status.edit_text(f"❌ Ошибка распознавания: {exc}")
        return

    await status.edit_text(
        f"📝 Распознал: «{text[:200]}{'…' if len(text) > 200 else ''}»\nГенерирую КП…"
    )
    await _generate_and_reply(message, text)


@router.message(F.document | F.photo)
async def handle_file(message: Message, bot: Bot) -> None:
    """PDF/изображение → vision → бриф → пайплайн."""
    if message.document:
        doc = message.document
        filename = doc.file_name or "upload.bin"
    else:
        # фото без имени, считаем jpg
        doc = message.photo[-1]
        filename = "photo.jpg"

    file = await bot.get_file(doc.file_id)
    buf = io.BytesIO()
    await bot.download_file(file.file_path, buf)
    data = buf.getvalue()

    status = await message.answer(f"📎 Распознаю «{filename}»…")
    try:
        if is_audio_filename(filename):
            text = await asyncio.to_thread(transcribe_audio, filename, data)
        else:
            text, method = await asyncio.to_thread(extract_brief_from_file, filename, data)
            log.info("vision method=%s len=%d", method, len(text))
    except Exception as exc:  # noqa: BLE001
        await status.edit_text(f"❌ Ошибка распознавания: {exc}")
        return

    await status.edit_text(f"📝 Извлёк {len(text)} символов. Генерирую КП…")
    await _generate_and_reply(message, text)


@router.message(F.text & ~F.text.startswith("/"))
async def handle_text(message: Message) -> None:
    """Текст: новый бриф ИЛИ правка словами (если reply на сообщение бота)."""
    user_id = message.from_user.id
    text = message.text.strip()

    # Если reply на сообщение бота — это правка словами к последнему run_id
    if (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.is_bot
    ):
        run_id = _user_last_run.get(user_id)
        if not run_id:
            await message.answer("Нет активного КП для правки. Сначала пришли бриф.")
            return

        status = await message.answer(f"✏️ Применяю правку к {run_id}…")
        try:
            state = state_store.load_state(run_id)
            updated = await asyncio.to_thread(revise, state, text)
            state_store.snapshot(run_id)
            state_store.save_state(run_id, updated)
        except Exception as exc:  # noqa: BLE001
            await status.edit_text(f"❌ Ошибка ревизора: {exc}")
            return

        await status.edit_text(_format_summary(run_id, updated), parse_mode="HTML")
        # отдаём обновлённый PDF
        try:
            pdf_bytes = await _render_pdf_bytes(run_id)
            await message.answer_document(
                BufferedInputFile(pdf_bytes, filename=_pdf_filename(updated)),
                caption="<i>Обновлённый PDF после правки.</i>",
                parse_mode="HTML",
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("pdf render error after revise")
            await message.answer(f"⚠️ PDF не отрендерился: <code>{exc}</code>", parse_mode="HTML")
        return

    # Обычный текст = новый бриф
    if len(text) < 20:
        await message.answer(
            "Бриф слишком короткий — пришли хотя бы несколько фраз о клиенте и задаче. "
            "Можно прислать голосовое или PDF."
        )
        return
    await _generate_and_reply(message, text)


async def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_BOT_TOKEN не задан в .env")
    bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)
    log.info("Telegram-бот запущен, polling...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
