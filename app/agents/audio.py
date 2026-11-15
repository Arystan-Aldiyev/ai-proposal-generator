"""Audio-агент: голосовое сообщение → текст брифа (OpenAI Whisper).

Реальный кейс CNS: клиенты часто шлют WhatsApp-голосовые вместо
текстового брифа. Раньше менеджеру приходилось переслушивать и
расшифровывать вручную. Теперь — загружаем файл (или записываем прямо
в браузере) и получаем текст, который идёт в обычный пайплайн.
"""

from __future__ import annotations

import io

from app.llm import _client

WHISPER_MODEL = "whisper-1"

# поддерживаемые форматы согласно документации Whisper API
SUPPORTED_AUDIO_EXT = (".mp3", ".mp4", ".m4a", ".wav", ".webm", ".ogg", ".oga", ".flac")


def is_audio_filename(filename: str) -> bool:
    return filename.lower().endswith(SUPPORTED_AUDIO_EXT)


def transcribe_audio(filename: str, data: bytes) -> str:
    """Расшифровывает голосовое сообщение в текст. Подсказка языка — ru."""
    buf = io.BytesIO(data)
    buf.name = filename  # Whisper API определяет формат по имени файла
    response = _client().audio.transcriptions.create(
        model=WHISPER_MODEL,
        file=buf,
        language="ru",
        # response_format по умолчанию "json" → берём .text напрямую
    )
    return (response.text or "").strip()
