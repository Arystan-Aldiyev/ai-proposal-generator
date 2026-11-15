"""FastAPI-приложение Proposal Generator.

Запуск:  uvicorn app.main:app --reload

Поток: /generate (SSE) собирает КП и сохраняет состояние → /editor
показывает редактируемое превью → правки идут через PATCH /state и
POST /revise → /pdf рендерит PDF из текущего состояния.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import subprocess
import sys
import threading
import traceback
from typing import AsyncIterator

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, ValidationError

from app import config
from app import state as state_store
from app.agents.audio import is_audio_filename, transcribe_audio
from app.agents.reviser import revise
from app.agents.vision import extract_brief_from_file
from app.pipeline import continue_after_spec, start_with_hitl
from app.render.html import context_from_state, render_editor

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 МБ — лимит на загружаемое ТЗ

_basic = HTTPBasic(auto_error=False)


def require_auth(
    request: Request, credentials: HTTPBasicCredentials | None = Depends(_basic)
) -> None:
    """HTTP Basic для всех страниц и API, если в .env задан APP_PASSWORD.

    Без пароля приложение открыто — это режим локального запуска на своём
    компьютере. /health всегда открыт для проверок живости (Docker).
    """
    if not config.APP_PASSWORD or request.url.path == "/health":
        return
    ok = (
        credentials is not None
        and secrets.compare_digest(credentials.username, config.APP_USER)
        and secrets.compare_digest(credentials.password, config.APP_PASSWORD)
    )
    if not ok:
        raise HTTPException(status_code=401, headers={"WWW-Authenticate": "Basic"})


app = FastAPI(title="CNS Proposal Generator", dependencies=[Depends(require_auth)])


def _check_run(run_id: str):
    if not state_store.RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="Некорректный run_id")
    if not (state_store.run_dir(run_id) / "state.json").exists():
        raise HTTPException(status_code=404, detail="КП не найдено")


class FieldPatch(BaseModel):
    field: str
    value: str


class ReviseBody(BaseModel):
    instruction: str


class ContinueBody(BaseModel):
    spec_patch: dict | None = None


# ─────────────────────────────────────────────────────────────────────────
# Страницы
# ─────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (config.WEB_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/editor/{run_id}", response_class=HTMLResponse)
def editor(run_id: str) -> str:
    _check_run(run_id)
    state = state_store.load_state(run_id)
    context = context_from_state(state, run_id=run_id)
    rag_path = state_store.run_dir(run_id) / "rag.json"
    context["rag"] = json.loads(rag_path.read_text(encoding="utf-8")) if rag_path.exists() else []
    return render_editor(context)


# ─────────────────────────────────────────────────────────────────────────
# Генерация (SSE-стрим этапов)
# ─────────────────────────────────────────────────────────────────────────
def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _stream_in_thread(generator_fn) -> AsyncIterator[str]:
    """Универсальная обёртка: блокирующий генератор → SSE async iterator.

    Цепочка блокирующая (OpenAI calls). Если отдавать sync-генератор прямо
    в StreamingResponse, Starlette крутит его next() по разным потокам пула,
    что ломает долгие блокирующие вызовы. Прогон — в одном потоке, события
    через asyncio.Queue.
    """

    async def _gen() -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for event in generator_fn():
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                loop.call_soon_threadsafe(queue.put_nowait, {"step": "error", "detail": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            event = await queue.get()
            if event is None:
                break
            yield _sse(event)

    return _gen()


@app.post("/generate")
async def generate(brief: str = Form(...)) -> StreamingResponse:
    """Фаза 1 HITL: запускаем граф до interrupt_after=["extract"].

    Возвращает SSE-стрим с событием прогресса extract и финальным
    spec_ready с распознанным Spec — UI показывает его для проверки.
    Граф приостанавливается в checkpointer; продолжить — через /continue.
    """
    return StreamingResponse(
        _stream_in_thread(lambda: start_with_hitl(brief)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/continue/{run_id}")
async def continue_run(run_id: str, body: ContinueBody) -> StreamingResponse:
    """Фаза 2 HITL: продолжаем граф после правки/подтверждения Spec.

    spec_patch может содержать любые поля Spec для перезаписи (project_type,
    client_industry, budget_kzt, scope, ...). Граф продолжает с этим
    Spec — retrieve → estimate → risk → triage → write.
    """
    return StreamingResponse(
        _stream_in_thread(lambda: continue_after_spec(run_id, body.spec_patch)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/upload-tz")
async def upload_tz(file: UploadFile = File(...)) -> JSONResponse:
    """Принимает PDF/изображение с ТЗ, извлекает текст (vision при скане),
    возвращает строку для подстановки в поле брифа.

    Это мультимодальная точка входа: текстовые PDF идут детерминированным
    парсером, сканы/фото — через GPT-4o vision. Внутри одна точка решения,
    наружу — единый «текст брифа».
    """
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Файл больше {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Пустой файл")
    filename = file.filename or "upload.bin"
    try:
        if is_audio_filename(filename):
            text = transcribe_audio(filename, data)
            method = "audio_whisper"
        else:
            text, method = extract_brief_from_file(filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ошибка распознавания: {exc}") from exc
    if not text.strip():
        raise HTTPException(status_code=422, detail="Из файла не удалось извлечь текст")
    return JSONResponse({"text": text, "method": method, "filename": filename})


# ─────────────────────────────────────────────────────────────────────────
# Правки состояния
# ─────────────────────────────────────────────────────────────────────────
@app.patch("/state/{run_id}")
def patch_state(run_id: str, patch: FieldPatch) -> dict:
    """Инлайн-правка одного поля из редактируемого превью."""
    _check_run(run_id)
    state = state_store.load_state(run_id)
    try:
        updated = state_store.apply_field_patch(state, patch.field, patch.value)
    except (KeyError, IndexError, ValueError, ValidationError) as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось применить правку: {exc}") from exc
    state_store.snapshot(run_id)  # снимок только при успехе
    state_store.save_state(run_id, updated)
    return {"ok": True}


@app.post("/revise/{run_id}")
def revise_state(run_id: str, body: ReviseBody) -> dict:
    """Правка КП текстовой инструкцией — один проход агента-ревизора."""
    _check_run(run_id)
    instruction = body.instruction.strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="Пустая инструкция правки")

    state = state_store.load_state(run_id)
    try:
        updated = revise(state, instruction)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail="Ревизор вернул некорректную структуру. Переформулируйте правку.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    state_store.snapshot(run_id)
    state_store.save_state(run_id, updated)
    return {"ok": True}


@app.post("/revert/{run_id}")
def revert_state(run_id: str) -> dict:
    """Откат на одну правку назад."""
    _check_run(run_id)
    if not state_store.revert(run_id):
        raise HTTPException(status_code=400, detail="Откатывать нечего")
    return {"ok": True}


# ─────────────────────────────────────────────────────────────────────────
# PDF — рендерится из ТЕКУЩЕГО состояния отдельным процессом
# (sync Playwright не запускается из рабочего потока uvicorn на Windows)
# ─────────────────────────────────────────────────────────────────────────
@app.get("/pdf/{run_id}")
def pdf(run_id: str) -> FileResponse:
    _check_run(run_id)
    try:
        subprocess.run(
            [sys.executable, "-m", "scripts.render_pdf", run_id],
            cwd=str(config.ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "")[-600:]
        raise HTTPException(status_code=500, detail=f"Ошибка рендера PDF: {detail}") from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Рендер PDF превысил таймаут") from exc

    pdf_path = state_store.run_dir(run_id) / "proposal.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=500, detail="PDF не создан")
    return FileResponse(pdf_path, media_type="application/pdf", filename=f"KP-CNS-{run_id}.pdf")
