"""Рендер PDF из сохранённого состояния КП.

Запускается ОТДЕЛЬНЫМ процессом (см. эндпоинт /pdf). Причина: sync API
Playwright на Windows не может запустить браузер из рабочего потока
uvicorn — SelectorEventLoop не умеет subprocess (NotImplementedError).
В отдельном процессе рендер идёт в главном потоке и работает штатно.

Запуск:  python -m scripts.render_pdf <run_id>
"""

from __future__ import annotations

import sys

from app import state as state_store
from app.render.html import context_from_state, render_html
from app.render.pdf import html_to_pdf


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m scripts.render_pdf <run_id>", file=sys.stderr)
        sys.exit(2)

    run_id = sys.argv[1]
    state = state_store.load_state(run_id)
    html = render_html(context_from_state(state, run_id=run_id))
    pdf_path = state_store.run_dir(run_id) / "proposal.pdf"
    html_to_pdf(html, pdf_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
