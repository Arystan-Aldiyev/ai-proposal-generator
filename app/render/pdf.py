"""HTML → PDF через Playwright (headless Chromium).

Колонтитулы (шапка/футер с номером страницы) — через header/footer-шаблоны
Playwright, так как Chromium не поддерживает CSS Paged Media margin-boxes.
"""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

from app import config

_YEAR = config.AGENCY["year"]
_LOGO = config.AGENCY["logo_text"]

# Шапка: одно лого-место + название документа (из <title>).
# Лого выводится РОВНО один раз — текст документа уже не содержит "CNS".
_HEADER = f"""
<div style="font-size:8px; width:100%; padding:2mm 16mm 0; color:#9aa3af;
            font-family:Arial,Helvetica,sans-serif; -webkit-print-color-adjust:exact;">
  <span style="color:#3b4cca; font-weight:700; letter-spacing:0.5px;">{_LOGO}</span>
  <span style="margin-left:6px; color:#c2c8d0;">&middot;</span>
  <span style="margin-left:6px; color:#aab1bd;" class="title"></span>
</div>
"""

_FOOTER = f"""
<div style="font-size:8px; width:100%; padding:3px 16mm 0; color:#9aa3af;
            font-family:Arial,Helvetica,sans-serif; border-top:1px solid #e5e7eb;
            -webkit-print-color-adjust:exact;">
  © CNS Agency, {_YEAR} &nbsp;·&nbsp; Конфиденциально &nbsp;·&nbsp;
  Стр. <span class="pageNumber"></span>
</div>
"""


def html_to_pdf(html: str, out_path: Path) -> Path:
    """Рендерит готовый HTML-документ КП в PDF формата A4."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        # channel="chromium" — полноценный Chromium (new headless), а не
        # headless-shell: у shell на Windows ломается кириллический
        # font-fallback (regular-текст уезжает в Times New Roman).
        browser = p.chromium.launch(channel="chromium")
        page = browser.new_page()
        page.set_content(html, wait_until="load")
        page.pdf(
            path=str(out_path),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template=_HEADER,
            footer_template=_FOOTER,
            margin={"top": "22mm", "bottom": "18mm", "left": "16mm", "right": "16mm"},
        )
        browser.close()
    return out_path
