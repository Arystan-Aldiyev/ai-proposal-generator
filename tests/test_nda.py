"""NDA-защита: фильтр портфолио, фильтр свободного текста и разбор имён из данных."""

from __future__ import annotations

from app import config
from app.agents import writer
from app.rag import dataset
from app.rag.portfolio import PortfolioProject
from app.schemas import PortfolioItem
from tests.conftest import make_proposal

PUBLIC = [
    PortfolioProject(
        public_label="Каталог для поставщика", name_disclosure="none", service_disclosure="full"
    ),
    PortfolioProject(
        public_label="Бренд одежды", name_disclosure="full", service_disclosure="none"
    ),
]


def test_portfolio_keeps_only_public_labels():
    proposal = make_proposal(
        portfolio=[
            PortfolioItem(title="Каталог для поставщика", blurb="Фильтры по параметрам"),
            PortfolioItem(title="ООО Секретный Клиент", blurb="Придумано моделью"),
        ]
    )
    out = writer._enforce_portfolio_nda(proposal, PUBLIC)
    assert [i.title for i in out.portfolio] == ["Каталог для поставщика"]


def test_portfolio_title_match_ignores_case_and_spaces():
    proposal = make_proposal(
        portfolio=[PortfolioItem(title="  каталог для ПОСТАВЩИКА ", blurb="x")]
    )
    assert len(writer._enforce_portfolio_nda(proposal, PUBLIC).portfolio) == 1


def test_portfolio_blurb_cleared_when_service_is_secret():
    proposal = make_proposal(
        portfolio=[PortfolioItem(title="Бренд одежды", blurb="Что именно делали")]
    )
    out = writer._enforce_portfolio_nda(proposal, PUBLIC)
    assert out.portfolio[0].blurb == ""


def _patch_terms(monkeypatch, terms):
    def find(text):
        return [t for t in terms if t.lower() in text.lower()]

    monkeypatch.setattr(writer, "find_nda_terms", find)


def test_text_filter_replaces_names_everywhere(monkeypatch):
    _patch_terms(monkeypatch, ["Acme", "Acme Holding"])
    proposal = make_proposal(
        intro="Похожий сайт мы делали для Acme Holding.",
        included=["Тот же объём, что у ACME"],
        next_step="Отзыв acme пришлём.",
    )
    out = writer._enforce_text_nda(proposal)
    text = out.model_dump_json().lower()
    assert "acme" not in text
    # длинное имя заменяется целиком, а не по частям
    assert "один из наших клиентов holding" not in out.intro


def test_text_filter_leaves_clean_proposal_untouched(monkeypatch):
    _patch_terms(monkeypatch, ["Acme"])
    proposal = make_proposal()
    assert writer._enforce_text_nda(proposal) is proposal


def test_nda_terms_are_read_from_data(tmp_path, monkeypatch):
    metadata = tmp_path / "projects_metadata.txt"
    metadata.write_text(
        "---\n\n[PROJECT_ID: project-a]\nname_disclosure: none\nclient_name: Acme Holding (AH)\n\n---\n\n"
        "[PROJECT_ID: project-b]\nname_disclosure: full\nclient_name: Public Corp\n\n---\n",
        encoding="utf-8",
    )
    portfolio = tmp_path / "portfolio.md"
    portfolio.write_text(
        "## 4. Каталог для поставщика (Secretco — обезличенно)\n", encoding="utf-8"
    )
    monkeypatch.setattr(config, "PROJECTS_METADATA", metadata)
    monkeypatch.setattr(config, "PORTFOLIO", portfolio)

    terms = dataset.get_nda_terms()
    assert "Acme Holding" in terms and "Secretco" in terms
    assert "Public Corp" not in terms
    assert "AH" not in terms  # короче 3 символов — слишком много ложных совпадений
    assert dataset.find_nda_terms("проект для SECRETCO") == ["Secretco"]
