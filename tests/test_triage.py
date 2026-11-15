"""Правила triage: когда КП пишется осторожным тоном."""

from __future__ import annotations

from app.pipeline import BUDGET_CONFLICT_RATIO, WEAK_RAG_SCORE, node_triage, route_after_triage
from tests.conftest import make_analogue, make_spec


def _triage(spec, analogues, estimation):
    return node_triage({"spec": spec, "analogues": analogues, "estimation": estimation})


def test_normal_brief_goes_standard(two_variants):
    out = _triage(make_spec(), [make_analogue(0.6)], two_variants)
    assert out == {"needs_clarification": False, "triage_reasons": []}
    assert route_after_triage(out) == "write_standard"


def test_low_info_brief_goes_cautious(two_variants):
    out = _triage(make_spec(scope=["сайт"]), [make_analogue(0.6)], two_variants)
    assert out["triage_reasons"] == ["low_info"]
    assert route_after_triage(out) == "write_cautious"


def test_no_analogue_below_threshold(two_variants):
    out = _triage(make_spec(), [make_analogue(WEAK_RAG_SCORE - 0.01)], two_variants)
    assert out["triage_reasons"] == ["no_analogue"]


def test_score_at_threshold_is_not_weak(two_variants):
    out = _triage(make_spec(), [make_analogue(WEAK_RAG_SCORE)], two_variants)
    assert out["triage_reasons"] == []


def test_no_rag_ablation_does_not_trigger_no_analogue(two_variants):
    # в абляции без RAG аналогов нет вообще — это не сигнал «вне профиля»
    out = _triage(make_spec(), [], two_variants)
    assert "no_analogue" not in out["triage_reasons"]


def test_budget_conflict(two_variants):
    cheapest = min(v.total_kzt for v in two_variants.variants)
    budget = int(cheapest / BUDGET_CONFLICT_RATIO) - 1
    out = _triage(make_spec(budget_kzt=budget), [make_analogue(0.6)], two_variants)
    assert out["triage_reasons"] == ["budget_conflict"]


def test_budget_close_to_estimate_is_fine(two_variants):
    cheapest = min(v.total_kzt for v in two_variants.variants)
    out = _triage(make_spec(budget_kzt=cheapest), [make_analogue(0.6)], two_variants)
    assert out["triage_reasons"] == []


def test_several_reasons_are_all_reported(two_variants):
    out = _triage(make_spec(scope=["сайт"], budget_kzt=1000), [make_analogue(0.1)], two_variants)
    assert out["triage_reasons"] == ["low_info", "no_analogue", "budget_conflict"]
