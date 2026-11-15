"""Проверка вариантов сметы, пересчёт итогов и формат run_id."""

from __future__ import annotations

from app import state
from app.agents.estimator import _variants_indistinguishable
from tests.conftest import make_estimation, make_variant


def test_copy_pasted_variants_are_rejected():
    est = make_estimation(
        [
            make_variant("Эконом", ["Аналитика", "Дизайн", "Разработка"], [100_000], [2, 3, 5]),
            make_variant(
                "Корпоративный", ["аналитика", "Дизайн.", "Разработка"], [300_000], [4, 6, 10]
            ),
        ]
    )
    reason = _variants_indistinguishable(est)
    assert reason is not None and "Эконом" in reason


def test_variants_with_unique_stages_pass(two_variants):
    assert _variants_indistinguishable(two_variants) is None


def test_single_variant_is_not_checked():
    est = make_estimation([make_variant("Единственный", ["Разработка"], [100_000], [5])])
    assert _variants_indistinguishable(est) is None


def test_recompute_totals(proposal_state):
    variant = proposal_state.estimation.variants[1]
    variant.total_kzt = 1  # заведомо неверные итоги
    variant.work_days = 1
    state.recompute_totals(proposal_state)
    assert variant.total_kzt == 1_500_000
    assert variant.total_our_tax_kzt == 1_650_000  # ×1.1, округление до 50 000
    assert variant.work_days == 30
    assert variant.duration  # срок словами пересчитан


def test_run_ids_are_random_and_safe():
    a, b = state.new_run_id(), state.new_run_id()
    assert a != b
    assert state.RUN_ID_RE.match(a)
    for bad in ("../etc", "a/b", "x.json", ""):
        assert not state.RUN_ID_RE.match(bad)
