"""Regression cases that distinguish the headline from legacy range-only counts."""

import numpy as np
import pandas as pd
import pytest

from paper1_r2.paper1_feasibility import (
    classify_feasibility, origin_constrained_weights, potential_components,
    summarise_potential,
)


def test_weighted_screened_headline_and_tiers_are_distinct_from_range_only():
    c = potential_components([100] * 4, [.5, .2, .8, .7],
                             [.3, .8, .9, .5], [40, 80, 20, 81])
    classes = classify_feasibility([.3, .8, .9, .5], [40, 80, 20, 81],
                                   [1, 1, 1, 0], [0, 0, 1, 1])
    assert classes.tolist() == ["immediately_feasible", "constrained",
                                "infeasible", "infeasible"]
    s = summarise_potential(c, classes)
    assert s["screened_share"] == pytest.approx(.175)
    assert s["raw_range_share"] == pytest.approx(.75)
    assert s["weighted_range_share"] == pytest.approx(.375)
    assert s["immediate_share"] + s["conditional_share"] == pytest.approx(.175)
    assert s["excluded_high_share"] == pytest.approx(.2)


def test_medium_capacity_and_no_charging_split_potential_only():
    classes = classify_feasibility([.5] * 4, [40, 40.01, 80, 20],
                                   [1, 1, 1, 0], [0, 0, 1, 0])
    assert classes.tolist() == ["immediately_feasible", "constrained",
                                "immediately_feasible", "constrained"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1, 1.1])
def test_invalid_ap_fails(bad):
    with pytest.raises(ValueError):
        potential_components([1], [.5], [bad], [20])


def test_saved_categories_must_match_the_actual_gate():
    c = potential_components([1], [.5], [.1], [20])
    with pytest.raises(ValueError, match="disagrees"):
        summarise_potential(c, ["constrained"])


def test_zero_denominator_and_missing_capacity_fail():
    with pytest.raises(ValueError, match="positive"):
        summarise_potential(potential_components([0], [.5], [.5], [0]))
    with pytest.raises(ValueError, match="finite"):
        classify_feasibility([.5], [50], [1], [np.nan])


def test_constrained_reweighting_preserves_each_origin_purpose():
    v = np.array([10., 20., 100., 200.])
    groups = ["r:o1:p", "r:o1:p", "r:o2:p", "r:o2:p"]
    result = origin_constrained_weights(v, [4, 1, 4, 1], groups)
    assert result[:2].sum() == pytest.approx(30)
    assert result[2:].sum() == pytest.approx(300)
    assert result[0] > v[0]
    assert origin_constrained_weights(v, np.ones(4), groups) == pytest.approx(v)
