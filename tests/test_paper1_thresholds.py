"""Regression checks for the threshold study's fitting and held-out selection."""

import numpy as np
import pytest

from functions.paper1_thresholds import (
    best_knots, breakpoint_scan, hinge_design, loo_predictions,
    nested_threshold_cv, weighted_fit, local_regime_design, local_regime_scan,
)


def test_piecewise_scan_recovers_known_knots() -> None:
    x = np.linspace(.02, .98, 1000)
    y = hinge_design(x, (.3, .8)) @ np.array([.1, .2, .7, -.5])
    grid = np.array([.2, .3, .4, .6, .7, .8, .9])
    scan = breakpoint_scan(x, y, np.linspace(1, 4, len(x)), grid)
    assert best_knots(scan, 2) == (.3, .8)
    assert scan.sse.min() < 1e-9


def test_press_predictions_equal_explicit_leave_one_out() -> None:
    x = np.linspace(0, 1, 15)
    y = np.sin(x*3)
    design = hinge_design(x, (.3, .8))
    explicit = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        explicit.append(design[i] @ weighted_fit(design[keep], y[keep], np.ones(keep.sum())))
    np.testing.assert_allclose(loo_predictions(design, y), explicit, atol=1e-10)


def test_outer_observation_cannot_influence_its_threshold_selection() -> None:
    x = np.linspace(.1, .9, 12)
    predictors = np.column_stack([x, x*x, np.maximum(x-.5, 0)])
    y = np.sin(x*2)
    controls = np.ones((len(y), 1))
    _, first = nested_threshold_cv(predictors, y, controls, ["linear", "square", "hinge"], 0)
    y[3] += 100
    _, second = nested_threshold_cv(predictors, y, controls, ["linear", "square", "hinge"], 0)
    a = first[(first.omitted_index == 3) & first.model.eq("nested_selected")].iloc[0]
    b = second[(second.omitted_index == 3) & second.model.eq("nested_selected")].iloc[0]
    assert a.selected == b.selected
    assert a.prediction == pytest.approx(b.prediction)


def test_invalid_weights_and_knots_fail() -> None:
    with pytest.raises(ValueError, match="increasing"):
        hinge_design(np.array([.2, .4]), (.8, .3))
    with pytest.raises(ValueError, match="positive"):
        weighted_fit(np.ones((3, 1)), np.ones(3), np.array([1., 0., 1.]))


def test_local_regime_scan_recovers_a_composition_jump() -> None:
    x = np.linspace(.15, .45, 300)
    y = local_regime_design(x, .3) @ np.array([.2, .1, -.15, .4])
    scan = local_regime_scan(x, y, np.ones(len(y)), np.array([.25, .3, .35]))
    row = scan.loc[scan.sse.idxmin()]
    assert row.threshold == .3
    assert row.level_change == pytest.approx(-.15)
