"""Check the population and threshold assumptions behind chaining diagnostics."""

import pytest

from paper1_r2.paper1_chaining import chaining_loss_budget, two_stop_geometry


def test_chaining_loss_affects_only_the_weighted_home_subset():
    result = chaining_loss_budget([100] * 4, [.5] * 4, [.5, .5, .5, .9],
                                  [40, 60, 20, 20], [.5, .8, .2, .8], 1, 1)
    assert result["screened_share"] == pytest.approx(.375)
    assert result["home_subset_share"] == pytest.approx(.125)
    assert result["adjusted_share"] == pytest.approx(.25)
    assert result["conditional_subset_floor"] == pytest.approx(.25)


def test_zero_exposure_and_zero_failure_reproduce_the_headline():
    for exposure, failure in [(0, 1), (1, 0)]:
        result = chaining_loss_budget([100], [.6], [.5], [40], [.5], exposure, failure)
        assert result["adjusted_share"] == pytest.approx(.6)


def test_geometric_extremes_include_exact_range_and_never_fail_collinear():
    assert two_stop_geometry([20], [1]) == {"collinear_failure": 0., "opposed_failure": 0.}
    assert two_stop_geometry([40], [1]) == {"collinear_failure": 0., "opposed_failure": 1.}
    assert two_stop_geometry([10, 30], [1, 1])["opposed_failure"] == pytest.approx(.25)


def test_geometry_rejects_distances_outside_the_claimed_subset():
    with pytest.raises(ValueError, match="home-feasible"):
        two_stop_geometry([40.01], [1])
    with pytest.raises(ValueError, match="lie in"):
        chaining_loss_budget([1], [.5], [.5], [20], [.5], 1.01, .5)
