"""Boundary and denominator checks for the Paper 1 vehicle component grid."""

import numpy as np
import pandas as pd
import pytest

from functions.paper1_grid4 import assign_grid4, summarize_grid4


def test_children_bonus_is_strict_and_scores_are_bounded() -> None:
    factors = pd.DataFrame({"household_size_factor": [1., 1., 0.],
        "age_factor": [1., 1., 0.], "children_factor": [.2, .200001, 1.]})
    scores = assign_grid4(factors, (.6, .3, .1), .6, .2)
    assert scores.loc[0, "four_seater_score"] == pytest.approx(.02)
    assert scores.loc[1, "four_seater_score"] == pytest.approx(.6200001)
    assert scores.loc[2, "four_seater_score"] == 1.


def test_binary_and_all_oa_denominators_are_distinct() -> None:
    result = summarize_grid4(pd.Series(["2-seater", "4-seater", "mixed", "mixed"]))
    assert result["total_areas"] == 4
    assert result["two_seater_share_all"] == .25
    assert result["binary_denominator"] == 2
    assert result["two_seater_share_binary"] == .5
    assert result["chi2_binary"] == pytest.approx(3.5555555555555554)


def test_all_mixed_has_no_binary_statistic() -> None:
    result = summarize_grid4(pd.Series(["mixed", "mixed"]))
    assert result["two_seater_share_all"] == 0
    assert np.isnan(result["chi2_binary"])
    assert np.isnan(result["two_seater_share_binary"])


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.01])
def test_invalid_factors_rejected(bad: float) -> None:
    factors = pd.DataFrame({"household_size_factor": [bad], "age_factor": [.5], "children_factor": [.2]})
    with pytest.raises(ValueError, match="component factors"):
        assign_grid4(factors, (.6, .3, .1), 0., .1)


def test_unknown_assignment_rejected() -> None:
    with pytest.raises(ValueError, match="known classes"):
        summarize_grid4(pd.Series(["not_applicable"]))
