"""Check that changing the screened population recalculates shared capacity."""

import numpy as np
import pandas as pd
import pytest

from paper1_r2.paper1_scenarios import charging_screen_scenario


@pytest.fixture
def competing_flows() -> pd.DataFrame:
    """Two groups compete for one charger; the second is outside the AP band."""
    return pd.DataFrame({
        "vol": [15.0, 15.0], "w": [0.5, 0.5], "ap": [0.5, 0.9],
        "d": [50.0, 50.0], "home_score": [0.8, 0.8], "P_i": [0, 0],
        "P_j": [1, 1], "occupancy": [1.0, 1.0],
        "charging_station_position": [0, 0],
    })


def test_removing_ap_can_overload_a_previously_adequate_site(
    competing_flows: pd.DataFrame,
) -> None:
    """A larger candidate population must compete for the same point-hours."""
    before, _, _ = charging_screen_scenario(competing_flows, [1], purpose_weighted=False)
    after, trips, sites = charging_screen_scenario(
        competing_flows, [1], band=(0, 1), purpose_weighted=False,
    )
    assert before["immediate_volume"] == 15
    assert after["potential_volume"] == 30
    assert after["immediate_volume"] == 0
    assert after["conditional_volume"] == 30
    assert sites.D_j.iloc[0] > sites.C_avail_hours.iloc[0]
    assert not trips.immediate.any()


def test_raw_and_weighted_loads_have_distinct_capacity_interpretations(
    competing_flows: pd.DataFrame,
) -> None:
    """Weights change actual load as well as the reported weighted volume."""
    original = competing_flows.copy(deep=True)
    weighted, _, _ = charging_screen_scenario(competing_flows, [1], band=(0, 1))
    raw, _, _ = charging_screen_scenario(
        competing_flows, [1], band=(0, 1), purpose_weighted=False,
    )
    assert weighted["immediate_volume"] == 15
    assert raw["immediate_volume"] == 0
    assert weighted["baseline_volume"] == raw["baseline_volume"] == 30
    pd.testing.assert_frame_equal(competing_flows, original)


def test_production_rescaling_recomputes_denominator_and_capacity(
    competing_flows: pd.DataFrame,
) -> None:
    """Reducing productions changes the denominator and resolves the overload."""
    result, _, _ = charging_screen_scenario(
        competing_flows, [1], band=(0, 1), purpose_weighted=False,
        production_factors=[0.25, 0.25],
    )
    assert result["baseline_volume"] == 7.5
    assert result["immediate_volume"] == 7.5


def test_distance_and_band_boundaries_do_not_require_short_trip_capacity() -> None:
    """Inclusive thresholds preserve short-trip access and medium-trip capacity rules."""
    flows = pd.DataFrame({
        "vol": [10.0] * 4, "w": [1.0] * 4, "ap": [0.3, 0.8, 0.5, 0.5],
        "d": [40.0, 80.0, 80.01, 40.01], "home_score": [0.5] * 4,
        "P_i": [0] * 4, "P_j": [0, 1, 1, 0], "occupancy": [2.0] * 4,
        "charging_station_position": [-1, 0, 0, -1],
    })
    totals, trips, _ = charging_screen_scenario(flows, [1])
    assert totals["potential_volume"] == 30
    assert trips.immediate.tolist() == [True, True, False, False]
    assert totals["immediate_volume"] == 20


@pytest.mark.parametrize("factors", [[1], [-1, 1], [np.nan, 1], [0, 0]])
def test_invalid_production_loads_fail(
    competing_flows: pd.DataFrame, factors: list[float],
) -> None:
    """Reject misaligned, non-finite, negative or empty-demand scenarios."""
    with pytest.raises(ValueError):
        charging_screen_scenario(competing_flows, [1], production_factors=factors)


def test_inconsistent_destination_assignment_fails(competing_flows: pd.DataFrame) -> None:
    """A destination-access flag must agree with its station assignment."""
    competing_flows.loc[0, "P_j"] = 0
    with pytest.raises(ValueError, match="assignments disagree"):
        charging_screen_scenario(competing_flows, [1])
