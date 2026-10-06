"""Check that changing the screened population recalculates shared capacity."""

import numpy as np
import pandas as pd
import pytest

from functions.paper1_scenarios import charging_screen_scenario


@pytest.fixture
def competing_flows() -> pd.DataFrame:
    """Two groups compete for one charger; the second is outside the AP band."""
    return pd.DataFrame({
        "vol": [15.0, 15.0], "w": [0.5, 0.5], "ap": [0.5, 0.9],
        "d": [50.0, 50.0], "home_score": [0.8, 0.8], "P_i": [0, 0],
        "P_j": [1, 1], "occupancy": [1.0, 1.0],
        "charging_station_position": [0, 0],
        "origin_charging_station_position": [-1, -1],
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


def test_distance_and_band_boundaries_retain_home_access() -> None:
    """Inclusive thresholds preserve short-trip access and medium-trip capacity rules."""
    flows = pd.DataFrame({
        "vol": [10.0] * 4, "w": [1.0] * 4, "ap": [0.3, 0.8, 0.5, 0.5],
        "d": [40.0, 80.0, 80.01, 40.01], "home_score": [0.5] * 4,
        "P_i": [0] * 4, "P_j": [0, 1, 1, 0], "occupancy": [2.0] * 4,
        "charging_station_position": [-1, 0, 0, -1],
        "origin_charging_station_position": [-1] * 4,
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


def test_frw_separates_missing_access_from_capacity_and_keeps_short_trips() -> None:
    """N=fRW includes access, has no purpose gate and retains distances <=3 km."""
    flows = pd.DataFrame({
        "vol": [10., 10., 100., 10., 10.], "w": [.2, .5, .5, .5, .5],
        "ap": [.3, .8, .5, .5, .5], "d": [0., 3., 50., 50., 80.01],
        "home_score": [.5, .1, .5, .5, .5], "P_i": [0, 0, 0, 0, 0],
        "P_j": [0, 0, 1, 0, 1], "occupancy": [1.] * 5,
        "charging_station_position": [-1, -1, 0, -1, 0],
        "origin_charging_station_position": [-1] * 5,
    })
    total, trips, _ = charging_screen_scenario(
        flows, [1], require_charging_access=True,
    )
    assert total["baseline_volume"] == 140
    assert total["potential_volume"] == 52
    assert total["immediate_volume"] == 2
    assert total["conditional_volume"] == 50
    assert total["access_excluded_volume"] == 10
    np.testing.assert_allclose(trips.potential_volume, flows.vol * flows.w * trips.R)
    assert total["potential_volume"] == total["immediate_volume"] + total["conditional_volume"]
    assert total["screened_pre_access_volume"] == 62


def test_public_origin_capacity_applies_to_short_and_medium_trips() -> None:
    """Public origin access qualifies R but does not guarantee capacity."""
    flows = pd.DataFrame({
        "vol": [100., 10.], "w": [1., 1.], "ap": [.5, .5],
        "d": [20., 60.], "home_score": [.1, .1], "P_i": [1, 1],
        "P_j": [0, 1], "occupancy": [1., 1.],
        "origin_charging_station_position": [0, 0],
        "charging_station_position": [-1, 1],
    })
    total, trips, _ = charging_screen_scenario(flows, [1, 10], require_charging_access=True)
    assert trips.R.all()
    assert trips.Cap_j.iloc[1] == 1
    assert not trips.Cap_i.any()
    assert total["potential_volume"] == total["conditional_volume"] == 110
    assert total["immediate_volume"] == 0


def test_inconsistent_origin_assignment_fails(competing_flows: pd.DataFrame) -> None:
    """Origin access cannot exist without a linked physical station."""
    competing_flows.loc[0, "P_i"] = 1
    with pytest.raises(ValueError, match="Origin access"):
        charging_screen_scenario(competing_flows, [1])


def test_numeric_binary_access_flags_accept_float_storage(competing_flows: pd.DataFrame) -> None:
    """CSV-derived 0.0/1.0 flags have exactly the same meaning as integers."""
    expected, _, _ = charging_screen_scenario(competing_flows, [1])
    numeric = competing_flows.astype({"P_i": float, "P_j": float})
    actual, _, _ = charging_screen_scenario(numeric, [1])
    assert actual == expected


@pytest.mark.parametrize("occupancy", [0., .5, np.nan])
def test_invalid_occupancy_fails(competing_flows: pd.DataFrame, occupancy: float) -> None:
    """A travelling vehicle cannot carry fewer than one person."""
    competing_flows.loc[0, "occupancy"] = occupancy
    with pytest.raises(ValueError, match="Occupancy"):
        charging_screen_scenario(competing_flows, [1])
