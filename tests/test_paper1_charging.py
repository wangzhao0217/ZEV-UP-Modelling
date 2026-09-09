"""Guard charging units, shared-site capacity and geographic access."""

import numpy as np
import pytest

from paper1_r2.paper1_charging import destination_capacity, nearest_charger


def test_projected_distance_and_empty_network():
    d, position = nearest_charger(np.array([[300000, 600000]]),
                                  np.array([[300300, 600400]]))
    assert d[0] == 500
    assert position[0] == 0
    d, position = nearest_charger(np.array([[0, 0]]), np.empty((0, 2)))
    assert np.isinf(d[0]) and position[0] == -1


def test_shared_station_capacity_is_not_repeated_for_different_destinations():
    trips, sites = destination_capacity([20, 20], [1, 1], [1, 1],
        [50, 50], [1, 1], [0, 0], [1])
    assert len(sites) == 1
    assert sites.D_j.iloc[0] == pytest.approx(40 * 50 * .065 / 7)
    assert sites.C_avail_hours.iloc[0] == pytest.approx(8.8)
    assert trips.Cap_j.tolist() == [0, 0]


def test_occupancy_converts_people_to_vehicles_and_can_change_capacity_tier():
    args = ([20], [1], [2], [50], [1], [0], [1])
    trips, sites = destination_capacity(*args)
    assert trips.vehicle_equivalent_volume.iloc[0] == 10
    assert trips.Cap_j.iloc[0] == 1
    assert sites.D_j.iloc[0] == pytest.approx(10 * 50 * .065 / 7)


def test_demand_is_pre_capacity_and_only_medium_eligible_flows_contribute():
    trips, sites = destination_capacity([10] * 4, [.5] * 4, [1] * 4,
        [40, 80, 81, 60], [1, 1, 0, 0], [-1, 0, -1, 0], [1])
    assert trips.demand_contrib.to_numpy() == pytest.approx([0, 5 * 80 * .065 / 7, 0, 0])
    assert sites.D_j.sum() == pytest.approx(trips.demand_contrib.sum())


def test_no_network_does_not_create_capacity():
    trips, sites = destination_capacity([10], [.5], [1], [20], [0], [-1], [])
    assert trips.Cap_j.iloc[0] == 0 and sites.empty
    with pytest.raises(ValueError, match="lacks an accessible"):
        destination_capacity([10], [.5], [1], [50], [1], [-1], [])


@pytest.mark.parametrize("occupancy,points", [(0.8, 1), (1, 1.5), (float('nan'), 1)])
def test_invalid_units_fail(occupancy, points):
    with pytest.raises(ValueError):
        destination_capacity([10], [.5], [occupancy], [50], [1], [0], [points])
