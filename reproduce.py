"""Recompute four charging-screen cases from explicit prepared trip attributes.

The included fixture is fictional. No national result is inferred from it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paper1_r2.paper1_scenarios import charging_screen_scenario


def read_frame(path: Path) -> pd.DataFrame:
    """Read CSV, or Parquet when its optional engine is installed."""
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    raise ValueError('Inputs must be CSV or Parquet files')


def calculate(flows_path: Path, stations_path: Path) -> pd.DataFrame:
    """Return four results, recomputing shared-site capacity in every case."""
    flows, stations = read_frame(flows_path), read_frame(stations_path)
    if not {'station_position', 'N_points'} <= set(stations.columns):
        raise ValueError('Stations require station_position and N_points')
    if not np.array_equal(stations.station_position.to_numpy(), np.arange(len(stations))):
        raise ValueError('Station rows must be uniquely ordered at positions 0, 1, ...')
    rows = []
    for weighted in (False, True):
        for screened, band in ((False, (0.0, 1.0)), (True, (0.30, 0.80))):
            totals, _, _ = charging_screen_scenario(
                flows, stations.N_points, band=band, purpose_weighted=weighted,
            )
            row = {'loading': 'purpose_weighted' if weighted else 'raw',
                   'ap_screen': '0.30-0.80' if screened else 'none', **totals}
            for key in ('potential', 'immediate', 'conditional'):
                row[key + '_share'] = totals[key + '_volume'] / totals['baseline_volume']
            if not np.isclose(row['potential_volume'], row['immediate_volume'] + row['conditional_volume']):
                raise ValueError('Charging tiers fail the numerator identity')
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flows', type=Path, required=True)
    parser.add_argument('--stations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = calculate(args.flows, args.stations)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
        print(json.dumps({'status': 'calculated', 'cases': len(result),
                          'flows': str(args.flows), 'stations': str(args.stations),
                          'output': str(args.output),
                          'note': 'These are supplied-input results, not national validation.'}))
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f'Cannot reproduce supplied inputs: {error}\n')


if __name__ == '__main__':
    main()
