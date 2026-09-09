"""Run the four charging comparisons on explicitly supplied attribute files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .scenarios import charging_screen_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flows', type=Path, required=True)
    parser.add_argument('--stations', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-kind', choices=['illustrative', 'user-supplied'], required=True,
                        help='Explicit provenance label; never infers that inputs are national.')
    args = parser.parse_args()
    try:
        flows = pd.read_csv(args.flows)
        stations = pd.read_csv(args.stations)
        if not {'station_position', 'N_points'} <= set(stations):
            raise ValueError('Stations require station_position and N_points.')
        if not np.array_equal(stations.station_position.to_numpy(), np.arange(len(stations))):
            raise ValueError('Station positions must be unique, ordered integers 0, 1, ... .')
        results = []
        for weighted, band in [(False, (0., 1.)), (False, (.3, .8)),
                               (True, (0., 1.)), (True, (.3, .8))]:
            totals, _, _ = charging_screen_scenario(
                flows, stations.N_points.to_numpy(), band=band, purpose_weighted=weighted)
            row = {'purpose_weighted': weighted, 'ap_band': list(band), **totals}
            for key in ['potential', 'immediate', 'conditional']:
                row[key + '_share'] = totals[key + '_volume'] / totals['baseline_volume']
            results.append(row)
        record = {'input_kind': args.input_kind, 'national_reproduction_claim': False,
                  'flows_file': args.flows.name, 'station_file': args.stations.name,
                  'flow_rows': len(flows), 'scenarios': results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f'Input or calculation error: {error}\n')
    print(f'Wrote four comparisons to {args.output}. Input kind: {args.input_kind}.')


if __name__ == '__main__':
    main()
