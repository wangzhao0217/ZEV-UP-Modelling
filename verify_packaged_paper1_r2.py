"""Independently reconcile the compact Paper 1 R2 computational archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Check archive hashes and reconstruct the published numerical identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--package-manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.package_manifest:
        record = json.loads(args.package_manifest.read_text())
        for item in record['files']:
            path = ROOT / item['path']
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != item['sha256']:
                raise ValueError(f'Archive file changed: {path}')
    release = args.run_dir / 'final'
    metrics = json.loads((release / 'national_metrics.json').read_text())
    totals = np.zeros(8)
    oa = pd.read_parquet(release / 'adoption_attributes.parquet')
    if not oa.geo_code.is_unique or len(oa) != metrics['oa_count']:
        raise ValueError('OA count/uniqueness mismatch')
    scores = oa.set_index('geo_code').adoption_propensity
    for region in metrics['regions']:
        f = pd.read_parquet(release / 'attributes' / f'{region}.parquet')
        if not np.allclose(f.ap, f.origin_geo_code.map(scores), rtol=0, atol=1e-12):
            raise ValueError(f'{region}: origin score mismatch')
        eligible = (f.d <= 80).astype(float)
        gate = f.ap.between(.3, .8).astype(float)
        weighted = f.vol * f.w * eligible
        screened = weighted * gate
        home = f.home_score >= .5
        short = f.d <= 40
        charging = ((short & (home | f.P_i.astype(bool))) |
                    (~short & home & f.P_j.astype(bool) & f.Cap_j.astype(bool)))
        immediate = screened * charging
        if not np.allclose(f.N_ij, screened, rtol=1e-12, atol=1e-12):
            raise ValueError(f'{region}: stored numerator differs from equation')
        if not np.allclose(immediate, screened * f.cat.eq('fully_feasible'), rtol=1e-12, atol=1e-12):
            raise ValueError(f'{region}: immediate tier differs from charging rule')
        totals += [f.vol.sum(), (f.vol * eligible).sum(), weighted.sum(), screened.sum(),
                   immediate.sum(), (screened - immediate).sum(),
                   (weighted * (f.ap < .3)).sum(), (weighted * (f.ap > .8)).sum()]
    keys = ['baseline_volume', 'raw_range_volume', 'weighted_range_volume', 'screened_volume',
            'immediate_volume', 'conditional_volume', 'excluded_low_volume', 'excluded_high_volume']
    for key, value in zip(keys, totals):
        if not np.isclose(value, metrics[key], rtol=1e-12, atol=1e-6):
            raise ValueError(f'{key}: independent total differs')
    if int(scores.between(.3, .8).sum()) != metrics['oa_in_band']:
        raise ValueError('National OA market-band count differs')
    result = {'status': 'passed', 'regions': metrics['regions'],
              'method': 'direct equations reconstructed independently of the shared helper',
              'checks': ['origin AP joins', 'range/weight/screen numerator', 'charging/tier rule',
                         'national totals', 'OA uniqueness and inclusive band'],
              'reconstructed_totals': dict(zip(keys, totals.tolist()))}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
