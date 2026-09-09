"""Independent totals for the eight-flow fixture; no national-data assertion."""
from pathlib import Path
import json
import subprocess
import sys

import pandas as pd
import pytest
from paper1_r2.scenarios import charging_screen_scenario

ROOT = Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('weighted,band,expected', [
    (False, (0, 1), (84, 36, 48)),
    (False, (.3, .8), (63, 45, 18)),
    (True, (0, 1), (48.8, 37, 11.8)),
    (True, (.3, .8), (38.3, 26.5, 11.8)),
])
def test_illustrative_totals_by_hand(weighted, band, expected):
    flows = pd.read_csv(ROOT / 'data/paper1_r2/illustrative_flows.csv')
    stations = pd.read_csv(ROOT / 'data/paper1_r2/illustrative_stations.csv')
    totals, _, _ = charging_screen_scenario(flows, stations.N_points,
                                           purpose_weighted=weighted, band=band)
    assert totals['baseline_volume'] == 88
    assert tuple(totals[x + '_volume'] for x in ['potential', 'immediate', 'conditional']) == pytest.approx(expected)


def test_cli_labels_fixture_and_does_not_claim_national_reproduction(tmp_path):
    output = tmp_path / 'example.json'
    result = subprocess.run([sys.executable, '-m', 'paper1_r2',
        '--flows', str(ROOT / 'data/paper1_r2/illustrative_flows.csv'),
        '--stations', str(ROOT / 'data/paper1_r2/illustrative_stations.csv'),
        '--input-kind', 'illustrative', '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text())
    assert data['input_kind'] == 'illustrative'
    assert data['national_reproduction_claim'] is False
    assert data['flow_rows'] == 8
    assert len(data['scenarios']) == 4


def test_cli_missing_data_fails_instead_of_using_fallback(tmp_path):
    output = tmp_path / 'bad.json'
    result = subprocess.run([sys.executable, '-m', 'paper1_r2',
        '--flows', str(tmp_path / 'missing.csv'),
        '--stations', str(ROOT / 'data/paper1_r2/illustrative_stations.csv'),
        '--input-kind', 'user-supplied', '--output', str(output)],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert not output.exists()
