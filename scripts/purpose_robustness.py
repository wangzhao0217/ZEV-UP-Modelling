"""Reproduce Appendix B.5 for the published R2 purpose-specific specification.

This small diagnostic needs only tracked scoring inputs, not OD flows. It does
not regenerate or certify the absent frozen national v4 calculation. The R1
constant-input producer is preserved unchanged for historical reproducibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'functions'))
from functions.trip_purpose import TripPurposeAnalyzer

PURPOSES = (
    'Commuting', 'Education', 'Shopping', 'Health visits', 'Escort',
    'Other personal business', 'Business', 'Eating/drinking',
    'Sport/entertainment', 'Visiting friends or relatives', 'Other journey',
    'Holiday/daytrip',
)
RANK_DECIMALS = 12

RATING_KEYS = ('regularity', 'predictability', 'dwell_flexibility', 'temporal_sensitivity')


def inputs(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read explicit values only; never substitute common charging defaults."""
    p = config['trip_purpose_parameters']
    ratings = np.array([[p['input_parameters'][name][key] for key in RATING_KEYS]
                        for name in PURPOSES], dtype=float)
    infra = np.array([p['baseline_destination_compatibility'][name] for name in PURPOSES], dtype=float)
    home = np.array([p['baseline_home_charging_feasibility'][name] for name in PURPOSES], dtype=float)
    top = np.array([p['suitability_weights'][key] for key in
                    ('regularity', 'predictability', 'charging_flexibility')], dtype=float)
    sub = np.array([p['charging_flexibility_weights'][key] for key in
                    ('dwell', 'timing', 'destination', 'home')], dtype=float)
    for value in (ratings, infra, home, top, sub):
        if not np.isfinite(value).all() or ((value < 0) | (value > 1)).any():
            raise ValueError('All purpose inputs must be finite and in [0,1]')
    for weights in (top, sub):
        if not np.isclose(weights.sum(), 1., rtol=0, atol=1e-12):
            raise ValueError('Each weight vector must sum to one')
    return ratings, infra, home, top, sub


def scores(ratings: np.ndarray, infra: np.ndarray, home: np.ndarray,
           top: np.ndarray, sub: np.ndarray) -> np.ndarray:
    """Vectorised counterpart of the actual Stage 3 scoring equation."""
    cf = np.clip(sub[0]*ratings[..., 2] + sub[1]*(1-ratings[..., 3])
                 + sub[2]*infra + sub[3]*home, 0, 1)
    return np.clip(top[0]*ratings[..., 0] + top[1]*ratings[..., 1] + top[2]*cf, 0, 1)


def weight_grid() -> np.ndarray:
    """Restricted simplex: each weight 0.10--0.80, step 0.05, sum one."""
    return np.array([(a/20, b/20, (20-a-b)/20)
                     for a in range(2, 17) for b in range(2, 17)
                     if 2 <= 20-a-b <= 16], dtype=float)


def rank_statistics(samples: np.ndarray, baseline: np.ndarray) -> pd.DataFrame:
    """Retention and tau-b with one ranking-only numerical tie policy.

    Round scores to RANK_DECIMALS only for comparisons. Equal ranking scores
    retain PURPOSES order for top-k and count as ties in tau-b. Input arrays
    and the suitability weights used in the trip model are not modified.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    baseline = np.asarray(baseline, dtype=float)
    if (baseline.ndim != 1 or samples.ndim != 2
            or samples.shape[1] != baseline.size or baseline.size < 2):
        raise ValueError('Expected one sample column per baseline purpose')
    if not np.isfinite(samples).all() or not np.isfinite(baseline).all():
        raise ValueError('Ranking scores must be finite')
    samples = np.round(samples, decimals=RANK_DECIMALS)
    baseline = np.round(baseline, decimals=RANK_DECIMALS)
    base_order = np.argsort(-baseline, kind='stable')
    order = np.argsort(-samples, axis=1, kind='stable')
    pairs = np.triu_indices(len(baseline), 1)
    bs = np.sign(baseline[pairs[0]] - baseline[pairs[1]])
    ss = np.sign(samples[:, pairs[0]] - samples[:, pairs[1]])
    den = np.sqrt(np.count_nonzero(bs)*np.count_nonzero(ss, axis=1))
    tau = np.divide((ss*bs).sum(axis=1), den,
                    out=np.full(len(samples), np.nan), where=den > 0)
    result = {'top1': order[:, 0] == base_order[0]}
    for key, selected in [('top2', slice(0, 2)), ('top3', slice(0, 3)),
                          ('bottom2', slice(-2, None))]:
        result[key] = (np.sort(order[:, selected], axis=1)
                       == np.sort(base_order[selected])).all(axis=1)
    return pd.DataFrame({**result, 'tau_b': tau})


def calculate(config: dict, draws: int = 10000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Check all twelve baseline weights against the actual v4 scoring class."""
    if draws < 1:
        raise ValueError('draws must be positive')
    ratings, infra, home, top, sub = inputs(config)
    baseline = scores(ratings, infra, home, top, sub)
    implementation = TripPurposeAnalyzer(scoring_weights=config)
    implemented = np.array([implementation.get_suitability_weight(p) for p in PURPOSES])
    np.testing.assert_allclose(baseline, implemented, rtol=0, atol=1e-14)
    grid = weight_grid()
    plausible = (abs(grid-top) <= .10 + 1e-12).all(axis=1)
    broad = rank_statistics(np.stack([scores(ratings, infra, home, w, sub) for w in grid]), baseline)
    results = [('weights_plausible', broad[plausible]), ('weights_restricted', broad)]
    draws_hashes = {}
    for eps in (.05, .10, .15):
        # Reset seed per amplitude as in the original experiment.
        rng = np.random.default_rng(seed)
        jittered = np.clip(ratings + rng.uniform(-eps, eps, (draws, *ratings.shape)), 0, 1)
        stats = rank_statistics(scores(jittered, infra, home, top, sub), baseline)
        draws_hashes[f'{eps:.2f}'] = hashlib.sha256(stats.to_csv(index=False).encode()).hexdigest()
        results.append((f'ratings_{eps:.2f}', stats))
    rows = [{'experiment': name, 'variants': len(data),
             **{key: float(data[key].mean()) for key in ('top1', 'top2', 'top3', 'bottom2')},
             'tau_b_median': float(data.tau_b.median())}
            for name, data in results]
    detail = pd.DataFrame(ratings, columns=RATING_KEYS)
    detail.insert(0, 'purpose', PURPOSES)
    detail['destination_compatibility'], detail['home_charging_feasibility'] = infra, home
    detail['equation_weight'], detail['implementation_weight'] = baseline, implemented
    detail['rank'] = pd.Series(np.round(baseline, RANK_DECIMALS)).rank(method='first', ascending=False).astype(int)
    metadata = {'schema_version': 2, 'ranking_decimals': RANK_DECIMALS, 'seed': seed, 'draws_per_amplitude': draws,
                'rating_amplitudes': [.05, .10, .15], 'grid_step': .05,
                'grid_bounds': [.10, .80], 'grid_size': len(grid),
                'plausible_grid_size': int(plausible.sum()),
                'baseline_order': [PURPOSES[i] for i in np.argsort(-np.round(baseline, RANK_DECIMALS), kind='stable')],
                'baseline_implementation_max_abs_error': float(abs(baseline-implemented).max()),
                'draw_statistics_sha256': draws_hashes,
                'scope': 'Tracked R2 scoring specification; no OD or national result regeneration',
                'tie_rule': 'Scores rounded to 12 decimal places for ranking comparisons only; equal ranking scores retain PURPOSES order for top-k and count as ties in Kendall tau-b',
                'top_weights': top.tolist(), 'charging_weights': sub.tolist()}
    return pd.DataFrame(rows), detail, metadata
