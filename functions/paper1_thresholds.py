"""Pure regression diagnostics for the Paper 1 adoption-band study."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def hinge_design(x: np.ndarray, knots: tuple[float, ...] = ()) -> np.ndarray:
    """Return a continuous piecewise-linear design with the supplied knots."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("x must be a finite one-dimensional array")
    if tuple(sorted(set(knots))) != knots or not np.isfinite(knots).all():
        raise ValueError("Knots must be finite, distinct and increasing")
    return np.column_stack([np.ones(len(x)), x] + [np.maximum(x-k, 0) for k in knots])


def weighted_fit(design: np.ndarray, y: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Fit least squares; reject nonfinite data and nonpositive weights."""
    design, y, weight = (np.asarray(a, dtype=float) for a in (design, y, weight))
    if design.ndim != 2 or y.shape != weight.shape or y.shape != (len(design),):
        raise ValueError("Design, outcome and weights have incompatible shapes")
    if not all(np.isfinite(a).all() for a in (design, y, weight)) or (weight <= 0).any():
        raise ValueError("Regression data must be finite and weights positive")
    root = np.sqrt(weight / weight.mean())
    return np.linalg.lstsq(design*root[:, None], y*root, rcond=None)[0]


def breakpoint_scan(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray,
    grid: np.ndarray, min_segment_fraction: float = .03,
) -> pd.DataFrame:
    """Scan zero, one and two continuous knots, requiring supported segments.

    Moment matrices avoid repeatedly multiplying the full OA design. Weights
    determine the loss, while support is assessed by the number of spatial units.
    Returned fits are descriptive, not tests of a causal discontinuity.
    """
    x, y, weight, grid = (np.asarray(a, dtype=float) for a in (x, y, weight, grid))
    if not 0 < min_segment_fraction < .5:
        raise ValueError("min_segment_fraction must lie between zero and one half")
    basis = hinge_design(x, tuple(grid))
    weighted_fit(basis[:, :2], y, weight)
    weight = weight / weight.mean()
    gram = basis.T @ (basis * weight[:, None])
    rhs = basis.T @ (y*weight)
    yy = float(np.dot(y*weight, y))
    total = float(np.dot(weight, (y-np.average(y, weights=weight))**2))
    rows = []
    for size in (0, 1, 2):
        for indexes in combinations(range(len(grid)), size):
            knots = tuple(grid[list(indexes)])
            if size == 2 and knots[1]-knots[0] < .099999:
                continue
            counts = np.bincount(np.searchsorted(knots, x), minlength=size+1)
            if counts.min() < max(5, min_segment_fraction*len(x)):
                continue
            columns = [0, 1] + [i+2 for i in indexes]
            matrix = gram[np.ix_(columns, columns)]
            vector = rhs[columns]
            beta = np.linalg.lstsq(matrix, vector, rcond=None)[0]
            sse = max(0., yy-2*float(beta@vector)+float(beta@matrix@beta))
            rows.append({"n_knots": size, "lower": knots[0] if size else np.nan,
                         "upper": knots[1] if size == 2 else np.nan,
                         "sse": sse, "rmse": np.sqrt(sse/weight.sum()),
                         "r2": 1-sse/total if total > 0 else np.nan})
    return pd.DataFrame(rows)


def best_knots(scan: pd.DataFrame, n_knots: int) -> tuple[float, ...]:
    """Extract the minimum-loss knot set of a specified model size."""
    candidates = scan[scan.n_knots.eq(n_knots)]
    if candidates.empty:
        raise ValueError("No supported breakpoint model of the requested size")
    row = candidates.loc[candidates.sse.idxmin()]
    return () if not n_knots else (float(row.lower),) if n_knots == 1 else (
        float(row.lower), float(row.upper))


def regional_piecewise_cv(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray,
    groups: np.ndarray, grid: np.ndarray,
) -> pd.DataFrame:
    """Select knots within training regions and predict each omitted region."""
    x, y, weight, groups = (np.asarray(a) for a in (x, y, weight, groups))
    rows = []
    for group in np.unique(groups):
        test = groups == group
        if test.all():
            raise ValueError("At least two groups are required")
        scan = breakpoint_scan(x[~test], y[~test], weight[~test], grid)
        choices = {"linear": (), "quadratic": (), "fixed_030_080": (.3, .8),
                   "estimated_one": best_knots(scan, 1),
                   "estimated_two": best_knots(scan, 2)}
        for name, knots in choices.items():
            design = np.column_stack([np.ones(len(x)), x, x*x]) if name == "quadratic" else hinge_design(x, knots)
            beta = weighted_fit(design[~test], y[~test], weight[~test])
            pred = design[test] @ beta
            rows.append({"group": str(group), "model": name,
                         "lower": knots[0] if knots else np.nan,
                         "upper": knots[1] if len(knots) == 2 else np.nan,
                         "n_test": int(test.sum()), "weight": float(weight[test].sum()),
                         "weighted_sse": float(np.sum(weight[test]*(y[test]-pred)**2))})
    return pd.DataFrame(rows)


def loo_predictions(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return exact OLS leave-one-out predictions using the PRESS identity."""
    design, y = np.asarray(design, float), np.asarray(y, float)
    beta = weighted_fit(design, y, np.ones(len(y)))
    pinv = np.linalg.pinv(design)
    leverage = np.sum(design * pinv.T, axis=1)
    if (1-leverage < 1e-8).any():
        raise ValueError("Leave-one-out fit has unit leverage")
    return y-(y-design@beta)/(1-leverage)


def local_regime_design(x: np.ndarray, threshold: float) -> np.ndarray:
    """Allow a level and slope change at a descriptive composition boundary."""
    x = np.asarray(x, dtype=float)
    centered = x-threshold
    return np.column_stack([np.ones(len(x)), centered, x >= threshold, np.maximum(centered, 0)])


def local_regime_scan(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray, grid: np.ndarray,
) -> pd.DataFrame:
    """Scan a fixed local sample for level/slope changes, without causal claims."""
    rows = []
    for threshold in grid:
        if min(np.sum(x < threshold), np.sum(x >= threshold)) < 5:
            continue
        design = local_regime_design(x, float(threshold))
        beta = weighted_fit(design, y, weight)
        rows.append({"threshold": float(threshold), "level_change": beta[2],
                     "left_slope": beta[1], "right_slope": beta[1]+beta[3],
                     "sse": float(np.sum(weight*(y-design@beta)**2))})
    return pd.DataFrame(rows)


def nested_threshold_cv(
    predictors: np.ndarray, y: np.ndarray, controls: np.ndarray,
    labels: list[str], fixed_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fixed and internally selected thresholds on omitted areas.

    Threshold selection uses inner leave-one-area-out error inside each outer
    training set. Controls must already include the intercept. This evaluates
    threshold selection conditional on the supplied, previously built AP score.
    """
    predictors, y, controls = (np.asarray(a, float) for a in (predictors, y, controls))
    if predictors.ndim != 2 or len(labels) != predictors.shape[1]:
        raise ValueError("One label is required for each candidate predictor")
    if len(y) < 6 or not 0 <= fixed_index < len(labels):
        raise ValueError("Insufficient areas or invalid fixed candidate")
    rows = []
    for i in range(len(y)):
        train = np.arange(len(y)) != i
        errors = []
        for j in range(len(labels)):
            design = np.column_stack([controls[train], predictors[train, j]])
            prediction = loo_predictions(design, y[train])
            errors.append(float(np.mean((y[train]-prediction)**2)))
        selected = int(np.argmin(errors))
        for model, choice in (("fixed", fixed_index), ("nested_selected", selected), ("controls_only", None)):
            design = controls if choice is None else np.column_stack([controls, predictors[:, choice]])
            beta = weighted_fit(design[train], y[train], np.ones(train.sum()))
            rows.append({"omitted_index": i, "model": model,
                         "selected": "none" if choice is None else labels[choice],
                         "observed": float(y[i]), "prediction": float(design[i]@beta),
                         "inner_mse": np.nan if choice is None else errors[choice]})
    predictions = pd.DataFrame(rows)
    summary = predictions.groupby("model").apply(
        lambda frame: pd.Series({"n": len(frame),
            "rmse": np.sqrt(np.mean((frame.observed-frame.prediction)**2)),
            "mae": np.mean(np.abs(frame.observed-frame.prediction)),
            "r2": 1-np.sum((frame.observed-frame.prediction)**2)/np.sum((y-y.mean())**2)}),
        include_groups=False).reset_index()
    return summary, predictions
