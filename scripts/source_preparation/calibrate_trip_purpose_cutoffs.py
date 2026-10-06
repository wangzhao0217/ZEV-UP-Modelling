#!/usr/bin/env python3
"""
Calibrate Stage 3 trip-purpose classification cutoffs using data-driven procedures.

This script estimates low/high cutoffs for purpose suitability score W_p
(moderate and high thresholds) using three methods:

1) Method A: Supervised weighted grid-search maximizing weighted macro-F1
2) Method B: Weighted piecewise-constant segmentation (2 breakpoints)
3) Method C: Weighted quantile mapping on W_p

It supports leave-one-region-out CV, bootstrap uncertainty intervals, and
optional config writing to scoring_weights.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import glob

import numpy as np
import pandas as pd


CLASSES = np.array(["low", "medium", "high"])


@dataclass
class Thresholds:
    low: float
    high: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate trip-purpose classification cutoffs."
    )
    parser.add_argument(
        "--input-glob",
        default="output/*/*_purpose_replacement_summary.csv",
        help="Glob for regional purpose replacement summary CSV files.",
    )
    parser.add_argument(
        "--weights-file",
        default="scoring_weights.json",
        help="Path to scoring_weights.json.",
    )
    parser.add_argument(
        "--outdir",
        default="output/calibration/trip_purpose_cutoffs",
        help="Output directory for calibration artifacts.",
    )
    parser.add_argument(
        "--cv",
        choices=["logo", "none"],
        default="logo",
        help="Cross-validation strategy.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for CI estimation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.005,
        help="Grid step for method A threshold search.",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.05,
        help="Minimum gap between low/high thresholds.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["tercile", "quantile", "fixed"],
        default="tercile",
        help="Pseudo-labeling mode based on observed replacement rate.",
    )
    parser.add_argument(
        "--q-low",
        type=float,
        default=1.0 / 3.0,
        help="Low quantile for quantile/fixed label mode.",
    )
    parser.add_argument(
        "--q-high",
        type=float,
        default=2.0 / 3.0,
        help="High quantile for quantile/fixed label mode.",
    )
    parser.add_argument(
        "--fixed-low",
        type=float,
        default=0.33,
        help="Fixed observed-rate low boundary for label-mode=fixed.",
    )
    parser.add_argument(
        "--fixed-high",
        type=float,
        default=0.66,
        help="Fixed observed-rate high boundary for label-mode=fixed.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="If set, write recommended thresholds back into scoring_weights.json.",
    )
    return parser.parse_args()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(values) == 0:
        raise ValueError("Cannot compute weighted quantile on empty array.")
    q = float(np.clip(q, 0.0, 1.0))
    sorter = np.argsort(values)
    v = values[sorter]
    w = weights[sorter]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.quantile(v, q))
    cutoff = q * cw[-1]
    idx = np.searchsorted(cw, cutoff, side="left")
    idx = int(np.clip(idx, 0, len(v) - 1))
    return float(v[idx])


def classify_by_thresholds(w: np.ndarray, thr: Thresholds) -> np.ndarray:
    labels = np.full(len(w), "medium", dtype=object)
    labels[w < thr.low] = "low"
    labels[w >= thr.high] = "high"
    return labels.astype(str)


def build_target_labels(
    observed_rate: np.ndarray,
    weights: np.ndarray,
    mode: str,
    q_low: float,
    q_high: float,
    fixed_low: float,
    fixed_high: float,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    if mode in {"tercile", "quantile"}:
        low_b = weighted_quantile(observed_rate, weights, q_low)
        high_b = weighted_quantile(observed_rate, weights, q_high)
    elif mode == "fixed":
        low_b, high_b = fixed_low, fixed_high
    else:
        raise ValueError(f"Unknown label mode: {mode}")

    labels = np.full(len(observed_rate), "medium", dtype=object)
    labels[observed_rate < low_b] = "low"
    labels[observed_rate >= high_b] = "high"
    return labels.astype(str), (float(low_b), float(high_b))


def macro_f1_weighted(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    f1s = []
    for cls in CLASSES:
        tp = np.sum(weights[(y_true == cls) & (y_pred == cls)])
        fp = np.sum(weights[(y_true != cls) & (y_pred == cls)])
        fn = np.sum(weights[(y_true == cls) & (y_pred != cls)])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1s.append(f1)
    return float(np.mean(f1s))


def balanced_accuracy_weighted(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> float:
    recalls = []
    for cls in CLASSES:
        tp = np.sum(weights[(y_true == cls) & (y_pred == cls)])
        fn = np.sum(weights[(y_true == cls) & (y_pred != cls)])
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    return float(np.mean(recalls))


def monotonicity_ok(
    observed_rate: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> bool:
    means: Dict[str, float] = {}
    for cls in CLASSES:
        mask = y_pred == cls
        if not np.any(mask):
            return False
        means[cls] = float(np.average(observed_rate[mask], weights=weights[mask]))
    return means["high"] > means["medium"] > means["low"]


def method_a_grid_search(
    w: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    observed_rate: np.ndarray,
    grid_step: float = 0.005,
    min_gap: float = 0.05,
) -> Thresholds:
    candidates = np.arange(0.2, 0.951, grid_step)
    best_thr = Thresholds(0.6, 0.8)
    best_score = -np.inf

    for low in candidates:
        high_start = low + min_gap
        highs = candidates[candidates >= high_start]
        for high in highs:
            thr = Thresholds(float(low), float(high))
            pred = classify_by_thresholds(w, thr)
            f1 = macro_f1_weighted(y_true, pred, weights)
            bal = balanced_accuracy_weighted(y_true, pred, weights)
            mono = monotonicity_ok(observed_rate=observed_rate, y_pred=pred, weights=weights)
            # Use class agreement as primary objective; lightly penalize non-monotonic splits.
            score = f1 + 0.2 * bal - (0.05 if not mono else 0.0)
            if score > best_score:
                best_score = score
                best_thr = thr
    return best_thr


def method_b_segmented(
    w: np.ndarray,
    observed_rate: np.ndarray,
    weights: np.ndarray,
    min_gap: float = 0.05,
) -> Thresholds:
    """
    Weighted piecewise-constant segmentation with two breakpoints minimizing SSE.
    """
    grid = np.unique(np.round(w, 3))
    grid = grid[(grid > 0.15) & (grid < 0.95)]
    if len(grid) < 3:
        return Thresholds(0.6, 0.8)

    best_thr = Thresholds(0.6, 0.8)
    best_sse = np.inf
    total_w = float(np.sum(weights))

    for low in grid:
        for high in grid:
            if high <= low + min_gap:
                continue
            seg1 = w < low
            seg2 = (w >= low) & (w < high)
            seg3 = w >= high

            if not (np.any(seg1) and np.any(seg2) and np.any(seg3)):
                continue

            w1, w2, w3 = np.sum(weights[seg1]), np.sum(weights[seg2]), np.sum(weights[seg3])
            # Avoid degenerate segments
            if min(w1, w2, w3) < 0.05 * total_w:
                continue

            m1 = np.average(observed_rate[seg1], weights=weights[seg1])
            m2 = np.average(observed_rate[seg2], weights=weights[seg2])
            m3 = np.average(observed_rate[seg3], weights=weights[seg3])

            sse = (
                np.sum(weights[seg1] * (observed_rate[seg1] - m1) ** 2)
                + np.sum(weights[seg2] * (observed_rate[seg2] - m2) ** 2)
                + np.sum(weights[seg3] * (observed_rate[seg3] - m3) ** 2)
            )
            if sse < best_sse:
                best_sse = sse
                best_thr = Thresholds(float(low), float(high))
    return best_thr


def method_c_quantile(w: np.ndarray, weights: np.ndarray) -> Thresholds:
    low = weighted_quantile(w, weights, 1.0 / 3.0)
    high = weighted_quantile(w, weights, 2.0 / 3.0)
    if high <= low:
        high = min(0.95, low + 0.1)
    return Thresholds(float(low), float(high))


def evaluate_thresholds(
    w: np.ndarray,
    observed_rate: np.ndarray,
    y_true: np.ndarray,
    weights: np.ndarray,
    thr: Thresholds,
) -> Dict[str, float]:
    pred = classify_by_thresholds(w, thr)
    return {
        "macro_f1": macro_f1_weighted(y_true, pred, weights),
        "balanced_accuracy": balanced_accuracy_weighted(y_true, pred, weights),
        "monotonic_ok": float(monotonicity_ok(observed_rate, pred, weights)),
    }


def canonicalize_region_files(paths: Iterable[Path]) -> Tuple[List[Path], List[Tuple[str, Path]]]:
    grouped: Dict[str, List[Path]] = {}
    for p in paths:
        stem = p.name.replace("_purpose_replacement_summary.csv", "")
        grouped.setdefault(stem, []).append(p)

    selected: List[Path] = []
    dropped: List[Tuple[str, Path]] = []
    for region, plist in grouped.items():
        plist_sorted = sorted(
            plist,
            key=lambda x: (
                0 if x.parent.name == region else 1,
                len(str(x)),
                str(x),
            ),
        )
        pick = plist_sorted[0]
        selected.append(pick)
        for d in plist_sorted[1:]:
            dropped.append((region, d))
    return sorted(selected), dropped


def load_dataset(input_glob: str) -> Tuple[pd.DataFrame, List[Tuple[str, Path]]]:
    paths = [Path(p) for p in glob.glob(input_glob)]
    if not paths:
        raise FileNotFoundError(f"No files found for input glob: {input_glob}")

    selected, dropped = canonicalize_region_files(paths)
    frames = []
    for p in selected:
        region = p.name.replace("_purpose_replacement_summary.csv", "")
        df = pd.read_csv(p)
        df["region"] = region
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    required_any = ["avg_purpose_weight", "replacement_rate_pct", "replaceable_trips"]
    for col in required_any:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    data["W_p"] = pd.to_numeric(data["avg_purpose_weight"], errors="coerce")

    if "replacement_rate_pct" in data.columns:
        data["observed_rate"] = pd.to_numeric(data["replacement_rate_pct"], errors="coerce") / 100.0
    else:
        denom = pd.to_numeric(data.get("car_trips", data.get("total_trips")), errors="coerce")
        num = pd.to_numeric(data["replaceable_trips"], errors="coerce")
        data["observed_rate"] = np.where(denom > 0, num / denom, np.nan)

    if "car_trips" in data.columns:
        data["sample_weight"] = pd.to_numeric(data["car_trips"], errors="coerce")
    elif "total_trips" in data.columns:
        data["sample_weight"] = pd.to_numeric(data["total_trips"], errors="coerce")
    else:
        data["sample_weight"] = 1.0

    data = data.dropna(subset=["W_p", "observed_rate", "sample_weight", "region", "purpose"]).copy()
    data = data[(data["sample_weight"] > 0) & (data["W_p"] >= 0) & (data["W_p"] <= 1)]

    return data, dropped


def calibrate_method(
    method: str,
    train: pd.DataFrame,
    y_train_labels: np.ndarray,
    grid_step: float,
    min_gap: float,
) -> Thresholds:
    w = train["W_p"].to_numpy()
    obs = train["observed_rate"].to_numpy()
    wt = train["sample_weight"].to_numpy()

    if method == "A":
        return method_a_grid_search(
            w=w,
            y_true=y_train_labels,
            weights=wt,
            observed_rate=obs,
            grid_step=grid_step,
            min_gap=min_gap,
        )
    if method == "B":
        return method_b_segmented(w=w, observed_rate=obs, weights=wt, min_gap=min_gap)
    if method == "C":
        return method_c_quantile(w=w, weights=wt)
    raise ValueError(f"Unknown method: {method}")


def run_cv(
    data: pd.DataFrame,
    y_labels: np.ndarray,
    methods: List[str],
    baseline: Thresholds,
    cv_mode: str,
    grid_step: float,
    min_gap: float,
) -> pd.DataFrame:
    rows = []
    if cv_mode == "none":
        folds = [("all", data.index.to_numpy(), data.index.to_numpy())]
    else:
        regions = sorted(data["region"].unique())
        folds = []
        for r in regions:
            test_idx = data.index[data["region"] == r].to_numpy()
            train_idx = data.index[data["region"] != r].to_numpy()
            folds.append((r, train_idx, test_idx))

    for fold_name, train_idx, test_idx in folds:
        train = data.loc[train_idx]
        test = data.loc[test_idx]
        y_train = y_labels[train_idx]
        y_test = y_labels[test_idx]

        # Baseline
        base_metrics = evaluate_thresholds(
            test["W_p"].to_numpy(),
            test["observed_rate"].to_numpy(),
            y_test,
            test["sample_weight"].to_numpy(),
            baseline,
        )
        rows.append(
            {
                "fold": fold_name,
                "method": "baseline",
                "low_threshold": baseline.low,
                "high_threshold": baseline.high,
                **base_metrics,
            }
        )

        for method in methods:
            thr = calibrate_method(
                method=method,
                train=train,
                y_train_labels=y_train,
                grid_step=grid_step,
                min_gap=min_gap,
            )
            metrics = evaluate_thresholds(
                test["W_p"].to_numpy(),
                test["observed_rate"].to_numpy(),
                y_test,
                test["sample_weight"].to_numpy(),
                thr,
            )
            rows.append(
                {
                    "fold": fold_name,
                    "method": method,
                    "low_threshold": thr.low,
                    "high_threshold": thr.high,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def summarize_methods(cv_df: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for method in sorted(cv_df["method"].unique()):
        subset = cv_df[cv_df["method"] == method]
        mean_f1 = subset["macro_f1"].mean()
        std_f1 = subset["macro_f1"].std(ddof=0)
        mean_bal = subset["balanced_accuracy"].mean()
        mono_rate = subset["monotonic_ok"].mean()
        mean_low = subset["low_threshold"].mean()
        mean_high = subset["high_threshold"].mean()
        std_low = subset["low_threshold"].std(ddof=0)
        std_high = subset["high_threshold"].std(ddof=0)

        # Stability-aware score without allowing variance penalty to dominate
        # mean predictive performance.
        score = mean_f1 - 0.25 * std_f1 - 0.05 * (1.0 - mono_rate)
        summary_rows.append(
            {
                "method": method,
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
                "mean_balanced_accuracy": mean_bal,
                "monotonic_pass_rate": mono_rate,
                "mean_low_threshold": mean_low,
                "mean_high_threshold": mean_high,
                "std_low_threshold": std_low,
                "std_high_threshold": std_high,
                "selection_score": score,
            }
        )
    return pd.DataFrame(summary_rows).sort_values("selection_score", ascending=False)


def bootstrap_thresholds(
    data: pd.DataFrame,
    method: str,
    y_labels: np.ndarray,
    n_bootstrap: int,
    seed: int,
    grid_step: float,
    min_gap: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(data)

    rows = []
    for b in range(n_bootstrap):
        # Standard non-parametric bootstrap on rows. Sample weights remain
        # part of the estimator, so we avoid probability-weighted resampling
        # here to prevent double-weighting.
        idx = rng.choice(np.arange(n), size=n, replace=True)
        sample = data.iloc[idx].copy()
        y_sample = y_labels[idx]
        thr = calibrate_method(
            method=method,
            train=sample,
            y_train_labels=y_sample,
            grid_step=grid_step,
            min_gap=min_gap,
        )
        rows.append({"bootstrap_id": b, "low_threshold": thr.low, "high_threshold": thr.high})
    return pd.DataFrame(rows)


def maybe_make_plots(
    outdir: Path,
    cv_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    selected_method: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    methods = sorted(cv_df["method"].unique())
    x = np.arange(len(methods))
    means = [cv_df[cv_df["method"] == m]["macro_f1"].mean() for m in methods]
    errs = [cv_df[cv_df["method"] == m]["macro_f1"].std(ddof=0) for m in methods]
    ax.bar(x, means, yerr=errs, color=["#4c78a8" if m != selected_method else "#f58518" for m in methods])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("CV weighted macro-F1")
    ax.set_title("Method comparison (mean ± std)")
    fig.tight_layout()
    fig.savefig(outdir / "cv_method_performance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(bootstrap_df["low_threshold"], bins=30, alpha=0.7, label="low threshold")
    ax.hist(bootstrap_df["high_threshold"], bins=30, alpha=0.7, label="high threshold")
    ax.set_title("Bootstrap threshold distributions")
    ax.set_xlabel("Threshold value")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "bootstrap_threshold_distributions.png", dpi=150)
    plt.close(fig)


def load_baseline_thresholds(weights_file: Path) -> Thresholds:
    with weights_file.open() as f:
        cfg = json.load(f)
    tp = cfg.get("trip_purpose_parameters", {})
    thr = tp.get("classification_thresholds", {})
    return Thresholds(float(thr.get("moderate", 0.6)), float(thr.get("high", 0.8)))


def write_config_thresholds(weights_file: Path, thr: Thresholds) -> None:
    with weights_file.open() as f:
        cfg = json.load(f)
    cfg.setdefault("trip_purpose_parameters", {})
    cfg["trip_purpose_parameters"].setdefault("classification_thresholds", {})
    cfg["trip_purpose_parameters"]["classification_thresholds"]["moderate"] = round(thr.low, 4)
    cfg["trip_purpose_parameters"]["classification_thresholds"]["high"] = round(thr.high, 4)
    with weights_file.open("w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    weights_file = Path(args.weights_file)

    data, dropped = load_dataset(args.input_glob)
    baseline = load_baseline_thresholds(weights_file)

    y_labels, label_bounds = build_target_labels(
        observed_rate=data["observed_rate"].to_numpy(),
        weights=data["sample_weight"].to_numpy(),
        mode=args.label_mode,
        q_low=args.q_low,
        q_high=args.q_high,
        fixed_low=args.fixed_low,
        fixed_high=args.fixed_high,
    )

    methods = ["A", "B", "C"]
    cv_df = run_cv(
        data=data,
        y_labels=y_labels,
        methods=methods,
        baseline=baseline,
        cv_mode=args.cv,
        grid_step=args.grid_step,
        min_gap=args.min_gap,
    )
    cv_df.to_csv(outdir / "cv_results.csv", index=False)

    summary_df = summarize_methods(cv_df)
    summary_df.to_csv(outdir / "method_comparison.csv", index=False)
    selected_method = summary_df.iloc[0]["method"]

    # Guardrail: selected calibrated method should not underperform baseline
    # in mean CV macro-F1.
    baseline_row = summary_df[summary_df["method"] == "baseline"]
    baseline_f1 = float(baseline_row["mean_macro_f1"].iloc[0]) if not baseline_row.empty else -np.inf
    calibrated = summary_df[summary_df["method"] != "baseline"].copy()
    if selected_method == "baseline" or float(summary_df.iloc[0]["mean_macro_f1"]) < baseline_f1:
        calibrated_sorted = calibrated.sort_values(
            ["mean_macro_f1", "selection_score"], ascending=[False, False]
        )
        selected_method = str(calibrated_sorted.iloc[0]["method"])

    final_thr = calibrate_method(
        method=selected_method,
        train=data,
        y_train_labels=y_labels,
        grid_step=args.grid_step,
        min_gap=args.min_gap,
    )

    bootstrap_df = bootstrap_thresholds(
        data=data,
        method=selected_method,
        y_labels=y_labels,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
        grid_step=args.grid_step,
        min_gap=args.min_gap,
    )
    bootstrap_df.to_csv(outdir / "bootstrap_thresholds.csv", index=False)

    low_ci = np.quantile(bootstrap_df["low_threshold"], [0.025, 0.975]).tolist()
    high_ci = np.quantile(bootstrap_df["high_threshold"], [0.025, 0.975]).tolist()

    baseline_eval = evaluate_thresholds(
        data["W_p"].to_numpy(),
        data["observed_rate"].to_numpy(),
        y_labels,
        data["sample_weight"].to_numpy(),
        baseline,
    )
    final_eval = evaluate_thresholds(
        data["W_p"].to_numpy(),
        data["observed_rate"].to_numpy(),
        y_labels,
        data["sample_weight"].to_numpy(),
        final_thr,
    )

    recommended = {
        "selected_method": selected_method,
        "recommended_thresholds": {
            "moderate": round(final_thr.low, 4),
            "high": round(final_thr.high, 4),
        },
        "bootstrap_95ci": {
            "moderate": [round(low_ci[0], 4), round(low_ci[1], 4)],
            "high": [round(high_ci[0], 4), round(high_ci[1], 4)],
        },
        "baseline_thresholds": {"moderate": baseline.low, "high": baseline.high},
        "label_mode": args.label_mode,
        "label_bounds_observed_rate": {
            "low_boundary": round(label_bounds[0], 4),
            "high_boundary": round(label_bounds[1], 4),
        },
        "performance_full_data": {
            "baseline": baseline_eval,
            "recommended": final_eval,
        },
        "dataset": {
            "rows": int(len(data)),
            "regions": int(data["region"].nunique()),
            "purposes": int(data["purpose"].nunique()),
        },
        "duplicates_dropped": [
            {"region": r, "path": str(p)} for r, p in dropped
        ],
    }
    with (outdir / "recommended_thresholds.json").open("w") as f:
        json.dump(recommended, f, indent=2)
        f.write("\n")

    report_lines = [
        "# Trip-Purpose Cutoff Calibration Report",
        "",
        "## Inputs",
        f"- Input glob: `{args.input_glob}`",
        f"- Rows used: {len(data)}",
        f"- Regions: {data['region'].nunique()}",
        f"- Purposes: {data['purpose'].nunique()}",
        f"- Label mode: `{args.label_mode}`",
        f"- Observed-rate boundaries for pseudo-labels: "
        f"low<{label_bounds[0]:.4f}, high>={label_bounds[1]:.4f}",
        "",
        "## Method Selection",
        f"- Selected method: **{selected_method}**",
        f"- Recommended moderate threshold (low cutoff): **{final_thr.low:.4f}**",
        f"- Recommended high threshold: **{final_thr.high:.4f}**",
        f"- 95% bootstrap CI (moderate): [{low_ci[0]:.4f}, {low_ci[1]:.4f}]",
        f"- 95% bootstrap CI (high): [{high_ci[0]:.4f}, {high_ci[1]:.4f}]",
        "",
        "## Full-Data Performance",
        f"- Baseline macro-F1: {baseline_eval['macro_f1']:.4f}",
        f"- Recommended macro-F1: {final_eval['macro_f1']:.4f}",
        f"- Baseline balanced accuracy: {baseline_eval['balanced_accuracy']:.4f}",
        f"- Recommended balanced accuracy: {final_eval['balanced_accuracy']:.4f}",
        "",
        "## Artifacts",
        "- `cv_results.csv`",
        "- `method_comparison.csv`",
        "- `bootstrap_thresholds.csv`",
        "- `recommended_thresholds.json`",
    ]
    (outdir / "calibration_report.md").write_text("\n".join(report_lines))

    maybe_make_plots(outdir, cv_df, bootstrap_df, selected_method)

    if args.write_config:
        write_config_thresholds(weights_file, final_thr)

    print(f"Calibration complete. Outputs written to: {outdir}")
    print(
        f"Recommended thresholds -> moderate: {final_thr.low:.4f}, high: {final_thr.high:.4f} "
        f"(method {selected_method})"
    )


if __name__ == "__main__":
    main()
