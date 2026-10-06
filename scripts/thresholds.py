"""Threshold diagnostics using the manuscript's unchanged statistical functions."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from functions.paper1_thresholds import (
    best_knots, breakpoint_scan, hinge_design, loo_predictions,
    nested_threshold_cv, regional_piecewise_cv, weighted_fit,
    local_regime_design, local_regime_scan,
)
AP = "final_adoption_propensity"
CAR = ["No.cars_Car_or_van_availability_2022", "One.car_Car_or_van_availability_2022",
       "Two.or.more.cars_Car_or_van_availability_2022"]
GRID = np.round(np.arange(.05, .951, .025), 3)

def profiles(oa: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Describe raw census counts and model charging scores without fitting gates."""
    rows = []
    bands = {"Below 0.30": oa[AP] < .3, "0.30-0.80": oa[AP].between(.3, .8), "Above 0.80": oa[AP] > .8,
             "0.25-0.30": oa[AP].between(.25, .3, inclusive="left"),
             "0.30-0.35": oa[AP].between(.3, .35, inclusive="left"),
             "0.75-0.80": oa[AP].between(.75, .8, inclusive="left"),
             "0.80-0.85": oa[AP].between(.8, .85, inclusive="left")}
    for label, mask in bands.items():
        d = oa.loc[mask]
        rows.append({"band": label, "oa_count": len(d), "households": d.households.sum(),
            "no_car_pct": 100*d[CAR[0]].sum()/d.households.sum(),
            "multi_car_pct": 100*d[CAR[2]].sum()/d.households.sum(),
            "abc1_pct": 100*d.abc1_count.sum()/d.social_total.sum(),
            "flat_pct": 100*d.flat_count.sum()/d.accommodation_total.sum(),
            "home_charging_mean": np.average(d.home_charging_feasibility, weights=d.households),
            "home_charging_feasible_pct": 100*np.average(d.home_charging_feasibility >= .5, weights=d.households)})
    result = pd.DataFrame(rows)
    result.to_csv(out / "demographic_band_profiles.csv", index=False)
    bins = []
    for index, d in oa.groupby(np.minimum((oa[AP]/.025).astype(int), 39)):
        bins.append({"ap_mid": (index+.5)*.025, "oa_count": len(d),
            **{col: np.average(d[col], weights=d.households) for col in
               ["no_car_share", "multi_car_share", "abc1_share", "flat_share", "home_charging_feasibility"]}})
    pd.DataFrame(bins).to_csv(out / "demographic_curves.csv", index=False)
    return result


def spatial_models(oa: pd.DataFrame, current_hpi: pd.DataFrame, out: Path,
                   observed_datazone_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare free and fixed slope changes, including omitted-region tests."""
    # This table preserves the original joins, suppression and price exclusions.
    dz = pd.read_csv(observed_datazone_path, float_precision="round_trip")
    dz.to_csv(out / "datazone_study_inputs.csv", index=False)
    targets = [(name, oa, AP, "households") for name in
               ["no_car_share", "multi_car_share", "abc1_share", "flat_share", "home_charging_feasibility"]]
    targets += [("log_price", dz, "ap", None), ("bev_rate", dz, "ap", None)]
    summaries, all_folds = [], []
    for name, frame, xcol, wcol in targets:
        d = frame.dropna(subset=[xcol, name, "region"])
        x, y = d[xcol].to_numpy(), d[name].to_numpy()
        w = np.ones(len(d)) if wcol is None else d[wcol].to_numpy()
        scan = breakpoint_scan(x, y, w, GRID)
        scan.to_csv(out / f"breakpoint_scan_{name}.csv", index=False)
        cv = regional_piecewise_cv(x, y, w, d.region.to_numpy(), GRID)
        cv["outcome"] = name
        all_folds.append(cv)
        for model, knots in {"linear": (), "quadratic": (), "fixed_030_080": (.3, .8),
                             "estimated_one": best_knots(scan, 1), "estimated_two": best_knots(scan, 2)}.items():
            design = np.column_stack([np.ones(len(x)), x, x*x]) if model == "quadratic" else hinge_design(x, knots)
            pred = design @ weighted_fit(design, y, w)
            folds = cv[cv.model.eq(model)]
            summaries.append({"outcome": name, "model": model, "n": len(d),
                "lower": knots[0] if knots else np.nan, "upper": knots[1] if len(knots) == 2 else np.nan,
                "fit_r2": 1-np.sum(w*(y-pred)**2)/np.sum(w*(y-np.average(y, weights=w))**2),
                "fit_rmse": np.sqrt(np.average((y-pred)**2, weights=w)),
                "region_cv_rmse": np.sqrt(folds.weighted_sse.sum()/folds.weight.sum()),
                "fold_lower_min": folds.lower.min(), "fold_lower_max": folds.lower.max(),
                "fold_upper_min": folds.upper.min(), "fold_upper_max": folds.upper.max()})
        print(f"Breakpoint fits and regional CV: {name} ({len(d):,} units)", flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "breakpoint_summary.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(out / "regional_breakpoint_folds.csv", index=False)
    price_bands = []
    for label, mask in {"Below 0.30": dz.ap < .3, "0.30-0.80": dz.ap.between(.3, .8), "Above 0.80": dz.ap > .8}.items():
        d = dz.loc[mask]
        price_bands.append({"band": label, "all_dz": len(d), "price_dz": d.log_price.notna().sum(),
                            "median_of_dz_median_price": d.median_price_2022.median(),
                            "observed_bev_dz": d.bev.notna().sum(), "observed_bev": d.bev.sum()})
    pd.DataFrame(price_bands).to_csv(out / "datazone_band_profiles.csv", index=False)
    return summary, dz


def postcode_models(oa: pd.DataFrame, counts: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test band alignment on a fixed sample with size and house-price controls."""
    d = oa.dropna(subset=["pc_area"]).copy()
    d["ap_hh"] = d[AP]*d.households
    pc = d.groupby("pc_area").agg(households=("households", "sum"), oa_count=("geo_code", "size"),
                                 score_sum=(AP, "sum"), ap_hh=("ap_hh", "sum")).reset_index()
    pc["ap_mean"] = pc.ap_hh/pc.households
    # Allocate DZ sales totals by the household share in each postcode, once.
    total_hh = oa.groupby("DZ22").households.sum()
    d["dz_fraction"] = d.households / d.DZ22.map(total_hh)
    d["allocated_value"] = d.value_1920*d.dz_fraction
    d["allocated_sales"] = d.sales_1920*d.dz_fraction
    prices = d.groupby("pc_area").agg(value=("allocated_value", "sum"), sales=("allocated_sales", "sum"))
    pc = pc.merge(prices, on="pc_area", validate="one_to_one").merge(counts, on="pc_area", validate="one_to_one")
    pc["mean_price_1920"] = pc.value/pc.sales
    pc = pc[pc.mean_price_1920.gt(0) & pc.matched_registrations.gt(0)].sort_values("pc_area").reset_index(drop=True)
    pc["lower_rate"] = 1000*pc.lower_price_registrations/pc.households
    pc["premium_rate"] = 1000*pc.premium_registrations/pc.households
    pc["premium_share"] = pc.premium_registrations/pc.matched_registrations
    pc["total_rate"] = 1000*pc.total_registrations/pc.households
    pc.to_csv(out / "postcode_study_inputs.csv", index=False)
    scans, summaries, folds = [], [], []
    settings = [("lower_price", [(float(lo), float(hi)) for lo in np.round(np.arange(.15, .501, .025), 3)
                                for hi in np.round(np.arange(.60, .951, .025), 3)]),
                ("premium", [(0., float(hi)) for hi in np.round(np.arange(.60, .951, .025), 3)])]
    for segment, candidates in settings:
        labels, predictor_columns, raw_columns = [], [], []
        for lo, hi in candidates:
            mask = d[AP].between(lo, hi) if segment == "lower_price" else d[AP] > hi
            labels.append(f"{lo:.3f}/{hi:.3f}")
            weighted = (d.ap_hh*mask).groupby(d.pc_area).sum().reindex(pc.pc_area, fill_value=0).to_numpy()
            raw = (d[AP]*mask).groupby(d.pc_area).sum().reindex(pc.pc_area, fill_value=0).to_numpy()
            predictor_columns.append(weighted/pc.households.to_numpy())
            raw_columns.append(raw)
        predictors = np.column_stack(predictor_columns)
        raw_predictors = np.column_stack(raw_columns)
        target = pc.lower_rate.to_numpy() if segment == "lower_price" else pc.premium_share.to_numpy()
        raw_target = pc.lower_price_registrations if segment == "lower_price" else pc.premium_registrations
        fixed = labels.index("0.300/0.800" if segment == "lower_price" else "0.000/0.800")
        for control_name in ("intercept", "house_price"):
            controls = np.ones((len(pc), 1)) if control_name == "intercept" else np.column_stack(
                [np.ones(len(pc)), np.log(pc.mean_price_1920.to_numpy())])
            summary, pred = nested_threshold_cv(predictors, target, controls, labels, fixed)
            summary["segment"], summary["control"] = segment, control_name
            summaries.append(summary)
            pred["pc_area"] = pred.omitted_index.map(pc.pc_area)
            pred["segment"], pred["control"] = segment, control_name
            folds.append(pred)
            for j, (lo, hi) in enumerate(candidates):
                design = np.column_stack([controls, predictors[:, j]])
                beta = weighted_fit(design, target, np.ones(len(pc)))
                prediction = loo_predictions(design, target)
                scans.append({"segment": segment, "control": control_name, "lower": lo, "upper": hi,
                    "n": len(pc), "fit_r2": 1-np.sum((target-design@beta)**2)/np.sum((target-target.mean())**2),
                    "loo_rmse": np.sqrt(np.mean((target-prediction)**2)), "slope": beta[-1],
                    "raw_count_r2": pearsonr(raw_predictors[:, j], raw_target).statistic**2,
                    "raw_count_r": pearsonr(raw_predictors[:, j], raw_target).statistic})
            design = np.column_stack([controls, pc.ap_mean.to_numpy()])
            continuous_prediction = loo_predictions(design, target)
            summaries.append(pd.DataFrame([{"model": "continuous_ap", "n": len(pc),
                "rmse": np.sqrt(np.mean((target-continuous_prediction)**2)),
                "mae": np.mean(abs(target-continuous_prediction)),
                "r2": 1-np.sum((target-continuous_prediction)**2)/np.sum((target-target.mean())**2),
                "segment": segment, "control": control_name}]))
        print(f"Nested postcode-area CV: {segment}, {len(candidates)} candidates, {len(pc)} areas", flush=True)
    scan = pd.DataFrame(scans)
    scan.to_csv(out / "mot_threshold_scan.csv", index=False)
    result = pd.concat(summaries, ignore_index=True)
    result.to_csv(out / "mot_cross_validation.csv", index=False)
    pd.concat(folds, ignore_index=True).to_csv(out / "mot_nested_folds.csv", index=False)
    correlation = []
    for x, y in [("ap_mean", "mean_price_1920"), ("mean_price_1920", "premium_share"),
                 ("ap_mean", "premium_share"), ("households", "total_registrations")]:
        correlation.append({"x": x, "y": y, "n": len(pc), "pearson_r": pearsonr(pc[x], pc[y]).statistic,
                            "spearman_rho": spearmanr(pc[x], pc[y]).statistic})
    pd.DataFrame(correlation).to_csv(out / "postcode_associations.csv", index=False)
    return result, scan


def local_transitions(oa: pd.DataFrame, out: Path) -> None:
    """Distinguish local composition changes from global continuous gradients."""
    rows = []
    for label, domain, grid, fixed in [
        ("lower", (.15, .45), np.round(np.arange(.20, .401, .01), 3), .3),
        ("upper", (.65, .95), np.round(np.arange(.70, .901, .01), 3), .8),
    ]:
        d = oa[oa[AP].between(*domain)]
        x, w, groups = d[AP].to_numpy(), d.households.to_numpy(), d.region.to_numpy()
        for outcome in ["no_car_share", "multi_car_share", "abc1_share", "flat_share", "home_charging_feasibility"]:
            y = d[outcome].to_numpy()
            scan = local_regime_scan(x, y, w, grid)
            scan.to_csv(out / f"local_scan_{label}_{outcome}.csv", index=False)
            best = scan.loc[scan.sse.idxmin()]
            specified = scan[scan.threshold.eq(fixed)].iloc[0]
            selected_knots, fixed_loss, selected_loss = [], 0., 0.
            for region in np.unique(groups):
                test = groups == region
                training = local_regime_scan(x[~test], y[~test], w[~test], grid)
                selected = float(training.loc[training.sse.idxmin(), "threshold"])
                selected_knots.append(selected)
                for model, threshold in (("fixed", fixed), ("selected", selected)):
                    design = local_regime_design(x, threshold)
                    beta = weighted_fit(design[~test], y[~test], w[~test])
                    loss = float(np.sum(w[test]*(y[test]-design[test]@beta)**2))
                    if model == "fixed":
                        fixed_loss += loss
                    else:
                        selected_loss += loss
            rows.append({"edge": label, "outcome": outcome, "n": len(d),
                "sample_min": domain[0], "sample_max": domain[1], "best_threshold": best.threshold,
                "fixed_threshold": fixed, "level_change_at_fixed": specified.level_change,
                "best_level_change": best.level_change,
                "fold_threshold_min": min(selected_knots), "fold_threshold_max": max(selected_knots),
                "fixed_cv_rmse": np.sqrt(fixed_loss/w.sum()),
                "selected_cv_rmse": np.sqrt(selected_loss/w.sum())})
    pd.DataFrame(rows).to_csv(out / "local_transition_summary.csv", index=False)


def figures(dz: pd.DataFrame, scan: pd.DataFrame, out: Path, figure_output: Path) -> None:
    """Plot demographic gradients and external diagnostics without maps."""
    plt.rcParams.update({"font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    curves = pd.read_csv(out / "demographic_curves.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for field, label, color in [("no_car_share", "No car/van", "#b54448"),
                               ("multi_car_share", "Two or more cars/vans", "#207b73"),
                               ("home_charging_feasibility", "Home-charging score", "#336aaa")]:
        axes[0, 0].plot(curves.ap_mid, 100*curves[field], color=color, label=label, lw=1.8)
    axes[0, 0].set(title="(a) Car ownership and charging", xlabel="OA adoption propensity", ylabel="Household share or score (%)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    for field, label, color in [("abc1_share", "ABC1 social grade", "#6a5293"), ("flat_share", "Flatted accommodation", "#777777")]:
        axes[0, 1].plot(curves.ap_mid, 100*curves[field], label=label, color=color, lw=1.8)
    axes[0, 1].set(title="(b) Demographic composition", xlabel="OA adoption propensity", ylabel="Weighted share (%)")
    axes[0, 1].legend(frameon=False, fontsize=8)
    price = dz.dropna(subset=["median_price_2022"]).copy()
    price["bin"] = np.minimum((price.ap/.05).astype(int), 19)
    grouped = price.groupby("bin").median_price_2022
    median, q25, q75 = grouped.median(), grouped.quantile(.25), grouped.quantile(.75)
    centres = (median.index.to_numpy()+.5)*.05
    axes[1, 0].fill_between(centres, q25/1000, q75/1000, color="#207b73", alpha=.18)
    axes[1, 0].plot(centres, median/1000, color="#207b73", lw=1.8)
    axes[1, 0].set(title="(c) Observed 2022 house prices", xlabel="Household-weighted Data-Zone AP", ylabel="Data-Zone median price (GBP thousand)")
    for ax in axes.flat[:3]:
        for threshold in (.3, .8):
            ax.axvline(threshold, color="#b54448", ls="--", lw=.9)
        ax.set_xlim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    surface = scan[scan.segment.eq("lower_price") & scan.control.eq("house_price")].pivot(index="lower", columns="upper", values="loo_rmse")
    im = axes[1, 1].imshow(surface.to_numpy(), origin="lower", aspect="auto", cmap="viridis",
                          extent=[.5875, .9625, .1375, .5125])
    axes[1, 1].plot(.8, .3, marker="x", color="white", ms=9, mew=2)
    axes[1, 1].set(title="(d) Lower-price MOT: area-level sensitivity", xlabel="Upper AP edge", ylabel="Lower AP edge")
    fig.colorbar(im, ax=axes[1, 1], label="LOO RMSE (vehicles per 1,000 households)", shrink=.78)
    fig.savefig(out / "threshold_evidence.png", dpi=220)
    fig.savefig(figure_output, dpi=220)
    fig.savefig(out / "threshold_evidence.pdf")
    plt.close(fig)

def run_thresholds(root: Path, out: Path, figure_dir: Path) -> dict:
    """Rebuild all threshold tables and compare with the manuscript references."""
    from reproduce import compare_csv
    out.mkdir(parents=True, exist_ok=True)
    oa = pd.read_parquet(root / "data/thresholds/oa_study_inputs.parquet")
    counts = pd.read_csv(root / "data/thresholds/price_segment_counts.csv")
    profiles(oa, out)
    _, dz = spatial_models(oa, pd.DataFrame(), out, root / "data/thresholds/datazone_study_inputs.csv")
    local_transitions(oa, out)
    _, scan = postcode_models(oa, counts, out)
    figures(dz, scan, out, figure_dir / "threshold_evidence.png")
    checked = 0
    for path in sorted(out.glob("*.csv")):
        reference = root / "data/reference/thresholds" / path.name
        if not reference.is_file():
            raise ValueError(f"Unmapped threshold result: {path.name}")
        compare_csv(path, reference, rtol=1e-7, atol=1e-7)
        checked += 1
    metadata = {"observed_bev_datazones": int(dz.bev.notna().sum()), "oa_count": len(oa),
                "scope": "Conditional on the frozen OA scores; aggregated source inputs."}
    (out / "study_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"tables_verified": checked, **metadata}
