"""Pure vehicle-assignment calculations for the Paper 1 component grid."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_grid4(
    factors: pd.DataFrame,
    weights: tuple[float, float, float],
    children_bonus: float,
    children_threshold: float,
    assignment_margin: float = 0.3,
    two_seater_threshold: float = 0.7,
    four_seater_threshold: float = 0.3,
) -> pd.DataFrame:
    """Apply the existing three-class selector to validated OA component factors.

    The inputs are household-size, age and dependent-child factors in [0, 1].
    Returns independent scores and assignments. Raises ValueError for invalid
    factors or settings. The fixed thresholds match the Grid 4 producer.
    """
    names = ["household_size_factor", "age_factor", "children_factor"]
    values = factors[names].to_numpy(dtype=float)
    settings = np.asarray([*weights, children_bonus, children_threshold,
                           assignment_margin, two_seater_threshold,
                           four_seater_threshold], dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("All component factors must be finite and in [0, 1].")
    if not np.isfinite(settings).all() or ((settings < 0) | (settings > 1)).any():
        raise ValueError("All settings must be finite and in [0, 1].")
    if not np.isclose(sum(weights), 1.0):
        raise ValueError("Assignment component weights must sum to one.")
    hsf, age, children = values.T
    wh, wa, wc = weights
    two = np.clip(wh * hsf + wa * age + wc * (1 - children), 0, 1)
    four = np.clip(wh * (1 - hsf) + wa * (1 - age) + wc * children
                   + children_bonus * (children > children_threshold), 0, 1)
    gap = four - two
    two_mask = (two >= two_seater_threshold) & (gap < -assignment_margin)
    four_mask = (four >= four_seater_threshold) & (
        (gap > assignment_margin) | (np.abs(gap) <= assignment_margin)
    )
    assignment = np.select([two_mask, four_mask], ["2-seater", "4-seater"],
                           default="mixed")
    return pd.DataFrame({"two_seater_score": two, "four_seater_score": four,
                         "assignment": assignment}, index=factors.index)


def summarize_grid4(assignments: pd.Series, target: float = 0.10) -> dict[str, float | int]:
    """Count assignments and keep all-OA and binary-only denominators explicit.

    Chi-squared is a descriptive discrepancy on the binary subset, not an
    inferential test of observed vehicle purchases. Empty input, unknown classes
    and invalid targets raise ValueError.
    """
    if assignments.empty or not assignments.isin(["2-seater", "4-seater", "mixed"]).all():
        raise ValueError("Assignments must contain known classes and be nonempty.")
    if not 0 < target < 1:
        raise ValueError("The target must be strictly between zero and one.")
    counts = assignments.value_counts()
    two, four, mixed = (int(counts.get(k, 0)) for k in ["2-seater", "4-seater", "mixed"])
    total = two + four + mixed
    binary = two + four
    expected = binary * np.array([target, 1 - target])
    chi2 = float(np.sum((np.array([two, four]) - expected) ** 2 / expected)) if binary else float("nan")
    return {"total_areas": total, "two_seater_count": two,
            "four_seater_count": four, "mixed_count": mixed,
            "two_seater_share_all": two / total,
            "four_seater_share_all": four / total,
            "mixed_share_all": mixed / total, "binary_denominator": binary,
            "two_seater_share_binary": two / binary if binary else float("nan"),
            "chi2_binary": chi2, "target_binary_share": target}
