"""Inclusive AP bands and membership stability."""
import numpy as np

def assign_bands(ap, tau_min: float, tau_max: float) -> np.ndarray:
    """0 = Below_Min (< tau_min), 1 = Feasible_Range ([tau_min, tau_max]), 2 = Above_Max (> tau_max)."""
    ap = np.asarray(ap, dtype=float)
    labels = np.ones(ap.shape, dtype=int)
    labels[ap < tau_min] = 0
    labels[ap > tau_max] = 2
    return labels


def membership_stability(base, variant) -> float:
    """Share of points whose band label is unchanged between two cut-off specifications."""
    base = np.asarray(base)
    variant = np.asarray(variant)
    return float((base == variant).mean())
