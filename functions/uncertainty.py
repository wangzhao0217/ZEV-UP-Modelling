"""
Uncertainty Quantification Module for EV Modelling

Implements Monte Carlo simulation for propagating weight uncertainty
through adoption propensity calculations.

Per @eq-uncertainty-propagation in paper.qmd:
σ²(AP) = Σᵢ (∂AP/∂wᵢ)² · σ²(wᵢ)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class UncertaintyResult:
    """Results from uncertainty quantification analysis."""
    mean: np.ndarray
    std: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    coefficient_of_variation: np.ndarray
    n_simulations: int
    confidence_level: float


class MonteCarloUncertainty:
    """
    Monte Carlo uncertainty quantification for adoption propensity.

    Propagates weight uncertainty through the adoption calculation
    to produce confidence intervals for each area's score.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: str = "scoring_weights.json"
    ):
        """
        Initialize Monte Carlo uncertainty analyzer.

        Args:
            config: Configuration dictionary (optional)
            config_path: Path to scoring_weights.json if config not provided
        """
        if config is None:
            config = self._load_config(config_path)

        self.config = config

        # Get uncertainty parameters
        unc_params = config.get('uncertainty_parameters', {})
        self.n_simulations = unc_params.get('n_simulations', 1000)
        self.weight_std = unc_params.get('weight_uncertainty_std', 0.05)
        self.confidence_level = unc_params.get('confidence_level', 0.95)

        # Get base weights
        self.demographic_weights = config.get('demographic_weights', {})
        self.home_charging_blend = config.get('home_charging_blend', {})
        self.final_adoption_blend = config.get('final_adoption_blend', )

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _perturb_weights(
        self,
        weights: Dict[str, float],
        std: float
    ) -> Dict[str, float]:
        """
        Perturb weights with Gaussian noise and renormalize.

        Args:
            weights: Original weight dictionary
            std: Standard deviation for perturbation

        Returns:
            Perturbed and renormalized weights
        """
        perturbed = {}
        for key, value in weights.items():
            if value > 0:  # Only perturb active weights
                perturbed[key] = np.clip(
                    value + np.random.normal(0, std),
                    0.01, 0.99
                )
            else:
                perturbed[key] = 0.0

        # Renormalize to sum to 1
        total = sum(perturbed.values())
        if total > 0:
            perturbed = {k: v / total for k, v in perturbed.items()}

        return perturbed

    def _calculate_base_score(
        self,
        scores: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Calculate weighted adoption score."""
        result = np.zeros(len(next(iter(scores.values()))))

        for key, weight in weights.items():
            if key in scores and weight > 0:
                result += scores[key] * weight

        return np.clip(result, 0.0, 1.0)

    def run_monte_carlo(
        self,
        component_scores: Dict[str, np.ndarray],
        n_simulations: Optional[int] = None
    ) -> UncertaintyResult:
        """
        Run Monte Carlo simulation for adoption uncertainty.

        Args:
            component_scores: Dict mapping weight keys to score arrays
            n_simulations: Override default simulation count

        Returns:
            UncertaintyResult with statistics
        """
        n_sims = n_simulations or self.n_simulations
        n_areas = len(next(iter(component_scores.values())))

        # Store results from each simulation
        results = np.zeros((n_sims, n_areas))

        for i in range(n_sims):
            perturbed = self._perturb_weights(
                self.demographic_weights,
                self.weight_std
            )
            results[i] = self._calculate_base_score(
                component_scores,
                perturbed
            )

        # Calculate statistics
        alpha = 1 - self.confidence_level
        return UncertaintyResult(
            mean=results.mean(axis=0),
            std=results.std(axis=0),
            ci_lower=np.percentile(results, alpha/2 * 100, axis=0),
            ci_upper=np.percentile(results, (1-alpha/2) * 100, axis=0),
            coefficient_of_variation=results.std(axis=0) / (results.mean(axis=0) + 1e-10),
            n_simulations=n_sims,
            confidence_level=self.confidence_level
        )

    def add_uncertainty_to_dataframe(
        self,
        df: pd.DataFrame,
        result: UncertaintyResult,
        prefix: str = 'adoption_propensity'
    ) -> pd.DataFrame:
        """Add uncertainty columns to DataFrame."""
        df = df.copy()
        df[f'{prefix}_mean'] = result.mean
        df[f'{prefix}_std'] = result.std
        df[f'{prefix}_ci_lower'] = result.ci_lower
        df[f'{prefix}_ci_upper'] = result.ci_upper
        df[f'{prefix}_confidence'] = 1 - result.coefficient_of_variation
        return df


def monte_carlo_adoption_uncertainty(
    component_scores: Dict[str, np.ndarray],
    config: Optional[Dict] = None,
    n_simulations: int = 1000
) -> pd.DataFrame:
    """
    Convenience function for Monte Carlo uncertainty analysis.

    Args:
        component_scores: Dict of score arrays by weight key
        config: Configuration dictionary
        n_simulations: Number of simulations

    Returns:
        DataFrame with uncertainty statistics
    """
    analyzer = MonteCarloUncertainty(config=config)
    result = analyzer.run_monte_carlo(component_scores, n_simulations)

    return pd.DataFrame({
        'adoption_mean': result.mean,
        'adoption_std': result.std,
        'adoption_ci_lower': result.ci_lower,
        'adoption_ci_upper': result.ci_upper,
        'coefficient_of_variation': result.coefficient_of_variation
    })
