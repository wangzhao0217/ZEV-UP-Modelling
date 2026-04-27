"""
Spatial Autocorrelation Module for EV Modelling

Implements spatial analysis including Moran's I and spatial spillover effects
for adoption propensity scores.

Per @eq-spatial-lag in paper.qmd:
AP_spatial = ρ·W·AP + AP_base
where W is spatial weights matrix, ρ is spatial lag coefficient
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import json
from pathlib import Path
import warnings


@dataclass
class SpatialAutocorrelationResult:
    """Results from spatial autocorrelation analysis."""
    morans_i: float
    morans_p_value: float
    morans_z_score: float
    interpretation: str
    is_significant: bool


@dataclass
class SpatialLagResult:
    """Results from spatial lag model."""
    rho: float  # Spatial lag coefficient
    rho_p_value: float
    r_squared: float
    aic: float


class SpatialAutocorrelationAnalyzer:
    """
    Analyzes spatial autocorrelation in adoption propensity scores.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: str = "scoring_weights.json"
    ):
        """Initialize spatial analyzer."""
        if config is None:
            config = self._load_config(config_path)

        self.config = config
        spatial_params = config.get('spatial_parameters', {})
        self.k_neighbors = spatial_params.get('k_neighbors', 8)
        self.spillover_coefficient = spatial_params.get('spillover_coefficient', 0.15)
        self.morans_threshold = spatial_params.get('morans_i_threshold', 0.3)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _create_weights_matrix(
        self,
        gdf: gpd.GeoDataFrame,
        k: Optional[int] = None
    ) -> np.ndarray:
        """
        Create k-nearest neighbors spatial weights matrix.

        Args:
            gdf: GeoDataFrame with geometry
            k: Number of neighbors (default from config)

        Returns:
            Row-standardized weights matrix
        """
        k = k or self.k_neighbors
        n = len(gdf)

        # Get centroids
        centroids = gdf.geometry.centroid
        coords = np.array([[p.x, p.y] for p in centroids])

        # Calculate distance matrix
        from scipy.spatial import distance_matrix
        dist_matrix = distance_matrix(coords, coords)

        # Create k-NN weights
        weights = np.zeros((n, n))
        for i in range(n):
            # Get k nearest neighbors (excluding self)
            distances = dist_matrix[i]
            nearest_idx = np.argsort(distances)[1:k+1]
            weights[i, nearest_idx] = 1.0

        # Row-standardize
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights = weights / row_sums

        return weights

    def calculate_morans_i(
        self,
        gdf: gpd.GeoDataFrame,
        score_column: str = 'adoption_propensity'
    ) -> SpatialAutocorrelationResult:
        """
        Calculate Global Moran's I statistic.

        Args:
            gdf: GeoDataFrame with scores
            score_column: Column name for scores

        Returns:
            SpatialAutocorrelationResult
        """
        y = gdf[score_column].values
        n = len(y)
        W = self._create_weights_matrix(gdf)

        # Calculate Moran's I
        y_mean = y.mean()
        y_dev = y - y_mean

        numerator = n * np.sum(W * np.outer(y_dev, y_dev))
        denominator = np.sum(W) * np.sum(y_dev ** 2)

        I = numerator / denominator if denominator != 0 else 0

        # Expected value and variance under randomization
        E_I = -1 / (n - 1)
        S0 = np.sum(W)
        S1 = 0.5 * np.sum((W + W.T) ** 2)
        S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)

        k = (np.sum(y_dev ** 4) / n) / ((np.sum(y_dev ** 2) / n) ** 2)

        var_I = (
            (n * ((n**2 - 3*n + 3) * S1 - n * S2 + 3 * S0**2) -
             k * (n * (n - 1) * S1 - 2 * n * S2 + 6 * S0**2)) /
            ((n - 1) * (n - 2) * (n - 3) * S0**2) - E_I**2
        )

        z_score = (I - E_I) / np.sqrt(var_I) if var_I > 0 else 0

        # Two-tailed p-value
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return SpatialAutocorrelationResult(
            morans_i=I,
            morans_p_value=p_value,
            morans_z_score=z_score,
            interpretation='positive' if I > 0 else 'negative',
            is_significant=p_value < 0.05
        )

    def apply_spatial_spillover(
        self,
        gdf: gpd.GeoDataFrame,
        score_column: str = 'adoption_propensity',
        lambda_coef: Optional[float] = None
    ) -> gpd.GeoDataFrame:
        """
        Apply spatial spillover effect to scores.

        Per @eq-spillover-adjustment:
        AP_adjusted = (1-λ)·AP_base + λ·mean(AP_neighbors)

        Args:
            gdf: GeoDataFrame with scores
            score_column: Column name for scores
            lambda_coef: Spillover coefficient (default from config)

        Returns:
            GeoDataFrame with spatial columns added
        """
        gdf = gdf.copy()
        lam = lambda_coef or self.spillover_coefficient

        y = gdf[score_column].values
        W = self._create_weights_matrix(gdf)

        # Calculate neighbor mean (spatial lag)
        neighbor_mean = W @ y

        # Apply spillover
        gdf[f'{score_column}_spatial'] = (1 - lam) * y + lam * neighbor_mean
        gdf['neighbor_mean_adoption'] = neighbor_mean

        return gdf

    def calculate_local_morans_i(
        self,
        gdf: gpd.GeoDataFrame,
        score_column: str = 'adoption_propensity'
    ) -> gpd.GeoDataFrame:
        """
        Calculate Local Moran's I (LISA) for cluster detection.

        Args:
            gdf: GeoDataFrame with scores
            score_column: Column name for scores

        Returns:
            GeoDataFrame with LISA columns added
        """
        gdf = gdf.copy()
        y = gdf[score_column].values
        n = len(y)
        W = self._create_weights_matrix(gdf)

        y_mean = y.mean()
        y_dev = y - y_mean
        m2 = np.sum(y_dev ** 2) / n

        # Local Moran's I
        local_i = (y_dev / m2) * (W @ y_dev)
        gdf['local_morans_i'] = local_i

        # Classify clusters
        neighbor_mean = W @ y
        clusters = []
        for i in range(n):
            if y[i] > y_mean and neighbor_mean[i] > y_mean:
                clusters.append('High-High')
            elif y[i] < y_mean and neighbor_mean[i] < y_mean:
                clusters.append('Low-Low')
            elif y[i] > y_mean and neighbor_mean[i] < y_mean:
                clusters.append('High-Low')
            else:
                clusters.append('Low-High')

        gdf['spatial_cluster_type'] = clusters
        return gdf


def spatial_autocorrelation_analysis(
    gdf: gpd.GeoDataFrame,
    score_column: str = 'adoption_propensity',
    config: Optional[Dict] = None
) -> Tuple[gpd.GeoDataFrame, Dict]:
    """
    Convenience function for full spatial analysis.

    Args:
        gdf: GeoDataFrame with scores
        score_column: Column name for scores
        config: Configuration dictionary

    Returns:
        Tuple of (enhanced GeoDataFrame, analysis results dict)
    """
    analyzer = SpatialAutocorrelationAnalyzer(config=config)

    # Calculate global Moran's I
    moran_result = analyzer.calculate_morans_i(gdf, score_column)

    # Apply spillover
    gdf = analyzer.apply_spatial_spillover(gdf, score_column)

    # Calculate local Moran's I
    gdf = analyzer.calculate_local_morans_i(gdf, score_column)

    results = {
        'morans_i': moran_result.morans_i,
        'morans_p_value': moran_result.morans_p_value,
        'morans_z_score': moran_result.morans_z_score,
        'interpretation': moran_result.interpretation,
        'is_significant': moran_result.is_significant
    }

    return gdf, results
