"""
ML Validation Module for EV Modelling

Validates expert weights against data-driven feature importance
using machine learning models and SHAP analysis.

Per @eq-weight-validation in paper.qmd:
Alignment = 1 - |w_expert - w_ml| / max(w_expert, w_ml)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import json
from pathlib import Path
import warnings


@dataclass
class WeightAlignment:
    """Alignment between expert and ML-derived weights."""
    feature_name: str
    expert_weight: float
    ml_weight: float
    alignment_score: float
    mean_shap: Optional[float] = None


@dataclass
class MLValidationResult:
    """Results from ML weight validation."""
    alignments: List[WeightAlignment]
    overall_alignment: float
    model_r2: float
    feature_importances: Dict[str, float]


class MLWeightValidator:
    """
    Validates expert weights using ML feature importance.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        config_path: str = "scoring_weights.json"
    ):
        """Initialize ML validator."""
        if config is None:
            config = self._load_config(config_path)

        self.config = config
        self.expert_weights = config.get('demographic_weights', {})

        ml_params = config.get('ml_validation_parameters', {})
        self.n_estimators = ml_params.get('n_estimators', 100)
        self.max_depth = ml_params.get('max_depth', 4)
        self.alignment_threshold = ml_params.get('alignment_threshold', 0.7)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        path = Path(config_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _calculate_alignment(
        self,
        w_expert: float,
        w_ml: float
    ) -> float:
        """Calculate alignment score between weights."""
        if max(w_expert, w_ml) == 0:
            return 1.0 if w_expert == w_ml else 0.0
        return 1 - abs(w_expert - w_ml) / max(w_expert, w_ml)

    def validate_weights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_mapping: Optional[Dict[str, str]] = None
    ) -> MLValidationResult:
        """
        Validate expert weights against ML feature importance.

        Args:
            X: Feature DataFrame with score columns
            y: Target variable (e.g., EV share)
            feature_mapping: Map from X columns to expert weight keys

        Returns:
            MLValidationResult with alignment scores
        """
        from sklearn.ensemble import GradientBoostingRegressor

        # Default mapping
        if feature_mapping is None:
            feature_mapping = {
                'social_grade_score': 'social_grade',
                'education_score': 'education',
                'car_ownership_score': 'car_ownership',
                'housing_score': 'housing'
            }

        # Filter to available features
        available = [c for c in feature_mapping.keys() if c in X.columns]
        X_filtered = X[available].dropna()
        y_filtered = y.loc[X_filtered.index]

        # Train model
        model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        model.fit(X_filtered, y_filtered)

        # Get feature importance
        importances = dict(zip(available, model.feature_importances_))

        # Normalize
        total = sum(importances.values())
        if total > 0:
            ml_weights = {k: v/total for k, v in importances.items()}
        else:
            ml_weights = importances

        # Calculate alignments
        alignments = []
        for ml_col, expert_key in feature_mapping.items():
            if ml_col in ml_weights:
                w_ml = ml_weights[ml_col]
                w_expert = self.expert_weights.get(expert_key, 0)
                alignment = self._calculate_alignment(w_expert, w_ml)
                alignments.append(WeightAlignment(
                    feature_name=expert_key,
                    expert_weight=w_expert,
                    ml_weight=w_ml,
                    alignment_score=alignment
                ))

        overall = np.mean([a.alignment_score for a in alignments])
        r2 = model.score(X_filtered, y_filtered)

        return MLValidationResult(
            alignments=alignments,
            overall_alignment=overall,
            model_r2=r2,
            feature_importances=ml_weights
        )


def ml_weight_validation(
    demographic_data: pd.DataFrame,
    target_column: str,
    config: Optional[Dict] = None
) -> Dict:
    """
    Convenience function for ML weight validation.

    Args:
        demographic_data: DataFrame with score columns
        target_column: Column name for target variable
        config: Configuration dictionary

    Returns:
        Dictionary with validation results
    """
    validator = MLWeightValidator(config=config)

    feature_cols = [
        'social_grade_score', 'education_score',
        'car_ownership_score', 'housing_score'
    ]
    available = [c for c in feature_cols if c in demographic_data.columns]

    if not available or target_column not in demographic_data.columns:
        return {'error': 'Missing required columns'}

    X = demographic_data[available]
    y = demographic_data[target_column]

    result = validator.validate_weights(X, y)

    return {
        'overall_alignment': result.overall_alignment,
        'model_r2': result.model_r2,
        'alignments': {
            a.feature_name: {
                'expert': a.expert_weight,
                'ml': a.ml_weight,
                'alignment': a.alignment_score
            }
            for a in result.alignments
        },
        'feature_importances': result.feature_importances
    }
