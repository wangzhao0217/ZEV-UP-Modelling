"""
Adoption Analysis Module for Frugal EV Analysis

This module provides classes and functions for calculating demographic-based EV adoption propensity
and assessing home charging feasibility based on dwelling characteristics.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_config_path(path_value: Union[str, Path]) -> Path:
    """Resolve config paths from cwd first, then fall back to repo root."""
    path = Path(path_value)
    if path.exists() or path.is_absolute():
        return path

    repo_relative = REPO_ROOT / path
    if repo_relative.exists():
        return repo_relative

    return path


@dataclass
class AdoptionScores:
    """Container for adoption propensity scores and components"""
    overall_score: float
    social_grade_score: float
    education_score: float
    car_ownership_score: float
    housing_score: float
    age_score: float
    economic_activity_score: float
    household_composition_score: float
    population_density_score: float


@dataclass
class InteractionEffects:
    """Container for demographic interaction effects"""
    income_education_synergy: float
    multicar_advantage: float
    age_income_interaction: float
    overall_interaction_multiplier: float


@dataclass
class GeographicAdjustment:
    """Container for geographic context adjustments"""
    density_adjustment: float
    community_engagement_adjustment: float
    rural_penalty: float
    urban_penalty: float
    suburban_bonus: float
    overall_geographic_multiplier: float


@dataclass
class EnhancedAdoptionScore:
    """Enhanced container for adoption propensity scores with interaction effects and geographic adjustments"""
    # Core scores with literature-based weights
    social_grade_score: float          # 35% weight
    education_score: float             # 25% weight  
    car_ownership_score: float         # 20% weight
    housing_score: float               # 20% weight
    
    # Legacy scores (zero weight but maintained for compatibility)
    age_score: float = 0.0
    economic_activity_score: float = 0.0
    household_composition_score: float = 0.0
    population_density_score: float = 0.0
    
    # Interaction effects
    interaction_effects: Optional[InteractionEffects] = None
    
    # Geographic adjustments
    geographic_adjustment: Optional[GeographicAdjustment] = None
    
    # Final scores
    base_adoption_propensity: float = 0.0
    adjusted_adoption_propensity: float = 0.0
    home_charging_feasibility: float = 0.0
    
    # Validation metrics
    validation_confidence: float = 0.0
    pattern_match_score: float = 0.0
    
    @property
    def overall_score(self) -> float:
        """Backward compatibility property for overall score"""
        return self.adjusted_adoption_propensity


class InteractionEffectsEngine:
    """
    Engine for calculating demographic interaction effects in EV adoption propensity.
    
    Implements research-backed interaction effects between demographic factors:
    - Income-education synergy effects
    - Multi-car household advantages
    - Age-income interaction patterns
    """
    
    def __init__(self, interaction_parameters: Optional[Dict[str, float]] = None):
        """
        Initialize interaction effects engine with configuration parameters.
        
        Args:
            interaction_parameters: Dictionary of interaction effect parameters
        """
        # Default interaction parameters based on literature review
        # Values aligned with scoring_weights.json and paper documentation
        # References paper @eq-income-education-multiplier (lines 530-538),
        # @eq-age-income-multiplier (lines 542-553), @eq-multicar-advantage (lines 555-565)
        self.default_parameters = {
            "high_synergy_bonus": 1.60,      # High income + high education (@eq-income-education-multiplier line 532)
            "education_penalty": 0.70,       # High education, low income (@eq-income-education-multiplier line 534)
            "income_penalty": 0.70,          # High income, low education (@eq-income-education-multiplier line 533)
            "compound_penalty": 0.45,        # Low income + low education (@eq-income-education-multiplier line 535)
            "multicar_advantage_high": 1.55, # >50% multi-car households (@eq-multicar-advantage line 559)
            "multicar_advantage_medium": 1.30, # >30% multi-car households (@eq-multicar-advantage line 561)
            "peak_age_bonus": 1.35,          # Peak age (30-44) + high income (@eq-age-income-multiplier line 546)
            "aspirational_score": 0.55,      # Young + low income (@eq-age-income-multiplier line 547)
            "conservative_score": 0.35       # Elderly regardless of income (@eq-age-income-multiplier line 548)
        }
        
        # Use provided parameters or defaults
        self.parameters = interaction_parameters or self.default_parameters
        
        logger.info("Initialized InteractionEffectsEngine with interaction parameters")
    
    def apply_all_interactions(self, 
                             social_grade_score: pd.Series,
                             education_score: pd.Series,
                             car_ownership_score: pd.Series,
                             age_distribution: pd.DataFrame,
                             car_ownership_data: pd.DataFrame) -> pd.Series:
        """
        Apply all interaction effects to base demographic scores.
        
        Args:
            social_grade_score: Social grade/income scores [0,1]
            education_score: Education level scores [0,1]
            car_ownership_score: Car ownership scores [0,1]
            age_distribution: DataFrame with age distribution columns
            car_ownership_data: DataFrame with car ownership category columns
            
        Returns:
            Series of interaction effect multipliers to apply to base scores
        """
        # Start with neutral multiplier
        interaction_multiplier = pd.Series(1.0, index=social_grade_score.index)
        
        # Apply income-education synergy
        income_education_effect = self.calculate_income_education_synergy(
            social_grade_score, education_score
        )
        interaction_multiplier *= income_education_effect
        
        # Apply multi-car household advantage
        multicar_effect = self.calculate_multicar_advantage(
            car_ownership_score, car_ownership_data
        )
        interaction_multiplier *= multicar_effect
        
        # Apply age-income interaction
        age_income_effect = self.calculate_age_income_interaction(
            social_grade_score, age_distribution
        )
        interaction_multiplier *= age_income_effect
        
        logger.info(f"Applied interaction effects - mean multiplier: {interaction_multiplier.mean():.3f}")
        logger.info(f"Interaction effect range: [{interaction_multiplier.min():.3f}, {interaction_multiplier.max():.3f}]")
        
        return interaction_multiplier
    
    def calculate_income_education_synergy(self, 
                                         social_grade_score: pd.Series, 
                                         education_score: pd.Series) -> pd.Series:
        """
        Calculate income-education synergy effects based on literature findings.
        
        Research shows that high income + high education creates synergistic effects,
        while mismatched combinations (high income/low education or vice versa) 
        create penalties, and low income + low education compounds disadvantages.
        
        Args:
            social_grade_score: Social grade/income scores [0,1]
            education_score: Education level scores [0,1]
            
        Returns:
            Series of synergy multipliers [0.8, 1.15]
        """
        synergy_multiplier = pd.Series(1.0, index=social_grade_score.index)
        
        # High income + high education synergy (both scores > 0.7)
        high_both_mask = (social_grade_score > 0.7) & (education_score > 0.7)
        synergy_multiplier.loc[high_both_mask] = self.parameters["high_synergy_bonus"]
        
        # High income, low education penalty (income > 0.7, education < 0.4)
        high_income_low_edu_mask = (social_grade_score > 0.7) & (education_score < 0.4)
        synergy_multiplier.loc[high_income_low_edu_mask] = self.parameters["income_penalty"]
        
        # Low income, high education penalty (income < 0.4, education > 0.7)
        low_income_high_edu_mask = (social_grade_score < 0.4) & (education_score > 0.7)
        synergy_multiplier.loc[low_income_high_edu_mask] = self.parameters["education_penalty"]
        
        # Low income + low education compounding penalty (both < 0.4)
        low_both_mask = (social_grade_score < 0.4) & (education_score < 0.4)
        synergy_multiplier.loc[low_both_mask] = self.parameters["compound_penalty"]
        
        # Log the distribution of synergy effects
        synergy_counts = {
            "high_synergy": high_both_mask.sum(),
            "income_penalty": high_income_low_edu_mask.sum(),
            "education_penalty": low_income_high_edu_mask.sum(),
            "compound_penalty": low_both_mask.sum(),
            "neutral": len(synergy_multiplier) - (high_both_mask.sum() + high_income_low_edu_mask.sum() + 
                                                 low_income_high_edu_mask.sum() + low_both_mask.sum())
        }
        
        logger.info(f"Income-education synergy effects: {synergy_counts}")
        
        return synergy_multiplier
    
    def calculate_multicar_advantage(self, 
                                   car_ownership_score: pd.Series,
                                   car_ownership_data: pd.DataFrame) -> pd.Series:
        """
        Calculate multi-car household advantage effects.
        
        Research shows that areas with high concentrations of multi-car households
        have significantly higher EV adoption feasibility due to reduced range anxiety
        and ability to maintain ICE backup vehicle.
        
        Args:
            car_ownership_score: Base car ownership scores [0,1]
            car_ownership_data: DataFrame with car ownership category columns
            
        Returns:
            Series of multi-car advantage multipliers [1.0, 1.3]
        """
        multicar_multiplier = pd.Series(1.0, index=car_ownership_score.index)
        
        # Find multi-car columns in the data
        multicar_cols = [col for col in car_ownership_data.columns 
                        if 'Two.or.more.cars' in col or 'Two.cars' in col]
        
        if not multicar_cols:
            logger.warning("No multi-car columns found in car ownership data")
            return multicar_multiplier
        
        # Calculate total households for normalization
        all_car_cols = [col for col in car_ownership_data.columns 
                       if any(pattern in col for pattern in ['No.cars', 'One.car', 'Two.or.more.cars', 'Two.cars'])]
        
        if not all_car_cols:
            logger.warning("No car ownership columns found for normalization")
            return multicar_multiplier
        
        # Calculate multi-car household ratio
        multicar_households = car_ownership_data[multicar_cols].sum(axis=1)
        total_households = car_ownership_data[all_car_cols].sum(axis=1)
        
        # Avoid division by zero
        multicar_ratio = multicar_households / (total_households + 1e-6)
        
        # Apply advantages based on multi-car concentration thresholds
        # High concentration areas (>50% multi-car households)
        high_multicar_mask = multicar_ratio > 0.5
        multicar_multiplier.loc[high_multicar_mask] = self.parameters["multicar_advantage_high"]
        
        # Medium concentration areas (>30% multi-car households)
        medium_multicar_mask = (multicar_ratio > 0.3) & (multicar_ratio <= 0.5)
        multicar_multiplier.loc[medium_multicar_mask] = self.parameters["multicar_advantage_medium"]
        
        # Log the distribution of multi-car advantages
        advantage_counts = {
            "high_advantage": high_multicar_mask.sum(),
            "medium_advantage": medium_multicar_mask.sum(),
            "no_advantage": len(multicar_multiplier) - (high_multicar_mask.sum() + medium_multicar_mask.sum())
        }
        
        logger.info(f"Multi-car household advantages: {advantage_counts}")
        logger.info(f"Mean multi-car ratio: {multicar_ratio.mean():.3f}")
        
        return multicar_multiplier
    
    def calculate_age_income_interaction(self, 
                                       social_grade_score: pd.Series,
                                       age_distribution: pd.DataFrame) -> pd.Series:
        """
        Calculate age-income interaction effects based on generational adoption patterns.
        
        Research shows different adoption patterns by age-income combinations:
        - Peak age (30-44) + high income: highest adoption potential
        - Young + low income: aspirational adopters with limited current capacity
        - Elderly + high income: conservative adopters with financial capacity but technology hesitancy
        
        Args:
            social_grade_score: Social grade/income scores [0,1]
            age_distribution: DataFrame with age distribution columns
            
        Returns:
            Series of age-income interaction multipliers [0.4, 1.2]
        """
        age_income_multiplier = pd.Series(1.0, index=social_grade_score.index)
        
        # Find age columns in the data
        peak_age_cols = [col for col in age_distribution.columns 
                        if any(age_pattern in col for age_pattern in ['X30.to.34', 'X35.to.39', 'X40.to.44'])]
        
        young_age_cols = [col for col in age_distribution.columns 
                         if any(age_pattern in col for age_pattern in ['X20.to.24', 'X25.to.29'])]
        
        elderly_age_cols = [col for col in age_distribution.columns 
                           if 'X65.and.over' in col or any(age_pattern in col for age_pattern in 
                              ['X65.to.69', 'X70.to.74', 'X75.to.79', 'X80.to.84', 'X85.and.over'])]
        
        # Calculate total population for normalization
        # Include individual elderly age bands (X65.to.69 ... X85.and.over)
        # so 2022 data with disaggregated elderly columns is counted correctly
        all_age_cols = [col for col in age_distribution.columns 
                       if any(pattern in col for pattern in ['X20.to.24', 'X25.to.29', 'X30.to.34', 'X35.to.39', 
                                                           'X40.to.44', 'X45.to.49', 'X50.to.54', 'X55.to.59',
                                                           'X60.to.64', 'X65.and.over',
                                                           'X65.to.69', 'X70.to.74', 'X75.to.79',
                                                           'X80.to.84', 'X85.and.over'])]
        
        if not all_age_cols:
            logger.warning("No age distribution columns found for age-income interaction")
            return age_income_multiplier
        
        total_population = age_distribution[all_age_cols].sum(axis=1)
        
        # Calculate age group ratios
        peak_age_ratio = pd.Series(0.0, index=social_grade_score.index)
        young_ratio = pd.Series(0.0, index=social_grade_score.index)
        elderly_ratio = pd.Series(0.0, index=social_grade_score.index)
        
        if peak_age_cols:
            peak_age_population = age_distribution[peak_age_cols].sum(axis=1)
            peak_age_ratio = peak_age_population / (total_population + 1e-6)
        
        if young_age_cols:
            young_population = age_distribution[young_age_cols].sum(axis=1)
            young_ratio = young_population / (total_population + 1e-6)
        
        if elderly_age_cols:
            elderly_population = age_distribution[elderly_age_cols].sum(axis=1)
            elderly_ratio = elderly_population / (total_population + 1e-6)
        
        # Apply age-income interaction effects
        
        # Peak age (30-44) + high income: maximum adoption potential
        peak_high_income_mask = (peak_age_ratio > 0.3) & (social_grade_score > 0.7)
        age_income_multiplier.loc[peak_high_income_mask] = self.parameters["peak_age_bonus"]
        
        # Young + low income: aspirational adopters with limited capacity
        young_low_income_mask = (young_ratio > 0.3) & (social_grade_score < 0.4)
        age_income_multiplier.loc[young_low_income_mask] = self.parameters["aspirational_score"]
        
        # Elderly + high income: conservative adopters with financial capacity but tech hesitancy
        elderly_high_income_mask = (elderly_ratio > 0.3) & (social_grade_score > 0.7)
        age_income_multiplier.loc[elderly_high_income_mask] = self.parameters["conservative_score"]
        
        # Log the distribution of age-income interactions
        interaction_counts = {
            "peak_age_high_income": peak_high_income_mask.sum(),
            "young_low_income": young_low_income_mask.sum(),
            "elderly_high_income": elderly_high_income_mask.sum(),
            "neutral": len(age_income_multiplier) - (peak_high_income_mask.sum() + 
                                                   young_low_income_mask.sum() + 
                                                   elderly_high_income_mask.sum())
        }
        
        logger.info(f"Age-income interaction effects: {interaction_counts}")
        
        return age_income_multiplier


class GeographicContextEngine:
    """
    Engine for applying Scottish-specific geographic context adjustments to EV adoption propensity.
    
    Implements research-backed geographic effects:
    - Population density effects (rural penalties, urban constraints, suburban advantages)
    - Community engagement factors (local initiative bonuses)
    - Scottish-specific contextual adjustments
    """
    
    def __init__(self, geographic_parameters: Optional[Dict[str, float]] = None):
        """
        Initialize geographic context engine with configuration parameters.
        
        Args:
            geographic_parameters: Dictionary of geographic adjustment parameters
        """
        # Default geographic parameters based on literature review
        # References paper @eq-density-multiplier (lines 582-591)
        self.default_parameters = {
            "rural_penalty": 0.85,           # Density < 50 people/km² penalty (@eq-density-multiplier line 585)
            "very_rural_penalty": 0.75,      # Density < 20 people/km² penalty (@eq-density-multiplier line 584)
            "remote_penalty": 0.70,          # Density < 10 people/km² penalty (@eq-density-multiplier line 583)
            "high_density_penalty": 0.92,    # Density > 2000 + high flat ratio penalty (@eq-density-multiplier line 587)
            "high_density_moderate_penalty": 0.98,  # Density > 2000, moderate flat ratio (@eq-density-multiplier line 588)
            "suburban_bonus": 1.08,          # Moderate density (200-1000) bonus (@eq-density-multiplier line 586)
            "community_engagement_bonus": 1.25, # High community engagement bonus (@eq-engagement-multiplier lines 597-602)
            "rural_threshold": 50,           # Rural density threshold (people/km²)
            "very_rural_threshold": 20,      # Very rural threshold (people/km²)
            "remote_threshold": 10,          # Remote threshold (people/km²)
            "high_density_threshold": 2000,  # High density threshold (people/km²)
            "suburban_min": 200,             # Suburban density minimum
            "suburban_max": 1000,            # Suburban density maximum
            "flat_ratio_threshold": 0.7,     # High flat ratio threshold
            "community_engagement_threshold": 0.8  # High community engagement threshold
        }
        
        # Use provided parameters or defaults
        self.parameters = geographic_parameters or self.default_parameters
        
        logger.info("Initialized GeographicContextEngine with geographic parameters")
    
    def apply_scottish_adjustments(self, 
                                 base_scores: pd.Series,
                                 population_density: pd.Series,
                                 geographic_context: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """
        Apply Scottish-specific geographic adjustments to base adoption scores.
        
        Literature shows population density has complex effects:
        - EU-wide: negative correlation (high density = lower adoption)
        - England: positive correlation (metropolitan effect)
        - Scotland: context-dependent (infrastructure vs. housing constraints)
        
        Args:
            base_scores: Base adoption propensity scores [0,1]
            population_density: Population density values (people/km²)
            geographic_context: Optional dictionary with additional context data:
                - 'flat_ratio': Proportion of flat/apartment dwellings
                - 'community_engagement_score': Community engagement metric [0,1]
                - 'infrastructure_score': Local charging infrastructure metric [0,1]
                
        Returns:
            Series of geographically adjusted adoption scores
        """
        # Start with base scores
        adjusted_scores = base_scores.copy()
        
        # Initialize geographic context if not provided
        if geographic_context is None:
            geographic_context = {}
        
        # Apply population density effects
        density_adjustment = self._calculate_density_effects(
            population_density, geographic_context.get('flat_ratio')
        )
        adjusted_scores *= density_adjustment
        
        # Apply community engagement effects
        if 'community_engagement_score' in geographic_context:
            engagement_adjustment = self._calculate_community_engagement_effects(
                geographic_context['community_engagement_score']
            )
            adjusted_scores *= engagement_adjustment
        
        # Ensure scores remain in valid [0,1] range
        adjusted_scores = np.clip(adjusted_scores, 0.0, 1.0)
        
        # Log adjustment statistics
        self._log_adjustment_statistics(base_scores, adjusted_scores, density_adjustment)
        
        return adjusted_scores
    
    def _calculate_density_effects(self, 
                                 population_density: pd.Series,
                                 flat_ratio: Optional[pd.Series] = None) -> pd.Series:
        """
        Calculate population density effects on EV adoption propensity.
        
        Args:
            population_density: Population density values (people/km²)
            flat_ratio: Optional proportion of flat/apartment dwellings [0,1]
            
        Returns:
            Series of density adjustment multipliers
        """
        density_multiplier = pd.Series(1.0, index=population_density.index)
        
        # Tiered rural adjustments for range anxiety and infrastructure constraints
        # Remote areas (< 10 people/km²)
        remote_mask = population_density < self.parameters.get("remote_threshold", 10)
        density_multiplier.loc[remote_mask] = self.parameters.get("remote_penalty", 0.70)
        
        # Very rural areas (10-20 people/km²)
        very_rural_mask = (population_density >= self.parameters.get("remote_threshold", 10)) & \
                          (population_density < self.parameters.get("very_rural_threshold", 20))
        density_multiplier.loc[very_rural_mask] = self.parameters.get("very_rural_penalty", 0.75)
        
        # Rural areas (20-50 people/km²)
        rural_mask = (population_density >= self.parameters.get("very_rural_threshold", 20)) & \
                     (population_density < self.parameters["rural_threshold"])
        density_multiplier.loc[rural_mask] = self.parameters["rural_penalty"]
        
        # High-density urban penalty (charging infrastructure dependency)
        high_density_mask = population_density > self.parameters["high_density_threshold"]
        
        # Apply additional penalty if high flat ratio is available
        if flat_ratio is not None:
            high_flat_mask = flat_ratio > self.parameters["flat_ratio_threshold"]
            combined_penalty_mask = high_density_mask & high_flat_mask
            # High density with high flat ratio (@eq-density-multiplier line 587)
            density_multiplier.loc[combined_penalty_mask] = self.parameters["high_density_penalty"]
            
            # Apply moderate penalty for high density without high flat ratio
            # (@eq-density-multiplier line 588)
            high_density_only_mask = high_density_mask & ~high_flat_mask
            density_multiplier.loc[high_density_only_mask] = self.parameters.get(
                "high_density_moderate_penalty", 0.98
            )
        else:
            # Apply standard high-density penalty without flat ratio consideration
            density_multiplier.loc[high_density_mask] = self.parameters["high_density_penalty"]
        
        # Suburban advantage (moderate density sweet spot)
        suburban_mask = (
            (population_density >= self.parameters["suburban_min"]) & 
            (population_density <= self.parameters["suburban_max"])
        )
        density_multiplier.loc[suburban_mask] = self.parameters.get("suburban_bonus", 1.08)
        
        # Log density effect distribution
        density_effects = {
            "remote_penalty": remote_mask.sum(),
            "very_rural_penalty": very_rural_mask.sum(),
            "rural_penalty": rural_mask.sum(),
            "suburban_bonus": suburban_mask.sum(),
            "high_density_penalty": high_density_mask.sum(),
            "neutral": len(density_multiplier) - (remote_mask.sum() + very_rural_mask.sum() + 
                                                  rural_mask.sum() + high_density_mask.sum() + suburban_mask.sum())
        }
        
        logger.info(f"Population density effects: {density_effects}")
        logger.info(f"Mean population density: {population_density.mean():.1f} people/km²")
        
        return density_multiplier
    
    def _calculate_community_engagement_effects(self, 
                                              community_engagement_score: pd.Series) -> pd.Series:
        """
        Calculate community engagement effects on EV adoption propensity.
        
        Areas with high community engagement (like Orkney) often show higher
        EV adoption rates due to local initiatives and social influence.
        
        Args:
            community_engagement_score: Community engagement metric [0,1]
            
        Returns:
            Series of community engagement adjustment multipliers
        """
        engagement_multiplier = pd.Series(1.0, index=community_engagement_score.index)
        
        # High community engagement bonus (Orkney-type areas)
        high_engagement_mask = community_engagement_score > self.parameters["community_engagement_threshold"]
        engagement_multiplier.loc[high_engagement_mask] = self.parameters["community_engagement_bonus"]
        
        # Log community engagement effects
        engagement_effects = {
            "high_engagement_bonus": high_engagement_mask.sum(),
            "neutral": len(engagement_multiplier) - high_engagement_mask.sum()
        }
        
        logger.info(f"Community engagement effects: {engagement_effects}")
        logger.info(f"Mean community engagement score: {community_engagement_score.mean():.3f}")
        
        return engagement_multiplier
    
    def _log_adjustment_statistics(self, 
                                 base_scores: pd.Series,
                                 adjusted_scores: pd.Series,
                                 density_adjustment: pd.Series) -> None:
        """Log statistics about geographic adjustments applied"""
        
        mean_adjustment = (adjusted_scores / base_scores).mean()
        adjustment_range = [
            (adjusted_scores / base_scores).min(),
            (adjusted_scores / base_scores).max()
        ]
        
        logger.info(f"Geographic adjustments applied:")
        logger.info(f"  Mean adjustment factor: {mean_adjustment:.3f}")
        logger.info(f"  Adjustment range: [{adjustment_range[0]:.3f}, {adjustment_range[1]:.3f}]")
        logger.info(f"  Mean density adjustment: {density_adjustment.mean():.3f}")
        
        # Count areas with significant adjustments
        significant_increase = ((adjusted_scores / base_scores) > 1.02).sum()
        significant_decrease = ((adjusted_scores / base_scores) < 0.98).sum()
        
        logger.info(f"  Areas with >2% increase: {significant_increase}")
        logger.info(f"  Areas with >2% decrease: {significant_decrease}")


class AdoptionPropensityCalculator:
    """
    Main analysis engine for calculating demographic-based EV adoption propensity.
    
    Implements literature-validated weighted scoring framework as documented in paper
    Stage 2: Frugal EV Adoption Propensity Assessment (lines 440-621).
    
    Core methodology:
    - Base demographic scoring (@eq-demographic-score, line 450): weighted aggregation
    - Interaction effects (@eq-interaction-adjusted, lines 524-573): synergistic multipliers  
    - Geographic adjustments (@eq-geographic-adjusted, lines 575-612): Scottish context
    - Home charging integration (@eq-final-adoption, lines 614-621): feasibility weighting
    
    Implements weighted scoring using 8 demographic datasets:
    - Social grade (45% weight) - income proxy
    - Education levels (25% weight)
    - Car availability (20% weight)
    - Accommodation type (10% weight)
    - Age/sex distribution (interaction effects)
    - Economic activity (legacy, 0% weight)
    - Household composition (legacy, 0% weight)
    - Population density (geographic adjustments)
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, target_columns_file: str = "target_columns.json", weights_file: str = "scoring_weights.json"):
        """
        Initialize calculator with demographic factor weights.
        
        Loads configuration from scoring_weights.json and target_columns.json to ensure
        alignment with paper specifications.
        
        Args:
            weights: Dictionary of weights for each demographic factor (overrides file)
            target_columns_file: Path to JSON file containing target column definitions
            weights_file: Path to JSON file containing scoring weights
        """
        # Load scoring weights from file (contains all paper-aligned parameters)
        self.scoring_weights = self._load_scoring_weights(weights_file)
        
        # Resolve demographic weights strictly (no silent defaults)
        expected_keys = ['social_grade','education','car_ownership','housing','economic_activity','age','household_composition','population_density']
        if isinstance(weights, dict):
            # Full scoring configuration?
            if ('demographic_weights' in weights) or any(isinstance(v, dict) for v in weights.values()):
                self.scoring_weights = weights
                if 'demographic_weights' not in self.scoring_weights:
                    raise ValueError("scoring_weights dict provided to AdoptionPropensityCalculator is missing 'demographic_weights'. Supply it explicitly.")
                self.weights = {k: float(v) for k, v in self.scoring_weights['demographic_weights'].items()}
            else:
                # Flat dict of numeric weights
                try:
                    self.weights = {k: float(v) for k, v in weights.items()}
                except Exception:
                    raise ValueError("Non-numeric values provided in weights; expected a dict[str, float].")
        else:
            # No explicit weights provided – must be present in weights file
            demo_w = self.scoring_weights.get('demographic_weights')
            if demo_w is None:
                raise ValueError("No 'demographic_weights' found in scoring weights file and no weights provided. Provide weights explicitly.")
            self.weights = {k: float(v) for k, v in demo_w.items()}
        
        # Validate required keys are present
        missing = [k for k in expected_keys if k not in self.weights]
        if missing:
            raise ValueError(f"Missing demographic weight keys: {missing}. Provide all required keys explicitly.")
        
        # Load target columns if file exists
        self.target_columns = self._load_target_columns(target_columns_file)
        
        # Initialize interaction effects engine (@eq-interaction-adjusted lines 567-573)
        interaction_params = self.scoring_weights.get('interaction_parameters', {})
        self.interaction_engine = InteractionEffectsEngine(interaction_params)
        
        # Initialize geographic context engine (@eq-geographic-adjusted lines 606-612)
        geographic_params = self.scoring_weights.get('geographic_parameters', {})
        self.geographic_engine = GeographicContextEngine(geographic_params)
        
        # Initialize validation engine
        validation_params = self.scoring_weights.get('validation_parameters', {})
        self.validation_engine = ValidationEngine(validation_params)
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0, rtol=1e-3):
            logger.warning(f"Weights sum to {weight_sum:.3f}, normalizing to 1.0")
            self.weights = {k: v/weight_sum for k, v in self.weights.items()}
    
    def _load_scoring_weights(self, weights_file: str) -> Dict[str, Any]:
        """Load scoring weights from JSON file"""
        try:
            resolved_path = _resolve_config_path(weights_file)
            if resolved_path.exists():
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    weights_data = json.load(f)
                    logger.info(f"Loaded scoring weights from {resolved_path}")
                    return weights_data
            else:
                logger.warning(
                    f"Scoring weights file {weights_file} not found, using default weights"
                )
                return {}
        except Exception as e:
            logger.warning(f"Error loading scoring weights: {e}, using default weights")
            return {}
    
    def _load_target_columns(self, target_columns_file: str) -> Dict[str, List[str]]:
        """Load target column definitions from JSON file"""
        try:
            resolved_path = _resolve_config_path(target_columns_file)
            if resolved_path.exists():
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    target_data = json.load(f)
                    return target_data.get('columns', {})
            else:
                logger.warning(
                    f"Target columns file {target_columns_file} not found, using fallback column patterns"
                )
                return {}
        except Exception as e:
            logger.warning(f"Error loading target columns: {e}, using fallback patterns")
            return {}
    
    def calculate_propensity_scores(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> gpd.GeoDataFrame:
        """
        Calculate EV adoption propensity scores for all geographic areas.
        
        Args:
            demographics: GeoDataFrame with merged demographic data
            category_lists: Optional dictionary mapping category names to column lists
                          (e.g., {'Age': [...], 'Education': [...]})
            
        Returns:
            GeoDataFrame with adoption propensity scores added
            
        Raises:
            ValueError: If required demographic columns are missing
        """
        logger.info("Calculating EV adoption propensity scores")
        effective_category_lists = category_lists or self.target_columns or None
        
        # Validate input data
        self._validate_demographic_data(demographics, category_lists)
        
        # Create copy to avoid modifying original data
        result = demographics.copy()
        
        # Preprocess data to handle mixed types and missing values
        result = self._preprocess_demographic_data(result)
        
        # Calculate individual component scores
        social_grade_scores = self._calculate_social_grade_scores(result, effective_category_lists)
        education_scores = self._calculate_education_scores(result, effective_category_lists)
        car_ownership_scores = self._calculate_car_ownership_scores(result, effective_category_lists)
        housing_scores = self._calculate_housing_scores(result, effective_category_lists)
        age_scores = self._calculate_age_scores(result, effective_category_lists)
        economic_activity_scores = self._calculate_economic_activity_scores(result, effective_category_lists)
        household_composition_scores = self._calculate_household_composition_scores(result, effective_category_lists)
        population_density_scores = self._calculate_population_density_scores(result, effective_category_lists)
        
        # Calculate base weighted overall scores
        base_scores = (
            social_grade_scores * self.weights['social_grade'] +
            education_scores * self.weights['education'] +
            car_ownership_scores * self.weights['car_ownership'] +
            housing_scores * self.weights['housing'] +
            age_scores * self.weights['age'] +
            economic_activity_scores * self.weights['economic_activity'] +
            household_composition_scores * self.weights['household_composition'] +
            population_density_scores * self.weights['population_density']
        )
        # Expose base (pre-interaction, pre-geo) score for auditing
        result['base_adoption_propensity'] = base_scores
        
        # Apply interaction effects to enhance base scores
        try:
            # Prepare age distribution data for interaction calculations
            age_cols = [col for col in result.columns if any(pattern in col for pattern in 
                       ['X20.to.24', 'X25.to.29', 'X30.to.34', 'X35.to.39', 'X40.to.44', 
                        'X45.to.49', 'X50.to.54', 'X55.to.59', 'X60.to.64', 'X65.and.over'])]
            
            car_ownership_cols = [col for col in result.columns if any(pattern in col for pattern in 
                                 ['No.cars', 'One.car', 'Two.or.more.cars', 'Two.cars'])]
            
            if age_cols and car_ownership_cols:
                age_data = result[age_cols]
                car_data = result[car_ownership_cols]
                
                interaction_multiplier = self.interaction_engine.apply_all_interactions(
                    social_grade_scores, education_scores, car_ownership_scores, 
                    age_data, car_data
                )
                
                # Apply interaction effects to base scores
                overall_scores = base_scores * interaction_multiplier
                
                # Store interaction multiplier for analysis
                result['interaction_multiplier'] = interaction_multiplier
                
                logger.info("Successfully applied interaction effects to adoption scores")
            else:
                logger.warning("Missing age or car ownership data for interaction effects, using base scores")
                overall_scores = base_scores
                result['interaction_multiplier'] = pd.Series(1.0, index=result.index)
                
        except Exception as e:
            logger.warning(f"Error applying interaction effects: {e}, using base scores")
            overall_scores = base_scores
            result['interaction_multiplier'] = pd.Series(1.0, index=result.index)
        
        # Apply geographic context adjustments after interaction effects
        try:
            # Extract population density data for geographic adjustments
            density_cols = [col for col in result.columns 
                           if 'Density..number.of.persons.per.hectare' in col or 'density' in col.lower()]
            
            if density_cols:
                # Extract density (in persons per hectare from census data)
                density_per_hectare = result[density_cols[0]].fillna(0)
                
                # Convert to persons per km² (1 hectare = 0.01 km², so multiply by 100)
                # This ensures alignment with paper thresholds (@eq-density-multiplier):
                # remote <10, very rural <20, rural <50, suburban 200-1000, urban >2000 people/km²
                population_density = density_per_hectare * 100
                
                logger.info(f"Converted density: mean {density_per_hectare.mean():.1f} persons/hectare "
                           f"→ {population_density.mean():.1f} persons/km²")
                
                # Prepare geographic context data
                geographic_context = {}
                
                # Add flat ratio if housing data is available
                flat_cols = [col for col in result.columns 
                            if 'Flat' in col and ('maisonette' in col or 'apartment' in col)]
                house_cols = [col for col in result.columns 
                             if 'house' in col.lower() or 'bungalow' in col.lower()]
                
                if flat_cols and house_cols:
                    total_dwellings = result[flat_cols + house_cols].sum(axis=1)
                    flat_ratio = result[flat_cols].sum(axis=1) / (total_dwellings + 1e-6)
                    geographic_context['flat_ratio'] = flat_ratio
                
                # Apply geographic adjustments
                overall_scores = self.geographic_engine.apply_scottish_adjustments(
                    overall_scores, population_density, geographic_context
                )
                
                logger.info("Successfully applied geographic context adjustments")
            else:
                logger.warning("No population density data found for geographic adjustments")
                
        except Exception as e:
            logger.warning(f"Error applying geographic context adjustments: {e}")
        
        # Normalize to [0,1] range and handle edge cases
        overall_scores = np.clip(overall_scores, 0.0, 1.0)
        overall_scores = np.nan_to_num(overall_scores, nan=0.0)
        
        # Store intermediate geo-adjusted scores (before home charging integration)
        result['geo_adjusted_adoption'] = overall_scores
        result['social_grade_score'] = social_grade_scores
        result['education_score'] = education_scores
        result['car_ownership_score'] = car_ownership_scores
        result['housing_score'] = housing_scores
        result['age_score'] = age_scores
        result['economic_activity_score'] = economic_activity_scores
        result['household_composition_score'] = household_composition_scores
        result['population_density_score'] = population_density_scores
        
        # Integrate home charging feasibility assessment
        try:
            charging_assessor = ImprovedChargingAssessor(self.scoring_weights)
            
            # Extract car ownership data for parking inference
            car_ownership_cols = [col for col in result.columns if any(pattern in col for pattern in 
                                 ['No.cars', 'One.car', 'Two.or.more.cars', 'Two.cars'])]
            
            if car_ownership_cols:
                car_data = result[car_ownership_cols]
                result = charging_assessor.assess_home_charging_feasibility(result, car_data)
                logger.info("Successfully integrated home charging feasibility assessment")
            else:
                logger.warning("No car ownership data found for charging feasibility assessment")
                result = charging_assessor.assess_home_charging_feasibility(result)
                
        except Exception as e:
            logger.warning(f"Error in charging feasibility assessment: {e}")
            # Add default feasibility scores if assessment fails
            result['accommodation_feasibility'] = pd.Series(0.5, index=result.index)
            result['parking_multiplier'] = pd.Series(1.0, index=result.index)
            result['home_charging_feasibility'] = pd.Series(0.5, index=result.index)
        
        # Integrate home charging feasibility into final adoption propensity
        try:
            # Get integration weights from scoring_weights
            integration_weights = self.scoring_weights.get("final_adoption_blend", {
                "adoption_propensity": 0.7,
                "home_charging_feasibility": 0.3
            })
            
            w_adoption = integration_weights["adoption_propensity"]
            w_charging = integration_weights["home_charging_feasibility"]
            
            # Normalize weights to sum to 1.0
            w_sum = w_adoption + w_charging
            w_adoption_norm = w_adoption / w_sum
            w_charging_norm = w_charging / w_sum
            
            # Method selection (additive vs multiplicative)
            integration_method = self.scoring_weights.get("final_adoption_method", "additive")
            
            if integration_method == "multiplicative":
                # Multiplicative integration: adoption × (0.5 + 0.5 × charging)
                charging_modifier = 0.5 + 0.5 * result['home_charging_feasibility']
                final_adoption_propensity = (overall_scores * charging_modifier).clip(0.0, 1.0)
                logger.info("Applied multiplicative integration for final adoption propensity")
            else:
                # Additive integration: weighted average (default)
                final_adoption_propensity = (
                    w_adoption_norm * overall_scores + 
                    w_charging_norm * result['home_charging_feasibility']
                ).clip(0.0, 1.0)
                logger.info(f"Applied additive integration for final adoption propensity (weights: {w_adoption_norm:.2f}/{w_charging_norm:.2f})")
            
            # Set final scores
            result['final_adoption_propensity'] = final_adoption_propensity
            result['adoption_propensity'] = final_adoption_propensity  # Updated to include charging integration
            
            logger.info(f"Final adoption propensity integration completed - mean: {final_adoption_propensity.mean():.3f}")
            
        except Exception as e:
            logger.warning(f"Error integrating home charging into final adoption propensity: {e}")
            # Fallback to geo-adjusted scores only
            result['final_adoption_propensity'] = overall_scores
            result['adoption_propensity'] = overall_scores

        # Convenience alias to match external docs/examples
        if 'parking_multiplier' in result.columns and 'parking_multiplier_calculated' not in result.columns:
            result['parking_multiplier_calculated'] = result['parking_multiplier']
        
        # Validate results using basic validation
        self._validate_propensity_scores(result)
        
        # Run comprehensive validation using ValidationEngine
        try:
            validation_results = self.validation_engine.validate_against_known_patterns(result)
            
            # Add validation results to the dataframe for analysis
            result.attrs['validation_results'] = validation_results
            result.attrs['validation_confidence'] = validation_results.validation_confidence
            result.attrs['pattern_match_score'] = validation_results.pattern_match_score
            
            logger.info(f"Validation completed - Confidence: {validation_results.validation_confidence:.3f}, "
                       f"Pattern Match: {validation_results.pattern_match_score:.3f}")
            
            # Log any validation issues
            if validation_results.issues:
                logger.warning(f"Validation issues found: {len(validation_results.issues)}")
                for issue in validation_results.issues[:3]:  # Log first 3 issues
                    logger.warning(f"  - {issue}")
            
            # Log recommendations
            if validation_results.recommendations:
                logger.info(f"Validation recommendations: {len(validation_results.recommendations)}")
                for rec in validation_results.recommendations[:2]:  # Log first 2 recommendations
                    logger.info(f"  - {rec}")
                    
        except Exception as e:
            logger.warning(f"Error in comprehensive validation: {e}")
            # Add default validation results
            result.attrs['validation_confidence'] = 0.5
            result.attrs['pattern_match_score'] = 0.5
        
        logger.info(f"Calculated adoption propensity for {len(result)} areas")
        logger.info(f"Mean adoption propensity: {overall_scores.mean():.3f}")
        logger.info(f"Score range: [{overall_scores.min():.3f}, {overall_scores.max():.3f}]")
        
        return result

    def compute_adoption_propensity(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> pd.Series:
        """
        Backward-compatible wrapper that returns the final adoption propensity series.
        Many callers expect a simple Series; this wraps calculate_propensity_scores.
        """
        df = self.calculate_propensity_scores(demographics, category_lists)
        if isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
            if 'adoption_propensity' in df.columns:
                return df['adoption_propensity']
            if 'final_adoption_propensity' in df.columns:
                return df['final_adoption_propensity']
        # Fallback: return a neutral 0.5 series
        return pd.Series(0.5, index=demographics.index)
    
    def _validate_demographic_data(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> None:
        """Validate that required demographic columns are present"""
        if category_lists:
            # If category_lists provided, validate that required categories exist
            required_categories = ['Social Grade', 'Education', 'Car Ownership', 'Housing']
            missing_categories = []
            
            for category in required_categories:
                if category not in category_lists or not category_lists[category]:
                    missing_categories.append(category)
            
            if missing_categories:
                logger.warning(f"Missing category lists: {missing_categories}, falling back to pattern matching")
        
        # Fallback pattern validation (more flexible)
        required_patterns = [
            ['AB.', 'AB_', 'social_grade', 'Social Grade'],  # Social grade data
            ['Level', 'education', 'Education'],             # Education data  
            ['car', 'Car', 'ownership'],                     # Car ownership data
            ['House', 'house', 'housing', 'accommodation'],  # Housing type data
            ['aged', 'Age', 'X20', 'X30'],                   # Age data
            ['Full', 'Economic', 'activity'],                # Economic activity data
            ['One', 'household', 'Household'],               # Household composition data
            ['Density', 'density', 'population']             # Population density data
        ]
        
        available_columns = demographics.columns.tolist()
        missing_patterns = []
        
        for pattern_group in required_patterns:
            if not any(any(pattern in col for col in available_columns) for pattern in pattern_group):
                missing_patterns.append(pattern_group[0])  # Report first pattern as missing
        
        if missing_patterns:
            logger.warning(f"Some demographic data patterns not found: {missing_patterns}")
            # Only raise error if critical patterns are missing
            critical_patterns = ['Level', 'car', 'House']
            critical_missing = [p for p in missing_patterns if any(cp in p for cp in critical_patterns)]
            if critical_missing:
                raise ValueError(f"Missing demographic data patterns: {critical_missing}")
    
    def _preprocess_demographic_data(self, demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Preprocess demographic data to handle mixed types and missing values"""
        logger.info("Preprocessing demographic data for numeric operations")
        
        # Get all numeric columns (exclude geo_code, geometry, label, name)
        exclude_cols = ['geo_code', 'geometry', 'label', 'name']
        numeric_cols = [col for col in demographics.columns if col not in exclude_cols]
        
        # Convert to numeric, replacing non-numeric values with 0
        for col in numeric_cols:
            demographics[col] = pd.to_numeric(demographics[col], errors='coerce').fillna(0)
        
        logger.info(f"Preprocessed {len(numeric_cols)} demographic columns")
        return demographics

    def _to_numeric_col(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s):
            return s
        s = s.astype(str).str.strip()
        # common cleanups: thousands separators, blanks, dashes
        s = s.replace({'': np.nan, '-': np.nan, 'na': np.nan, 'n/a': np.nan, 'None': np.nan}, regex=False)
        s = s.str.replace(',', '', regex=False)
        return pd.to_numeric(s, errors='coerce')

    def _calculate_social_grade_scores(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> pd.Series:
        """Calculate social grade scores from grade-specific columns."""
        social_grade_weights = self.scoring_weights.get('social_grade_weights', {})
        scores = pd.Series(0.0, index=demographics.index)

        if category_lists and 'Social Grade' in category_lists:
            social_cols = [col for col in category_lists['Social Grade'] if col in demographics.columns]
        elif self.target_columns and 'Social Grade' in self.target_columns:
            social_cols = [col for col in self.target_columns['Social Grade'] if col in demographics.columns]
            if not social_cols:
                social_cols = [
                    col for col in demographics.columns
                    if col.startswith('AB.') or col.startswith('AB_')
                    or col.startswith('C1.') or col.startswith('C1_')
                    or col.startswith('C2.') or col.startswith('C2_')
                    or col.startswith('DE.') or col.startswith('DE_')
                ]
        else:
            social_cols = [
                col for col in demographics.columns
                if col.startswith('AB.') or col.startswith('AB_')
                or col.startswith('C1.') or col.startswith('C1_')
                or col.startswith('C2.') or col.startswith('C2_')
                or col.startswith('DE.') or col.startswith('DE_')
            ]

        if not social_cols:
            logger.warning("No social grade columns found, returning default score of 0.5")
            return pd.Series(0.5, index=demographics.index)

        total_weighted = pd.Series(0.0, index=demographics.index)
        total_population = pd.Series(0.0, index=demographics.index)

        for grade, weight in social_grade_weights.items():
            grade_cols = [
                col for col in social_cols
                if col.startswith(f'{grade}.') or col.startswith(f'{grade}_')
            ]
            if grade_cols:
                grade_total = demographics[grade_cols].sum(axis=1)
                total_weighted += grade_total * weight
                total_population += grade_total

        if total_population.sum() > 0:
            scores = total_weighted / (total_population + 1e-6)
        else:
            logger.warning("No usable social grade totals found, returning default score of 0.5")
            return pd.Series(0.5, index=demographics.index)

        return np.clip(scores, 0.0, 1.0)
    
    def _calculate_education_scores(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> pd.Series:
        """Calculate education level scores"""
        education_weights = self.scoring_weights.get('education_weights', {})

        scores = pd.Series(0.0, index=demographics.index)

        if category_lists and 'Education' in category_lists:
            education_cols = [col for col in category_lists['Education'] if col in demographics.columns]
        elif self.target_columns and 'Education' in self.target_columns:
            education_cols = [col for col in self.target_columns['Education'] if col in demographics.columns]
            if not education_cols:
                education_cols = [
                    col for col in demographics.columns
                    if 'Level.' in col or 'No.qualifications' in col
                ]
        else:
            education_cols = [
                col for col in demographics.columns
                if 'Level.' in col or 'No.qualifications' in col
            ]

        if not education_cols:
            return np.clip(scores, 0.0, 1.0)

        for level, weight in education_weights.items():
            level_cols = [col for col in education_cols if level in col]
            if level_cols:
                level_total = demographics[level_cols].sum(axis=1)
                scores += level_total * weight

        total_pop = demographics[education_cols].sum(axis=1)
        scores = scores / (total_pop + 1e-6)

        return np.clip(scores, 0.0, 1.0)
    
    def _calculate_car_ownership_scores(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> pd.Series:
        """Calculate car ownership scores"""
        car_weights = self.scoring_weights.get('car_ownership_weights', {})
        
        scores = pd.Series(0.0, index=demographics.index)
        
        # Use category_lists if provided, otherwise fallback to target_columns or pattern matching
        if category_lists and 'Car Ownership' in category_lists:
            car_cols = [col for col in category_lists['Car Ownership'] if col in demographics.columns]
        elif self.target_columns and 'Car Ownership' in self.target_columns:
            car_cols = [col for col in self.target_columns['Car Ownership'] if col in demographics.columns]
            if not car_cols:
                car_cols = [col for col in demographics.columns if any(cat in col for cat in car_weights.keys())]
        else:
            car_cols = [col for col in demographics.columns if any(cat in col for cat in car_weights.keys())]

        if not car_cols:
            return np.clip(scores, 0.0, 1.0)

        for car_category, weight in car_weights.items():
            category_cols = [col for col in car_cols if car_category in col]
            if category_cols:
                car_total = demographics[category_cols].sum(axis=1)
                scores += car_total * weight

        total_households = demographics[car_cols].sum(axis=1)
        scores = scores / (total_households + 1e-6)
        
        return np.clip(scores, 0.0, 1.0)
    
    def _calculate_housing_scores(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> pd.Series:
        """
        Calculate housing-type score (home charging feasibility).

        Score = (Σ type_total * weight) / (Σ type_total), over ONLY the housing types
        that (1) have a weight in `housing_weights`, and (2) have matching columns
        among the provided target columns in category_lists['Housing'].

        - Excludes any columns containing "_Total_".
        - Robust to "Flat.maisonette..." vs "Flat..maisonette..." token differences.
        - Normalises by the sum of the actually used types (e.g., Houses + Flats), not by extra columns.
        """
        import re
        import numpy as np
        import pandas as pd
        import logging

        logger = logging.getLogger(__name__)
        eps = 1e-6

        # --- 1) Load weights and set sensible defaults for common keys ---
        housing_weights = self.scoring_weights.get("housing_weights", {}) or {}
        # Accept both key variants for flats
        if ("Flat..maisonette.or.apartment" not in housing_weights
            and "Flat.maisonette.or.apartment" in housing_weights):
            housing_weights["Flat..maisonette.or.apartment"] = housing_weights["Flat.maisonette.or.apartment"]

        if not isinstance(housing_weights, dict) or not housing_weights:
            logger.warning("No housing_weights provided; returning default 0.6")
            return pd.Series(0.6, index=demographics.index, dtype=float)

        # --- 2) Resolve candidate housing columns (mapping first, then fallback scan) ---
        candidate_cols: list[str] = []
        if category_lists and "Housing" in category_lists and category_lists["Housing"]:
            candidate_cols = [c for c in category_lists["Housing"] if c in demographics.columns]
        elif self.target_columns and "Housing" in self.target_columns:
            candidate_cols = [c for c in self.target_columns["Housing"] if c in demographics.columns]

        if not candidate_cols:
            fragments = [
                "Whole.house.or.bungalow",
                "House.or.bungalow",
                "Flat..maisonette.or.apartment",
                "Flat.maisonette.or.apartment",
            ]
            candidate_cols = [c for c in demographics.columns if any(f in c for f in fragments)]

        # Exclude roll-up totals
        candidate_cols = [c for c in candidate_cols if "_Total_" not in c]

        if not candidate_cols:
            logger.warning("After filtering, no usable housing columns; returning default 0.6")
            return pd.Series(0.6, index=demographics.index, dtype=float)

        # --- 3) Robust matching helpers (collapse dot runs, lowercase) ---
        def norm(s: str) -> str:
            return re.sub(r"\.+", ".", s).lower().strip()

        # Canonical prefixes actually present in the dataset
        CANON_FLAT_PREFIX  = "flat.maisonette.or.apartment"     # normalized token
        CANON_HOUSE_PREFIX = "whole.house.or.bungalow"          # normalized token

        # Map weight keys -> normalized matcher tokens (fallback: normalized key itself)
        def key_to_token(k: str) -> str:
            nk = norm(k)
            if "flat.maisonette.or.apartment" in nk:
                return CANON_FLAT_PREFIX
            if "house.or.bungalow" in nk or "whole.house.or.bungalow" in nk:
                return "house.or.bungalow"
            return nk

        # Build normalized lookup for columns (normalized -> original)
        norm_col_map = {norm(c): c for c in candidate_cols}

        # --- 4) For each weighted housing type, find matching columns (normalized substring) ---
        type_to_cols: dict[str, list[str]] = {}
        for housing_type, weight in housing_weights.items():
            token = key_to_token(housing_type)
            matched_norm = [nc for nc in norm_col_map.keys() if token in nc]
            if matched_norm:
                type_to_cols[housing_type] = [norm_col_map[nc] for nc in matched_norm]
            else:
                logger.debug("No columns matched for key=%r (token=%r)", housing_type, token)

        if not type_to_cols:
            logger.warning("No columns matched any housing_weights keys; returning default 0.6")
            return pd.Series(0.6, index=demographics.index, dtype=float)

        # --- 5) Ensure numeric for all used columns ---
        used_cols = sorted({c for cols in type_to_cols.values() for c in cols})
        demo_num = demographics.copy()
        demo_num[used_cols] = demo_num[used_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # --- 6) Compute weighted numerator and aligned denominator (only used types) ---
        numerator   = pd.Series(0.0, index=demo_num.index, dtype=float)
        denominator = pd.Series(0.0, index=demo_num.index, dtype=float)

        for housing_type, cols in type_to_cols.items():
            type_total = demo_num[cols].sum(axis=1)
            w = float(housing_weights.get(housing_type, 0.0))
            numerator   += type_total * w
            denominator += type_total

        # --- 7) Safe final score; default for zero-denominator rows; clip to [0,1] ---
        score = numerator / (denominator.replace(0, np.nan))
        score = score.fillna(0.6).clip(0.0, 1.0).astype(float)

        return score

    
    def _calculate_age_scores(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> pd.Series:
        """Calculate age distribution scores"""
        age_weights = self.scoring_weights.get('age_weights', {})
        
        scores = pd.Series(0.0, index=demographics.index)

        def _effective_age_weights(age_cols: list[str]) -> dict:
            effective = dict(age_weights)
            has_combined_85 = any('X85.and.over' in col for col in age_cols)
            has_split_85 = any(
                token in col
                for token in ['X85.to.89', 'X90.to.94', 'X95.and.over']
                for col in age_cols
            )
            if has_combined_85 and not has_split_85:
                for token in ['X85.to.89', 'X90.to.94', 'X95.and.over']:
                    effective.pop(token, None)
                effective['X85.and.over'] = 0.05
            return effective
        
        # Use category_lists if provided, otherwise fallback to target_columns or pattern matching
        if category_lists and 'Age' in category_lists:
            age_cols = [col for col in category_lists['Age'] if col in demographics.columns]
            
            for age_group, weight in _effective_age_weights(age_cols).items():
                group_cols = [col for col in age_cols if age_group in col]
                
                if group_cols:
                    age_total = demographics[group_cols].sum(axis=1)
                    scores += age_total * weight
            
            # Normalize by total age population
            if age_cols:
                total_pop = demographics[age_cols].sum(axis=1)
                scores = scores / (total_pop + 1e-6)
        elif self.target_columns and 'Age' in self.target_columns:
            age_cols = [col for col in self.target_columns['Age'] if col in demographics.columns]
            
            for age_group, weight in _effective_age_weights(age_cols).items():
                group_cols = [col for col in age_cols if age_group in col]
                
                if group_cols:
                    age_total = demographics[group_cols].sum(axis=1)
                    scores += age_total * weight
            
            # Normalize by total age population
            if age_cols:
                total_pop = demographics[age_cols].sum(axis=1)
                scores = scores / (total_pop + 1e-6)
        else:
            # Fallback to pattern matching
            for age_group, weight in age_weights.items():
                age_cols = [col for col in demographics.columns if age_group.replace('X', '').replace('.', '.') in col.lower()]
                
                if age_cols:
                    age_total = demographics[age_cols].sum(axis=1)
                    scores += age_total * weight
            
            # Normalize by total population
            total_pop_cols = [col for col in demographics.columns if 'All.people' in col and 'aged' in col]
            if total_pop_cols:
                total_pop = demographics[total_pop_cols[0]]
                scores = scores / (total_pop + 1e-6)
        
        return np.clip(scores, 0.0, 1.0)
    
    def _calculate_economic_activity_scores(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> pd.Series:
        """Calculate economic activity scores"""
        activity_weights = self.scoring_weights.get('economic_activity_weights', {})
        
        scores = pd.Series(0.0, index=demographics.index)
        
        # Use category_lists if provided, otherwise fallback to target_columns or pattern matching
        if category_lists and 'Economic Activity' in category_lists:
            activity_cols = [col for col in category_lists['Economic Activity'] if col in demographics.columns]
            
            for activity, weight in activity_weights.items():
                activity_specific_cols = [col for col in activity_cols if activity in col]
                
                if activity_specific_cols:
                    activity_total = demographics[activity_specific_cols].sum(axis=1)
                    scores += activity_total * weight
            
            # Normalize by total economic activity population
            if activity_cols:
                total_active = demographics[activity_cols].sum(axis=1)
                scores = scores / (total_active + 1e-6)
        elif self.target_columns and 'Economic Activity' in self.target_columns:
            activity_cols = [col for col in self.target_columns['Economic Activity'] if col in demographics.columns]
            
            for activity, weight in activity_weights.items():
                activity_specific_cols = [col for col in activity_cols if activity in col]
                
                if activity_specific_cols:
                    activity_total = demographics[activity_specific_cols].sum(axis=1)
                    scores += activity_total * weight
            
            # Normalize by total economic activity population
            if activity_cols:
                total_active = demographics[activity_cols].sum(axis=1)
                scores = scores / (total_active + 1e-6)
        else:
            # Fallback to pattern matching
            for activity, weight in activity_weights.items():
                activity_cols = [col for col in demographics.columns if activity in col]
                
                if activity_cols:
                    activity_total = demographics[activity_cols].sum(axis=1)
                    scores += activity_total * weight
            
            # Normalize by economically active population
            all_activity_cols = [col for col in demographics.columns if any(act in col for act in activity_weights.keys())]
            if all_activity_cols:
                total_active = demographics[all_activity_cols].sum(axis=1)
                scores = scores / (total_active + 1e-6)
        
        return np.clip(scores, 0.0, 1.0)
    
    def _calculate_household_composition_scores(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> pd.Series:
        """Calculate household composition scores"""
        household_weights = self.scoring_weights.get('household_composition_weights', {})
        
        scores = pd.Series(0.0, index=demographics.index)
        
        # Use category_lists if provided, otherwise fallback to target_columns or pattern matching
        if category_lists and 'Household Composition' in category_lists:
            household_cols = [col for col in category_lists['Household Composition'] if col in demographics.columns]
            logger.info(f"Found {len(household_cols)} household composition columns in category_lists")
            
            total_matches = 0
            for household_type, weight in household_weights.items():
                type_cols = [col for col in household_cols if household_type in col]
                
                if type_cols:
                    household_total = demographics[type_cols].sum(axis=1)
                    scores += household_total * weight
                    total_matches += len(type_cols)
                    logger.info(f"Matched {len(type_cols)} columns for pattern '{household_type}'")
            
            logger.info(f"Total household composition matches: {total_matches}")
            
            # Normalize by total households
            if household_cols:
                total_households = demographics[household_cols].sum(axis=1)
                scores = scores / (total_households + 1e-6)
            else:
                logger.warning("No household composition columns found in category_lists, returning default score of 0.6")
                return pd.Series(0.6, index=demographics.index)  # Default moderate score
        return np.clip(scores, 0.0, 1.0)
    
    def _calculate_population_density_scores(self, demographics: gpd.GeoDataFrame, category_lists: dict = None) -> pd.Series:
        """Calculate population density scores (minimal weight in current model)"""
        # Since population density has 0% weight, return neutral scores
        # But keep implementation for potential future use
        
        # Use category_lists if provided, otherwise fallback to target_columns or pattern matching
        density_col = None
        if category_lists and 'Population Density' in category_lists:
            density_cols = [col for col in category_lists['Population Density'] if col in demographics.columns]

            # Look for the density per hectare column specifically
            for col in density_cols:
                if 'Density..number.of.persons.per.hectare' in col or 'Density.number.of.persons.per.hectare' in col:
                    density_col = col
                    break
        elif self.target_columns and 'Population Density' in self.target_columns:
            density_cols = [col for col in self.target_columns['Population Density'] if col in demographics.columns]

            # Look for the density per hectare column specifically
            for col in density_cols:
                if 'Density..number.of.persons.per.hectare' in col or 'Density.number.of.persons.per.hectare' in col:
                    density_col = col
                    break

        if not density_col:
            # Fallback to pattern matching (also used when mapped columns do not resolve)
            density_cols = [col for col in demographics.columns if 'Density..number.of.persons.per.hectare' in col or 'Density.number.of.persons.per.hectare' in col]
            density_col = density_cols[0] if density_cols else None
        
        if not density_col:
            logger.info("Population density column not found, using neutral score (weight=0% anyway)")
            return pd.Series(0.5, index=demographics.index)
        
        density = demographics[density_col].fillna(0)
        
        # Simplified scoring function (since weight is 0%)
        # Moderate density still considered optimal for EV adoption
        scores = np.where(
            density < 5,    # Rural: 0.4
            0.4,
            np.where(
                density < 20,   # Suburban: 0.8
                0.8,
                np.where(
                    density < 50,   # Urban: 0.6
                    0.6,
                    0.5             # Very dense urban: 0.5
                )
            )
        )
        
        return pd.Series(scores, index=demographics.index)
    
    def _validate_propensity_scores(self, result: gpd.GeoDataFrame) -> None:
        """Validate calculated propensity scores"""
        scores = result['adoption_propensity']
        
        # Check range
        if scores.min() < 0 or scores.max() > 1:
            logger.error(f"Scores out of range [0,1]: [{scores.min():.3f}, {scores.max():.3f}]")
            raise ValueError("Adoption propensity scores must be in range [0,1]")
        
        # Check for excessive NaN values
        nan_count = scores.isna().sum()
        if nan_count > len(scores) * 0.1:  # More than 10% NaN
            logger.warning(f"High number of NaN scores: {nan_count}/{len(scores)}")
        
        # Check for unrealistic distributions
        if scores.std() < 0.01:  # Very low variance
            logger.warning("Very low variance in adoption scores - check input data")
        
        # Validate charging feasibility scores if present
        charging_feasibility_cols = ['accommodation_feasibility', 'parking_multiplier', 'home_charging_feasibility']
        for col in charging_feasibility_cols:
            if col in result.columns:
                col_scores = result[col]
                
                # Check ranges based on column type
                if col == 'parking_multiplier':
                    # Parking multipliers should be [1.0, 1.3]
                    if col_scores.min() < 1.0 or col_scores.max() > 1.3:
                        logger.warning(f"{col} outside expected range [1.0, 1.3]: [{col_scores.min():.3f}, {col_scores.max():.3f}]")
                else:
                    # Other feasibility scores should be [0, 1]
                    if col_scores.min() < 0 or col_scores.max() > 1:
                        logger.warning(f"{col} outside valid range [0, 1]: [{col_scores.min():.3f}, {col_scores.max():.3f}]")
                
                # Check for excessive NaN values
                nan_ratio = col_scores.isna().sum() / len(col_scores)
                if nan_ratio > 0.1:
                    logger.warning(f"High proportion of NaN values in {col}: {nan_ratio:.1%}")
        
        logger.info("Propensity score validation passed")


class HomeChargingAssessor:
    """
    Specialized class for assessing home charging feasibility based on dwelling type
    and parking availability.
    """
    
    def __init__(self, weights_file: str = "scoring_weights.json"):
        """Initialize home charging assessor with parameters from weights file"""
        # Load weights from file
        self.scoring_weights = self._load_scoring_weights(weights_file)
        
        # Charging feasibility by accommodation type
        self.accommodation_feasibility = self.scoring_weights.get('accommodation_feasibility', {})
        
        # Parking availability multipliers by car ownership
        self.parking_multipliers = self.scoring_weights.get('parking_multipliers', {})
    
    def _load_scoring_weights(self, weights_file: str) -> Dict[str, Any]:
        """Load scoring weights from JSON file"""
        try:
            resolved_path = _resolve_config_path(weights_file)
            if resolved_path.exists():
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    weights_data = json.load(f)
                    return weights_data
            else:
                logger.warning(
                    f"Scoring weights file {weights_file} not found, using default weights"
                )
                return {}
        except Exception as e:
            logger.warning(f"Error loading scoring weights: {e}, using default weights")
            return {}
    
    def assess_home_charging_feasibility(self, demographics: gpd.GeoDataFrame, target_columns: Optional[dict] = None) -> gpd.GeoDataFrame:
        """
        Assess home charging feasibility for all geographic areas.
        
        Args:
            demographics: GeoDataFrame with accommodation and car ownership data
            
        Returns:
            GeoDataFrame with home charging feasibility scores added
        """
        logger.info("Assessing home charging feasibility")
        
        result = demographics.copy()
        
        # Preprocess data to handle mixed types and missing values
        result = self._preprocess_data_for_charging_assessment(result)
        
        # Calculate base feasibility from accommodation type
        base_feasibility = self._calculate_accommodation_feasibility(result, target_columns)
        
        # Adjust for parking availability (inferred from car ownership)
        parking_adjustment = self._calculate_parking_availability(result, target_columns)
        
        # Combined feasibility score
        home_charging_feasibility = base_feasibility * parking_adjustment
        home_charging_feasibility = np.clip(home_charging_feasibility, 0.0, 1.0)
        
        result['home_charging_feasibility'] = home_charging_feasibility
        result['accommodation_feasibility'] = base_feasibility
        result['parking_availability'] = parking_adjustment
        
        logger.info(f"Mean home charging feasibility: {home_charging_feasibility.mean():.3f}")
        
        return result
    
    def _calculate_accommodation_feasibility(
        self,
        demographics: gpd.GeoDataFrame,
        category_lists: dict | None = None
    ) -> pd.Series:
        """
        Feasibility from accommodation type, using self.scoring_weights['accommodation_feasibility'].
        Matches both 'Whole.house.or.bungalow' and 'House.or.bungalow', and both
        'Flat..maisonette.or.apartment' and 'Flat.maisonette.or.apartment'.
        """
        # ---- 1) Load feasibility weights from scoring_weights
        feas_raw = dict(self.scoring_weights.get("accommodation_feasibility", {
 "House.or.bungalow":0.92,
 "Flat.maisonette.or.apartment":0.22,
 "Other":0.5,
 }))

        # Canonicalize keys to the column fragments used in your dataset
        canon = {}
        for k, v in feas_raw.items():
            k_stripped = str(k).strip()
            if k_stripped in (
                "Whole.house.or.bungalow", "Whole house or bungalow",
                "House.or.bungalow", "House or bungalow"
            ):
                canon["House.or.bungalow"] = float(v)
            elif k_stripped in (
                "Flat..maisonette.or.apartment", "Flat.maisonette.or.apartment",
                "Flat, maisonette or apartment"
            ):
                canon["Flat..maisonette.or.apartment"] = float(v)
            elif k_stripped == "Other":
                canon["Other"] = float(v)
            else:
                # keep unknowns as-is in case you have custom types
                canon[k_stripped] = float(v)

        default_other = float(canon.get("Other", 0.5))

        # ---- 2) Decide which housing columns to use
        if category_lists and "Housing" in category_lists:
            housing_cols = [c for c in category_lists["Housing"] if c in demographics.columns]
        else:
            housing_cols = []

        if not housing_cols:
            # Fallback: scan for known fragments
            fragments = [
                "Whole.house.or.bungalow",
                "House.or.bungalow",
                "Flat..maisonette.or.apartment",
                "Flat.maisonette.or.apartment",
            ]
            housing_cols = [c for c in demographics.columns if any(f in c for f in fragments)]

        if not housing_cols:
            logger.warning("Accommodation feasibility: no housing columns found; returning default %.2f", default_other)
            return pd.Series(default_other, index=demographics.index)

        # ---- 3) Accumulate feasibility
        scores = pd.Series(0.0, index=demographics.index)

        for accom_frag, feasibility in canon.items():
            if accom_frag == "Other":
                continue
            type_cols = [c for c in housing_cols if accom_frag in c]
            if not type_cols:
                continue
            accom_total = demographics[type_cols].sum(axis=1).astype(float)
            scores = scores.add(accom_total * float(feasibility), fill_value=0.0)

        # ---- 4) Normalize by total households across all housing columns
        total_households = demographics[housing_cols].sum(axis=1).astype(float)
        scores = (scores / (total_households + 1e-6)).fillna(0.0).clip(0.0, 1.0)

        logger.info(
            "Accommodation feasibility: used %d housing cols; score range [%.3f, %.3f]",
            len(housing_cols), scores.min(), scores.max()
        )
        return scores


    
    def _preprocess_data_for_charging_assessment(self, demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Preprocess demographic data for charging assessment"""
        # Get all numeric columns (exclude geo_code, geometry, label, name)
        exclude_cols = ['geo_code', 'geometry', 'label', 'name']
        numeric_cols = [col for col in demographics.columns if col not in exclude_cols]
        
        # Convert to numeric, replacing non-numeric values with 0
        for col in numeric_cols:
            demographics[col] = pd.to_numeric(demographics[col], errors='coerce').fillna(0)
        
        return demographics
        
    def _calculate_parking_availability(
        self,
        demographics: gpd.GeoDataFrame,
        target_columns: Optional[dict] = None,
        *,
        as_multiplier: bool = False,   # set True to return the widened multiplier directly
    ) -> pd.Series:
        """
        Parking availability from car-ownership mix.

        Base score ∈ [0, 1]:
            score = Σ(share(category) * multiplier(category))

        Then, optionally map to a multiplier range:
            multiplier = min_mult + score * (max_mult - min_mult)

        Config used:
        - self.scoring_weights['parking_multipliers']  (e.g., {"No.cars":0.1,"One.car":0.7,"Two.or.more.cars":0.95})
        - self.scoring_weights['parking_multiplier_range']  (optional, e.g., {"min":0.7,"max":1.3})
        """
        import numpy as np
        import pandas as pd
        import logging

        logger = logging.getLogger(__name__)
        eps = 1e-6

        # --- 1) Pull the exact columns provided (defensive: exclude roll-ups) ---
        car_cols = [c for c in (target_columns or {}).get("Car Ownership", []) if c in demographics.columns]
        car_cols = [c for c in car_cols if "_Total_" not in c]

        if not car_cols:
            fallback_patterns = [
                "No.cars.or.vans",
                "One.car.or.van",
                "Two.or.more.cars.or.vans",
                "Two.cars.or.vans",
                "Three.or.more.cars.or.vans",
            ]
            car_cols = [
                c for c in demographics.columns
                if "_Total_" not in c and any(p in c for p in fallback_patterns)
            ]

        if not car_cols:
            logger.warning("Parking availability: no usable 'Car Ownership' columns; returning neutral 0.6")
            base = pd.Series(0.6, index=demographics.index, dtype=float)
            return self._map_score_to_multiplier_if_requested(base, as_multiplier)

        # --- 2) Load category multipliers from config ---
        multipliers_cfg = dict(
 self.scoring_weights.get("parking_multipliers")
 or self.scoring_weights.get("parking_multipliers_DEPRECATED")
 or {
 "No.cars":0.1,
 "One.car":0.7,
 "Two.or.more.cars":0.95,
 }
 )
        if not multipliers_cfg:
            logger.warning("Parking availability: no 'parking_multipliers' in config; returning neutral 0.6")
            base = pd.Series(0.6, index=demographics.index, dtype=float)
            return self._map_score_to_multiplier_if_requested(base, as_multiplier)

        # Robust alias map (covers common label variants found in census outputs)
        alias_map = {
            "No.cars": ("No.cars", "No.cars.or.vans"),
            "One.car": ("One.car", "One.car.or.van"),
            # Treat 2+ buckets together
            "Two.or.more.cars": ("Two.or.more.cars", "Two.or.more.cars.or.vans", "Two.cars.or.vans", "Three.or.more.cars.or.vans"),
        }

        # --- 3) Ensure numeric for all candidate columns ---
        demo = demographics.copy()
        demo[car_cols] = demo[car_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # --- 4) Accumulate weighted totals using only matched categories ---
        weighted = pd.Series(0.0, index=demo.index, dtype=float)
        total    = pd.Series(0.0, index=demo.index, dtype=float)

        any_match = False
        for cat, mult in multipliers_cfg.items():
            patterns = alias_map.get(cat, (cat,))  # fall back to the raw key if no alias
            type_cols = [c for c in car_cols if any(p in c for p in patterns)]
            if not type_cols:
                continue
            any_match = True
            cat_total = demo[type_cols].sum(axis=1).astype(float)
            weighted += cat_total * float(mult)
            total    += cat_total

        if not any_match or (total <= eps).all():
            logger.warning("Parking availability: no matching buckets (or zero totals); returning neutral 0.6")
            base = pd.Series(0.6, index=demo.index, dtype=float)
        else:
            base = (weighted / (total + eps)).clip(0.0, 1.0)

        # --- 5) Optionally map base score to a wider multiplier range (e.g., [0.7, 1.3]) ---
        return self._map_score_to_multiplier_if_requested(base, as_multiplier)


    def _map_score_to_multiplier_if_requested(self, score: pd.Series, as_multiplier: bool) -> pd.Series:
        """
        Helper: if as_multiplier=True, map score∈[0,1] to a multiplier range from config.
        Falls back to {'min': 0.7, 'max': 1.3} if not provided.
        """
        if not as_multiplier:
            return score

        cfg = getattr(self, 'scoring_weights', {}) or {}
        rng = dict(cfg.get("parking_multiplier_range", {}))
        min_mult = float(rng.get("min", 0.7))  # widened default
        max_mult = float(rng.get("max", 1.3))

        # Linear map: min + s*(max-min)
        mult = min_mult + score * (max_mult - min_mult)
        return mult.clip(min_mult, max_mult).astype(float)


def validate_adoption_scores(adoption_data: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Validate adoption propensity scores for consistency and realism.
    
    Args:
        adoption_data: GeoDataFrame with adoption propensity scores
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating adoption propensity scores")
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'statistics': {}
    }
    
    if 'adoption_propensity' not in adoption_data.columns:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Missing adoption_propensity column")
        return validation_results
    
    scores = adoption_data['adoption_propensity']
    
    # Range validation
    if scores.min() < 0 or scores.max() > 1:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Scores out of range [0,1]: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Distribution validation
    validation_results['statistics'] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'nan_count': scores.isna().sum(),
        'zero_count': (scores == 0).sum(),
        'one_count': (scores == 1).sum()
    }
    
    # Check for unrealistic distributions
    if scores.std() < 0.01:
        validation_results['issues'].append("Very low variance in scores")
    
    if scores.isna().sum() > len(scores) * 0.1:
        validation_results['issues'].append(f"High proportion of NaN values: {scores.isna().sum()}/{len(scores)}")
    
    # Check component scores if available
    component_cols = [col for col in adoption_data.columns if col.endswith('_score')]
    for col in component_cols:
        component_scores = adoption_data[col]
        if component_scores.min() < 0 or component_scores.max() > 1:
            validation_results['issues'].append(f"{col} out of range [0,1]")
    
    if validation_results['issues']:
        validation_results['is_valid'] = False
    
    logger.info(f"Validation {'passed' if validation_results['is_valid'] else 'failed'}")
    
    return validation_results

class ImprovedChargingAssessor:
    """
    Enhanced home charging feasibility assessment with improved parking inference
    and accommodation-based feasibility scoring.
    
    Implements research-backed accommodation feasibility weights and multi-car
    household parking availability inference as specified in the literature review.
    """
    
    def __init__(self, scoring_weights: Optional[Dict[str, Any]] = None):
        """
        Initialize ImprovedChargingAssessor with enhanced scoring parameters.
        
        Args:
            scoring_weights: Dictionary containing accommodation feasibility weights
                           and parking multipliers from scoring_weights.json
        """
        # Load scoring weights from file if not provided
        if scoring_weights is None:
            scoring_weights = self._load_scoring_weights()
        
        # Persist scoring configuration for later use (validation, ranges)
        self.scoring_weights = scoring_weights

        # Extract weights from scoring_weights.json configuration
        # References paper @eq-accommodation-feasibility (line 495), 
        # @eq-parking-multiplier (lines 503-504), @eq-home-charging-feasibility-raw (line 515)
        self.accommodation_weights = scoring_weights.get('accommodation_feasibility', {
            "House.or.bungalow": 0.92,       # @eq-accommodation-feasibility w_h,acc = 0.92
            "Flat.maisonette.or.apartment": 0.22,  # @eq-accommodation-feasibility w_f,acc = 0.22
            "Other": 0.5                     # @eq-accommodation-feasibility w_o,acc = 0.50
        })
        
        # Legacy parking multipliers (DEPRECATED - only used by old HomeChargingAssessor)
        # Current implementation uses parking_inference method below
        self.parking_multipliers = scoring_weights.get('parking_multipliers', 
            scoring_weights.get('parking_multipliers_DEPRECATED', {
                "No.cars": 0.1,
                "One.car": 0.7,
                "Two.or.more.cars": 0.95
            })
        )
        
        # Parking inference parameters from @eq-parking-multiplier (line 503)
        # This is the ACTIVE method used by ImprovedChargingAssessor
        self.parking_inference = scoring_weights.get('parking_inference', {
            "bonus_rate": 0.3,              # @eq-parking-multiplier: 0.3 × (TC_i/TH_i)
            "maximum_multiplier": 1.3       # @eq-parking-multiplier: min(..., 1.3)
        })
        
        # Home charging blend weights from @eq-home-charging-feasibility-raw (line 515)
        self.home_charging_blend = scoring_weights.get('home_charging_blend', {
            "accommodation": 0.2,            # @eq-home-charging-feasibility-raw w_a = 0.2
            "parking": 0.8                   # @eq-home-charging-feasibility-raw w_p = 0.8
        })
            
        logger.info("Initialized ImprovedChargingAssessor with weights from scoring_weights.json")
    
    def _load_scoring_weights(self, weights_file: str = "scoring_weights.json") -> Dict[str, Any]:
        """Load scoring weights from JSON file"""
        try:
            resolved_path = _resolve_config_path(weights_file)
            if resolved_path.exists():
                with open(resolved_path, 'r', encoding='utf-8') as f:
                    weights_data = json.load(f)
                    logger.info(f"Loaded scoring weights from {resolved_path}")
                    return weights_data
            else:
                logger.warning(
                    f"Scoring weights file {weights_file} not found, using minimal defaults"
                )
                return {}
        except Exception as e:
            logger.warning(f"Error loading scoring weights: {e}, using minimal defaults")
            return {}
    
    def calculate_parking_inference(self, car_ownership_data: pd.DataFrame) -> pd.Series:
        """
        Infer parking availability from car ownership patterns with multi-car advantage.
        
        Research shows that multi-car households likely have better parking access,
        which is crucial for home charging feasibility. This method applies parking
        multipliers based on multi-car household ratios.
        
        Implements @eq-parking-multiplier (line 503): M_i = min(1.0 + 0.3 × TC_i/TH_i, 1.3)
        
        Args:
            car_ownership_data: DataFrame with car ownership category columns
            
        Returns:
            Series of parking availability multipliers [1.0, max_multiplier]
        """
        logger.info("Calculating parking availability inference from car ownership patterns")
        
        # Get parking inference parameters from config
        bonus_rate = self.parking_inference.get("bonus_rate", 0.3)
        max_multiplier = self.parking_inference.get("maximum_multiplier", 1.3)
        
        # Find car ownership columns in the data
        car_cols = [col for col in car_ownership_data.columns 
                   if any(pattern in col for pattern in ['No.cars', 'One.car', 'Two.or.more.cars', 'Two.cars'])]
        
        if not car_cols:
            logger.warning("No car ownership columns found, using default parking multiplier")
            return pd.Series(1.0, index=car_ownership_data.index)
        
        # Find multi-car columns specifically (TC_i in @eq-parking-multiplier)
        multicar_cols = [col for col in car_cols 
                        if 'Two.or.more.cars' in col or 'Two.cars' in col]
        
        if not multicar_cols:
            logger.warning("No multi-car columns found, using default parking multiplier")
            return pd.Series(1.0, index=car_ownership_data.index)
        
        # Calculate total households for normalization (TH_i in @eq-parking-multiplier)
        total_households = car_ownership_data[car_cols].sum(axis=1)
        
        # Calculate multi-car household ratio (TC_i/TH_i in @eq-parking-multiplier)
        multicar_households = car_ownership_data[multicar_cols].sum(axis=1)
        multicar_ratio = multicar_households / (total_households + 1e-6)  # Avoid division by zero
        
        # Apply parking multiplier: M_i = 1.0 + bonus_rate × (TC_i/TH_i)
        # Multi-car households likely have better parking (driveways, garages)
        parking_multiplier = 1.0 + (multicar_ratio * bonus_rate)
        
        # Cap at maximum multiplier: M_i = min(M_i, max_multiplier)
        parking_multiplier = np.minimum(parking_multiplier, max_multiplier)
        
        # Log statistics
        logger.info(f"Mean multi-car ratio: {multicar_ratio.mean():.3f}")
        logger.info(f"Parking inference params: bonus_rate={bonus_rate}, max_multiplier={max_multiplier}")
        logger.info(f"Mean parking multiplier: {parking_multiplier.mean():.3f}")
        logger.info(f"Parking multiplier range: [{parking_multiplier.min():.3f}, {parking_multiplier.max():.3f}]")
        
        return pd.Series(parking_multiplier, index=car_ownership_data.index)
    
    def assess_home_charging_feasibility(self, 
                                       demographics: gpd.GeoDataFrame,
                                       car_ownership_data: Optional[pd.DataFrame] = None,
                                       target_columns: Optional[Dict[str, List[str]]] = None) -> gpd.GeoDataFrame:
        """
        Assess home charging feasibility using enhanced accommodation weights
        and parking availability inference.
        
        Args:
            demographics: GeoDataFrame with demographic and housing data
            car_ownership_data: Optional DataFrame with car ownership data for parking inference
                              If None, will extract from demographics
            
        Returns:
            GeoDataFrame with home charging feasibility scores added
        """
        logger.info("Assessing home charging feasibility with enhanced methodology")
        
        result = demographics.copy()
        
        # Extract car ownership data if not provided
        if car_ownership_data is None:
            car_cols = [col for col in demographics.columns 
                       if any(pattern in col for pattern in ['No.cars', 'One.car', 'Two.or.more.cars', 'Two.cars'])]
            
            if car_cols:
                car_ownership_data = demographics[car_cols]
            else:
                logger.warning("No car ownership data found for parking inference")
                car_ownership_data = pd.DataFrame(index=demographics.index)
        
        # Calculate accommodation-based feasibility scores
        accommodation_scores = self._calculate_accommodation_feasibility(demographics, target_columns=target_columns)
        
        # Calculate parking availability multipliers
        parking_multipliers = self.calculate_parking_inference(car_ownership_data)
        
        # Derive parking score in [0,1]
        # Priority: use existing normalized 'parking_availability' if present;
        # otherwise normalize any 'parking_multiplier' (using config range if provided),
        # otherwise use derived parking_multipliers from car ownership.
        if 'parking_availability' in result.columns:
            parking_score = pd.to_numeric(result['parking_availability'], errors='coerce').fillna(0.0).clip(0.0, 1.0)
            raw_parking_signal = result['parking_availability']
        else:
            # Choose raw signal
            if 'parking_multiplier' in result.columns:
                raw_parking_signal = pd.to_numeric(result['parking_multiplier'], errors='coerce').fillna(0.0)
            else:
                raw_parking_signal = parking_multipliers

            # Normalize raw signal to [0,1]
            rng_cfg = self.scoring_weights.get('parking_signal_range', {})
            pmin = rng_cfg.get('min', None)
            pmax = rng_cfg.get('max', None)
            if pmin is None or pmax is None:
                # dynamic robust min/max (1st–99th percentile) to reduce outlier effects
                try:
                    pmin, pmax = np.nanpercentile(raw_parking_signal, [1, 99])
                except Exception:
                    pmin, pmax = float(np.nanmin(raw_parking_signal)), float(np.nanmax(raw_parking_signal))

            # Avoid zero division
            if pmax is None or pmin is None or pmax <= pmin:
                parking_score = pd.Series(0.0, index=result.index)
            else:
                parking_score = ((raw_parking_signal - pmin) / (pmax - pmin)).clip(0.0, 1.0)
        w_acc = float(self.home_charging_blend.get("accommodation", 0.2))
        w_park = float(self.home_charging_blend.get("parking", 0.8))
        # Normalize weights to sum to 1
        w_sum = max(w_acc + w_park, 1e-9)
        w_acc /= w_sum
        w_park /= w_sum
        # Raw blended score - direct weighted average without max normalization
        # This preserves the effect of blend weights on the final score
        home_charging_feasibility = (w_acc * accommodation_scores) + (w_park * parking_score)
        # Clip to [0, 1] range (both inputs are already in [0, 1])
        home_charging_feasibility = home_charging_feasibility.clip(0.0, 1.0)
        
        # Add results to dataframe
        result['accommodation_feasibility'] = accommodation_scores
        result['parking_multiplier'] = raw_parking_signal if 'raw_parking_signal' in locals() else parking_multipliers
        result['parking_availability'] = parking_score
        result['home_charging_feasibility'] = home_charging_feasibility
        
        # Validate results
        self._validate_feasibility_scores(result)
        
        logger.info(f"Calculated home charging feasibility for {len(result)} areas")
        logger.info(f"Mean feasibility score: {home_charging_feasibility.mean():.3f}")
        logger.info(f"Feasibility range: [{home_charging_feasibility.min():.3f}, {home_charging_feasibility.max():.3f}]")
        
        return result
    
    def _calculate_accommodation_feasibility(self, demographics: gpd.GeoDataFrame, *, car_owning_only: bool = True, target_columns: Optional[Dict[str, List[str]]] = None) -> pd.Series:
        """
        Accommodation-based charging feasibility using precise column matching.
        If car_owning_only=True, ignores 'No cars or vans' buckets.
        """

        import re
        import numpy as np
        import pandas as pd

        # Canonical prefixes
        HOUSE_PREFIX = r"^Whole\.house\.or\.bungalow\."
        FLAT_PREFIX  = r"^Flat\.\.maisonette\.or\.apartment\.\.or\.mobile\.temporary\.accommodation\."
        OTHER_PREFIX = r"^Other\.accommodation\."

        # Car-ownership filters
        car_blocks = [
            r"_Number\.of\.cars\.or\.vans\.in\.household\.\.One\.car\.or\.van_",
            r"_Number\.of\.cars\.or\.vans\.in\.household\.\.Two\.or\.more\.cars\.or\.vans_",
        ]
        if not car_owning_only:
            car_blocks = car_blocks + [r"_Number\.of\.cars\.or\.vans\.in\.household\.\.No\.cars\.or\.vans_"]

        def keep_col(col: str, root_prefix: str) -> bool:
            if not re.search(root_prefix, col):
                return False
            # Must belong to one of the selected car-ownership blocks
            return any(b in col for b in (block.replace("\\", "") for block in car_blocks))

        if target_columns and 'Housing' in target_columns:
            housing_cols_list = [c for c in target_columns['Housing'] if c in demographics.columns]
            house_cols = [c for c in housing_cols_list if re.search(HOUSE_PREFIX, c)]
            flat_cols  = [c for c in housing_cols_list if re.search(FLAT_PREFIX, c)]
            other_cols = [c for c in housing_cols_list if re.search(OTHER_PREFIX, c)]
        else:
            house_cols = [c for c in demographics.columns if keep_col(c, HOUSE_PREFIX)]
            flat_cols  = [c for c in demographics.columns if keep_col(c, FLAT_PREFIX)]
            other_cols = [c for c in demographics.columns if keep_col(c, OTHER_PREFIX)]

        # Coerce numeric
        use_cols = house_cols + flat_cols + other_cols
        if use_cols:
            demographics[use_cols] = demographics[use_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # Weights (from config, with sensible defaults)
        w_house = float(self.accommodation_weights.get("House.or.bungalow", 0.92))
        w_flat  = float(self.accommodation_weights.get("Flat.maisonette.or.apartment", 0.22))
        w_other = float(self.accommodation_weights.get("Other", 0.50))

        # Weighted sum
        scores = pd.Series(0.0, index=demographics.index)
        total  = pd.Series(0.0, index=demographics.index)

        if house_cols:
            hsum = demographics[house_cols].sum(axis=1)
            scores += hsum * w_house
            total  += hsum
        if flat_cols:
            fsum = demographics[flat_cols].sum(axis=1)
            scores += fsum * w_flat
            total  += fsum
        if other_cols:
            osum = demographics[other_cols].sum(axis=1)
            scores += osum * w_other
            total  += osum

        # Normalise and clip
        out = (scores / total.replace(0, np.nan)).fillna(0.0).clip(0.0, 1.0)
        return out

    
    def _validate_feasibility_scores(self, result: gpd.GeoDataFrame) -> None:
        """Validate calculated feasibility scores"""
        
        # Validate accommodation feasibility
        if 'accommodation_feasibility' in result.columns:
            acc_scores = result['accommodation_feasibility']
            if acc_scores.min() < 0 or acc_scores.max() > 1:
                logger.error(f"Accommodation feasibility out of range [0,1]: [{acc_scores.min():.3f}, {acc_scores.max():.3f}]")
                raise ValueError("Accommodation feasibility scores must be in range [0,1]")
        
        # Validate parking multiplier/signal (range may be project-specific)
        if 'parking_multiplier' in result.columns:
            parking_mult = pd.to_numeric(result['parking_multiplier'], errors='coerce')
            cfg = getattr(self, 'scoring_weights', {}) or {}
            rng_cfg = cfg.get('parking_multiplier_range') or cfg.get('parking_signal_range')
            if isinstance(rng_cfg, dict) and 'min' in rng_cfg and 'max' in rng_cfg:
                mn, mx = float(rng_cfg['min']), float(rng_cfg['max'])
                if (parking_mult.min() < mn) or (parking_mult.max() > mx):
                    logger.warning(
                        f"Parking multiplier outside configured range [{mn}, {mx}]: "
                        f"[{parking_mult.min():.3f}, {parking_mult.max():.3f}]"
                    )
            else:
                # No configured range: ensure finite and non-negative; warn on anomalies
                if not np.isfinite(parking_mult).all():
                    logger.warning("Non-finite values detected in parking_multiplier")
                if (parking_mult < 0).any():
                    logger.warning("Negative values detected in parking_multiplier; expected non-negative signal")
        
        # Validate home charging feasibility
        if 'home_charging_feasibility' in result.columns:
            charging_scores = result['home_charging_feasibility']
            if charging_scores.min() < 0 or charging_scores.max() > 1:
                logger.error(f"Home charging feasibility out of range [0,1]: [{charging_scores.min():.3f}, {charging_scores.max():.3f}]")
                raise ValueError("Home charging feasibility scores must be in range [0,1]")
        
        logger.info("Feasibility score validation passed")


@dataclass
class ValidationResults:
    """Container for validation results and metrics"""
    overall_validation_score: float
    early_adopter_profile_match: Dict[str, float]
    geographic_correlation: Dict[str, float]
    demographic_distribution: Dict[str, float]
    validation_confidence: float
    pattern_match_score: float
    issues: List[str]
    recommendations: List[str]


@dataclass
class EarlyAdopterProfile:
    """Expected characteristics of early EV adopters based on literature"""
    high_social_grade_threshold: float = 0.7
    high_education_threshold: float = 0.7
    multicar_threshold: float = 0.8
    house_dwelling_threshold: float = 0.8
    target_profile_match: float = 0.7  # Target match score for validation


class ValidationEngine:
    """
    Engine for validating EV adoption propensity results against known patterns and literature.
    
    Implements comprehensive validation including:
    - Early adopter profile validation against literature characteristics
    - Geographic pattern validation against known high-adoption regions
    - Demographic distribution validation for realism
    - Correlation validation with actual EV registration data where available
    """
    
    def __init__(self, validation_parameters: Optional[Dict[str, Any]] = None):
        """
        Initialize validation engine with configuration parameters.
        
        Args:
            validation_parameters: Dictionary of validation thresholds and parameters
        """
        # Default validation parameters based on literature review
        self.default_parameters = {
            "r_squared_target": 0.8,           # Target correlation with actual data
            "profile_match_threshold": 0.7,    # Target early adopter profile match
            "geographic_correlation": 0.75,    # Target geographic correlation
            "high_score_threshold": 0.7,       # Threshold for high-scoring areas
            "low_score_threshold": 0.3,        # Threshold for low-scoring areas
            "variance_threshold": 0.01,        # Minimum acceptable variance
            "nan_threshold": 0.1,              # Maximum acceptable NaN proportion
            "confidence_weights": {             # Weights for confidence calculation
                "profile_match": 0.4,
                "geographic_correlation": 0.3,
                "distribution_realism": 0.3
            }
        }
        
        # Use provided parameters or defaults
        self.parameters = validation_parameters or self.default_parameters
        
        # Initialize early adopter profile expectations
        self.early_adopter_profile = EarlyAdopterProfile()
        
        # Known high-adoption regions for geographic validation
        self.known_high_adoption_regions = [
            "Edinburgh", "Glasgow", "Aberdeen", "Stirling", "Perth",
            "East Renfrewshire", "East Dunbartonshire", "Aberdeenshire"
        ]
        
        logger.info("Initialized ValidationEngine with validation parameters")
    
    def validate_against_known_patterns(self, results_df: gpd.GeoDataFrame) -> ValidationResults:
        """
        Validate results against known UK/Scottish adoption patterns and literature findings.
        
        This is the main validation method that orchestrates all validation checks
        and produces a comprehensive validation report.
        
        Args:
            results_df: GeoDataFrame with calculated adoption propensity scores
            
        Returns:
            ValidationResults object with comprehensive validation metrics
        """
        logger.info("Starting comprehensive validation against known patterns")
        
        # Initialize validation results
        issues = []
        recommendations = []
        
        # Validate basic data quality first
        basic_validation = self._validate_basic_data_quality(results_df)
        if not basic_validation["is_valid"]:
            issues.extend(basic_validation["issues"])
        
        # Perform early adopter profile validation
        try:
            profile_validation = self._check_early_adopter_profile(results_df)
            logger.info("Early adopter profile validation completed")
        except Exception as e:
            logger.error(f"Error in early adopter profile validation: {e}")
            profile_validation = {
                "overall_match_score": 0.0,
                "high_social_grade": 0.0,
                "high_education": 0.0,
                "multicar_households": 0.0,
                "house_dwelling": 0.0
            }
            issues.append(f"Early adopter profile validation failed: {e}")
        
        # Perform geographic pattern validation
        try:
            geographic_validation = self._check_geographic_patterns(results_df)
            logger.info("Geographic pattern validation completed")
        except Exception as e:
            logger.error(f"Error in geographic pattern validation: {e}")
            geographic_validation = {
                "correlation_score": 0.0,
                "high_adoption_region_match": 0.0,
                "distribution_realism": 0.0
            }
            issues.append(f"Geographic pattern validation failed: {e}")
        
        # Perform demographic distribution validation
        try:
            distribution_validation = self._check_demographic_distribution(results_df)
            logger.info("Demographic distribution validation completed")
        except Exception as e:
            logger.error(f"Error in demographic distribution validation: {e}")
            distribution_validation = {
                "variance_check": 0.0,
                "range_check": 1.0,
                "outlier_check": 1.0
            }
            issues.append(f"Demographic distribution validation failed: {e}")
        
        # Calculate overall validation metrics
        validation_confidence = self._calculate_validation_confidence(
            profile_validation, geographic_validation, distribution_validation
        )
        
        pattern_match_score = self._calculate_pattern_match_score(
            profile_validation, geographic_validation
        )
        
        overall_validation_score = (validation_confidence + pattern_match_score) / 2
        
        # Generate recommendations based on validation results
        recommendations.extend(self._generate_recommendations(
            profile_validation, geographic_validation, distribution_validation
        ))
        
        # Create comprehensive validation results
        validation_results = ValidationResults(
            overall_validation_score=overall_validation_score,
            early_adopter_profile_match=profile_validation,
            geographic_correlation=geographic_validation,
            demographic_distribution=distribution_validation,
            validation_confidence=validation_confidence,
            pattern_match_score=pattern_match_score,
            issues=issues,
            recommendations=recommendations
        )
        
        # Log validation summary
        self._log_validation_summary(validation_results)
        
        logger.info("Comprehensive validation completed")
        
        return validation_results
    
    def _check_early_adopter_profile(self, results_df: gpd.GeoDataFrame) -> Dict[str, float]:
        """
        Verify high-scoring areas match literature profile of early EV adopters.
        
        Literature shows early adopters are typically:
        "middle-aged, male, higher social grade, higher education, multi-car homeowners"
        
        Args:
            results_df: GeoDataFrame with adoption scores and demographic components
            
        Returns:
            Dictionary with profile match metrics
        """
        logger.info("Checking early adopter profile characteristics")
        
        # Identify high-scoring areas (top scoring areas above threshold)
        high_score_threshold = self.parameters["high_score_threshold"]
        high_score_areas = results_df[results_df['adoption_propensity'] > high_score_threshold]
        
        if len(high_score_areas) == 0:
            logger.warning(f"No areas found with adoption scores > {high_score_threshold}")
            return {
                "overall_match_score": 0.0,
                "high_social_grade": 0.0,
                "high_education": 0.0,
                "multicar_households": 0.0,
                "house_dwelling": 0.0
            }
        
        # Check each characteristic of the early adopter profile
        profile_metrics = {}
        
        # High social grade (income) - should be >70% in high-scoring areas
        if 'social_grade_score' in high_score_areas.columns:
            high_social_grade_ratio = (
                high_score_areas['social_grade_score'] > self.early_adopter_profile.high_social_grade_threshold
            ).mean()
            profile_metrics['high_social_grade'] = high_social_grade_ratio
        else:
            profile_metrics['high_social_grade'] = 0.0
            logger.warning("Social grade scores not found for profile validation")
        
        # High education - should be >70% in high-scoring areas
        if 'education_score' in high_score_areas.columns:
            high_education_ratio = (
                high_score_areas['education_score'] > self.early_adopter_profile.high_education_threshold
            ).mean()
            profile_metrics['high_education'] = high_education_ratio
        else:
            profile_metrics['high_education'] = 0.0
            logger.warning("Education scores not found for profile validation")
        
        # Multi-car households - should be >80% in high-scoring areas
        if 'car_ownership_score' in high_score_areas.columns:
            multicar_ratio = (
                high_score_areas['car_ownership_score'] > self.early_adopter_profile.multicar_threshold
            ).mean()
            profile_metrics['multicar_households'] = multicar_ratio
        else:
            profile_metrics['multicar_households'] = 0.0
            logger.warning("Car ownership scores not found for profile validation")
        
        # House dwelling - should be >80% in high-scoring areas
        if 'housing_score' in high_score_areas.columns:
            house_dwelling_ratio = (
                high_score_areas['housing_score'] > self.early_adopter_profile.house_dwelling_threshold
            ).mean()
            profile_metrics['house_dwelling'] = house_dwelling_ratio
        else:
            profile_metrics['house_dwelling'] = 0.0
            logger.warning("Housing scores not found for profile validation")
        
        # Calculate overall profile match score
        available_metrics = [v for v in profile_metrics.values() if v > 0]
        if available_metrics:
            overall_match_score = sum(available_metrics) / len(available_metrics)
        else:
            overall_match_score = 0.0
        
        profile_metrics['overall_match_score'] = overall_match_score
        
        # Log profile validation results
        logger.info(f"Early adopter profile validation results:")
        logger.info(f"  High-scoring areas: {len(high_score_areas)}")
        logger.info(f"  High social grade ratio: {profile_metrics['high_social_grade']:.3f}")
        logger.info(f"  High education ratio: {profile_metrics['high_education']:.3f}")
        logger.info(f"  Multi-car households ratio: {profile_metrics['multicar_households']:.3f}")
        logger.info(f"  House dwelling ratio: {profile_metrics['house_dwelling']:.3f}")
        logger.info(f"  Overall profile match: {overall_match_score:.3f}")
        
        return profile_metrics
    
    def _check_geographic_patterns(self, results_df: gpd.GeoDataFrame) -> Dict[str, float]:
        """
        Validate correlation with known high-adoption regions and realistic distribution.
        
        Checks for:
        - Correlation with known high-adoption regions (Edinburgh, Glasgow, etc.)
        - Realistic geographic distribution across Scotland
        - Absence of unrealistic clustering or gaps
        
        Args:
            results_df: GeoDataFrame with adoption scores and geographic identifiers
            
        Returns:
            Dictionary with geographic validation metrics
        """
        logger.info("Checking geographic pattern validation")
        
        geographic_metrics = {}
        
        # Check correlation with known high-adoption regions
        high_adoption_region_match = self._validate_high_adoption_regions(results_df)
        geographic_metrics['high_adoption_region_match'] = high_adoption_region_match
        
        # Check distribution realism across Scotland
        distribution_realism = self._validate_distribution_realism(results_df)
        geographic_metrics['distribution_realism'] = distribution_realism
        
        # Calculate spatial correlation if geometry is available
        if 'geometry' in results_df.columns:
            spatial_correlation = self._calculate_spatial_correlation(results_df)
            geographic_metrics['spatial_correlation'] = spatial_correlation
        else:
            geographic_metrics['spatial_correlation'] = 0.5  # Neutral score
            logger.warning("No geometry column found for spatial correlation analysis")
        
        # Calculate overall correlation score
        correlation_score = (
            high_adoption_region_match * 0.4 +
            distribution_realism * 0.4 +
            geographic_metrics['spatial_correlation'] * 0.2
        )
        geographic_metrics['correlation_score'] = correlation_score
        
        # Log geographic validation results
        logger.info(f"Geographic pattern validation results:")
        logger.info(f"  High adoption region match: {high_adoption_region_match:.3f}")
        logger.info(f"  Distribution realism: {distribution_realism:.3f}")
        logger.info(f"  Spatial correlation: {geographic_metrics['spatial_correlation']:.3f}")
        logger.info(f"  Overall correlation score: {correlation_score:.3f}")
        
        return geographic_metrics
    
    def _validate_high_adoption_regions(self, results_df: gpd.GeoDataFrame) -> float:
        """
        Validate that known high-adoption regions have high scores.
        
        Args:
            results_df: GeoDataFrame with adoption scores
            
        Returns:
            Float score [0,1] indicating match with known high-adoption regions
        """
        # Look for region identifiers in the data
        region_cols = [col for col in results_df.columns 
                      if any(pattern in col.lower() for pattern in ['name', 'label', 'region', 'area'])]
        
        if not region_cols:
            logger.warning("No region identifier columns found for high-adoption region validation")
            return 0.5  # Neutral score when validation cannot be performed
        
        # Use the first available region column
        region_col = region_cols[0]
        
        # Find areas matching known high-adoption regions
        matched_regions = []
        region_scores = []
        
        for known_region in self.known_high_adoption_regions:
            # Look for partial matches in region names
            region_mask = results_df[region_col].astype(str).str.contains(
                known_region, case=False, na=False
            )
            
            if region_mask.any():
                matched_areas = results_df[region_mask]
                mean_score = matched_areas['adoption_propensity'].mean()
                matched_regions.append(known_region)
                region_scores.append(mean_score)
                
                logger.info(f"Found {region_mask.sum()} areas matching '{known_region}' with mean score {mean_score:.3f}")
        
        if not region_scores:
            logger.warning("No known high-adoption regions found in data")
            return 0.5  # Neutral score
        
        # Calculate validation score based on how many known regions have high scores
        high_score_threshold = self.parameters["high_score_threshold"]
        high_scoring_regions = sum(1 for score in region_scores if score > high_score_threshold)
        
        validation_score = high_scoring_regions / len(self.known_high_adoption_regions)
        
        logger.info(f"High-adoption region validation: {high_scoring_regions}/{len(self.known_high_adoption_regions)} regions have high scores")
        
        return validation_score
    
    def _validate_distribution_realism(self, results_df: gpd.GeoDataFrame) -> float:
        """
        Validate that the score distribution is realistic across Scotland.
        
        Args:
            results_df: GeoDataFrame with adoption scores
            
        Returns:
            Float score [0,1] indicating distribution realism
        """
        scores = results_df['adoption_propensity']
        
        # Check for reasonable variance (not all areas the same)
        variance_check = 1.0 if scores.std() > self.parameters["variance_threshold"] else 0.0
        
        # Check for reasonable range (not all extreme values)
        range_check = 1.0 if 0.1 < scores.mean() < 0.9 else 0.5
        
        # Check for reasonable distribution shape (not too skewed)
        skewness = scores.skew()
        skewness_check = 1.0 if abs(skewness) < 2.0 else 0.5
        
        # Check for absence of unrealistic clustering (coefficient of variation)
        cv = scores.std() / (scores.mean() + 1e-6)
        cv_check = 1.0 if 0.2 < cv < 2.0 else 0.5
        
        # Combine checks
        realism_score = (variance_check + range_check + skewness_check + cv_check) / 4
        
        logger.info(f"Distribution realism checks:")
        logger.info(f"  Variance check: {variance_check} (std: {scores.std():.3f})")
        logger.info(f"  Range check: {range_check} (mean: {scores.mean():.3f})")
        logger.info(f"  Skewness check: {skewness_check} (skew: {skewness:.3f})")
        logger.info(f"  CV check: {cv_check} (CV: {cv:.3f})")
        logger.info(f"  Overall realism score: {realism_score:.3f}")
        
        return realism_score
    
    def _calculate_spatial_correlation(self, results_df: gpd.GeoDataFrame) -> float:
        """
        Calculate spatial correlation metrics for geographic validation.
        
        Args:
            results_df: GeoDataFrame with geometry and adoption scores
            
        Returns:
            Float score [0,1] indicating spatial correlation quality
        """
        try:
            # Calculate spatial autocorrelation (Moran's I) if possible
            from libpysal.weights import Queen
            from esda.moran import Moran
            
            # Create spatial weights matrix
            w = Queen.from_dataframe(results_df)
            
            # Calculate Moran's I for adoption propensity
            moran = Moran(results_df['adoption_propensity'], w)
            
            # Convert Moran's I to [0,1] score
            # Moran's I ranges from -1 to 1, with positive values indicating positive spatial correlation
            spatial_score = max(0, (moran.I + 1) / 2)  # Normalize to [0,1]
            
            logger.info(f"Spatial autocorrelation (Moran's I): {moran.I:.3f} (p-value: {moran.p_norm:.3f})")
            
            return spatial_score
            
        except ImportError:
            logger.warning("Spatial analysis libraries not available, using simplified spatial validation")
            return self._simplified_spatial_validation(results_df)
        except Exception as e:
            logger.warning(f"Error in spatial correlation calculation: {e}, using simplified validation")
            return self._simplified_spatial_validation(results_df)
    
    def _simplified_spatial_validation(self, results_df: gpd.GeoDataFrame) -> float:
        """
        Simplified spatial validation when full spatial analysis is not available.
        
        Args:
            results_df: GeoDataFrame with adoption scores
            
        Returns:
            Float score [0,1] indicating basic spatial reasonableness
        """
        # Check for reasonable spatial distribution by examining score variance
        # across different parts of the dataset (proxy for spatial variation)
        
        scores = results_df['adoption_propensity']
        
        # Split data into chunks and check variance between chunks
        n_chunks = min(10, len(results_df) // 100)  # At least 100 areas per chunk
        
        if n_chunks < 2:
            return 0.5  # Neutral score for small datasets
        
        chunk_size = len(results_df) // n_chunks
        chunk_means = []
        
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_chunks - 1 else len(results_df)
            chunk_mean = scores.iloc[start_idx:end_idx].mean()
            chunk_means.append(chunk_mean)
        
        # Calculate variance between chunks
        chunk_variance = np.var(chunk_means)
        
        # Score based on reasonable variance between spatial chunks
        # Too low variance = unrealistic uniformity, too high = unrealistic clustering
        if 0.01 < chunk_variance < 0.1:
            spatial_score = 1.0
        elif 0.005 < chunk_variance < 0.2:
            spatial_score = 0.7
        else:
            spatial_score = 0.4
        
        logger.info(f"Simplified spatial validation: chunk variance = {chunk_variance:.4f}, score = {spatial_score:.3f}")
        
        return spatial_score
    
    def _check_demographic_distribution(self, results_df: gpd.GeoDataFrame) -> Dict[str, float]:
        """
        Check demographic distribution for realism and consistency.
        
        Args:
            results_df: GeoDataFrame with adoption scores and component scores
            
        Returns:
            Dictionary with demographic distribution validation metrics
        """
        logger.info("Checking demographic distribution validation")
        
        distribution_metrics = {}
        
        # Check variance in overall scores
        scores = results_df['adoption_propensity']
        variance_check = 1.0 if scores.std() > self.parameters["variance_threshold"] else 0.0
        distribution_metrics['variance_check'] = variance_check
        
        # Check score range reasonableness
        range_check = 1.0 if 0 <= scores.min() <= scores.max() <= 1 else 0.0
        distribution_metrics['range_check'] = range_check
        
        # Check for outliers (scores beyond 3 standard deviations)
        z_scores = np.abs((scores - scores.mean()) / (scores.std() + 1e-6))
        outlier_ratio = (z_scores > 3).mean()
        outlier_check = 1.0 if outlier_ratio < 0.05 else 0.5  # Less than 5% outliers is good
        distribution_metrics['outlier_check'] = outlier_check
        
        # Check component score consistency
        component_consistency = self._check_component_consistency(results_df)
        distribution_metrics['component_consistency'] = component_consistency
        
        # Log distribution validation results
        logger.info(f"Demographic distribution validation results:")
        logger.info(f"  Variance check: {variance_check} (std: {scores.std():.3f})")
        logger.info(f"  Range check: {range_check} (range: [{scores.min():.3f}, {scores.max():.3f}])")
        logger.info(f"  Outlier check: {outlier_check} (outlier ratio: {outlier_ratio:.3f})")
        logger.info(f"  Component consistency: {component_consistency:.3f}")
        
        return distribution_metrics
    
    def _check_component_consistency(self, results_df: gpd.GeoDataFrame) -> float:
        """
        Check consistency between component scores and overall scores.
        
        Args:
            results_df: GeoDataFrame with component scores
            
        Returns:
            Float score [0,1] indicating component consistency
        """
        component_cols = [col for col in results_df.columns if col.endswith('_score') and col != 'adoption_propensity']
        
        if len(component_cols) < 2:
            logger.warning("Insufficient component scores for consistency check")
            return 0.5
        
        consistency_scores = []
        
        for col in component_cols:
            if col in results_df.columns:
                # Check correlation between component and overall score
                correlation = results_df[col].corr(results_df['adoption_propensity'])
                
                # Component scores should be positively correlated with overall scores
                if correlation > 0.3:  # Reasonable positive correlation
                    consistency_scores.append(1.0)
                elif correlation > 0.1:  # Weak positive correlation
                    consistency_scores.append(0.7)
                else:  # Poor or negative correlation
                    consistency_scores.append(0.3)
                
                logger.info(f"  {col} correlation with overall: {correlation:.3f}")
        
        if consistency_scores:
            overall_consistency = sum(consistency_scores) / len(consistency_scores)
        else:
            overall_consistency = 0.5
        
        return overall_consistency
    
    def _validate_basic_data_quality(self, results_df: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate basic data quality requirements.
        
        Args:
            results_df: GeoDataFrame with adoption scores
            
        Returns:
            Dictionary with basic validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': []
        }
        
        # Check for required columns
        if 'adoption_propensity' not in results_df.columns:
            validation_result['is_valid'] = False
            validation_result['issues'].append("Missing adoption_propensity column")
        
        # Check for excessive NaN values
        if 'adoption_propensity' in results_df.columns:
            scores = results_df['adoption_propensity']
            nan_ratio = scores.isna().sum() / len(scores)
            
            if nan_ratio > self.parameters["nan_threshold"]:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f"Excessive NaN values: {nan_ratio:.1%}")
        
        # Check data size
        if len(results_df) < 100:
            validation_result['issues'].append("Small dataset size may affect validation reliability")
        
        return validation_result
    
    def _calculate_validation_confidence(self, 
                                       profile_validation: Dict[str, float],
                                       geographic_validation: Dict[str, float],
                                       distribution_validation: Dict[str, float]) -> float:
        """
        Calculate overall validation confidence score.
        
        Args:
            profile_validation: Early adopter profile validation results
            geographic_validation: Geographic pattern validation results
            distribution_validation: Demographic distribution validation results
            
        Returns:
            Float confidence score [0,1]
        """
        weights = self.parameters["confidence_weights"]
        
        profile_score = profile_validation.get('overall_match_score', 0.0)
        geographic_score = geographic_validation.get('correlation_score', 0.0)
        
        # Calculate distribution score from multiple metrics
        dist_metrics = [v for k, v in distribution_validation.items() if isinstance(v, (int, float))]
        distribution_score = sum(dist_metrics) / len(dist_metrics) if dist_metrics else 0.0
        
        confidence = (
            profile_score * weights["profile_match"] +
            geographic_score * weights["geographic_correlation"] +
            distribution_score * weights["distribution_realism"]
        )
        
        return confidence
    
    def _calculate_pattern_match_score(self, 
                                     profile_validation: Dict[str, float],
                                     geographic_validation: Dict[str, float]) -> float:
        """
        Calculate pattern match score based on literature alignment.
        
        Args:
            profile_validation: Early adopter profile validation results
            geographic_validation: Geographic pattern validation results
            
        Returns:
            Float pattern match score [0,1]
        """
        profile_score = profile_validation.get('overall_match_score', 0.0)
        geographic_score = geographic_validation.get('high_adoption_region_match', 0.0)
        
        # Weight profile matching more heavily as it's more directly validated by literature
        pattern_match = profile_score * 0.7 + geographic_score * 0.3
        
        return pattern_match
    
    def _generate_recommendations(self, 
                                profile_validation: Dict[str, float],
                                geographic_validation: Dict[str, float],
                                distribution_validation: Dict[str, float]) -> List[str]:
        """
        Generate recommendations based on validation results.
        
        Args:
            profile_validation: Early adopter profile validation results
            geographic_validation: Geographic pattern validation results
            distribution_validation: Demographic distribution validation results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Profile-based recommendations
        profile_score = profile_validation.get('overall_match_score', 0.0)
        if profile_score < self.parameters["profile_match_threshold"]:
            recommendations.append(
                f"Early adopter profile match is low ({profile_score:.2f}). "
                "Consider reviewing demographic factor weights or interaction effects."
            )
            
            # Specific profile component recommendations
            if profile_validation.get('high_social_grade', 0.0) < 0.6:
                recommendations.append("Low social grade correlation - review income/social grade weighting")
            
            if profile_validation.get('high_education', 0.0) < 0.6:
                recommendations.append("Low education correlation - review education factor weighting")
            
            if profile_validation.get('multicar_households', 0.0) < 0.7:
                recommendations.append("Low multi-car correlation - review car ownership scoring or interaction effects")
        
        # Geographic-based recommendations
        geographic_score = geographic_validation.get('correlation_score', 0.0)
        if geographic_score < self.parameters["geographic_correlation"]:
            recommendations.append(
                f"Geographic correlation is low ({geographic_score:.2f}). "
                "Consider adding geographic context adjustments or reviewing regional factors."
            )
        
        # Distribution-based recommendations
        if distribution_validation.get('variance_check', 1.0) < 1.0:
            recommendations.append("Low score variance detected - review input data quality and factor differentiation")
        
        if distribution_validation.get('component_consistency', 1.0) < 0.7:
            recommendations.append("Component score consistency is low - review factor calculation methods")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Validation results look good - model appears to be performing well against literature patterns")
        
        return recommendations
    
    def _log_validation_summary(self, validation_results: ValidationResults) -> None:
        """Log comprehensive validation summary"""
        
        logger.info("=== VALIDATION SUMMARY ===")
        logger.info(f"Overall Validation Score: {validation_results.overall_validation_score:.3f}")
        logger.info(f"Validation Confidence: {validation_results.validation_confidence:.3f}")
        logger.info(f"Pattern Match Score: {validation_results.pattern_match_score:.3f}")
        
        logger.info("\nEarly Adopter Profile Match:")
        for metric, value in validation_results.early_adopter_profile_match.items():
            logger.info(f"  {metric}: {value:.3f}")
        
        logger.info("\nGeographic Correlation:")
        for metric, value in validation_results.geographic_correlation.items():
            logger.info(f"  {metric}: {value:.3f}")
        
        logger.info("\nDemographic Distribution:")
        for metric, value in validation_results.demographic_distribution.items():
            logger.info(f"  {metric}: {value:.3f}")
        
        if validation_results.issues:
            logger.warning("\nValidation Issues:")
            for issue in validation_results.issues:
                logger.warning(f"  - {issue}")
        
        if validation_results.recommendations:
            logger.info("\nRecommendations:")
            for rec in validation_results.recommendations:
                logger.info(f"  - {rec}")
        
        logger.info("=== END VALIDATION SUMMARY ===")


def validate_adoption_scores(adoption_data: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Validate adoption propensity scores for consistency and realism.
    
    Args:
        adoption_data: GeoDataFrame with adoption propensity scores
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating adoption propensity scores")
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'statistics': {}
    }
    
    if 'adoption_propensity' not in adoption_data.columns:
        validation_results['is_valid'] = False
        validation_results['issues'].append("Missing adoption_propensity column")
        return validation_results
    
    scores = adoption_data['adoption_propensity']
    
    # Range validation
    if scores.min() < 0 or scores.max() > 1:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Scores out of range [0,1]: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Distribution validation
    validation_results['statistics'] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'nan_count': scores.isna().sum(),
        'zero_count': (scores == 0).sum(),
        'one_count': (scores == 1).sum()
    }
    
    # Check for unrealistic distributions
    if scores.std() < 0.01:
        validation_results['issues'].append("Very low variance in scores")
    
    if scores.isna().sum() > len(scores) * 0.1:
        validation_results['issues'].append(f"High proportion of NaN values: {scores.isna().sum()}/{len(scores)}")
    
    # Check component scores if available
    component_cols = [col for col in adoption_data.columns if col.endswith('_score')]
    for col in component_cols:
        component_scores = adoption_data[col]
        if component_scores.min() < 0 or component_scores.max() > 1:
            validation_results['issues'].append(f"{col} out of range [0,1]")
    
    if validation_results['issues']:
        validation_results['is_valid'] = False
    
    logger.info(f"Validation {'passed' if validation_results['is_valid'] else 'failed'}")
    
    return validation_results
