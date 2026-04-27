"""
EV Type Selection Module for Frugal EV Analysis

This module provides classes and functions for assigning appropriate EV types (2-seater vs 4-seater)
to households based on demographic characteristics, household composition, and affordability constraints.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DEFAULT_SCORING_WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "scoring_weights.json"
_DEFAULT_TARGET_COLUMNS_PATH = Path(__file__).resolve().parents[1] / "target_columns.json"


def _load_scoring_weights(config_path: Path = _DEFAULT_SCORING_WEIGHTS_PATH) -> Dict[str, Any]:
    """Load scoring weights configuration from disk."""
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_target_columns(config_path: Path = _DEFAULT_TARGET_COLUMNS_PATH) -> Dict[str, List[str]]:
    """Load target column mappings used across analytical stages."""
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and 'columns' in data:
        return data['columns']

    return data


def _normalize_token(token: str) -> str:
    """Normalize strings for pattern comparison (lowercase, remove non-alphanumerics)."""
    return re.sub(r"[^a-z0-9]", "", token.lower())


@dataclass
class EVTypeAssignment:
    """Container for EV type assignment results"""
    ev_type: str  # "2-seater", "4-seater", "mixed", or "not_applicable"
    household_size_factor: float
    affordability_factor: float
    assignment_confidence: float


class EVTypeSelector:
    """
    Main class for assigning 2-seater or 4-seater EVs based on household characteristics.
    
    Implements household size analysis from composition data and assignment logic
    based on family structure, considering both household needs and constraints.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, Any]] = None,
        assignment_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize EV type selector with scoring weights.

        Args:
            weights: Dictionary containing all scoring weights and preferences
            assignment_weights: Optional override for assignment weight configuration
        """
        if weights is None:
            weights = _load_scoring_weights()

        self.scoring_weights = weights  # Store full config for age_weights access
        self.target_columns = _load_target_columns()
        if assignment_weights is not None:
            self.assignment_weights = assignment_weights
        else:
            self.assignment_weights = weights.get(
                'assignment_weights', self._default_assignment_weights()
            )
        self.weights = weights  # Backwards-compatible attribute for legacy callers

    def assign_ev_type(self, demographics: gpd.GeoDataFrame, target_columns: Optional[Dict[str, List[str]]] = None, 
                       conversion_potential: Optional[pd.Series] = None) -> gpd.GeoDataFrame:
        """
        Assign appropriate EV types to households based on demographic characteristics.
        
        Per paper.qmd:968, Stage 7 only processes areas with replaceable trips from Stage 6.
        Areas with zero conversion potential receive 'not_applicable' assignment per @eq-vehicle-suitability-factor.
        
        Args:
            demographics: GeoDataFrame with household composition and demographic data
            target_columns: Dictionary mapping data categories to column names
            conversion_potential: Optional Series of Stage 6 conversion potential values.
                                If provided, areas with zero conversion receive 'not_applicable' assignment.
            
        Returns:
            GeoDataFrame with EV type assignments added
            
        Raises:
            ValueError: If required demographic columns are missing
        """
        logger.info("Assigning EV types based on household characteristics")

        if target_columns is None:
            target_columns = self.target_columns

        # Validate input data
        self._validate_demographic_data(demographics, target_columns)

        # Create copy to avoid modifying original data
        result = demographics.copy()

        # Configuration for Stage 6 filtering requirements
        stage_7_params = self.scoring_weights.get('stage_7_parameters', {})
        enable_not_applicable = stage_7_params.get('enable_not_applicable', False)
        min_conversion_threshold = stage_7_params.get('minimum_conversion_threshold', 0.0)
        require_stage6 = stage_7_params.get('require_stage_6_filtering', False)

        if conversion_potential is None and 'stage6_conversion_potential' in demographics.columns:
            conversion_potential = demographics['stage6_conversion_potential']

        if require_stage6 and conversion_potential is None:
            raise ValueError(
                "Stage 7 configuration requires Stage 6 conversion data. "
                "Provide conversion_potential when calling assign_ev_type."
            )

        # Align conversion potential with demographic index if provided
        conversion_series: Optional[pd.Series]
        if conversion_potential is not None:
            if not isinstance(conversion_potential, pd.Series):
                conversion_potential = pd.Series(conversion_potential, index=result.index)
            conversion_series = conversion_potential.reindex(result.index)
            conversion_series = conversion_series.fillna(0.0)
            result['stage6_conversion_potential'] = conversion_series
        else:
            conversion_series = None

        # Identify areas with no replaceable trips (per paper.qmd:968)
        if conversion_series is not None and enable_not_applicable:
            no_conversion_mask = conversion_series <= min_conversion_threshold
            logger.info(
                "Found %s areas with no replaceable trips (conversion ≤ %s)",
                int(no_conversion_mask.sum()),
                min_conversion_threshold,
            )
        else:
            no_conversion_mask = pd.Series(False, index=result.index)
        
        # Preprocess data to handle mixed types and missing values
        result = self._preprocess_demographic_data(result)
        
        # Calculate household size factors using target_columns
        household_size_factors = self._calculate_household_size_factor(result, target_columns)
        
        # Calculate age-based adjustments
        age_adjustments = self._calculate_age_adjustments(result, target_columns)
        
        # Calculate children presence indicators
        children_indicators = self._calculate_children_indicators(result, target_columns)
        
        # Combine factors to determine EV type preference
        ev_type_scores = self._calculate_ev_type_scores(
            household_size_factors, age_adjustments, children_indicators
        )
        
        # Make final EV type assignments
        ev_assignments = self._make_ev_type_assignments(ev_type_scores, household_size_factors)
        
        # Add results to dataframe
        result['ev_type_assignment'] = [assign.ev_type for assign in ev_assignments]
        result['household_size_factor'] = [assign.household_size_factor for assign in ev_assignments]
        result['assignment_confidence'] = [assign.assignment_confidence for assign in ev_assignments]
        result['two_seater_score'] = ev_type_scores['2-seater']
        result['four_seater_score'] = ev_type_scores['4-seater']
        
        # Override with 'not_applicable' for areas with no replaceable trips (per paper.qmd:968,1062)
        if no_conversion_mask.any():
            result.loc[no_conversion_mask, 'ev_type_assignment'] = 'not_applicable'
            result.loc[no_conversion_mask, 'assignment_confidence'] = 0.0
            if 'stage6_conversion_potential' in result.columns:
                result.loc[no_conversion_mask, 'stage6_conversion_potential'] = 0.0
            logger.info(
                "Set %s areas to 'not_applicable' due to zero conversion potential",
                int(no_conversion_mask.sum()),
            )
        
        # Validate assignments
        self._validate_ev_type_assignments(result)
        
        logger.info(f"Assigned EV types for {len(result)} areas")
        self._log_assignment_summary(result)
        
        return result

    @staticmethod
    def _default_assignment_weights() -> Dict[str, float]:
        """Fallback assignment weight configuration."""
        return {
            'household_size': 0.6,
            'age': 0.3,
            'children_present': 0.1,
            'children_bonus': 0.4,
            'children_threshold': 0.3,
            'assignment_threshold': 0.1,
        }
    
    def calculate_household_size_factor(self, composition: pd.Series, target_columns: Optional[Dict[str, List[str]]] = None) -> float:
        """
        Calculate household size factor for a single area's composition data.

        Args:
            composition: Series with household composition data for one area
            target_columns: Dictionary mapping data categories to column names
        
        Returns:
            Float representing household size factor (higher = larger households)
        """
        # This is a public method for individual calculations
        # Create a DataFrame with the composition data as a single row
        temp_df = pd.DataFrame([composition])
        temp_df.index = [0]  # Set a simple index

        if target_columns is None:
            target_columns = self.target_columns

        household_factors = self._calculate_household_size_factor(temp_df, target_columns)
        return household_factors.iloc[0] if len(household_factors) > 0 else 0.5
    
    @staticmethod
    def _dedupe_columns(columns: List[str]) -> List[str]:
        """Preserve order while removing duplicate column names."""
        return list(dict.fromkeys(col for col in columns if col))

    def _get_available_housing_columns(
        self,
        demographics: gpd.GeoDataFrame,
        target_columns: Dict[str, List[str]]
    ) -> List[str]:
        """Resolve housing columns across 2011 and 2022 naming variants."""
        target_housing_cols = target_columns.get('Housing', [])
        available_housing_cols = [col for col in target_housing_cols if col in demographics.columns]

        discovered_housing_cols = [
            col for col in demographics.columns
            if (
                'accommodation_type' in col.lower()
                or (
                    (
                        'whole.house.or.bungalow' in col.lower()
                        or 'flat..maisonette.or.apartment' in col.lower()
                    )
                    and 'cars.or.vans' in col.lower()
                )
            )
        ]

        return self._dedupe_columns(available_housing_cols + discovered_housing_cols)

    def _get_available_age_columns(
        self,
        demographics: gpd.GeoDataFrame,
        target_columns: Dict[str, List[str]]
    ) -> List[str]:
        """Resolve age columns from configured targets plus discovered age-series columns."""
        target_age_cols = target_columns.get('Age', [])
        available_age_cols = [col for col in target_age_cols if col in demographics.columns]

        discovered_age_cols = [
            col for col in demographics.columns
            if col.lower().endswith('_age_by_sex') or re.match(r'(?i)^x\d', col)
        ]

        return self._dedupe_columns(available_age_cols + discovered_age_cols)

    def _get_available_household_composition_columns(
        self,
        demographics: gpd.GeoDataFrame,
        target_columns: Dict[str, List[str]]
    ) -> List[str]:
        """Resolve household-composition columns across census schema variants."""
        target_household_cols = target_columns.get('Household Composition', [])
        available_household_cols = [col for col in target_household_cols if col in demographics.columns]

        discovered_household_cols = [
            col for col in demographics.columns
            if 'household_composition' in col.lower()
        ]

        return self._dedupe_columns(available_household_cols + discovered_household_cols)

    def _validate_demographic_data(
        self,
        demographics: gpd.GeoDataFrame,
        target_columns: Optional[Dict[str, List[str]]] = None
    ) -> None:
        """Validate that required demographic categories are present."""
        if target_columns is None:
            target_columns = self.target_columns

        household_cols = self._get_available_household_composition_columns(demographics, target_columns)
        age_cols = self._get_available_age_columns(demographics, target_columns)

        household_cols_lower = [col.lower() for col in household_cols]
        missing_patterns = []

        if not any('one.person.household' in col for col in household_cols_lower):
            missing_patterns.append('one.person.household')
        if not any('couple' in col for col in household_cols_lower):
            missing_patterns.append('couple')
        if not any('lone.parent' in col for col in household_cols_lower):
            missing_patterns.append('lone.parent')
        if not age_cols:
            missing_patterns.append('age')

        if missing_patterns:
            raise ValueError(f"Missing demographic data patterns for EV type selection: {missing_patterns}")
    
    def _preprocess_demographic_data(self, demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Preprocess demographic data to handle mixed types and missing values"""
        logger.info("Preprocessing demographic data for EV type selection")
        
        # Get all numeric columns (exclude geo_code, geometry, label, name)
        exclude_cols = ['geo_code', 'geometry', 'label', 'name']
        numeric_cols = [col for col in demographics.columns if col not in exclude_cols]
        
        # Convert to numeric, replacing non-numeric values with 0
        for col in numeric_cols:
            demographics[col] = pd.to_numeric(demographics[col], errors='coerce').fillna(0)
        
        logger.info(f"Preprocessed {len(numeric_cols)} demographic columns for EV type selection")
        return demographics
    
    def _calculate_household_size_factor(self, demographics: gpd.GeoDataFrame, target_columns: Dict[str, List[str]]) -> pd.Series:
        """Calculate household size factors based on housing data from target_columns"""
        household_size_scores = pd.Series(0.0, index=demographics.index)
        total_households = pd.Series(0.0, index=demographics.index)
        
        available_housing_cols = self._get_available_housing_columns(demographics, target_columns)
        
        if not available_housing_cols:
            logger.warning("No housing columns found in data, returning default household size factors")
            return pd.Series(0.5, index=demographics.index)
        
        # Define household size factors based on number of adults
        # Higher scores = larger households that prefer 4-seater EVs
        household_size_factors = {
            "No.people.aged.17.or.over.in.household": 0.2,
            "One.person.aged.17.or.over.in.household": 0.3,
            "Two.or.more.people.aged.17.or.over.in.household": 0.8
        }
        
        # Car ownership modifiers
        car_ownership_modifiers = {
            "No.cars.or.vans": 1.0,
            "One.car.or.van": 1.1,
            "Two.or.more.cars.or.vans": 1.2
        }
        
        # Accommodation type modifiers
        accommodation_modifiers = {
            "Whole.house.or.bungalow": 1.1,
            "Flat..maisonette.or.apartment": 0.9
        }
        
        # Process each housing column
        for col in available_housing_cols:
            col_lower = col.lower()
            
            # Extract household size factor
            size_factor = 0.5  # Default
            for adult_pattern, factor in household_size_factors.items():
                if adult_pattern.lower() in col_lower:
                    size_factor = factor
                    break
            
            # Apply car ownership modifier
            car_modifier = 1.0
            for car_pattern, modifier in car_ownership_modifiers.items():
                if car_pattern.lower() in col_lower:
                    car_modifier = modifier
                    break
            
            # Apply accommodation modifier
            acc_modifier = 1.0
            for acc_pattern, modifier in accommodation_modifiers.items():
                if acc_pattern.lower() in col_lower:
                    acc_modifier = modifier
                    break
            
            # Calculate final factor for this housing type
            final_factor = size_factor * car_modifier * acc_modifier
            
            # Add weighted contribution
            household_count = demographics[col]
            household_size_scores += household_count * final_factor
            total_households += household_count
        
        # Normalize by total households to get weighted average
        household_size_scores = household_size_scores / (total_households + 1e-6)
        
        return np.clip(household_size_scores, 0.0, 1.0)
    
    
    def _extract_age_band(self, column_name: str) -> Optional[str]:
        """Return configured age band key that matches the provided column name."""
        age_weights = getattr(self, 'scoring_weights', {}).get('age_weights', {})
        if age_weights:
            normalized_column = _normalize_token(column_name)
            sorted_bands = sorted(
                age_weights.keys(),
                key=lambda key: len(_normalize_token(key)),
                reverse=True
            )
            for band in sorted_bands:
                normalized_band = _normalize_token(band)
                if normalized_band and normalized_band in normalized_column:
                    return band

        # Fallback to legacy regex matching for unexpected patterns
        age_patterns = [
            r'X\d+\.to\.\d+',
            r'X\d+\.and\.over',
            r'X\d+\.and\.under'
        ]

        for pattern in age_patterns:
            match = re.search(pattern, column_name, re.IGNORECASE)
            if match:
                return match.group(0)

        return None
    
    def _calculate_age_adjustments(self, demographics: gpd.GeoDataFrame, target_columns: Dict[str, List[str]]) -> pd.Series:
        """
        Calculate age-based adjustments per @eq-age-adjustment using weighted sum over all age bands.
        
        Uses configured age_weights from scoring_weights for all cohorts, implementing the paper's
        full lifecycle weighting approach rather than focusing only on 30-40 cohort.
        """
        available_age_cols = self._get_available_age_columns(demographics, target_columns)
        
        if not available_age_cols:
            logger.warning("No age columns found in data, returning default age adjustments")
            return pd.Series(0.5, index=demographics.index)
        
        # Load age weights from configuration
        # Try to get from instance first, then fall back to module-level scoring_weights
        if hasattr(self, 'scoring_weights'):
            age_weights_config = self.scoring_weights.get('age_weights', {})
        else:
            # Fallback to default weights if not available
            age_weights_config = {
                'X20.to.24': 0.5, 'X25.to.29': 0.85, 'X30.to.34': 0.9, 'X35.to.39': 0.9,
                'X40.to.44': 0.85, 'X45.to.49': 0.8, 'X50.to.54': 0.7, 'X55.to.59': 0.6,
                'X60.to.64': 0.5, 'X65.to.69': 0.3, 'X70.to.74': 0.2
            }
        
        # Calculate weighted age adjustment per @eq-age-adjustment
        age_adjustment_score = pd.Series(0.0, index=demographics.index)
        total_population = pd.Series(0.0, index=demographics.index)
        
        for col in available_age_cols:
            # Extract age band from column name
            age_band = self._extract_age_band(col)
            
            if age_band:
                # Get weight for this age band (default 0.5 if not configured)
                weight = age_weights_config.get(age_band, 0.5)
                
                # Add weighted contribution
                population = demographics[col]
                age_adjustment_score += population * weight
                total_population += population
        
        # Normalize by total population per @eq-age-adjustment formula
        normalized_score = age_adjustment_score / (total_population + 1e-6)
        
        return np.clip(normalized_score, 0.0, 1.0)
    
    def _calculate_children_indicators(self, demographics: gpd.GeoDataFrame, target_columns: Dict[str, List[str]]) -> pd.Series:
        """Calculate indicators for presence of children in households"""
        
        household_cols = self._get_available_household_composition_columns(demographics, target_columns)
        
        # Build key household patterns using simple string matching (like in task4.qmd)
        key_household_patterns = {
            # Single-person households
            "Single Person (65+)": [
                c for c in household_cols
                if "one.person.household" in c.lower() and "65.and.over" in c.lower()
            ],
            "Single Person (<65)": [
                c for c in household_cols
                if "one.person.household" in c.lower() and "under.65" in c.lower()
            ],
            
            # Couple households
            "Couples - No Children": [
                c for c in household_cols
                if "couple" in c.lower() and ("no.children" in c.lower() or "without.children" in c.lower())
            ],
            "Couples - With Dependent Children": [
                c for c in household_cols
                if "couple" in c.lower() and "dependent.children" in c.lower() and "non.dependent" not in c.lower()
            ],
            "Couples - All Children Non-dependent": [
                c for c in household_cols
                if "couple" in c.lower() and "non.dependent" in c.lower()
            ],
            
            # Lone parent households
            "Lone Parent - With Dependent Children": [
                c for c in household_cols
                if ("lone.parent" in c.lower() or "one.family" in c.lower()) and "dependent" in c.lower() and "non.dependent" not in c.lower()
            ],
            "Lone Parent - All Children Non-dependent": [
                c for c in household_cols
                if ("lone.parent" in c.lower() or "one.family" in c.lower()) and "non.dependent" in c.lower()
            ],
            
            # Other households
            "Other Households - With Dependent Children": [
                c for c in household_cols
                if "other.households" in c.lower() and "dependent" in c.lower() and "non.dependent" not in c.lower()
            ],
            "One Family Only - All aged 65+": [
                c for c in household_cols
                if "one.family" in c.lower() and "65.and.over" in c.lower()
            ],
            "Other Households - All Full-time Students": [
                c for c in household_cols
                if "other.households" in c.lower() and "full.time.students" in c.lower()
            ],
            "Other Households - All aged 65+": [
                c for c in household_cols
                if "other.households" in c.lower() and "65.and.over" in c.lower()
            ],
            "Other Households - Other": [
                c for c in household_cols
                if "other.households..other_household_composition" in c.lower()
            ],
        }
        
        # Create household patterns rollups (remove duplicates)
        # Per @eq-children-factor: Only count DEPENDENT children (not non-dependent)
        household_patterns = {
            "Families with Children (Dependent)": list(set(
                key_household_patterns["Couples - With Dependent Children"]
                + key_household_patterns["Lone Parent - With Dependent Children"]
                + key_household_patterns["Other Households - With Dependent Children"]
            )),
            # Non-dependent children NOT included per paper definition (line 990-996)
            # "Families with Children (Non-dependent)": list(set(
            #     key_household_patterns["Couples - All Children Non-dependent"]
            #     + key_household_patterns["Lone Parent - All Children Non-dependent"]
            # )),
        }
        
        # Calculate children indicators using the household patterns
        children_indicators = pd.Series(0.0, index=demographics.index)
        
        # Sum households with DEPENDENT children only per @eq-children-factor
        for col in household_patterns["Families with Children (Dependent)"]:
            if col in demographics.columns:
                children_indicators += demographics[col]
        
        # Non-dependent children explicitly excluded per paper definition
        # Paper states: "H_{children,i} denotes households with dependent children"
        
        # Calculate total households from all household composition columns
        total_households = demographics[household_cols].sum(axis=1) if household_cols else pd.Series(1.0, index=demographics.index)
        
        # Calculate proportion of households with children
        children_proportion = children_indicators / (total_households + 1e-6)
        
        return np.clip(children_proportion, 0.0, 1.0)
    
    def _calculate_ev_type_scores(self, household_factors: pd.Series,
                                 age_adjustments: pd.Series,
                                 children_indicators: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate EV type preference scores using independent formulas.
        
        Implements @eq-two-seater-score and @eq-four-seater-score from paper.
        Both scores are calculated independently to allow genuine "mixed" assignments
        where both vehicle types are similarly suitable (captures household diversity).
        """
        
        # Get weights from config
        w_hs = self.assignment_weights['household_size']
        w_age = self.assignment_weights.get('age', self.assignment_weights.get('adult_composition', 0.3))
        w_child = self.assignment_weights['children_present']
        
        # 2-seater score: HSF + AAF + (1-CPF)
        # Higher for small households, younger demographics, no children
        two_seater_score = (
            household_factors * w_hs +
            age_adjustments * w_age +
            (1 - children_indicators) * w_child
        )
        
        # 4-seater score: (1-HSF) + (1-AAF) + CPF + bonus
        # Higher for large households, older demographics, with children
        four_seater_score = (
            (1 - household_factors) * w_hs +
            (1 - age_adjustments) * w_age +
            children_indicators * w_child
        )
        
        # Add conditional children bonus (β_child when CPF > 0.3)
        children_threshold = self.assignment_weights.get('children_threshold', 0.3)
        children_bonus = self.assignment_weights.get('children_bonus', 0.4)
        
        # Apply bonus only where children indicator exceeds threshold
        four_seater_score += np.where(
            children_indicators > children_threshold,
            children_bonus,
            0.0
        )
        
        # Normalize scores to [0, 1] range
        # Note: Scores do NOT sum to 1.0 - this allows mixed assignments
        two_seater_score = np.clip(two_seater_score, 0.0, 1.0)
        four_seater_score = np.clip(four_seater_score, 0.0, 1.0)
        
        return {
            '2-seater': two_seater_score,
            '4-seater': four_seater_score
        }
    
    def _calculate_assignment_confidence(self, two_score: float, four_score: float, 
                                         data_quality: float = 1.0) -> float:
        """
        Calculate assignment confidence per paper methodology (lines 1011-1017).
        
        Combines score gap and demographic data quality:
        - Larger gap between scores = higher confidence
        - Better data quality = higher confidence  
        - High confidence: >0.8 (clear demographic preference)
        - Low confidence: <0.6 (ambiguous or poor data)
        
        Args:
            two_score: 2-seater preference score
            four_score: 4-seater preference score
            data_quality: Data quality indicator (0-1 scale, default 1.0)
            
        Returns:
            Confidence score between 0 and 1
        """
        # Get confidence configuration if available
        if hasattr(self, 'scoring_weights'):
            confidence_config = self.scoring_weights.get('validation_parameters', {}) \
                .get('confidence_weights', {})
            gap_weight = confidence_config.get('gap_weight', 0.7)
            quality_weight = confidence_config.get('quality_weight', 0.3)
        else:
            # Fallback defaults
            gap_weight = 0.7
            quality_weight = 0.3
        
        # Calculate gap-based confidence (0-1 scale)
        # Larger gap = more confident assignment
        score_gap = abs(two_score - four_score)
        # Normalize: 0.5 gap = maximum confidence, scale linearly
        gap_confidence = min(1.0, score_gap / 0.5)
        
        # Combine gap and quality using configured weights
        confidence = gap_weight * gap_confidence + quality_weight * data_quality
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _make_ev_type_assignments(self, ev_type_scores: Dict[str, pd.Series], 
                                 household_size_factors: pd.Series) -> List[EVTypeAssignment]:
        """
        Make final EV type assignments using threshold-based classification.
        
        Implements paper's assignment logic using ±threshold:
        - 4-seater: S_4 - S_2 > threshold
        - 2-seater: S_2 - S_4 > threshold
        - Mixed: |S_4 - S_2| ≤ threshold (heterogeneous household compositions)
        
        Confidence scoring per paper (lines 1011-1017):
        - Based on score gap magnitude and data quality
        - >0.8 = high confidence (clear preference)
        - <0.6 = low confidence (ambiguous)
        
        Args:
            ev_type_scores: Dictionary with '2-seater' and '4-seater' score series
            household_size_factors: Series of HSF values from @eq-household-size-factor
        """
        assignments = []
        
        two_seater_scores = ev_type_scores['2-seater']
        four_seater_scores = ev_type_scores['4-seater']
        
        # Get score thresholds for assignment
        two_seater_threshold = self.assignment_weights.get('two_seater_threshold', 0.7)
        four_seater_threshold = self.assignment_weights.get('four_seater_threshold', 0.5)
        assignment_threshold = self.assignment_weights.get('assignment_threshold', 0.1)
        
        for i in range(len(two_seater_scores)):
            two_score = two_seater_scores.iloc[i]
            four_score = four_seater_scores.iloc[i]
            hsf_value = household_size_factors.iloc[i]
            
            # Calculate score difference
            score_diff = four_score - two_score

            # Assignment logic based on relative score comparison
            # Uses assignment_threshold as the margin required for clear preference
            # Default to 4-seater reflects real-world sales (~10:1 ratio)

            if two_score >= two_seater_threshold and score_diff < -assignment_threshold:
                # 2-seater clearly preferred (meets threshold and two_score > four_score + margin)
                ev_type = "2-seater"
            elif four_score >= four_seater_threshold and score_diff > assignment_threshold:
                # 4-seater clearly preferred (meets threshold and four_score > two_score + margin)
                ev_type = "4-seater"
            elif four_score >= four_seater_threshold and abs(score_diff) <= assignment_threshold:
                # Scores are close but 4-seater meets threshold - default to 4-seater
                ev_type = "4-seater"
            else:
                # Neither meets threshold clearly - assign mixed
                ev_type = "mixed"
            
            # Calculate confidence based on gap and data quality per paper
            confidence = self._calculate_assignment_confidence(two_score, four_score)
            
            assignment = EVTypeAssignment(
                ev_type=ev_type,
                household_size_factor=hsf_value,  # Actual HSF from @eq-household-size-factor
                affordability_factor=1.0,  # Will be updated by AffordabilityAssessor
                assignment_confidence=confidence
            )
            
            assignments.append(assignment)
        
        return assignments
    
    def _validate_ev_type_assignments(self, result: gpd.GeoDataFrame) -> None:
        """Validate EV type assignments"""
        assignments = result['ev_type_assignment']
        
        # Check valid assignment types
        valid_types = {'2-seater', '4-seater', 'mixed', 'not_applicable'}
        invalid_assignments = set(assignments) - valid_types
        
        if invalid_assignments:
            raise ValueError(f"Invalid EV type assignments found: {invalid_assignments}")
        
        # Check confidence scores
        confidences = result['assignment_confidence']
        if confidences.min() < 0 or confidences.max() > 1:
            logger.warning(f"Assignment confidence out of range [0,1]: [{confidences.min():.3f}, {confidences.max():.3f}]")
        
        logger.info("EV type assignment validation passed")
    
    def _log_assignment_summary(self, result: gpd.GeoDataFrame) -> None:
        """Log summary of EV type assignments"""
        assignments = result['ev_type_assignment']
        assignment_counts = assignments.value_counts()
        
        logger.info("EV Type Assignment Summary:")
        for ev_type, count in assignment_counts.items():
            percentage = (count / len(assignments)) * 100
            logger.info(f"  {ev_type}: {count} areas ({percentage:.1f}%)")
        
        mean_confidence = result['assignment_confidence'].mean()
        logger.info(f"Mean assignment confidence: {mean_confidence:.3f}")


class AffordabilityAssessor:
    """
    Specialized class for incorporating affordability constraints into EV type selection
    using income proxies and car ownership factors.
    """
    
    def __init__(self, scoring_weights: Optional[Dict[str, Any]] = None):
        """
        Initialize affordability assessor with scoring weights from configuration.

        Args:
            scoring_weights: Dictionary containing affordability_assessment configuration
        """
        if scoring_weights is None:
            scoring_weights = _load_scoring_weights()

        # Load affordability configuration or use defaults
        affordability_config = scoring_weights.get('affordability_assessment', {})
        
        # Affordability factors by social grade (income proxy)
        # Per @eq-income-affordability: AB (1.0), C1 (0.8), C2 (0.6), DE (0.4)
        self.affordability_factors = affordability_config.get('social_grade_factors', {
            'AB.': 1.0,   # High income - no constraints
            'C1.': 0.8,   # Upper middle - minor constraints
            'C2.': 0.6,   # Lower middle - moderate constraints  
            'DE.': 0.4    # Lower income - significant constraints
        })
        
        # Car ownership impact on affordability (existing car = lower EV affordability)
        # Per @eq-car-ownership-impact: No cars (1.0), One car (0.8), Two+ cars (0.6)
        # Three+ cars treated identically to Two+ cars (0.6)
        self.car_ownership_factors = affordability_config.get('car_ownership_factors', {
            'No.cars': 1.0,        # No existing car cost
            'One.car': 0.8,        # One car to potentially replace
            'Two.cars': 0.6,       # Multiple car costs
            'Two.or.more.cars': 0.6,  # Multiple car costs
            'Three.or.more.cars': 0.6  # Treated same as Two+ cars
        })
        
        # Affordability blend weights: A_i = 0.7*I_i + 0.3*C_i per @eq-affordability-factor
        blend_config = affordability_config.get('affordability_blend', {})
        self.income_weight = blend_config.get('income', 0.7)
        self.car_ownership_weight = blend_config.get('car_ownership', 0.3)
        
        # Constraint thresholds
        threshold_config = affordability_config.get('constraint_thresholds', {})
        self.critical_threshold = threshold_config.get('critical', 0.3)
        self.moderate_threshold = threshold_config.get('moderate', 0.5)
        
        # EV type cost assumptions (relative costs)
        self.ev_type_costs = {
            '2-seater': 0.7,  # Lower cost option
            '4-seater': 1.0   # Higher cost option
        }
    
    def apply_affordability_constraints(self, demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Apply affordability constraints to EV type assignments.
        
        Args:
            demographics: GeoDataFrame with EV type assignments and demographic data
            
        Returns:
            GeoDataFrame with affordability-adjusted EV type assignments
            
        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Applying affordability constraints to EV type assignments")
        
        # Validate input data
        self._validate_input_data(demographics)
        
        # Create copy to avoid modifying original data
        result = demographics.copy()
        
        # Preprocess data
        result = self._preprocess_data_for_affordability(result)
        
        # Calculate income-based affordability
        income_affordability = self._calculate_income_affordability(result)
        
        # Calculate car ownership impact
        car_ownership_impact = self._calculate_car_ownership_impact(result)
        
        # Combine affordability factors
        overall_affordability = self._combine_affordability_factors(
            income_affordability, car_ownership_impact
        )
        
        # Apply constraints to EV type assignments
        adjusted_assignments = self._apply_constraints_to_assignments(
            result, overall_affordability
        )
        
        # Update results
        result['affordability_factor'] = overall_affordability
        result['income_affordability'] = income_affordability
        result['car_ownership_impact'] = car_ownership_impact
        result['ev_type_assignment'] = adjusted_assignments
        
        # Validate results
        self._validate_affordability_results(result)
        
        logger.info("Applied affordability constraints")
        self._log_affordability_impact(result, demographics)
        
        return result
    
    def _validate_input_data(self, demographics: gpd.GeoDataFrame) -> None:
        """Validate input data has required columns"""
        required_columns = ['ev_type_assignment']
        required_patterns = ['AB.', 'C1.', 'C2.', 'DE.', 'car']
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in demographics.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for demographic patterns
        available_columns = demographics.columns.tolist()
        missing_patterns = []
        
        for pattern in required_patterns:
            if not any(pattern in col for col in available_columns):
                missing_patterns.append(pattern)
        
        if missing_patterns:
            raise ValueError(f"Missing demographic patterns for affordability assessment: {missing_patterns}")
    
    def _preprocess_data_for_affordability(self, demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Preprocess data for affordability calculations"""
        # Get all numeric columns (exclude geo_code, geometry, label, name)
        exclude_cols = ['geo_code', 'geometry', 'label', 'name', 'ev_type_assignment']
        numeric_cols = [col for col in demographics.columns if col not in exclude_cols]
        
        # Convert to numeric, replacing non-numeric values with 0
        for col in numeric_cols:
            demographics[col] = pd.to_numeric(demographics[col], errors='coerce').fillna(0)
        
        return demographics
    
    def _calculate_income_affordability(self, demographics: gpd.GeoDataFrame) -> pd.Series:
        """Calculate income-based affordability using social grade as proxy"""
        affordability_scores = pd.Series(0.0, index=demographics.index)
        total_population = pd.Series(0.0, index=demographics.index)
        
        for social_grade, affordability_factor in self.affordability_factors.items():
            # Find columns for this social grade
            grade_cols = [col for col in demographics.columns if social_grade in col]
            
            if grade_cols:
                grade_total = demographics[grade_cols].sum(axis=1)
                affordability_scores += grade_total * affordability_factor
                total_population += grade_total
        
        # Normalize by total population
        affordability_scores = affordability_scores / (total_population + 1e-6)
        
        return np.clip(affordability_scores, 0.0, 1.0)
    
    def _calculate_car_ownership_impact(self, demographics: gpd.GeoDataFrame) -> pd.Series:
        """Calculate car ownership impact on EV affordability"""
        ownership_scores = pd.Series(0.0, index=demographics.index)
        total_households = pd.Series(0.0, index=demographics.index)
        
        for car_category, ownership_factor in self.car_ownership_factors.items():
            # Find columns for this car ownership category
            car_cols = [col for col in demographics.columns if car_category in col]
            
            if car_cols:
                car_total = demographics[car_cols].sum(axis=1)
                ownership_scores += car_total * ownership_factor
                total_households += car_total
        
        # Normalize by total households
        ownership_scores = ownership_scores / (total_households + 1e-6)
        
        return np.clip(ownership_scores, 0.0, 1.0)
    
    def _combine_affordability_factors(self, income_affordability: pd.Series, 
                                     car_ownership_impact: pd.Series) -> pd.Series:
        """
        Combine income and car ownership factors.
        
        Per @eq-affordability-factor: A_i = 0.7*I_i + 0.3*C_i
        Uses configurable weights from affordability_assessment.affordability_blend
        """
        combined_affordability = (
            income_affordability * self.income_weight +
            car_ownership_impact * self.car_ownership_weight
        )
        
        return np.clip(combined_affordability, 0.0, 1.0)
    
    def _apply_constraints_to_assignments(self, demographics: gpd.GeoDataFrame, 
                                        affordability: pd.Series) -> pd.Series:
        """
        Apply affordability constraints to EV type assignments.
        
        Per paper Section 7:
        - If A < 0.3 and assignment = "4-seater": Adjust to "2-seater"
        - If 0.3 ≤ A < 0.5 and assignment = "4-seater": Adjust to "mixed"
        - If A < 0.3 and assignment = "mixed": Adjust to "2-seater"
        """
        original_assignments = demographics['ev_type_assignment'].copy()
        adjusted_assignments = original_assignments.copy()
        
        # Apply constraints based on affordability levels
        for i in range(len(adjusted_assignments)):
            current_assignment = original_assignments.iloc[i]
            affordability_score = affordability.iloc[i]
            
            # If affordability is low and assigned 4-seater, consider downgrading
            if current_assignment == '4-seater' and affordability_score < self.moderate_threshold:
                # Force to 2-seater if very low affordability (A < 0.3)
                if affordability_score < self.critical_threshold:
                    adjusted_assignments.iloc[i] = '2-seater'
                # Change to mixed if moderate affordability constraints (0.3 ≤ A < 0.5)
                else:
                    adjusted_assignments.iloc[i] = 'mixed'
            
            # If mixed assignment and very low affordability, force to 2-seater (A < 0.3)
            elif current_assignment == 'mixed' and affordability_score < self.critical_threshold:
                adjusted_assignments.iloc[i] = '2-seater'
        
        return adjusted_assignments
    
    def _validate_affordability_results(self, result: gpd.GeoDataFrame) -> None:
        """Validate affordability assessment results"""
        affordability_scores = result['affordability_factor']
        
        # Check score ranges
        if affordability_scores.min() < 0 or affordability_scores.max() > 1:
            logger.warning(f"Affordability scores out of range [0,1]: [{affordability_scores.min():.3f}, {affordability_scores.max():.3f}]")
        
        # Check for valid assignments
        assignments = result['ev_type_assignment']
        valid_types = {'2-seater', '4-seater', 'mixed', 'not_applicable'}
        invalid_assignments = set(assignments) - valid_types
        
        if invalid_assignments:
            raise ValueError(f"Invalid EV type assignments after affordability adjustment: {invalid_assignments}")
        
        logger.info("Affordability assessment validation passed")
    
    def _log_affordability_impact(self, result: gpd.GeoDataFrame, original: gpd.GeoDataFrame) -> None:
        """Log the impact of affordability constraints"""
        original_assignments = original['ev_type_assignment']
        adjusted_assignments = result['ev_type_assignment']
        
        # Count changes
        changes = (original_assignments != adjusted_assignments).sum()
        total = len(original_assignments)
        
        logger.info(f"Affordability constraints changed {changes}/{total} assignments ({changes/total*100:.1f}%)")
        
        # Log assignment distribution after constraints
        assignment_counts = adjusted_assignments.value_counts()
        logger.info("Final EV Type Assignment Distribution:")
        for ev_type, count in assignment_counts.items():
            percentage = (count / len(adjusted_assignments)) * 100
            logger.info(f"  {ev_type}: {count} areas ({percentage:.1f}%)")
        
        mean_affordability = result['affordability_factor'].mean()
        logger.info(f"Mean affordability factor: {mean_affordability:.3f}")


def validate_ev_type_assignments(ev_type_data: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Validate EV type assignments for consistency and realism.
    
    Args:
        ev_type_data: GeoDataFrame with EV type assignments
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating EV type assignments")
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'statistics': {}
    }
    
    required_columns = ['ev_type_assignment', 'assignment_confidence']
    missing_columns = [col for col in required_columns if col not in ev_type_data.columns]
    
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        return validation_results
    
    assignments = ev_type_data['ev_type_assignment']
    confidences = ev_type_data['assignment_confidence']
    
    # Validate assignment types
    valid_types = {'2-seater', '4-seater', 'mixed', 'not_applicable'}
    invalid_types = set(assignments) - valid_types
    
    if invalid_types:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Invalid EV types found: {invalid_types}")
    
    # Validate confidence scores
    if confidences.min() < 0 or confidences.max() > 1:
        validation_results['issues'].append(f"Confidence scores out of range [0,1]: [{confidences.min():.3f}, {confidences.max():.3f}]")
    
    # Calculate statistics
    assignment_counts = assignments.value_counts()
    validation_results['statistics'] = {
        'assignment_distribution': assignment_counts.to_dict(),
        'mean_confidence': confidences.mean(),
        'confidence_std': confidences.std(),
        'total_areas': len(assignments)
    }
    
    # Check for unrealistic distributions
    if len(assignment_counts) == 1:
        validation_results['issues'].append("All areas have same EV type assignment - check input data")
    
    if confidences.std() < 0.01:
        validation_results['issues'].append("Very low variance in assignment confidence")
    
    if validation_results['issues']:
        validation_results['is_valid'] = False
    
    logger.info(f"EV type assignment validation {'passed' if validation_results['is_valid'] else 'failed'}")

    return validation_results


# =============================================================================
# 5-Step Replaceability Logic Functions (per eq_revised.qmd)
# =============================================================================

def assign_ev_type_by_score(
    S_2seater: Union[float, pd.Series],
    S_4seater: Union[float, pd.Series],
    threshold: float = 0.1
) -> Union[str, pd.Series]:
    """
    Assign EV type based on score comparison per @eq-ev-type-assignment.

    EV_type_i = {
        "2-seater"  if S_2seater - S_4seater > threshold
        "4-seater"  if S_4seater - S_2seater > threshold
        "mixed"     otherwise
    }

    Args:
        S_2seater: 2-seater suitability score [0,1]
        S_4seater: 4-seater suitability score [0,1]
        threshold: Score difference threshold (default 0.1)

    Returns:
        EV type: "2-seater", "4-seater", or "mixed"
    """
    if isinstance(S_2seater, pd.Series):
        result = pd.Series("mixed", index=S_2seater.index)
        result.loc[S_2seater - S_4seater > threshold] = "2-seater"
        result.loc[S_4seater - S_2seater > threshold] = "4-seater"
        return result
    else:
        if S_2seater - S_4seater > threshold:
            return "2-seater"
        elif S_4seater - S_2seater > threshold:
            return "4-seater"
        else:
            return "mixed"
