"""
Trip Purpose Analysis Module for Frugal EV Analysis

This module provides classes and functions for analyzing trip purposes and their suitability
for EV conversion, including regularity assessment and charging flexibility evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripPurpose(Enum):
    """Enumeration of the 14 trip purposes with standardized names"""
    COMMUTING = "Commuting"
    SHOPPING = "Shopping"
    VISITING_FRIENDS_RELATIVES = "Visiting friends or relatives"
    EDUCATION = "Education"
    SPORT_ENTERTAINMENT = "Sport/entertainment"
    OTHER_PERSONAL_BUSINESS = "Other personal business"
    EATING_DRINKING = "Eating/drinking"
    OTHER_JOURNEY = "Other journey"
    BUSINESS = "Business"
    HEALTH_VISITS = "Health visits"
    ESCORT = "Escort"
    HOLIDAY_DAYTRIP = "Holiday/daytrip"
    RECREATION = "Recreation"
    SOCIAL_VISITS = "Social visits"


@dataclass
class TripPurposeProfile:
    """Stage 3 input profile for a given trip purpose."""

    purpose: str
    regularity: float
    predictability: float
    dwell_flexibility: float
    temporal_sensitivity: float
    destination_compatibility: float
    home_charging_feasibility: float


@dataclass
class TripPurposeMetrics:
    """Calculated Stage 3 scores for a trip purpose."""

    purpose: str
    charging_flexibility: float
    suitability_weight: float
    classification: str


@dataclass
class ChargingFlexibility:
    """Container for charging flexibility assessment by trip purpose"""

    purpose: str
    dwell_time_flexibility: float  # 0-1 scale
    timing_flexibility: float      # 0-1 scale
    destination_charging_suitable: bool
    home_charging_dependency: float  # 0-1 scale


class TripPurposeAnalyzer:
    """Stage 3 trip purpose analyzer driven by configuration values."""

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        scoring_weights: Optional[Dict[str, Any]] = None,
        destination_scores: Optional[Dict[str, float]] = None,
        home_feasibility: Optional[Dict[str, float]] = None,
    ):
        self._config_path = Path(config_path) if config_path else Path("scoring_weights.json")
        self.scoring_weights = scoring_weights or self._load_scoring_weights(self._config_path)

        self.trip_config = self.scoring_weights.get("trip_purpose_parameters")
        if not self.trip_config:
            raise ValueError(
                "Trip purpose configuration missing from scoring_weights.json."
            )

        self.integration_weights = self.trip_config.get(
            "charging_flexibility_weights",
            {"dwell": 0.4, "timing": 0.3, "destination": 0.2, "home": 0.1},
        )
        self.suitability_weights = self.trip_config.get(
            "suitability_weights",
            {"regularity": 0.35, "predictability": 0.35, "charging_flexibility": 0.30},
        )
        self.thresholds = self.trip_config.get(
            "classification_thresholds", {"high": 0.8, "moderate": 0.6}
        )

        self.input_parameters: Dict[str, Dict[str, float]] = self.trip_config.get(
            "input_parameters", {}
        )
        if not self.input_parameters:
            raise ValueError("No trip purpose input parameters defined in configuration.")

        defaults = self.trip_config.get("defaults", {})
        default_dest = defaults.get("destination_compatibility", 0.5)
        default_home = defaults.get("home_charging_feasibility", 0.5)

        baseline_dest = self.trip_config.get("baseline_destination_compatibility", {})
        baseline_home = self.trip_config.get("baseline_home_charging_feasibility", {})

        self.destination_scores = {
            purpose: (destination_scores or {}).get(
                purpose,
                baseline_dest.get(purpose, default_dest),
            )
            for purpose in self.input_parameters
        }

        self.home_feasibility_scores = {
            purpose: (home_feasibility or {}).get(
                purpose,
                baseline_home.get(purpose, default_home),
            )
            for purpose in self.input_parameters
        }

        self._purpose_lookup = {
            purpose.lower(): purpose for purpose in self.input_parameters
        }

        self.purpose_profiles = self._build_profiles()
        self.metrics_by_purpose = self._compute_metrics()
        self.charging_flexibility = self._build_charging_flexibility_summary()

    def _load_scoring_weights(self, config_path: Path) -> Dict[str, Any]:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with config_path.open() as config_file:
            return json.load(config_file)

    def _build_profiles(self) -> Dict[str, TripPurposeProfile]:
        profiles: Dict[str, TripPurposeProfile] = {}
        for purpose, params in self.input_parameters.items():
            profiles[purpose] = TripPurposeProfile(
                purpose=purpose,
                regularity=params.get("regularity", 0.5),
                predictability=params.get("predictability", 0.5),
                dwell_flexibility=params.get("dwell_flexibility", 0.5),
                temporal_sensitivity=params.get("temporal_sensitivity", 0.5),
                destination_compatibility=self.destination_scores.get(purpose, 0.5),
                home_charging_feasibility=self.home_feasibility_scores.get(purpose, 0.5),
            )
        return profiles

    def _compute_metrics(self) -> Dict[str, TripPurposeMetrics]:
        metrics: Dict[str, TripPurposeMetrics] = {}
        for purpose, profile in self.purpose_profiles.items():
            cf_score = self._calculate_charging_flexibility(profile)
            suitability = self._calculate_suitability_weight(profile, cf_score)
            metrics[purpose] = TripPurposeMetrics(
                purpose=purpose,
                charging_flexibility=cf_score,
                suitability_weight=suitability,
                classification=self._classify_suitability(suitability),
            )
        return metrics

    def _build_charging_flexibility_summary(self) -> Dict[str, ChargingFlexibility]:
        summary: Dict[str, ChargingFlexibility] = {}
        for purpose, profile in self.purpose_profiles.items():
            timing_flex = max(0.0, min(1.0, 1.0 - profile.temporal_sensitivity))
            destination_flag = self.destination_scores.get(purpose, 0.5) >= 0.6
            home_dependency = 1.0 - self.home_feasibility_scores.get(purpose, 0.5)
            home_dependency = max(0.0, min(1.0, home_dependency))
            summary[purpose] = ChargingFlexibility(
                purpose=purpose,
                dwell_time_flexibility=profile.dwell_flexibility,
                timing_flexibility=timing_flex,
                destination_charging_suitable=destination_flag,
                home_charging_dependency=home_dependency,
            )
        return summary

    def _calculate_charging_flexibility(self, profile: TripPurposeProfile) -> float:
        weights = self.integration_weights
        cf_score = (
            weights.get("dwell", 0.4) * profile.dwell_flexibility
            + weights.get("timing", 0.3) * (1.0 - profile.temporal_sensitivity)
            + weights.get("destination", 0.2)
            * self.destination_scores.get(profile.purpose, 0.5)
            + weights.get("home", 0.1)
            * self.home_feasibility_scores.get(profile.purpose, 0.5)
        )
        return max(0.0, min(1.0, cf_score))

    def _calculate_suitability_weight(
        self, profile: TripPurposeProfile, charging_flexibility: float
    ) -> float:
        weights = self.suitability_weights
        score = (
            weights.get("regularity", 0.35) * profile.regularity
            + weights.get("predictability", 0.35) * profile.predictability
            + weights.get("charging_flexibility", 0.30) * charging_flexibility
        )
        return max(0.0, min(1.0, score))

    def _classify_suitability(self, suitability_weight: float) -> str:
        high_threshold = self.thresholds.get("high", 0.8)
        moderate_threshold = self.thresholds.get("moderate", 0.6)
        if suitability_weight >= high_threshold:
            return "high"
        if suitability_weight >= moderate_threshold:
            return "moderate"
        return "low"

    def _resolve_purpose_key(self, purpose: str) -> Optional[str]:
        if purpose in self.purpose_profiles:
            return purpose
        purpose_lower = purpose.lower().strip()
        if purpose_lower in self._purpose_lookup:
            return self._purpose_lookup[purpose_lower]
        # Fallback fuzzy mapping for legacy names
        fuzzy = self._fuzzy_match_purpose(purpose)
        if fuzzy and fuzzy in self.purpose_profiles:
            return fuzzy
        return None

    def _get_profile(self, purpose: str) -> TripPurposeProfile:
        resolved = self._resolve_purpose_key(str(purpose))
        if resolved and resolved in self.purpose_profiles:
            return self.purpose_profiles[resolved]
        return TripPurposeProfile(
            purpose=str(purpose),
            regularity=0.5,
            predictability=0.5,
            dwell_flexibility=0.5,
            temporal_sensitivity=0.5,
            destination_compatibility=0.5,
            home_charging_feasibility=0.5,
        )

    def _get_metrics(self, purpose: str) -> TripPurposeMetrics:
        resolved = self._resolve_purpose_key(str(purpose))
        if resolved and resolved in self.metrics_by_purpose:
            return self.metrics_by_purpose[resolved]
        return TripPurposeMetrics(
            purpose=str(purpose),
            charging_flexibility=0.5,
            suitability_weight=0.5,
            classification='unknown',
        )

    def _fuzzy_match_purpose(self, purpose: str) -> Optional[str]:
        purpose_lower = purpose.lower().strip()
        purpose_mappings = {
            'visit hospital or other health': 'Health visits',
            'hospital visits': 'Health visits',
            'health': 'Health visits',
            'medical': 'Health visits',
            'shop': 'Shopping',
            'retail': 'Shopping',
            'eating/drinking': 'Eating/drinking',
            'eat/drink': 'Eating/drinking',
            'restaurant': 'Eating/drinking',
            'dining': 'Eating/drinking',
            'sport/entertainment': 'Sport/entertainment',
            'sports/entertainment': 'Sport/entertainment',
            'entertainment': 'Sport/entertainment',
            'leisure': 'Sport/entertainment',
            'other journey': 'Other journey',
            'journey': 'Other journey',
            'travel': 'Other journey',
            'personal business': 'Other personal business',
            'other personal': 'Other personal business',
            'errands': 'Other personal business',
            'social': 'Social visits',
            'visiting friends': 'Visiting friends or relatives',
            'visit friends': 'Visiting friends or relatives',
            'friends': 'Visiting friends or relatives',
            'relatives': 'Visiting friends or relatives',
            'work': 'Commuting',
            'commute': 'Commuting',
            'job': 'Commuting',
            'school': 'Education',
            'university': 'Education',
            'college': 'Education',
            'study': 'Education',
            'holiday': 'Holiday/daytrip',
            'vacation': 'Holiday/daytrip',
            'daytrip': 'Holiday/daytrip',
            'trip': 'Holiday/daytrip',
            'recreation': 'Recreation',
            'recreational': 'Recreation',
            'hobby': 'Recreation',
        }
        if purpose_lower in purpose_mappings:
            return purpose_mappings[purpose_lower]
        for variant, canonical in purpose_mappings.items():
            if variant in purpose_lower or purpose_lower in variant:
                return canonical
        return None

    def update_external_scores(
        self,
        destination_scores: Optional[Dict[str, float]] = None,
        home_feasibility: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update Stage 4 / Stage 2 inputs and recompute metrics."""

        if destination_scores:
            for key, value in destination_scores.items():
                resolved = self._resolve_purpose_key(key)
                if resolved:
                    self.destination_scores[resolved] = value
        if home_feasibility:
            for key, value in home_feasibility.items():
                resolved = self._resolve_purpose_key(key)
                if resolved:
                    self.home_feasibility_scores[resolved] = value

        self.purpose_profiles = self._build_profiles()
        self.metrics_by_purpose = self._compute_metrics()
        self.charging_flexibility = self._build_charging_flexibility_summary()
    
    def get_suitability_weight(self, purpose: str) -> float:
        """Return Stage 3 suitability weight for the provided purpose."""
        resolved = self._resolve_purpose_key(purpose)
        if resolved:
            return self.metrics_by_purpose[resolved].suitability_weight

        logger.warning(
            "Unknown trip purpose: '%s'. Using default weight 0.5", purpose
        )
        return 0.5

    def get_regularity_score(self, purpose: str) -> float:
        """
        Get the regularity score for a specific trip purpose.
        
        Args:
            purpose: Trip purpose name
            
        Returns:
            Regularity score (0-1 scale)
        """
        resolved = self._resolve_purpose_key(purpose)
        if resolved:
            return self.purpose_profiles[resolved].regularity

        logger.warning(
            "Unknown trip purpose: '%s'. Using default regularity 0.5", purpose
        )
        return 0.5

    def get_predictability_score(self, purpose: str) -> float:
        """
        Get the predictability score for a specific trip purpose.
        
        Args:
            purpose: Trip purpose name
            
        Returns:
            Predictability score (0-1 scale)
        """
        resolved = self._resolve_purpose_key(purpose)
        if resolved:
            return self.purpose_profiles[resolved].predictability

        logger.warning(
            "Unknown trip purpose: '%s'. Using default predictability 0.5",
            purpose,
        )
        return 0.5
    
    def analyze_trip_purposes(self, od_data: gpd.GeoDataFrame) -> Dict[str, Dict[str, float]]:
        """
        Analyze all trip purposes in the OD dataset and return comprehensive metrics.
        
        Args:
            od_data: GeoDataFrame with trip data including purpose column
            
        Returns:
            Dict with purpose-specific analysis results
        """
        results = {}
        
        # Identify the trip purpose column
        purpose_col = self._identify_purpose_column(od_data)
        if not purpose_col:
            raise ValueError("No trip purpose column found in OD data")
        
        # Get unique purposes in the data
        unique_purposes = od_data[purpose_col].unique()
        
        for purpose in unique_purposes:
            if pd.isna(purpose):
                continue
                
            purpose_data = od_data[od_data[purpose_col] == purpose]
            resolved = self._resolve_purpose_key(str(purpose))
            suitability_weight = self.get_suitability_weight(purpose)
            metrics = self.metrics_by_purpose.get(resolved) if resolved else None
            profile = self.purpose_profiles.get(resolved) if resolved else None

            results[purpose] = {
                'trip_count': len(purpose_data),
                'suitability_weight': suitability_weight,
                'regularity_score': self.get_regularity_score(purpose),
                'predictability_score': self.get_predictability_score(purpose),
                'charging_flexibility': self.get_charging_flexibility_score(purpose),
                'suitability_class': metrics.classification if metrics else 'unknown',
                'destination_compatibility': profile.destination_compatibility if profile else np.nan,
                'home_charging_feasibility': profile.home_charging_feasibility if profile else np.nan,
                'total_trip_volume': purpose_data.get('All', pd.Series([0])).sum() if 'All' in purpose_data.columns else len(purpose_data)
            }
        
        return results
    
    def _identify_purpose_column(self, od_data: gpd.GeoDataFrame) -> Optional[str]:
        """Identify the column containing trip purpose information"""
        possible_columns = ['purpose', 'trip_purpose', 'Purpose', 'Trip_Purpose', 'PURPOSE']
        
        for col in possible_columns:
            if col in od_data.columns:
                return col
        
        # Look for columns containing 'purpose' in the name
        for col in od_data.columns:
            if 'purpose' in col.lower():
                return col
        
        return None
    
    def get_charging_flexibility_score(self, purpose: str) -> float:
        """
        Get the overall charging flexibility score for a trip purpose.

        Args:
            purpose: Trip purpose name
            
        Returns:
            Overall charging flexibility score (0-1 scale)
        """
        resolved = self._resolve_purpose_key(purpose)
        if resolved:
            return self.metrics_by_purpose[resolved].charging_flexibility

        logger.warning(
            "Unknown trip purpose: '%s'. Using default flexibility 0.5", purpose
        )
        return 0.5

    def get_suitability_classification(self, purpose: str) -> str:
        """Return categorical suitability classification (high/moderate/low)."""

        resolved = self._resolve_purpose_key(purpose)
        if resolved:
            return self.metrics_by_purpose[resolved].classification

        logger.warning(
            "Unknown trip purpose: '%s'. Using default classification 'unknown'",
            purpose,
        )
        return 'unknown'
    
    def calculate_purpose_weights_for_od_data(self, od_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate suitability weights for all trips in the OD dataset.
        
        Args:
            od_data: GeoDataFrame with trip data
            
        Returns:
            GeoDataFrame with added suitability weight columns
        """
        result_data = od_data.copy()
        
        purpose_col = self._identify_purpose_column(od_data)
        if not purpose_col:
            raise ValueError("No trip purpose column found in OD data")
        
        # Add suitability weights
        result_data['ev_suitability_weight'] = result_data[purpose_col].apply(
            self.get_suitability_weight
        )
        
        # Add regularity and predictability scores
        result_data['regularity_score'] = result_data[purpose_col].apply(
            self.get_regularity_score
        )

        result_data['predictability_score'] = result_data[purpose_col].apply(
            self.get_predictability_score
        )

        # Add charging flexibility score
        result_data['charging_flexibility'] = result_data[purpose_col].apply(
            self.get_charging_flexibility_score
        )

        # Add destination/home configuration values for transparency
        result_data['destination_compatibility'] = result_data[purpose_col].apply(
            lambda value: self._get_profile(value).destination_compatibility
        )

        result_data['home_charging_feasibility'] = result_data[purpose_col].apply(
            lambda value: self._get_profile(value).home_charging_feasibility
        )

        # Add suitability classification
        result_data['suitability_class'] = result_data[purpose_col].apply(
            lambda value: self._get_metrics(value).classification
        )

        return result_data
    
    def validate_purpose_weights(self) -> Dict[str, Any]:
        """
        Validate that all purpose weights are within expected ranges.
        
        Returns:
            Dict with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        for purpose, profile in self.purpose_profiles.items():
            for attr_name in (
                'regularity',
                'predictability',
                'dwell_flexibility',
                'temporal_sensitivity',
                'destination_compatibility',
                'home_charging_feasibility',
            ):
                value = getattr(profile, attr_name)
                if not (0.0 <= value <= 1.0):
                    validation_results['errors'].append(
                        f"{purpose}: {attr_name} {value} not in [0,1]"
                    )
                    validation_results['valid'] = False

        for purpose, metrics in self.metrics_by_purpose.items():
            if not (0.0 <= metrics.charging_flexibility <= 1.0):
                validation_results['errors'].append(
                    f"{purpose}: charging_flexibility {metrics.charging_flexibility} not in [0,1]"
                )
                validation_results['valid'] = False
            if not (0.0 <= metrics.suitability_weight <= 1.0):
                validation_results['errors'].append(
                    f"{purpose}: suitability_weight {metrics.suitability_weight} not in [0,1]"
                )
                validation_results['valid'] = False

        weights_list = [metrics.suitability_weight for metrics in self.metrics_by_purpose.values()]
        validation_results['summary'] = {
            'total_purposes': len(self.metrics_by_purpose),
            'mean_suitability_weight': float(np.mean(weights_list)) if weights_list else np.nan,
            'min_suitability_weight': float(np.min(weights_list)) if weights_list else np.nan,
            'max_suitability_weight': float(np.max(weights_list)) if weights_list else np.nan,
            'weight_range': float(np.max(weights_list) - np.min(weights_list)) if weights_list else np.nan
        }
        
        return validation_results


class RegularityAssessor:
    """
    Specialized class for assessing trip regularity and predictability patterns.
    """
    
    def __init__(self):
        """Initialize the RegularityAssessor"""
        pass
    
    def assess_trip_regularity(self, purpose: str, trip_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Assess the regularity characteristics of trips for a given purpose.
        
        Args:
            purpose: Trip purpose name
            trip_data: Optional DataFrame with temporal trip data
            
        Returns:
            Dict with regularity assessment metrics
        """
        # Get base regularity from TripPurposeAnalyzer
        analyzer = TripPurposeAnalyzer()
        base_regularity = analyzer.get_regularity_score(purpose)
        base_predictability = analyzer.get_predictability_score(purpose)
        
        assessment = {
            'base_regularity_score': base_regularity,
            'base_predictability_score': base_predictability,
            'temporal_consistency': base_regularity,  # Would be calculated from trip_data if available
            'weekly_pattern_strength': self._estimate_weekly_pattern_strength(purpose),
            'seasonal_variation': self._estimate_seasonal_variation(purpose),
            'charging_planning_feasibility': base_regularity * base_predictability
        }
        
        return assessment
    
    def _estimate_weekly_pattern_strength(self, purpose: str) -> float:
        """Estimate how strong the weekly pattern is for a trip purpose"""
        strong_weekly_pattern = [
            TripPurpose.COMMUTING.value,      # Strong weekday pattern
            TripPurpose.EDUCATION.value,      # Strong weekday pattern
            TripPurpose.ESCORT.value,         # Strong school-day pattern
        ]
        
        moderate_weekly_pattern = [
            TripPurpose.SHOPPING.value,       # Weekend peaks
            TripPurpose.SPORT_ENTERTAINMENT.value,  # Weekend/evening peaks
            TripPurpose.EATING_DRINKING.value, # Weekend peaks
            TripPurpose.HEALTH_VISITS.value,  # Weekday preference
        ]
        
        if purpose in strong_weekly_pattern:
            return 0.9
        elif purpose in moderate_weekly_pattern:
            return 0.6
        else:
            return 0.3
    
    def _estimate_seasonal_variation(self, purpose: str) -> float:
        """Estimate seasonal variation in trip patterns (lower = more consistent)"""
        low_seasonal_variation = [
            TripPurpose.COMMUTING.value,      # Consistent year-round
            TripPurpose.SHOPPING.value,       # Consistent year-round
            TripPurpose.HEALTH_VISITS.value,  # Consistent year-round
            TripPurpose.OTHER_PERSONAL_BUSINESS.value,
        ]
        
        high_seasonal_variation = [
            TripPurpose.EDUCATION.value,      # Term-time only
            TripPurpose.HOLIDAY_DAYTRIP.value, # Summer peaks
            TripPurpose.RECREATION.value,     # Weather dependent
            TripPurpose.SPORT_ENTERTAINMENT.value, # Seasonal sports
        ]
        
        if purpose in low_seasonal_variation:
            return 0.2  # Low variation = more predictable
        elif purpose in high_seasonal_variation:
            return 0.8  # High variation = less predictable
        else:
            return 0.5


def validate_trip_purpose_weights(analyzer: TripPurposeAnalyzer) -> Dict[str, Any]:
    """
    Validate trip purpose weight assignments and calculations.
    
    Args:
        analyzer: TripPurposeAnalyzer instance to validate
        
    Returns:
        Dict with validation results
    """
    return analyzer.validate_purpose_weights()


def get_all_trip_purposes() -> List[str]:
    """
    Get list of all supported trip purposes.
    
    Returns:
        List of trip purpose names
    """
    return [purpose.value for purpose in TripPurpose]


class ChargingFlexibilityAssessor:
    """
    Comprehensive charging flexibility assessment system for different trip types.
    
    Analyzes charging time constraints, flexibility scoring, and time constraint handling
    for various trip purposes to support EV adoption feasibility analysis.
    """
    
    def __init__(self):
        """Initialize the ChargingFlexibilityAssessor"""
        self.time_constraints = self._initialize_time_constraints()
        self.charging_scenarios = self._initialize_charging_scenarios()
    
    def _initialize_time_constraints(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize time constraints for different trip purposes.
        
        Returns:
            Dict mapping purposes to time constraint characteristics
        """
        constraints = {
            TripPurpose.COMMUTING.value: {
                'departure_flexibility': 0.2,    # Low - fixed work start times
                'arrival_flexibility': 0.3,      # Low - fixed work end times
                'dwell_time_min': 480,           # 8 hours minimum at work
                'dwell_time_max': 600,           # 10 hours maximum typical
                'charging_urgency': 0.7,         # High - daily necessity
                'schedule_predictability': 0.95   # Very predictable
            },
            TripPurpose.SHOPPING.value: {
                'departure_flexibility': 0.8,    # High - flexible timing
                'arrival_flexibility': 0.9,      # Very high - no fixed arrival
                'dwell_time_min': 60,            # 1 hour minimum shopping
                'dwell_time_max': 240,           # 4 hours maximum typical
                'charging_urgency': 0.4,         # Low - can postpone
                'schedule_predictability': 0.6    # Moderate predictability
            },
            TripPurpose.EDUCATION.value: {
                'departure_flexibility': 0.3,    # Low - class start times
                'arrival_flexibility': 0.2,      # Very low - must arrive on time
                'dwell_time_min': 180,           # 3 hours minimum class time
                'dwell_time_max': 480,           # 8 hours full day
                'charging_urgency': 0.8,         # High - regular necessity
                'schedule_predictability': 0.9    # Very predictable
            },
            TripPurpose.BUSINESS.value: {
                'departure_flexibility': 0.4,    # Low-moderate - meeting times
                'arrival_flexibility': 0.2,      # Very low - punctuality critical
                'dwell_time_min': 60,            # 1 hour minimum meeting
                'dwell_time_max': 480,           # 8 hours full day meetings
                'charging_urgency': 0.9,         # Very high - professional image
                'schedule_predictability': 0.5    # Moderate - varies by business
            },
            TripPurpose.HEALTH_VISITS.value: {
                'departure_flexibility': 0.3,    # Low - appointment times
                'arrival_flexibility': 0.1,      # Very low - critical appointments
                'dwell_time_min': 30,            # 30 minutes quick consultation
                'dwell_time_max': 480,           # 8 hours day surgery/treatment
                'charging_urgency': 0.9,         # Very high - cannot miss appointments
                'schedule_predictability': 0.8    # High - scheduled appointments
            },
            TripPurpose.ESCORT.value: {
                'departure_flexibility': 0.2,    # Low - others depend on timing
                'arrival_flexibility': 0.1,      # Very low - punctuality critical
                'dwell_time_min': 5,             # 5 minutes quick drop-off
                'dwell_time_max': 60,            # 1 hour waiting
                'charging_urgency': 0.8,         # High - others depend on reliability
                'schedule_predictability': 0.9    # Very predictable routine
            },
            TripPurpose.VISITING_FRIENDS_RELATIVES.value: {
                'departure_flexibility': 0.9,    # Very high - social flexibility
                'arrival_flexibility': 0.8,      # High - social tolerance
                'dwell_time_min': 120,           # 2 hours minimum social visit
                'dwell_time_max': 720,           # 12 hours full day visit
                'charging_urgency': 0.3,         # Low - can postpone if needed
                'schedule_predictability': 0.4    # Low - irregular patterns
            },
            TripPurpose.SOCIAL_VISITS.value: {
                'departure_flexibility': 0.8,    # High - social flexibility
                'arrival_flexibility': 0.7,      # High - social tolerance
                'dwell_time_min': 120,           # 2 hours minimum social time
                'dwell_time_max': 480,           # 8 hours social event
                'charging_urgency': 0.3,         # Low - social not critical
                'schedule_predictability': 0.4    # Low - irregular patterns
            },
            TripPurpose.SPORT_ENTERTAINMENT.value: {
                'departure_flexibility': 0.5,    # Moderate - event start times
                'arrival_flexibility': 0.3,      # Low - miss start of event
                'dwell_time_min': 120,           # 2 hours minimum event
                'dwell_time_max': 360,           # 6 hours long events
                'charging_urgency': 0.5,         # Moderate - leisure activity
                'schedule_predictability': 0.7    # High - scheduled events
            },
            TripPurpose.RECREATION.value: {
                'departure_flexibility': 0.9,    # Very high - leisure flexibility
                'arrival_flexibility': 0.9,      # Very high - no fixed schedule
                'dwell_time_min': 120,           # 2 hours minimum recreation
                'dwell_time_max': 600,           # 10 hours full day recreation
                'charging_urgency': 0.2,         # Very low - leisure activity
                'schedule_predictability': 0.3    # Low - weather/mood dependent
            },
            TripPurpose.EATING_DRINKING.value: {
                'departure_flexibility': 0.7,    # High - meal timing flexible
                'arrival_flexibility': 0.6,      # Moderate - reservation times
                'dwell_time_min': 60,            # 1 hour minimum meal
                'dwell_time_max': 240,           # 4 hours long meal/drinks
                'charging_urgency': 0.4,         # Low-moderate - social activity
                'schedule_predictability': 0.5    # Moderate - some routine
            },
            TripPurpose.OTHER_PERSONAL_BUSINESS.value: {
                'departure_flexibility': 0.6,    # Moderate - varies by business
                'arrival_flexibility': 0.5,      # Moderate - varies by type
                'dwell_time_min': 30,            # 30 minutes quick errand
                'dwell_time_max': 240,           # 4 hours complex business
                'charging_urgency': 0.6,         # Moderate - important but flexible
                'schedule_predictability': 0.5    # Moderate - varies by type
            },
            TripPurpose.HOLIDAY_DAYTRIP.value: {
                'departure_flexibility': 0.8,    # High - leisure travel
                'arrival_flexibility': 0.9,      # Very high - no fixed schedule
                'dwell_time_min': 240,           # 4 hours minimum day trip
                'dwell_time_max': 720,           # 12 hours full day trip
                'charging_urgency': 0.1,         # Very low - leisure activity
                'schedule_predictability': 0.2    # Very low - irregular
            },
            TripPurpose.OTHER_JOURNEY.value: {
                'departure_flexibility': 0.5,    # Moderate - varies by purpose
                'arrival_flexibility': 0.5,      # Moderate - varies by purpose
                'dwell_time_min': 60,            # 1 hour minimum
                'dwell_time_max': 360,           # 6 hours maximum typical
                'charging_urgency': 0.5,         # Moderate - varies by purpose
                'schedule_predictability': 0.4    # Low-moderate - varies
            }
        }
        
        return constraints
    
    def _initialize_charging_scenarios(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize charging scenarios and their feasibility by trip purpose.
        
        Returns:
            Dict mapping purposes to charging scenario suitability scores
        """
        scenarios = {}
        
        for purpose in TripPurpose:
            purpose_name = purpose.value
            constraints = self.time_constraints[purpose_name]
            
            scenarios[purpose_name] = {
                'home_overnight_charging': self._calculate_home_overnight_suitability(constraints),
                'destination_charging': self._calculate_destination_charging_suitability(constraints),
                'rapid_charging_enroute': self._calculate_rapid_charging_suitability(constraints),
                'workplace_charging': self._calculate_workplace_charging_suitability(constraints),
                'public_slow_charging': self._calculate_public_slow_charging_suitability(constraints)
            }
        
        return scenarios
    
    def _calculate_home_overnight_suitability(self, constraints: Dict[str, float]) -> float:
        """Calculate suitability of home overnight charging for trip constraints"""
        # Home charging works best for regular, predictable trips that start/end at home
        predictability_factor = constraints['schedule_predictability']
        urgency_factor = 1.0 - constraints['charging_urgency']  # Less urgent = more suitable for slow charging
        
        return (predictability_factor * 0.6 + urgency_factor * 0.4)
    
    def _calculate_destination_charging_suitability(self, constraints: Dict[str, float]) -> float:
        """Calculate suitability of destination charging for trip constraints"""
        # Destination charging works best for long dwell times with flexible timing
        dwell_time_factor = min(1.0, constraints['dwell_time_min'] / 120.0)  # 2+ hours ideal
        flexibility_factor = (constraints['departure_flexibility'] + constraints['arrival_flexibility']) / 2
        
        return (dwell_time_factor * 0.7 + flexibility_factor * 0.3)
    
    def _calculate_rapid_charging_suitability(self, constraints: Dict[str, float]) -> float:
        """Calculate suitability of rapid charging for trip constraints"""
        # Rapid charging works for urgent trips with short dwell times
        urgency_factor = constraints['charging_urgency']
        short_dwell_factor = max(0.0, 1.0 - constraints['dwell_time_min'] / 120.0)  # Better for <2 hour dwell
        
        return (urgency_factor * 0.6 + short_dwell_factor * 0.4)
    
    def _calculate_workplace_charging_suitability(self, constraints: Dict[str, float]) -> float:
        """Calculate suitability of workplace charging for trip constraints"""
        # Workplace charging is mainly suitable for commuting and business trips
        # Check if this is a work-related trip by examining dwell time and predictability
        is_work_trip = (
            constraints['dwell_time_min'] >= 240 and  # At least 4 hours (work day)
            constraints['schedule_predictability'] >= 0.8  # High predictability (regular schedule)
        )
        
        if is_work_trip:
            dwell_time_factor = min(1.0, constraints['dwell_time_min'] / 240.0)  # 4+ hours ideal
            predictability_factor = constraints['schedule_predictability']
            return (dwell_time_factor * 0.6 + predictability_factor * 0.4)
        else:
            return 0.1  # Low suitability for non-work trips
    
    def _calculate_public_slow_charging_suitability(self, constraints: Dict[str, float]) -> float:
        """Calculate suitability of public slow charging for trip constraints"""
        # Public slow charging works for flexible trips with long dwell times
        dwell_time_factor = min(1.0, constraints['dwell_time_min'] / 180.0)  # 3+ hours ideal
        flexibility_factor = (constraints['departure_flexibility'] + constraints['arrival_flexibility']) / 2
        low_urgency_factor = 1.0 - constraints['charging_urgency']
        
        return (dwell_time_factor * 0.5 + flexibility_factor * 0.3 + low_urgency_factor * 0.2)
    
    def assess_charging_flexibility(self, purpose: str, trip_distance: Optional[float] = None) -> Dict[str, Any]:
        """
        Comprehensive charging flexibility assessment for a trip purpose.
        
        Args:
            purpose: Trip purpose name
            trip_distance: Optional trip distance in km for range considerations
            
        Returns:
            Dict with comprehensive flexibility assessment
        """
        if purpose not in self.time_constraints:
            logger.warning(f"Unknown trip purpose: {purpose}. Using default constraints.")
            purpose = TripPurpose.OTHER_JOURNEY.value
        
        constraints = self.time_constraints[purpose]
        scenarios = self.charging_scenarios[purpose]
        
        # Calculate overall flexibility score
        overall_flexibility = self._calculate_overall_flexibility(constraints)
        
        # Calculate time constraint severity
        time_constraint_severity = self._calculate_time_constraint_severity(constraints)
        
        # Calculate charging urgency level
        charging_urgency = constraints['charging_urgency']
        
        # Range considerations if distance provided
        range_considerations = {}
        if trip_distance is not None:
            range_considerations = self._assess_range_considerations(trip_distance, constraints)
        
        return {
            'purpose': purpose,
            'overall_flexibility_score': overall_flexibility,
            'time_constraint_severity': time_constraint_severity,
            'charging_urgency': charging_urgency,
            'departure_flexibility': constraints['departure_flexibility'],
            'arrival_flexibility': constraints['arrival_flexibility'],
            'dwell_time_range': {
                'min_minutes': constraints['dwell_time_min'],
                'max_minutes': constraints['dwell_time_max']
            },
            'schedule_predictability': constraints['schedule_predictability'],
            'charging_scenarios': scenarios,
            'recommended_charging_strategy': self._recommend_charging_strategy(scenarios),
            'range_considerations': range_considerations
        }
    
    def _calculate_overall_flexibility(self, constraints: Dict[str, float]) -> float:
        """Calculate overall charging flexibility score"""
        departure_flex = constraints['departure_flexibility']
        arrival_flex = constraints['arrival_flexibility']
        dwell_time_flex = min(1.0, constraints['dwell_time_min'] / 120.0)  # Normalize to 2 hours
        urgency_flex = 1.0 - constraints['charging_urgency']  # Inverse of urgency
        
        return (departure_flex * 0.25 + arrival_flex * 0.25 + dwell_time_flex * 0.3 + urgency_flex * 0.2)
    
    def _calculate_time_constraint_severity(self, constraints: Dict[str, float]) -> float:
        """Calculate severity of time constraints (higher = more constrained)"""
        timing_constraints = 1.0 - ((constraints['departure_flexibility'] + constraints['arrival_flexibility']) / 2)
        urgency_constraints = constraints['charging_urgency']
        predictability_constraints = constraints['schedule_predictability']  # High predictability = more constrained
        
        return (timing_constraints * 0.4 + urgency_constraints * 0.4 + predictability_constraints * 0.2)
    
    def _recommend_charging_strategy(self, scenarios: Dict[str, float]) -> str:
        """Recommend the best charging strategy based on scenario suitability"""
        best_scenario = max(scenarios.items(), key=lambda x: x[1])
        return best_scenario[0]
    
    def _assess_range_considerations(self, trip_distance: float, constraints: Dict[str, float]) -> Dict[str, Any]:
        """Assess range-related considerations for the trip"""
        # Assume 100km range for frugal EVs
        max_range = 100.0
        
        considerations = {
            'trip_distance_km': trip_distance,
            'max_ev_range_km': max_range,
            'range_sufficient': trip_distance <= max_range * 0.8,  # 20% buffer
            'requires_destination_charging': trip_distance > max_range * 0.4,  # >40km one-way
            'range_anxiety_level': min(1.0, trip_distance / max_range),
            'charging_necessity': 'optional' if trip_distance < 30 else 'recommended' if trip_distance < 60 else 'required'
        }
        
        return considerations
    
    def analyze_charging_constraints_by_purpose(self, od_data: gpd.GeoDataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Analyze charging constraints for all trip purposes in the dataset.
        
        Args:
            od_data: GeoDataFrame with trip data including purpose and distance
            
        Returns:
            Dict with constraint analysis by purpose
        """
        results = {}
        
        # Identify purpose column
        purpose_col = None
        for col in ['purpose', 'trip_purpose', 'Purpose', 'Trip_Purpose']:
            if col in od_data.columns:
                purpose_col = col
                break
        
        if not purpose_col:
            raise ValueError("No trip purpose column found in OD data")
        
        # Identify distance column
        distance_col = None
        for col in ['distance', 'trip_distance', 'route_distance', 'Distance']:
            if col in od_data.columns:
                distance_col = col
                break
        
        unique_purposes = od_data[purpose_col].dropna().unique()
        
        for purpose in unique_purposes:
            purpose_data = od_data[od_data[purpose_col] == purpose]
            
            # Calculate average distance if available
            avg_distance = None
            if distance_col:
                avg_distance = purpose_data[distance_col].mean()
            
            # Get flexibility assessment
            flexibility_assessment = self.assess_charging_flexibility(purpose, avg_distance)
            
            # Add purpose-specific statistics
            results[purpose] = {
                **flexibility_assessment,
                'trip_count': len(purpose_data),
                'average_distance_km': avg_distance,
                'distance_range': {
                    'min_km': purpose_data[distance_col].min() if distance_col else None,
                    'max_km': purpose_data[distance_col].max() if distance_col else None
                }
            }
        
        return results
    
    def validate_charging_flexibility_assessment(self) -> Dict[str, Any]:
        """
        Validate the charging flexibility assessment system.
        
        Returns:
            Dict with validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Validate time constraints
        for purpose, constraints in self.time_constraints.items():
            for key, value in constraints.items():
                if key in ['departure_flexibility', 'arrival_flexibility', 'charging_urgency', 'schedule_predictability']:
                    if not (0 <= value <= 1):
                        validation_results['errors'].append(
                            f"{purpose}: {key} value {value} not in [0,1] range"
                        )
                        validation_results['valid'] = False
                
                elif key in ['dwell_time_min', 'dwell_time_max']:
                    if value < 0:
                        validation_results['errors'].append(
                            f"{purpose}: {key} value {value} cannot be negative"
                        )
                        validation_results['valid'] = False
                    
                    if key == 'dwell_time_max' and value < constraints['dwell_time_min']:
                        validation_results['errors'].append(
                            f"{purpose}: dwell_time_max {value} less than dwell_time_min {constraints['dwell_time_min']}"
                        )
                        validation_results['valid'] = False
        
        # Validate charging scenarios
        for purpose, scenarios in self.charging_scenarios.items():
            for scenario, score in scenarios.items():
                if not (0 <= score <= 1):
                    validation_results['errors'].append(
                        f"{purpose}: {scenario} score {score} not in [0,1] range"
                    )
                    validation_results['valid'] = False
        
        # Summary statistics
        all_flexibility_scores = []
        for purpose in self.time_constraints:
            constraints = self.time_constraints[purpose]
            flexibility = self._calculate_overall_flexibility(constraints)
            all_flexibility_scores.append(flexibility)
        
        validation_results['summary'] = {
            'total_purposes': len(self.time_constraints),
            'mean_flexibility_score': np.mean(all_flexibility_scores),
            'min_flexibility_score': np.min(all_flexibility_scores),
            'max_flexibility_score': np.max(all_flexibility_scores),
            'flexibility_range': np.max(all_flexibility_scores) - np.min(all_flexibility_scores)
        }
        
        return validation_results


def calculate_weighted_trip_suitability(od_data: gpd.GeoDataFrame, 
                                      trip_volume_col: str = 'All') -> gpd.GeoDataFrame:
    """
    Calculate weighted trip suitability scores for OD data.
    
    Args:
        od_data: GeoDataFrame with trip data
        trip_volume_col: Column name containing trip volumes
        
    Returns:
        GeoDataFrame with weighted suitability scores
    """
    analyzer = TripPurposeAnalyzer()
    result_data = analyzer.calculate_purpose_weights_for_od_data(od_data)
    
    # Calculate weighted suitability if trip volume data is available
    if trip_volume_col in result_data.columns:
        result_data['weighted_suitability'] = (
            result_data['ev_suitability_weight'] * result_data[trip_volume_col]
        )
    
    return result_data
