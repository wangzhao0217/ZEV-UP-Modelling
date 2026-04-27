# Functions package for frugal EV analysis

# Import main classes for easy access
from .data_integration import DataIntegrator, DemographicLoader, SpatialJoiner, DataValidator, standardize_od_column_names
from .adoption import AdoptionPropensityCalculator, HomeChargingAssessor, ImprovedChargingAssessor, ValidationEngine, ValidationResults
from .ev_type_selection import EVTypeSelector, AffordabilityAssessor, EVTypeAssignment
from .trip_purpose import TripPurposeAnalyzer, RegularityAssessor, TripPurpose, ChargingFlexibility, ChargingFlexibilityAssessor
from .conversion import TripConversionEstimator, calculate_vehicle_suitability_factor

# Import validation functions
from .adoption import validate_adoption_scores
from .ev_type_selection import validate_ev_type_assignments
from .trip_purpose import validate_trip_purpose_weights, get_all_trip_purposes, calculate_weighted_trip_suitability

# Import propensity group analysis
from .propensity_group_analysis import PropensityGroupAnalyzer, analyze_propensity_groups

# Import trip purpose visualization
from .trip_purpose_visualization import (
    TripPurposeVisualizer, 
    visualize_trip_purpose_analysis,
    standardize_purpose_names,
    PURPOSE_NAME_MAPPING
)

# Import advanced charging infrastructure analysis
from .advanced_charging_analysis import (
    ConsecutiveCoverageAnalyzer,
    TripVolumeWeightedAnalyzer,
    DestinationChargingAnalyzer,
    InfrastructureGapClassifier,
    EquityAccessibilityAnalyzer
)

# Import charging infrastructure visualization
from .charging_infrastructure_visualization import (
    ChargingInfrastructureVisualizer,
    visualize_charging_infrastructure
)

# Import uncertainty quantification
from .uncertainty import MonteCarloUncertainty, monte_carlo_adoption_uncertainty

# Import spatial autocorrelation
from .spatial import SpatialAutocorrelationAnalyzer, spatial_autocorrelation_analysis

# Import ML validation
from .ml_validation import MLWeightValidator, ml_weight_validation

__all__ = [
    # Data integration classes
    'DataIntegrator',
    'DemographicLoader', 
    'SpatialJoiner',
    'DataValidator',
    'standardize_od_column_names',
    
    # Adoption analysis classes
    'AdoptionPropensityCalculator',
    'HomeChargingAssessor',
    'ImprovedChargingAssessor',
    'ValidationEngine',
    'ValidationResults',
    
    # EV type selection classes
    'EVTypeSelector',
    'AffordabilityAssessor',
    'EVTypeAssignment',
    
    # Trip purpose analysis classes
    'TripPurposeAnalyzer',
    'RegularityAssessor',
    'TripPurpose',
    'ChargingFlexibility',
    'ChargingFlexibilityAssessor',
    
    # Conversion analysis classes
    'TripConversionEstimator',
    
    # Validation functions
    'validate_adoption_scores',
    'validate_ev_type_assignments',
    'validate_trip_purpose_weights',
    
    # Utility functions
    'get_all_trip_purposes',
    'calculate_weighted_trip_suitability',
    'calculate_vehicle_suitability_factor',
    
    # Propensity group analysis
    'PropensityGroupAnalyzer',
    'analyze_propensity_groups',
    
    # Trip purpose visualization
    'TripPurposeVisualizer',
    'visualize_trip_purpose_analysis',
    'standardize_purpose_names',
    'PURPOSE_NAME_MAPPING',
    
    # Advanced charging infrastructure analysis
    'ConsecutiveCoverageAnalyzer',
    'TripVolumeWeightedAnalyzer',
    'DestinationChargingAnalyzer',
    'InfrastructureGapClassifier',
    'EquityAccessibilityAnalyzer',
    
    # Charging infrastructure visualization
    'ChargingInfrastructureVisualizer',
    'visualize_charging_infrastructure',

    # Uncertainty quantification
    'MonteCarloUncertainty',
    'monte_carlo_adoption_uncertainty',

    # Spatial autocorrelation
    'SpatialAutocorrelationAnalyzer',
    'spatial_autocorrelation_analysis',

    # ML validation
    'MLWeightValidator',
    'ml_weight_validation'
]