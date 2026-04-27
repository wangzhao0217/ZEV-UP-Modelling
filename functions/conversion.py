"""
Trip Conversion Estimation Module for Frugal EV Analysis

This module provides classes and functions for estimating trip conversion potential
by combining trip volumes, adoption propensity, and purpose weights.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from .data_integration import standardize_od_column_names

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversionFactors:
    """Container for factors used in trip conversion estimation"""
    trip_volume: float
    adoption_propensity: float
    purpose_weight: float
    infrastructure_factor: float = 1.0
    range_feasibility: float = 1.0


@dataclass
class ConversionResult:
    """Container for trip conversion estimation results"""
    origin_code: str
    destination_code: str
    trip_purpose: str
    base_trip_volume: float
    conversion_potential: float
    factors: ConversionFactors


class TripConversionEstimator:
    """
    Main calculation engine for estimating trip conversion potential to frugal EVs.
    
    Implements the core conversion formula: Trip Volume × Adoption Propensity × Purpose Weight
    with additional factors for infrastructure and range feasibility.
    """
    
    def __init__(self, 
                 infrastructure_weight: float = 1.0,
                 range_weight: float = 1.0,
                 min_conversion_threshold: float = 0.1,
                 default_adoption_propensity: float = 0.3):
        """
        Initialize TripConversionEstimator with weighting parameters.
        
        Args:
            infrastructure_weight: Weight for infrastructure availability factor
            range_weight: Weight for range feasibility factor
            min_conversion_threshold: Minimum conversion potential to consider viable
            default_adoption_propensity: Fallback value for missing adoption scores (default: 0.3)
        """
        self.infrastructure_weight = infrastructure_weight
        self.range_weight = range_weight
        self.min_conversion_threshold = min_conversion_threshold
        self.default_adoption_propensity = default_adoption_propensity
        
    def calculate_conversion_potential(self, 
                                     trip_volume: Union[float, pd.Series],
                                     adoption_propensity: Union[float, pd.Series],
                                     purpose_weight: Union[float, pd.Series],
                                     infrastructure_factor: Union[float, pd.Series] = 1.0,
                                     range_feasibility: Union[float, pd.Series] = 1.0) -> Union[float, pd.Series]:
        """
        Calculate conversion potential using the core formula.
        
        Args:
            trip_volume: Number of trips for the OD pair/purpose combination
            adoption_propensity: EV adoption propensity score [0,1]
            purpose_weight: Trip purpose suitability weight [0,1]
            infrastructure_factor: Infrastructure availability factor [0,1]
            range_feasibility: Range feasibility factor [0,1]
            
        Returns:
            Conversion potential (number of trips likely to convert to EV)
        """
        # Core conversion formula
        base_conversion = trip_volume * adoption_propensity * purpose_weight
        
        # Apply additional factors
        adjusted_conversion = (base_conversion * 
                             (infrastructure_factor ** self.infrastructure_weight) *
                             (range_feasibility ** self.range_weight))
        
        # Apply minimum threshold
        if isinstance(adjusted_conversion, pd.Series):
            adjusted_conversion = adjusted_conversion.where(
                adjusted_conversion >= self.min_conversion_threshold, 0.0
            )
        else:
            adjusted_conversion = adjusted_conversion if adjusted_conversion >= self.min_conversion_threshold else 0.0
        
        return adjusted_conversion
    
    def calculate_feasibility_factor(self, 
                                    range_classification: str,
                                    coverage_assessment: float = 1.0,
                                    available_capacity: float = 0.0,
                                    max_capacity: float = 1.0) -> float:
        """
        Calculate feasibility factor using piecewise function per @eq-feasibility-factor.
        
        Implements three-tier constraint handling:
        - Feasible routes: F = 1.0 (no constraints)
        - Constrained routes: F = I_cov × (C_avail / C_max) (partial feasibility)
        - Infeasible routes: F = 0.0 (range exceeded)
        
        Args:
            range_classification: Route classification ("feasible", "constrained", or "infeasible")
            coverage_assessment: Infrastructure coverage metric I_cov [0,1]
            available_capacity: Available charging capacity at destination C_avail
            max_capacity: Maximum charging capacity at destination C_max
            
        Returns:
            Feasibility factor [0, 1]
            
        Example:
            >>> estimator = TripConversionEstimator()
            >>> estimator.calculate_feasibility_factor("feasible")
            1.0
            >>> estimator.calculate_feasibility_factor("constrained", 0.8, 50, 100)
            0.4
            >>> estimator.calculate_feasibility_factor("infeasible")
            0.0
        """
        if range_classification == "feasible":
            return 1.0
        elif range_classification == "constrained":
            # Calculate capacity ratio, avoiding division by zero
            capacity_ratio = (available_capacity / max_capacity) if max_capacity > 0 else 0.0
            # Apply coverage-weighted capacity constraint
            return coverage_assessment * capacity_ratio
        else:  # "infeasible"
            return 0.0
    
    def estimate_od_conversion_potential(self, 
                                       od_data: gpd.GeoDataFrame,
                                       adoption_data: gpd.GeoDataFrame,
                                       purpose_weights: Dict[str, float]) -> gpd.GeoDataFrame:
        """
        Estimate conversion potential for all origin-destination pairs.
        
        Args:
            od_data: GeoDataFrame with trip data including volumes and purposes
            adoption_data: GeoDataFrame with adoption propensity scores by area
            purpose_weights: Dictionary mapping trip purposes to suitability weights
            
        Returns:
            GeoDataFrame with conversion potential estimates added
            
        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Estimating conversion potential for OD pairs")
        
        # Standardize column names
        od_data = standardize_od_column_names(od_data)
        
        # Validate required columns
        self._validate_od_data(od_data)
        self._validate_adoption_data(adoption_data)
        
        # Create result dataframe
        result = od_data.copy()
        
        # Identify trip volume and purpose columns
        volume_col = self._identify_volume_column(od_data)
        purpose_col = self._identify_purpose_column(od_data)
        
        if not volume_col:
            raise ValueError("No trip volume column found in OD data")
        if not purpose_col:
            raise ValueError("No trip purpose column found in OD data")
        
        # Attach adoption propensity for origins and destinations
        result = self._attach_adoption_scores(result, adoption_data)
        
        # Calculate purpose weights for each trip
        # Default to 0.5 (moderate suitability) for unrecognized trip purposes
        result['purpose_weight'] = result[purpose_col].map(
            lambda x: purpose_weights.get(x, 0.5)
        )
        
        # Log if any purposes were not found
        unknown_purposes = set(result[purpose_col].unique()) - set(purpose_weights.keys())
        if unknown_purposes:
            logger.warning(f"Unknown trip purposes defaulted to 0.5 weight: {unknown_purposes}")
        
        # Calculate conversion potential
        result['conversion_potential'] = self.calculate_conversion_potential(
            trip_volume=result[volume_col],
            adoption_propensity=result['adoption_propensity_combined'],
            purpose_weight=result['purpose_weight']
        )
        
        # Add factor breakdown for analysis
        result['base_trip_volume'] = result[volume_col]
        result['adoption_factor'] = result['adoption_propensity_combined']
        
        # Calculate summary statistics
        total_trips = result[volume_col].sum()
        total_conversion_potential = result['conversion_potential'].sum()
        conversion_rate = total_conversion_potential / total_trips if total_trips > 0 else 0
        
        logger.info(f"Total trips analyzed: {total_trips:,.0f}")
        logger.info(f"Total conversion potential: {total_conversion_potential:,.1f}")
        logger.info(f"Overall conversion rate: {conversion_rate:.1%}")
        
        return result
    
    def estimate_od_conversion_with_feasibility(self,
                                               od_data: gpd.GeoDataFrame,
                                               adoption_data: gpd.GeoDataFrame,
                                               purpose_weights: Dict[str, float],
                                               range_results: Optional[gpd.GeoDataFrame] = None,
                                               charging_results: Optional[gpd.GeoDataFrame] = None) -> gpd.GeoDataFrame:
        """
        Estimate conversion potential integrating range and charging feasibility.
        
        This method extends estimate_od_conversion_potential() by integrating
        piecewise feasibility classification from range and charging analysis.
        
        Note on weights: The piecewise feasibility_factor (F) combines both range 
        classification and infrastructure capacity per @eq-feasibility-factor. The 
        infrastructure_weight is applied to this combined F factor. The range_weight 
        parameter is not separately applied here because range constraints are already 
        encoded in the feasibility classification (feasible/constrained/infeasible).
        
        Args:
            od_data: GeoDataFrame with trip data
            adoption_data: GeoDataFrame with adoption propensity scores
            purpose_weights: Dictionary mapping trip purposes to weights
            range_results: Optional GeoDataFrame with range feasibility classification
            charging_results: Optional GeoDataFrame with charging infrastructure metrics
            
        Returns:
            GeoDataFrame with conversion potential including feasibility constraints
        """
        logger.info("Estimating conversion potential with feasibility integration")
        
        # Start with base conversion estimation
        result = self.estimate_od_conversion_potential(od_data, adoption_data, purpose_weights)
        
        # If feasibility data provided, apply piecewise function
        if range_results is not None:
            logger.info("Integrating range feasibility classification")
            
            # Merge range classification
            range_data = range_results[['origin_code', 'destination_code', 'feasibility_classification']].copy()
            result = result.merge(
                range_data,
                left_on=['origin_code', 'destination_code'],
                right_on=['origin_code', 'destination_code'],
                how='left'
            )
            
            # Get charging metrics if available
            if charging_results is not None:
                logger.info("Integrating charging infrastructure capacity")
                charging_data = charging_results[['destination_code', 'coverage_assessment', 
                                                'available_capacity', 'max_capacity']].copy()
                result = result.merge(
                    charging_data,
                    left_on='destination_code',
                    right_on='destination_code',
                    how='left'
                )
            else:
                # Use defaults if charging data not available
                # Note: Setting available_capacity=0 means constrained routes → F=0
                # This is conservative: without charging data, assume no destination charging
                logger.warning("No charging data provided; constrained routes will have zero feasibility")
                result['coverage_assessment'] = 1.0
                result['available_capacity'] = 0.0
                result['max_capacity'] = 1.0
            
            # Calculate piecewise feasibility factor
            result['feasibility_factor'] = result.apply(
                lambda row: self.calculate_feasibility_factor(
                    range_classification=row.get('feasibility_classification', 'feasible'),
                    coverage_assessment=row.get('coverage_assessment', 1.0),
                    available_capacity=row.get('available_capacity', 0.0),
                    max_capacity=row.get('max_capacity', 1.0)
                ),
                axis=1
            )
            
            # Recalculate conversion potential with feasibility factor
            # Note: feasibility_factor already incorporates range constraints via piecewise function
            # Apply infrastructure_weight to feasibility, range_weight = 1.0 as range is in feasibility_factor
            result['conversion_potential_with_feasibility'] = (
                result['base_trip_volume'] * 
                result['adoption_propensity_combined'] * 
                result['purpose_weight'] *
                (result['feasibility_factor'] ** self.infrastructure_weight)
            )
            
            # Apply minimum threshold filtering to feasibility-adjusted conversion
            result['conversion_potential_with_feasibility'] = result['conversion_potential_with_feasibility'].where(
                result['conversion_potential_with_feasibility'] >= self.min_conversion_threshold, 0.0
            )
            
            # Log impact statistics
            total_base = result['conversion_potential'].sum()
            total_feasibility = result['conversion_potential_with_feasibility'].sum()
            reduction_pct = (1 - total_feasibility / total_base) * 100 if total_base > 0 else 0
            
            logger.info(f"Feasibility constraints reduced conversion potential by {reduction_pct:.1f}%")
            logger.info(f"  Base conversion: {total_base:,.1f} trips/day")
            logger.info(f"  Feasibility-adjusted: {total_feasibility:,.1f} trips/day")
        
        return result
    
    def _validate_od_data(self, od_data: gpd.GeoDataFrame) -> None:
        """Validate OD data has required columns"""
        required_cols = ['origin_code', 'destination_code']
        missing_cols = [col for col in required_cols if col not in od_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required OD columns: {missing_cols}")
    
    def _validate_adoption_data(self, adoption_data: gpd.GeoDataFrame) -> None:
        """Validate adoption data has required columns"""
        required_cols = ['geo_code', 'adoption_propensity']
        missing_cols = [col for col in required_cols if col not in adoption_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required adoption columns: {missing_cols}")
    
    def _identify_volume_column(self, od_data: gpd.GeoDataFrame) -> Optional[str]:
        """Identify the column containing trip volume data"""
        possible_columns = ['All', 'total', 'volume', 'trips', 'count', 'Total']
        
        for col in possible_columns:
            if col in od_data.columns:
                return col
        
        # Look for numeric columns that might contain trip volumes
        numeric_cols = od_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if 'trip' in col.lower() or 'volume' in col.lower() or 'count' in col.lower():
                return col
        
        # If 'All' column exists, use it (common in census data)
        if 'All' in od_data.columns:
            return 'All'
            
        return None
    
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
    
    def _attach_adoption_scores(self, 
                              od_data: gpd.GeoDataFrame, 
                              adoption_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Attach adoption propensity scores to origin and destination areas"""
        logger.info("Attaching adoption propensity scores to OD data")
        
        # Prepare adoption data for joining
        adoption_scores = adoption_data[['geo_code', 'adoption_propensity']].copy()
        adoption_scores['geo_code'] = adoption_scores['geo_code'].astype(str)
        
        # Ensure OD codes are strings
        od_data['origin_code'] = od_data['origin_code'].astype(str)
        od_data['destination_code'] = od_data['destination_code'].astype(str)
        
        # Attach origin adoption scores
        origin_scores = adoption_scores.rename(columns={
            'geo_code': 'origin_code',
            'adoption_propensity': 'adoption_propensity_origin'
        })
        od_data = od_data.merge(origin_scores, on='origin_code', how='left')
        
        # Attach destination adoption scores
        dest_scores = adoption_scores.rename(columns={
            'geo_code': 'destination_code', 
            'adoption_propensity': 'adoption_propensity_dest'
        })
        od_data = od_data.merge(dest_scores, on='destination_code', how='left')
        
        # Calculate combined adoption propensity (average of origin and destination)
        # Handle missing values properly: only average non-missing values, then apply default if both missing
        origin_valid = od_data['adoption_propensity_origin'].notna()
        dest_valid = od_data['adoption_propensity_dest'].notna()
        
        # Case 1: Both available - take average
        both_valid = origin_valid & dest_valid
        od_data.loc[both_valid, 'adoption_propensity_combined'] = (
            od_data.loc[both_valid, 'adoption_propensity_origin'] + 
            od_data.loc[both_valid, 'adoption_propensity_dest']
        ) / 2
        
        # Case 2: Only origin available - use origin
        only_origin = origin_valid & ~dest_valid
        od_data.loc[only_origin, 'adoption_propensity_combined'] = od_data.loc[only_origin, 'adoption_propensity_origin']
        
        # Case 3: Only destination available - use destination
        only_dest = ~origin_valid & dest_valid
        od_data.loc[only_dest, 'adoption_propensity_combined'] = od_data.loc[only_dest, 'adoption_propensity_dest']
        
        # Case 4: Both missing - use default
        both_missing = ~origin_valid & ~dest_valid
        od_data.loc[both_missing, 'adoption_propensity_combined'] = self.default_adoption_propensity
        
        # Log matching statistics
        origin_matches = od_data['adoption_propensity_origin'].notna().sum()
        dest_matches = od_data['adoption_propensity_dest'].notna().sum()
        total_records = len(od_data)
        
        logger.info(f"Origin adoption matches: {origin_matches}/{total_records} ({origin_matches/total_records*100:.1f}%)")
        logger.info(f"Destination adoption matches: {dest_matches}/{total_records} ({dest_matches/total_records*100:.1f}%)")
        
        return od_data
    
    def validate_conversion_estimates(self, conversion_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate conversion potential estimates for consistency and realism.
        
        Args:
            conversion_data: GeoDataFrame with conversion estimates
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating conversion potential estimates")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        if 'conversion_potential' not in conversion_data.columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Missing conversion_potential column")
            return validation_results
        
        conversion_potential = conversion_data['conversion_potential']
        
        # Range validation
        if conversion_potential.min() < 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Negative conversion potential found: {conversion_potential.min():.3f}")
        
        # Check for unrealistic values (conversion potential > original trip volume)
        if 'base_trip_volume' in conversion_data.columns:
            base_volume = conversion_data['base_trip_volume']
            excessive_conversion = (conversion_potential > base_volume).sum()
            if excessive_conversion > 0:
                validation_results['issues'].append(f"{excessive_conversion} records have conversion > base volume")
        
        # Distribution validation
        validation_results['statistics'] = {
            'total_records': len(conversion_data),
            'mean_conversion': conversion_potential.mean(),
            'median_conversion': conversion_potential.median(),
            'std_conversion': conversion_potential.std(),
            'min_conversion': conversion_potential.min(),
            'max_conversion': conversion_potential.max(),
            'zero_conversion_count': (conversion_potential == 0).sum(),
            'total_conversion_potential': conversion_potential.sum()
        }
        
        # Check for unrealistic distributions
        if conversion_potential.std() == 0:
            validation_results['issues'].append("No variance in conversion potential")
        
        # Check for excessive zero values
        zero_rate = (conversion_potential == 0).sum() / len(conversion_potential)
        if zero_rate > 0.9:
            validation_results['issues'].append(f"High proportion of zero conversion: {zero_rate:.1%}")
        
        if validation_results['issues']:
            validation_results['is_valid'] = False
        
        logger.info(f"Conversion validation {'passed' if validation_results['is_valid'] else 'failed'}")
        
        return validation_results


class FactorIntegrator:
    """
    Multi-dimensional analysis engine for combining demographic and trip-based factors
    with geographic scale aggregation capabilities.
    """
    
    def __init__(self, aggregation_levels: Optional[List[str]] = None):
        """
        Initialize FactorIntegrator with aggregation level definitions.
        
        Args:
            aggregation_levels: List of geographic aggregation levels to support
        """
        self.aggregation_levels = aggregation_levels or [
            'output_area',    # Individual Output Areas
            'intermediate',   # Intermediate Zones  
            'council',        # Council Areas
            'regional',       # Regional level
            'national'        # Scotland-wide
        ]
        
    def integrate_demographic_factors(self, 
                                    conversion_data: gpd.GeoDataFrame,
                                    demographic_weights: Optional[Dict[str, float]] = None) -> gpd.GeoDataFrame:
        """
        Integrate multiple demographic factors into conversion estimates.
        
        Args:
            conversion_data: GeoDataFrame with conversion estimates and demographic data
            demographic_weights: Optional weights for different demographic factors
            
        Returns:
            GeoDataFrame with integrated demographic factor scores
        """
        logger.info("Integrating demographic factors into conversion estimates")
        
        # Default demographic weights if not provided
        # Note: Multi-factor analysis uses 4-factor model per @eq-integrated-demographic
        if demographic_weights is None:
            demographic_weights = {
                'social_grade': 0.30,
                'education': 0.25,
                'car_ownership': 0.25,
                'housing': 0.20
                # Note: age, economic_activity, household_composition, population_density
                # are excluded from multi-factor baseline to match paper specification
            }
        
        result = conversion_data.copy()
        
        # Calculate integrated demographic score
        demographic_score = pd.Series(0.0, index=result.index)
        
        for factor, weight in demographic_weights.items():
            # Look for demographic score columns
            score_cols = [col for col in result.columns if f'{factor}_score' in col]
            
            if score_cols:
                # Use origin and destination scores if available
                origin_cols = [col for col in score_cols if 'origin' in col]
                dest_cols = [col for col in score_cols if 'dest' in col]
                
                if origin_cols and dest_cols:
                    # Average origin and destination scores
                    origin_score = result[origin_cols[0]].fillna(0.5)
                    dest_score = result[dest_cols[0]].fillna(0.5)
                    factor_score = (origin_score + dest_score) / 2
                elif origin_cols:
                    factor_score = result[origin_cols[0]].fillna(0.5)
                elif dest_cols:
                    factor_score = result[dest_cols[0]].fillna(0.5)
                else:
                    factor_score = result[score_cols[0]].fillna(0.5)
                
                demographic_score += factor_score * weight
            else:
                logger.warning(f"No demographic score found for factor: {factor}")
        
        result['integrated_demographic_score'] = demographic_score
        
        # Recalculate conversion potential with integrated demographic factors
        if 'base_trip_volume' in result.columns and 'purpose_weight' in result.columns:
            result['conversion_potential_integrated'] = (
                result['base_trip_volume'] * 
                result['integrated_demographic_score'] * 
                result['purpose_weight']
            )
        
        logger.info(f"Integrated demographic factors for {len(result)} records")
        return result
    
    def integrate_trip_factors(self, 
                             conversion_data: gpd.GeoDataFrame,
                             trip_weights: Optional[Dict[str, float]] = None) -> gpd.GeoDataFrame:
        """
        Integrate trip-based factors (purpose, regularity, distance) into conversion estimates.
        
        Args:
            conversion_data: GeoDataFrame with conversion estimates and trip data
            trip_weights: Optional weights for different trip factors
            
        Returns:
            GeoDataFrame with integrated trip factor scores
        """
        logger.info("Integrating trip-based factors into conversion estimates")
        
        # Default trip weights if not provided
        if trip_weights is None:
            trip_weights = {
                'purpose_suitability': 0.4,    # Primary factor
                'regularity': 0.3,             # Trip predictability
                'distance_feasibility': 0.2,   # Range considerations
                'charging_flexibility': 0.1    # Charging constraints
            }
        
        result = conversion_data.copy()
        
        # Calculate integrated trip score
        trip_score = pd.Series(0.0, index=result.index)
        
        # Purpose suitability (already calculated)
        if 'purpose_weight' in result.columns:
            trip_score += result['purpose_weight'] * trip_weights['purpose_suitability']
        
        # Regularity score (if available)
        if 'regularity_score' in result.columns:
            trip_score += result['regularity_score'] * trip_weights['regularity']
        else:
            # Use default regularity based on purpose
            default_regularity = result.get('purpose_weight', pd.Series(0.5, index=result.index)) * 0.8
            trip_score += default_regularity * trip_weights['regularity']
        
        # Distance feasibility (if available)
        if 'distance_feasibility' in result.columns:
            trip_score += result['distance_feasibility'] * trip_weights['distance_feasibility']
        else:
            # Default distance feasibility (assume most trips are feasible)
            trip_score += 0.8 * trip_weights['distance_feasibility']
        
        # Charging flexibility (if available)
        if 'charging_flexibility' in result.columns:
            trip_score += result['charging_flexibility'] * trip_weights['charging_flexibility']
        else:
            # Default charging flexibility
            trip_score += 0.6 * trip_weights['charging_flexibility']
        
        result['integrated_trip_score'] = trip_score
        
        # Recalculate conversion potential with integrated trip factors
        if 'base_trip_volume' in result.columns and 'adoption_propensity_combined' in result.columns:
            result['conversion_potential_trip_integrated'] = (
                result['base_trip_volume'] * 
                result['adoption_propensity_combined'] * 
                result['integrated_trip_score']
            )
        
        logger.info(f"Integrated trip factors for {len(result)} records")
        return result
    
    def aggregate_to_geographic_scale(self, 
                                    conversion_data: gpd.GeoDataFrame,
                                    aggregation_level: str,
                                    area_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        Aggregate conversion estimates to different geographic scales.
        
        Args:
            conversion_data: GeoDataFrame with conversion estimates
            aggregation_level: Target aggregation level
            area_mapping: Optional mapping from Output Areas to target geography
            
        Returns:
            DataFrame with aggregated conversion estimates
            
        Raises:
            ValueError: If aggregation level is not supported
        """
        if aggregation_level not in self.aggregation_levels:
            raise ValueError(f"Unsupported aggregation level: {aggregation_level}")
        
        logger.info(f"Aggregating conversion estimates to {aggregation_level} level")
        
        # For now, implement basic aggregation by summing conversion potential
        # In a full implementation, this would use proper geographic hierarchies
        
        if aggregation_level == 'output_area':
            # No aggregation needed - return as is but add conversion_rate
            result = conversion_data.copy()
            if 'conversion_potential' in result.columns and 'base_trip_volume' in result.columns:
                result['conversion_rate'] = (
                    result['conversion_potential'] / result['base_trip_volume']
                ).fillna(0)
            return result
        
        # Group by origin and destination codes for higher-level aggregation
        if aggregation_level == 'council':
            # Extract council area codes (first 3 characters of geo_code)
            conversion_data['origin_council'] = conversion_data['origin_code'].str[:3]
            conversion_data['destination_council'] = conversion_data['destination_code'].str[:3]
            group_cols = ['origin_council', 'destination_council']
        elif aggregation_level == 'regional':
            # Extract regional codes (first 2 characters of geo_code)
            conversion_data['origin_region'] = conversion_data['origin_code'].str[:2]
            conversion_data['destination_region'] = conversion_data['destination_code'].str[:2]
            group_cols = ['origin_region', 'destination_region']
        elif aggregation_level == 'national':
            # National level - sum everything
            group_cols = []
        else:
            # Default to council level
            conversion_data['origin_council'] = conversion_data['origin_code'].str[:3]
            conversion_data['destination_council'] = conversion_data['destination_code'].str[:3]
            group_cols = ['origin_council', 'destination_council']
        
        # Aggregate conversion estimates
        if group_cols:
            aggregated = conversion_data.groupby(group_cols).agg({
                'base_trip_volume': 'sum',
                'conversion_potential': 'sum',
                'adoption_propensity_combined': 'mean',
                'purpose_weight': 'mean'
            }).reset_index()
        else:
            # National aggregation
            aggregated = pd.DataFrame({
                'level': ['national'],
                'base_trip_volume': [conversion_data['base_trip_volume'].sum()],
                'conversion_potential': [conversion_data['conversion_potential'].sum()],
                'adoption_propensity_combined': [conversion_data['adoption_propensity_combined'].mean()],
                'purpose_weight': [conversion_data['purpose_weight'].mean()]
            })
        
        # Calculate aggregated conversion rate
        aggregated['conversion_rate'] = (
            aggregated['conversion_potential'] / aggregated['base_trip_volume']
        ).fillna(0)
        
        logger.info(f"Aggregated to {len(aggregated)} {aggregation_level} records")
        return aggregated
    
    def create_multi_factor_conversion_estimates(self, 
                                               conversion_data: gpd.GeoDataFrame,
                                               demographic_weights: Optional[Dict[str, float]] = None,
                                               trip_weights: Optional[Dict[str, float]] = None) -> gpd.GeoDataFrame:
        """
        Create comprehensive conversion estimates integrating all available factors.
        
        Args:
            conversion_data: GeoDataFrame with base conversion estimates
            demographic_weights: Optional weights for demographic factors (from config)
            trip_weights: Optional weights for trip factors (from config)
            
        Returns:
            GeoDataFrame with multi-factor conversion estimates
        """
        logger.info("Creating multi-factor conversion estimates")
        
        # Integrate demographic factors with provided weights
        result = self.integrate_demographic_factors(conversion_data, demographic_weights)
        
        # Integrate trip factors with provided weights
        result = self.integrate_trip_factors(result, trip_weights)
        
        # Create final multi-factor conversion estimate
        # Per @eq-multi-factor-conversion: C = f × D × T × F
        if ('integrated_demographic_score' in result.columns and 
            'integrated_trip_score' in result.columns and
            'base_trip_volume' in result.columns):
            
            # Check if feasibility factor available (from range/charging integration)
            if 'feasibility_factor' in result.columns:
                # Include feasibility factor F as per equation
                result['conversion_potential_multi_factor'] = (
                    result['base_trip_volume'] * 
                    result['integrated_demographic_score'] * 
                    result['integrated_trip_score'] *
                    result['feasibility_factor']
                )
            else:
                # Without feasibility data, omit F (implicitly F=1.0)
                result['conversion_potential_multi_factor'] = (
                    result['base_trip_volume'] * 
                    result['integrated_demographic_score'] * 
                    result['integrated_trip_score']
                )
                logger.warning("Multi-factor conversion calculated without feasibility factor (F implicitly = 1.0)")
        
        # Calculate factor importance scores
        result['demographic_importance'] = result.get('integrated_demographic_score', 0.5)
        result['trip_importance'] = result.get('integrated_trip_score', 0.5)
        result['combined_factor_score'] = (
            result['demographic_importance'] * result['trip_importance']
        )
        
        logger.info("Multi-factor conversion estimates completed")
        return result


def validate_factor_integration(conversion_data: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Validate factor integration results for consistency and completeness.
    
    Args:
        conversion_data: GeoDataFrame with integrated factor scores
        
    Returns:
        Dictionary with validation results
    """
    logger.info("Validating factor integration")
    
    validation_results = {
        'is_valid': True,
        'issues': [],
        'statistics': {},
        'factor_coverage': {}
    }
    
    # Check for required columns
    required_cols = ['conversion_potential', 'base_trip_volume']
    missing_cols = [col for col in required_cols if col not in conversion_data.columns]
    if missing_cols:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Missing required columns: {missing_cols}")
    
    # Check factor score ranges
    factor_cols = [col for col in conversion_data.columns if 'score' in col]
    for col in factor_cols:
        scores = conversion_data[col]
        if scores.min() < 0 or scores.max() > 1:
            validation_results['issues'].append(f"{col} values out of range [0,1]")
    
    # Calculate factor coverage
    validation_results['factor_coverage'] = {
        'demographic_factors': len([col for col in conversion_data.columns if 'demographic' in col]),
        'trip_factors': len([col for col in conversion_data.columns if 'trip' in col]),
        'integrated_factors': len([col for col in conversion_data.columns if 'integrated' in col])
    }
    
    # Calculate summary statistics
    if 'conversion_potential' in conversion_data.columns:
        conversion_potential = conversion_data['conversion_potential']
        validation_results['statistics'] = {
            'total_records': len(conversion_data),
            'mean_conversion': conversion_potential.mean(),
            'total_conversion': conversion_potential.sum(),
            'non_zero_conversions': (conversion_potential > 0).sum()
        }
    
    if validation_results['issues']:
        validation_results['is_valid'] = False
    
    logger.info(f"Factor integration validation {'passed' if validation_results['is_valid'] else 'failed'}")
    
    return validation_results


def calculate_vehicle_suitability_factor(
    ev_type: str,
    trip_characteristics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate the vehicle-type suitability factor (V) for the conversion formula.
    
    Implements the V factor from F = AP × P × I × R × V, representing how well
    the assigned EV type matches trip characteristics.
    
    Args:
        ev_type: Assigned EV type ('2-seater', '4-seater', 'mixed', 'not_applicable')
        trip_characteristics: Dictionary with:
            - avg_distance_km: Average trip distance in km
            - family_purpose_ratio: Proportion of family/school/shopping trips [0,1]
            - confidence: Assignment confidence score [0,1] (optional)
        weights: Optional configuration weights for suitability adjustments
            - distance_penalty_per_km: Penalty per km for 2-seaters (default: 0.01)
            - family_trip_penalty: Penalty for 2-seaters on family trips (default: 0.3)
            - confidence_weight: Weight for confidence adjustment (default: 0.1)
            - baseline_distance_km: Baseline distance before penalties apply (default: 20.0)
    
    Returns:
        Vehicle suitability factor [0, 1.0]
        
    Examples:
        >>> # 2-seater for short commute trips
        >>> calculate_vehicle_suitability_factor('2-seater', 
        ...     {'avg_distance_km': 15, 'family_purpose_ratio': 0.1, 'confidence': 0.9})
        0.95
        
        >>> # 2-seater for long family trips (less suitable)
        >>> calculate_vehicle_suitability_factor('2-seater',
        ...     {'avg_distance_km': 50, 'family_purpose_ratio': 0.6, 'confidence': 0.7})
        0.35
        
        >>> # 4-seater is universally suitable
        >>> calculate_vehicle_suitability_factor('4-seater',
        ...     {'avg_distance_km': 50, 'family_purpose_ratio': 0.6, 'confidence': 0.9})
        1.0
    """
    # Default weights
    default_weights = {
        'distance_penalty_per_km': 0.01,
        'family_trip_penalty': 0.3,
        'confidence_weight': 0.1,
        'baseline_distance_km': 20.0,
    }
    
    if weights is None:
        weights = default_weights
    else:
        # Merge with defaults
        weights = {**default_weights, **weights}
    
    # Extract trip characteristics
    avg_distance = trip_characteristics.get(
        'avg_distance_km', weights.get('baseline_distance_km', 20.0)
    )
    family_ratio = trip_characteristics.get('family_purpose_ratio', 0.3)
    confidence = trip_characteristics.get('confidence', 1.0)
    
    # Base suitability by type
    if ev_type == '4-seater':
        # 4-seaters are universally suitable
        base_suitability = 1.0
        
    elif ev_type == '2-seater':
        # Calculate individual penalty components per @eq-vehicle-suitability-factor
        # δ_d: Distance penalty (1% per km beyond 20km baseline)
        baseline_distance = weights.get('baseline_distance_km', 20.0)
        delta_d = 0.0
        if avg_distance > baseline_distance:
            delta_d = (avg_distance - baseline_distance) * weights['distance_penalty_per_km']
        
        # δ_f: Family trip penalty (30% weighted by family trip proportion)
        delta_f = family_ratio * weights['family_trip_penalty']
        
        # Apply combined penalty with 0.5 cap per @eq-vehicle-suitability-factor
        # V = 1.0 - min(0.5, δ_d + δ_f)
        combined_penalty = min(0.5, delta_d + delta_f)
        base_suitability = 1.0 - combined_penalty
        
    elif ev_type == 'mixed':
        # Mixed deployments are generally suitable (deployment flexibility)
        base_suitability = 1.0
        
    elif ev_type == 'not_applicable':
        # No conversion potential
        return 0.0
        
    else:
        # Unknown type; conservative estimate
        logger.warning(f"Unknown EV type '{ev_type}', using conservative V=0.5")
        base_suitability = 0.5
    
    # Apply confidence adjustment (lower confidence = slight reduction)
    # If assignment confidence is low, slightly reduce suitability
    confidence_adjustment = (1.0 - confidence) * weights['confidence_weight']
    final_suitability = base_suitability - confidence_adjustment
    
    # Ensure within valid range [0, 1]
    final_suitability = max(0.0, min(1.0, final_suitability))
    
    return final_suitability


# =============================================================================
# 5-Step Replaceability Logic Functions (per eq_revised.qmd)
# =============================================================================

def calculate_replaceability(
    AP_i: Union[float, pd.Series],
    distance_km: Union[float, pd.Series],
    C_ij: Union[int, pd.Series],
    config: Optional[Dict[str, Any]] = None
) -> Union[int, pd.Series]:
    """
    Calculate binary replaceability indicator R_ij per @eq-feasibility-indicator.

    R_ij = 𝟙{0.3 ≤ AP_i ≤ 0.8} × 𝟙{d_ij ≤ 80 km} × C_ij

    A trip is replaceable if ALL conditions are met:
    1. Adoption propensity within target range [0.3, 0.8]
    2. Trip distance within effective range (≤80 km)
    3. Charging condition satisfied (C_ij = 1)

    Args:
        AP_i: Adoption propensity at origin [0,1]
        distance_km: Trip distance in km
        C_ij: Charging condition (binary)
        config: Optional config with replaceability_thresholds

    Returns:
        Binary: 1 if trip is replaceable, else 0
    """
    # Get thresholds from config or use defaults
    ap_min = 0.3
    ap_max = 0.8
    max_distance = 80.0
    if config and 'replaceability_thresholds' in config:
        rt = config['replaceability_thresholds']
        ap_min = rt.get('adoption_propensity_min', 0.3)
        ap_max = rt.get('adoption_propensity_max', 0.8)
        max_distance = rt.get('medium_trip_distance_km', 80.0)

    if isinstance(AP_i, pd.Series):
        ap_valid = ((AP_i >= ap_min) & (AP_i <= ap_max)).astype(int)
        dist_valid = (distance_km <= max_distance).astype(int)
        return ap_valid * dist_valid * C_ij
    else:
        ap_valid = 1 if (ap_min <= AP_i <= ap_max) else 0
        dist_valid = 1 if distance_km <= max_distance else 0
        return ap_valid * dist_valid * C_ij


def calculate_trip_conversion(
    f_ij: Union[float, pd.Series],
    R_ij: Union[int, pd.Series],
    W_p: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """
    Calculate converted trips per @eq-trip-conversion.

    N_ij^(p) = f_ij × R_ij × W^(p)

    Args:
        f_ij: Baseline trip volume
        R_ij: Binary replaceability indicator
        W_p: Trip purpose suitability weight [0,1]

    Returns:
        Number of converted trips
    """
    return f_ij * R_ij * W_p


def classify_trips(
    R_ij: Union[int, pd.Series],
    distance_km: Union[float, pd.Series],
    Cap_j: Union[int, pd.Series],
    config: Optional[Dict[str, Any]] = None
) -> Union[str, pd.Series]:
    """
    Classify trips into feasibility categories per @eq-trip-classification.

    Class_ij = {
        "immediately_feasible"  if R_ij=1 AND d_ij≤40km
        "constrained"           if R_ij=1 AND 40<d_ij≤80km AND Cap_j=1
        "infeasible"            otherwise
    }

    Args:
        R_ij: Binary replaceability indicator
        distance_km: Trip distance in km
        Cap_j: Capacity constraint indicator (binary)
        config: Optional config with replaceability_thresholds

    Returns:
        Classification: "immediately_feasible", "constrained", or "infeasible"
    """
    short_threshold = 40.0
    medium_threshold = 80.0
    if config and 'replaceability_thresholds' in config:
        rt = config['replaceability_thresholds']
        short_threshold = rt.get('short_trip_distance_km', 40.0)
        medium_threshold = rt.get('medium_trip_distance_km', 80.0)

    if isinstance(R_ij, pd.Series):
        result = pd.Series("infeasible", index=R_ij.index)
        # Immediately feasible: R_ij=1 AND d≤40km
        imm_mask = (R_ij == 1) & (distance_km <= short_threshold)
        result.loc[imm_mask] = "immediately_feasible"
        # Constrained: R_ij=1 AND 40<d≤80km AND Cap_j=1
        const_mask = (
            (R_ij == 1) &
            (distance_km > short_threshold) &
            (distance_km <= medium_threshold) &
            (Cap_j == 1)
        )
        result.loc[const_mask] = "constrained"
        return result
    else:
        if R_ij == 1 and distance_km <= short_threshold:
            return "immediately_feasible"
        elif R_ij == 1 and short_threshold < distance_km <= medium_threshold and Cap_j == 1:
            return "constrained"
        else:
            return "infeasible"
