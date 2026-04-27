"""
Range Feasibility Assessment Module for Frugal EV Analysis

This module provides classes and functions for evaluating trip feasibility within
the operational range constraints of frugal EVs, using realistic routing distances
from services like OSRM (Open Source Routing Machine).

Key Design Principles:
- Uses effective operational range (R_eff = 80km with default 0.8 buffer factor)
  rather than maximum theoretical range (R_max = 100km) for safety margins
- Applies safety buffer to account for range anxiety, environmental conditions,
  driving variations, and battery degradation
- Three-tier classification: Feasible (≤40km), Constrained (40-80km), Infeasible (>80km)
- Configurable parameters via scoring_weights.json for scenario analysis
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
from .data_integration import standardize_od_column_names

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RangeFeasibilityCategory(Enum):
    """
    Enumeration of range feasibility categories for frugal EV trips.
    
    Categories based on effective operational range (80km with default 0.8 buffer factor)
    and destination charging threshold (40km by default).
    """
    FEASIBLE = "feasible"                    # ≤40km one-way, no destination charging needed
    CONSTRAINED = "constrained"              # 40-80km one-way, destination charging required  
    INFEASIBLE = "infeasible"                # >80km one-way, beyond effective range capability


@dataclass
class RangeAssessment:
    """Container for range feasibility assessment results"""
    origin_code: str
    destination_code: str
    route_distance_km: float
    one_way_feasible: bool
    round_trip_feasible: bool
    destination_charging_required: bool
    feasibility_category: RangeFeasibilityCategory
    range_buffer_used: float  # Proportion of 100km range used
    

@dataclass
class RangeConstraints:
    """Container for range constraint parameters"""
    max_range_km: float = 100.0              # Maximum frugal EV range
    buffer_factor: float = 0.8               # Safety buffer (80% of max range)
    destination_charging_threshold: float = 40.0  # Distance requiring destination charging
    round_trip_factor: float = 2.0           # Multiplier for round-trip assessment


class RangeFeasibilityAssessor:
    """
    Main assessment engine for evaluating trip feasibility within frugal EV range constraints.
    
    Implements range constraint logic using realistic routing distances (e.g., from OSRM/OSM)
    with one-way and round-trip feasibility assessment capabilities. Uses effective operational
    range (80km by default with 0.8 buffer factor) rather than maximum theoretical range (100km)
    to ensure safety margins are respected in real-world conditions.
    """
    
    def __init__(self, 
                 max_range_km: float = 100.0,
                 buffer_factor: float = 0.8,
                 destination_charging_threshold: float = 40.0,
                 round_trip_factor: float = 2.0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize RangeFeasibilityAssessor with range parameters.
        
        Args:
            max_range_km: Maximum theoretical range of frugal EV in kilometers
            buffer_factor: Safety buffer as proportion of max range (0.8 = 80%)
            destination_charging_threshold: Distance above which destination charging is required (km)
            round_trip_factor: Multiplier for round-trip distance calculation (default: 2.0)
            config: Optional config dict containing 'range_feasibility' section from scoring_weights.json
        
        Note:
            If config is provided, it overrides individual parameters.
            The effective_range_km (max_range_km * buffer_factor) is used for all
            operational feasibility decisions to ensure safety margins are respected.
        """
        # Load from config if provided
        if config and 'range_feasibility' in config:
            rf_config = config['range_feasibility']
            max_range_km = rf_config.get('max_range_km', max_range_km)
            buffer_factor = rf_config.get('buffer_factor', buffer_factor)
            destination_charging_threshold = rf_config.get('destination_charging_threshold_km', destination_charging_threshold)
            round_trip_factor = rf_config.get('round_trip_factor', round_trip_factor)
        
        self.constraints = RangeConstraints(
            max_range_km=max_range_km,
            buffer_factor=buffer_factor,
            destination_charging_threshold=destination_charging_threshold,
            round_trip_factor=round_trip_factor
        )
        
        # Calculate effective range with safety buffer
        # This is the operational limit used for feasibility classification
        self.effective_range_km = self.constraints.max_range_km * self.constraints.buffer_factor
        
        logger.info(f"Initialized RangeFeasibilityAssessor with {max_range_km}km max range, "
                   f"{self.effective_range_km}km effective range (with {buffer_factor*100:.0f}% buffer)")
    
    def assess_single_trip_feasibility(self, 
                                     route_distance_km: float,
                                     origin_code: str = "",
                                     destination_code: str = "") -> RangeAssessment:
        """
        Assess range feasibility for a single trip using effective range with safety buffer.
        
        CRITICAL: All operational decisions use effective_range_km (80km with default buffer),
        NOT max_range_km, to ensure safety margins are respected in real-world conditions.
        
        Args:
            route_distance_km: Actual route distance from routing service (e.g., OSRM)
            origin_code: Origin area code (optional)
            destination_code: Destination area code (optional)
            
        Returns:
            RangeAssessment object with feasibility results
        """
        # One-way feasibility: Can complete trip within effective operational range
        one_way_feasible = route_distance_km <= self.effective_range_km
        
        # Round-trip feasibility: Can complete round-trip within effective range
        round_trip_distance = route_distance_km * self.constraints.round_trip_factor
        round_trip_feasible = round_trip_distance <= self.effective_range_km
        
        # Destination charging requirement: Needed for trips between threshold and effective range
        destination_charging_required = (
            route_distance_km > self.constraints.destination_charging_threshold and
            route_distance_km <= self.effective_range_km
        )
        
        # Determine feasibility category
        # BUG FIX: Use effective_range_km (not max_range_km) for infeasibility classification
        # This ensures safety buffer is applied to operational decisions
        if route_distance_km > self.effective_range_km:  # FIXED: was max_range_km
            category = RangeFeasibilityCategory.INFEASIBLE
        elif route_distance_km > self.constraints.destination_charging_threshold:
            category = RangeFeasibilityCategory.CONSTRAINED
        else:
            category = RangeFeasibilityCategory.FEASIBLE
        
        # Calculate range buffer usage (% of maximum theoretical range)
        # This is a diagnostic metric showing how much of theoretical max is used
        range_buffer_used = route_distance_km / self.constraints.max_range_km
        
        return RangeAssessment(
            origin_code=origin_code,
            destination_code=destination_code,
            route_distance_km=route_distance_km,
            one_way_feasible=one_way_feasible,
            round_trip_feasible=round_trip_feasible,
            destination_charging_required=destination_charging_required,
            feasibility_category=category,
            range_buffer_used=range_buffer_used
        )
    
    def assess_range_feasibility(self, routes_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Assess range feasibility for all routes in the dataset.
        
        Args:
            routes_data: GeoDataFrame with route data including distance information
            
        Returns:
            GeoDataFrame with range feasibility assessment results added
            
        Raises:
            ValueError: If required distance column is not found
        """
        logger.info(f"Assessing range feasibility for {len(routes_data)} routes")
        
        # Standardize column names
        routes_data = standardize_od_column_names(routes_data)
        
        # Validate input data
        distance_col = self._identify_distance_column(routes_data)
        if not distance_col:
            raise ValueError("No route distance column found in routes data")
        
        # Create result dataframe
        result = routes_data.copy()
        
        # Convert distance to kilometers if needed
        distances_km = self._convert_to_kilometers(result[distance_col], distance_col)
        
        # Assess feasibility for each route
        assessments = []
        for idx, row in result.iterrows():
            distance_km = distances_km.loc[idx] if isinstance(distances_km, pd.Series) else distances_km[idx]
            
            origin_code = str(row.get('origin_code', ''))
            dest_code = str(row.get('destination_code', ''))
            
            assessment = self.assess_single_trip_feasibility(
                route_distance_km=distance_km,
                origin_code=origin_code,
                destination_code=dest_code
            )
            assessments.append(assessment)
        
        # Add assessment results to dataframe
        result['route_distance_km'] = [a.route_distance_km for a in assessments]
        result['one_way_feasible'] = [a.one_way_feasible for a in assessments]
        result['round_trip_feasible'] = [a.round_trip_feasible for a in assessments]
        result['destination_charging_required'] = [a.destination_charging_required for a in assessments]
        result['feasibility_category'] = [a.feasibility_category.value for a in assessments]
        result['range_buffer_used'] = [a.range_buffer_used for a in assessments]
        
        # Calculate summary statistics
        self._log_feasibility_summary(result)
        
        return result
    
    def _identify_distance_column(self, routes_data: gpd.GeoDataFrame) -> Optional[str]:
        """Identify the column containing route distance data"""
        # Common distance column names from routing services (OSRM, Google Maps, etc.)
        possible_columns = [
            'route_distance_km',     # Standardized format
            'distance_km',           # Preferred format
            'distance',              # Generic distance
            'route_distance',        # Route-specific distance
            'driving_distance_km',   # Driving distance in km 
            'driving_distance',      # Driving distance
            'distance_meters',       # Distance in meters
            'distance_m',           # Distance in meters (short)
            'length_km',            # Length in km
            'length'                # Generic length
        ]
        
        for col in possible_columns:
            if col in routes_data.columns:
                return col
        
        # Look for columns containing 'distance' or 'length'
        for col in routes_data.columns:
            if 'distance' in col.lower() or 'length' in col.lower():
                return col
        
        return None
    
    def _convert_to_kilometers(self, distances: pd.Series, column_name: str) -> pd.Series:
        """Convert distance values to kilometers based on likely units"""
        # Check if values suggest meters (typically > 1000 for meaningful trips)
        if distances.median() > 1000:
            logger.info(f"Converting {column_name} from meters to kilometers")
            return distances / 1000.0
        else:
            logger.info(f"Using {column_name} values as kilometers")
            return distances
    
    def _log_feasibility_summary(self, result: gpd.GeoDataFrame) -> None:
        """Log summary statistics of feasibility assessment"""
        total_routes = len(result)
        
        # Count by feasibility category
        feasible_count = (result['feasibility_category'] == RangeFeasibilityCategory.FEASIBLE.value).sum()
        constrained_count = (result['feasibility_category'] == RangeFeasibilityCategory.CONSTRAINED.value).sum()
        infeasible_count = (result['feasibility_category'] == RangeFeasibilityCategory.INFEASIBLE.value).sum()
        
        # Round-trip feasibility
        round_trip_feasible = result['round_trip_feasible'].sum()
        
        # Destination charging requirements
        dest_charging_needed = result['destination_charging_required'].sum()
        
        logger.info(f"Range feasibility assessment completed:")
        logger.info(f"  Total routes: {total_routes:,}")
        logger.info(f"  Feasible (≤40km): {feasible_count:,} ({feasible_count/total_routes*100:.1f}%)")
        logger.info(f"  Constrained (40-80km): {constrained_count:,} ({constrained_count/total_routes*100:.1f}%)")
        logger.info(f"  Infeasible (>80km effective range): {infeasible_count:,} ({infeasible_count/total_routes*100:.1f}%)")
        logger.info(f"  Round-trip feasible: {round_trip_feasible:,} ({round_trip_feasible/total_routes*100:.1f}%)")
        logger.info(f"  Destination charging required: {dest_charging_needed:,} ({dest_charging_needed/total_routes*100:.1f}%)")
    
    def evaluate_round_trip_feasibility(self, route_distance: float) -> Dict[str, bool]:
        """
        Evaluate round-trip feasibility for a given route distance.
        
        Args:
            route_distance: One-way route distance in kilometers
            
        Returns:
            Dict with round-trip feasibility assessment
        """
        round_trip_distance = route_distance * self.constraints.round_trip_factor
        
        return {
            'one_way_within_range': route_distance <= self.effective_range_km,
            'round_trip_within_range': round_trip_distance <= self.effective_range_km,
            'one_way_within_max': route_distance <= self.constraints.max_range_km,
            'round_trip_within_max': round_trip_distance <= self.constraints.max_range_km,
            'destination_charging_sufficient': (
                route_distance <= self.effective_range_km and  # FIXED: was max_range_km
                route_distance > self.constraints.destination_charging_threshold
            ),
            'round_trip_distance_km': round_trip_distance,
            'range_utilization': route_distance / self.constraints.max_range_km
        }
    
    def get_feasibility_statistics(self, routes_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive feasibility statistics for the route dataset.
        
        Args:
            routes_data: GeoDataFrame with feasibility assessment results
            
        Returns:
            Dict with detailed feasibility statistics
        """
        if 'feasibility_category' not in routes_data.columns:
            raise ValueError("Routes data must contain feasibility assessment results")
        
        total_routes = len(routes_data)
        
        # Category counts
        category_counts = routes_data['feasibility_category'].value_counts()
        
        # Distance statistics
        distance_stats = routes_data['route_distance_km'].describe()
        
        # Range utilization statistics
        range_utilization = routes_data['range_buffer_used'].describe()
        
        # Trip volume analysis (if available)
        trip_volume_stats = {}
        if 'All' in routes_data.columns:
            # Weight statistics by trip volume
            total_trips = routes_data['All'].sum()
            
            for category in RangeFeasibilityCategory:
                category_mask = routes_data['feasibility_category'] == category.value
                category_trips = routes_data[category_mask]['All'].sum()
                trip_volume_stats[category.value] = {
                    'trip_count': category_trips,
                    'trip_percentage': category_trips / total_trips * 100 if total_trips > 0 else 0
                }
        
        return {
            'total_routes': total_routes,
            'category_distribution': {
                'feasible': category_counts.get(RangeFeasibilityCategory.FEASIBLE.value, 0),
                'constrained': category_counts.get(RangeFeasibilityCategory.CONSTRAINED.value, 0),
                'infeasible': category_counts.get(RangeFeasibilityCategory.INFEASIBLE.value, 0)
            },
            'category_percentages': {
                'feasible': category_counts.get(RangeFeasibilityCategory.FEASIBLE.value, 0) / total_routes * 100,
                'constrained': category_counts.get(RangeFeasibilityCategory.CONSTRAINED.value, 0) / total_routes * 100,
                'infeasible': category_counts.get(RangeFeasibilityCategory.INFEASIBLE.value, 0) / total_routes * 100
            },
            'distance_statistics': {
                'mean_km': distance_stats['mean'],
                'median_km': distance_stats['50%'],
                'min_km': distance_stats['min'],
                'max_km': distance_stats['max'],
                'std_km': distance_stats['std']
            },
            'range_utilization': {
                'mean_utilization': range_utilization['mean'],
                'max_utilization': range_utilization['max'],
                'routes_over_80_percent': (routes_data['range_buffer_used'] > 0.8).sum(),
                'routes_over_90_percent': (routes_data['range_buffer_used'] > 0.9).sum()
            },
            'round_trip_analysis': {
                'round_trip_feasible_count': routes_data['round_trip_feasible'].sum(),
                'round_trip_feasible_percentage': routes_data['round_trip_feasible'].mean() * 100,
                'destination_charging_required_count': routes_data['destination_charging_required'].sum(),
                'destination_charging_required_percentage': routes_data['destination_charging_required'].mean() * 100
            },
            'trip_volume_weighted': trip_volume_stats
        }
    
    def validate_range_calculations(self, routes_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate range feasibility calculations for accuracy and consistency.
        
        Args:
            routes_data: GeoDataFrame with range assessment results
            
        Returns:
            Dict with validation results
        """
        logger.info("Validating range feasibility calculations")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        required_cols = ['route_distance_km', 'feasibility_category', 'one_way_feasible', 
                        'round_trip_feasible', 'destination_charging_required']
        missing_cols = [col for col in required_cols if col not in routes_data.columns]
        
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
            return validation_results
        
        # Validate distance values
        distances = routes_data['route_distance_km']
        if distances.min() < 0:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Negative distances found: {distances.min():.2f}")
        
        if distances.max() > 1000:  # Unrealistic for typical trips
            validation_results['issues'].append(f"Very long distances found: {distances.max():.2f}km")
        
        # Validate feasibility logic consistency
        for idx, row in routes_data.iterrows():
            distance = row['route_distance_km']
            category = row['feasibility_category']
            one_way = row['one_way_feasible']
            round_trip = row['round_trip_feasible']
            dest_charging = row['destination_charging_required']
            
            # Check category consistency (using effective_range_km for operational decisions)
            if distance <= self.constraints.destination_charging_threshold:
                if category != RangeFeasibilityCategory.FEASIBLE.value:
                    validation_results['issues'].append(f"Row {idx}: Short trip ({distance:.1f}km) not marked as feasible")
            elif distance <= self.effective_range_km:  # FIXED: was max_range_km
                if category != RangeFeasibilityCategory.CONSTRAINED.value:
                    validation_results['issues'].append(f"Row {idx}: Medium trip ({distance:.1f}km) not marked as constrained")
            else:
                if category != RangeFeasibilityCategory.INFEASIBLE.value:
                    validation_results['issues'].append(f"Row {idx}: Long trip ({distance:.1f}km) not marked as infeasible")
            
            # Check one-way feasibility consistency
            expected_one_way = distance <= self.effective_range_km
            if one_way != expected_one_way:
                validation_results['issues'].append(f"Row {idx}: One-way feasibility inconsistent")
            
            # Check destination charging logic
            expected_dest_charging = (
                distance > self.constraints.destination_charging_threshold and 
                distance <= self.effective_range_km
            )
            if dest_charging != expected_dest_charging:
                validation_results['issues'].append(f"Row {idx}: Destination charging logic inconsistent")
        
        # Calculate validation statistics
        validation_results['statistics'] = {
            'total_routes_validated': len(routes_data),
            'logic_errors_found': len([issue for issue in validation_results['issues'] if 'Row' in issue]),
            'distance_range': f"{distances.min():.1f} - {distances.max():.1f} km",
            'feasibility_distribution': routes_data['feasibility_category'].value_counts().to_dict()
        }
        
        if validation_results['issues']:
            validation_results['is_valid'] = len(validation_results['issues']) <= 5  # Allow minor issues
        
        logger.info(f"Range calculation validation {'passed' if validation_results['is_valid'] else 'failed'}")
        
        return validation_results


class DestinationChargingEvaluator:
    """
    Specialized evaluator for destination charging requirements and long-trip analysis.
    
    Identifies trips requiring destination charging (>40km by default) and classifies
    infeasible trips (>80km effective range by default) for frugal EV adoption analysis.
    
    Note:
        Uses effective operational range (R_eff = 80km with default buffer), not
        maximum theoretical range (R_max = 100km), for infeasibility classification.
    """
    
    def __init__(self, 
                 destination_charging_threshold: float = 40.0,
                 max_feasible_distance: float = 80.0,  # CHANGED: Default to effective range
                 charging_time_minutes: float = 30.0,
                 moderate_severity_threshold: float = 150.0,
                 significant_severity_threshold: float = 200.0,
                 very_long_distance_threshold: float = 300.0,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize DestinationChargingEvaluator with charging parameters.
        
        Args:
            destination_charging_threshold: Distance above which destination charging is required (km)
            max_feasible_distance: Maximum feasible distance for frugal EVs (effective range, km)
            charging_time_minutes: Typical destination charging time required
            moderate_severity_threshold: Distance threshold for moderate infeasibility (km)
            significant_severity_threshold: Distance threshold for significant infeasibility (km)
            very_long_distance_threshold: Distance threshold for very long trips requiring air travel (km)
            config: Optional config dict containing 'range_feasibility' section
        
        Note:
            max_feasible_distance should typically be the effective range (R_eff = 80km),
            not the maximum theoretical range (R_max = 100km).
        """
        # Load from config if provided
        if config and 'range_feasibility' in config:
            rf_config = config['range_feasibility']
            destination_charging_threshold = rf_config.get('destination_charging_threshold_km', destination_charging_threshold)
            charging_time_minutes = rf_config.get('baseline_charging_time_minutes', charging_time_minutes)
            # Calculate effective range from config
            max_range = rf_config.get('max_range_km', 100.0)
            buffer_factor = rf_config.get('buffer_factor', 0.8)
            max_feasible_distance = max_range * buffer_factor  # Use effective range
            # Load severity band thresholds if available
            if 'infeasibility_severity_bands' in rf_config:
                severity_bands = rf_config['infeasibility_severity_bands']
                moderate_severity_threshold = severity_bands.get('moderate_threshold_km', moderate_severity_threshold)
                significant_severity_threshold = severity_bands.get('significant_threshold_km', significant_severity_threshold)
                very_long_distance_threshold = severity_bands.get('very_long_distance_km', very_long_distance_threshold)
        
        self.destination_charging_threshold = destination_charging_threshold
        self.max_feasible_distance = max_feasible_distance
        self.charging_time_minutes = charging_time_minutes
        self.moderate_severity_threshold = moderate_severity_threshold
        self.significant_severity_threshold = significant_severity_threshold
        self.very_long_distance_threshold = very_long_distance_threshold
        
        logger.info(f"Initialized DestinationChargingEvaluator with {destination_charging_threshold}km threshold, "
                   f"{max_feasible_distance}km max feasible distance (effective range)")
    
    def identify_destination_charging_requirements(self, routes_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Identify trips requiring destination charging based on distance thresholds.
        
        Args:
            routes_data: GeoDataFrame with route distance information
            
        Returns:
            GeoDataFrame with destination charging requirements added
            
        Raises:
            ValueError: If distance column is not found
        """
        logger.info(f"Identifying destination charging requirements for {len(routes_data)} routes")
        
        # Find distance column
        distance_col = self._identify_distance_column(routes_data)
        if not distance_col:
            raise ValueError("No route distance column found in routes data")
        
        result = routes_data.copy()
        
        # Convert distances to kilometers if needed
        distances_km = self._convert_to_kilometers(result[distance_col], distance_col)
        result['route_distance_km'] = distances_km
        
        # Identify destination charging requirements
        result['requires_destination_charging'] = (
            distances_km > self.destination_charging_threshold
        ) & (distances_km <= self.max_feasible_distance)
        
        # Classify charging scenarios
        result['charging_scenario'] = self._classify_charging_scenarios(distances_km)
        
        # Estimate charging time requirements
        result['estimated_charging_time_minutes'] = self._estimate_charging_time(distances_km)
        
        # Identify trip categories for EV feasibility
        result['trip_category'] = self._categorize_trips_for_ev_feasibility(distances_km)
        
        # Calculate destination charging statistics
        self._log_destination_charging_summary(result)
        
        return result
    
    def _classify_charging_scenarios(self, distances_km: pd.Series) -> pd.Series:
        """Classify trips into charging scenarios"""
        def classify_single_trip(distance):
            if distance <= self.destination_charging_threshold:
                return "home_charging_sufficient"
            elif distance <= self.max_feasible_distance:
                return "destination_charging_required"
            else:
                return "infeasible_for_frugal_ev"
        
        return distances_km.apply(classify_single_trip)
    
    def _estimate_charging_time(self, distances_km: pd.Series) -> pd.Series:
        """
        Estimate required charging time based on trip distance.
        
        Note:
            This provides distance-based time estimates only. Full implementation per
            @eq-charging-time in the paper should also check infrastructure capacity
            (C_{avail,j} > 0) and return infinity if no capacity available. Integration
            with Stage 4 capacity data (from ChargingCapacityAnalyzer) would enable
            this gating logic.
            
        Returns:
            Series with estimated charging times (minutes). Returns 0 for trips not
            requiring destination charging, finite positive values for constrained
            trips, and infinity for infeasible trips.
        """
        def estimate_time(distance):
            if distance <= self.destination_charging_threshold:
                return 0.0  # No destination charging needed
            elif distance <= self.max_feasible_distance:
                # Estimate charging time based on distance beyond threshold
                excess_distance = distance - self.destination_charging_threshold
                # Linear relationship: more distance = more charging time
                # TODO: Gate by infrastructure capacity (C_{avail,j} > 0) per @eq-charging-time
                return self.charging_time_minutes * (1 + excess_distance / 50.0)
            else:
                return float('inf')  # Infeasible - exceeds effective range
        
        return distances_km.apply(estimate_time)
    
    def _categorize_trips_for_ev_feasibility(self, distances_km: pd.Series) -> pd.Series:
        """Categorize trips for EV feasibility analysis"""
        def categorize_trip(distance):
            if distance <= self.destination_charging_threshold:
                return "feasible_no_destination_charging"
            elif distance <= self.max_feasible_distance:
                return "feasible_with_destination_charging"
            else:
                return "infeasible_exceeds_range"
        
        return distances_km.apply(categorize_trip)
    
    def _identify_distance_column(self, routes_data: gpd.GeoDataFrame) -> Optional[str]:
        """Identify the column containing route distance data"""
        possible_columns = [
            'distance_km', 'distance', 'route_distance', 'driving_distance',
            'distance_meters', 'distance_m', 'length_km', 'length'
        ]
        
        for col in possible_columns:
            if col in routes_data.columns:
                return col
        
        # Look for columns containing 'distance' or 'length'
        for col in routes_data.columns:
            if 'distance' in col.lower() or 'length' in col.lower():
                return col
        
        return None
    
    def _convert_to_kilometers(self, distances: pd.Series, column_name: str) -> pd.Series:
        """Convert distance values to kilometers based on likely units"""
        if distances.median() > 1000:
            logger.info(f"Converting {column_name} from meters to kilometers")
            return distances / 1000.0
        else:
            logger.info(f"Using {column_name} values as kilometers")
            return distances
    
    def _log_destination_charging_summary(self, result: gpd.GeoDataFrame) -> None:
        """Log summary statistics of destination charging analysis"""
        total_trips = len(result)
        
        # Count by charging scenario
        scenario_counts = result['charging_scenario'].value_counts()
        
        # Count by trip category
        category_counts = result['trip_category'].value_counts()
        
        # Destination charging statistics
        dest_charging_needed = result['requires_destination_charging'].sum()
        
        logger.info(f"Destination charging analysis completed:")
        logger.info(f"  Total trips: {total_trips:,}")
        logger.info(f"  Home charging sufficient: {scenario_counts.get('home_charging_sufficient', 0):,}")
        logger.info(f"  Destination charging required: {scenario_counts.get('destination_charging_required', 0):,}")
        logger.info(f"  Infeasible for frugal EV: {scenario_counts.get('infeasible_for_frugal_ev', 0):,}")
        logger.info(f"  Trips requiring destination charging: {dest_charging_needed:,} ({dest_charging_needed/total_trips*100:.1f}%)")
    
    def classify_infeasible_trips(self, routes_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Classify trips as infeasible for frugal EVs (exceeding effective operational range).
        
        By default, trips >80km (effective range with safety buffer) are infeasible.
        
        Args:
            routes_data: GeoDataFrame with route distance information
            
        Returns:
            GeoDataFrame with infeasibility classification added
        """
        logger.info(f"Classifying infeasible trips for {len(routes_data)} routes")
        
        # Find distance column
        distance_col = self._identify_distance_column(routes_data)
        if not distance_col:
            raise ValueError("No route distance column found in routes data")
        
        result = routes_data.copy()
        
        # Convert distances to kilometers if needed
        distances_km = self._convert_to_kilometers(result[distance_col], distance_col)
        result['route_distance_km'] = distances_km
        
        # Classify infeasible trips
        result['infeasible_for_frugal_ev'] = distances_km > self.max_feasible_distance
        
        # Classify infeasibility reasons
        result['infeasibility_reason'] = self._classify_infeasibility_reasons(distances_km)
        
        # Calculate infeasibility severity
        result['infeasibility_severity'] = self._calculate_infeasibility_severity(distances_km)
        
        # Identify alternative solutions
        result['alternative_solutions'] = self._suggest_alternative_solutions(distances_km)
        
        # Log infeasibility statistics
        self._log_infeasibility_summary(result)
        
        return result
    
    def _classify_infeasibility_reasons(self, distances_km: pd.Series) -> pd.Series:
        """
        Classify reasons for infeasibility using configurable severity bands.
        
        Note:
            For the 100km frugal EV study, all trips >80km are simply INFEASIBLE.
            These severity bands are implementation details for diagnostic purposes
            and alternative vehicle analysis, not core to the feasibility assessment.
        
        Bands:
        - feasible: ≤ max_feasible_distance (typically 80km)
        - moderately_exceeds_range: > max_feasible to moderate_threshold (default 150km)
        - significantly_exceeds_range: > moderate to significant_threshold (default 200km)
        - far_exceeds_range: > significant_threshold
        """
        def classify_reason(distance):
            if distance <= self.max_feasible_distance:
                return "feasible"
            elif distance <= self.moderate_severity_threshold:
                return "moderately_exceeds_range"
            elif distance <= self.significant_severity_threshold:
                return "significantly_exceeds_range"
            else:
                return "far_exceeds_range"
        
        return distances_km.apply(classify_reason)
    
    def _calculate_infeasibility_severity(self, distances_km: pd.Series) -> pd.Series:
        """Calculate severity of infeasibility (how much it exceeds max range)"""
        return (distances_km - self.max_feasible_distance) / self.max_feasible_distance
    
    def _suggest_alternative_solutions(self, distances_km: pd.Series) -> pd.Series:
        """
        Suggest alternative solutions for infeasible trips using configurable thresholds.
        
        Recommendations:
        - ≤ max_feasible: Frugal EV suitable
        - > max_feasible to moderate: Longer-range EV or public transport
        - > moderate to very_long: Conventional vehicle or public transport  
        - > very_long: Air travel or long-distance transport
        """
        def suggest_alternatives(distance):
            if distance <= self.max_feasible_distance:
                return "frugal_ev_suitable"
            elif distance <= self.moderate_severity_threshold:
                return "longer_range_ev_or_public_transport"
            elif distance <= self.very_long_distance_threshold:
                return "conventional_vehicle_or_public_transport"
            else:
                return "air_travel_or_long_distance_transport"
        
        return distances_km.apply(suggest_alternatives)
    
    def _log_infeasibility_summary(self, result: gpd.GeoDataFrame) -> None:
        """Log summary statistics of infeasibility analysis"""
        total_trips = len(result)
        infeasible_trips = result['infeasible_for_frugal_ev'].sum()
        
        # Count by infeasibility reason
        reason_counts = result['infeasibility_reason'].value_counts()
        
        logger.info(f"Infeasibility analysis completed:")
        logger.info(f"  Total trips: {total_trips:,}")
        logger.info(f"  Infeasible trips: {infeasible_trips:,} ({infeasible_trips/total_trips*100:.1f}%)")
        logger.info(f"  Feasible trips: {total_trips - infeasible_trips:,} ({(total_trips - infeasible_trips)/total_trips*100:.1f}%)")
        
        for reason, count in reason_counts.items():
            if reason != "feasible":
                logger.info(f"  {reason}: {count:,} ({count/total_trips*100:.1f}%)")
    
    def analyze_long_trip_patterns(self, routes_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in long trips requiring destination charging or infeasible for frugal EVs.
        
        Args:
            routes_data: GeoDataFrame with route and trip purpose information
            
        Returns:
            Dict with long trip pattern analysis results
        """
        logger.info("Analyzing long trip patterns")
        
        # Identify destination charging requirements
        analyzed_routes = self.identify_destination_charging_requirements(routes_data)
        
        # Filter for long trips (>40km by default)
        long_trips = analyzed_routes[
            analyzed_routes['route_distance_km'] > self.destination_charging_threshold
        ].copy()
        
        if len(long_trips) == 0:
            return {'message': 'No long trips found in dataset'}
        
        # Analyze by trip purpose (if available)
        purpose_analysis = {}
        if 'purpose' in long_trips.columns:
            purpose_analysis = self._analyze_by_trip_purpose(long_trips)
        
        # Analyze by distance bands
        distance_band_analysis = self._analyze_by_distance_bands(long_trips)
        
        # Analyze charging time requirements
        charging_time_analysis = self._analyze_charging_time_requirements(long_trips)
        
        # Calculate trip volume impacts (if available)
        volume_analysis = {}
        if 'All' in long_trips.columns:
            volume_analysis = self._analyze_trip_volume_impacts(long_trips)
        
        return {
            'total_long_trips': len(long_trips),
            'long_trip_percentage': len(long_trips) / len(analyzed_routes) * 100,
            'purpose_analysis': purpose_analysis,
            'distance_band_analysis': distance_band_analysis,
            'charging_time_analysis': charging_time_analysis,
            'volume_analysis': volume_analysis,
            'summary_statistics': {
                'mean_distance_km': long_trips['route_distance_km'].mean(),
                'median_distance_km': long_trips['route_distance_km'].median(),
                'max_distance_km': long_trips['route_distance_km'].max(),
                'destination_charging_required': long_trips['requires_destination_charging'].sum(),
                'infeasible_count': long_trips['infeasible_for_frugal_ev'].sum() if 'infeasible_for_frugal_ev' in long_trips.columns else 0
            }
        }
    
    def _analyze_by_trip_purpose(self, long_trips: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Analyze long trips by trip purpose"""
        purpose_stats = {}
        
        for purpose in long_trips['purpose'].unique():
            if pd.isna(purpose):
                continue
                
            purpose_trips = long_trips[long_trips['purpose'] == purpose]
            
            purpose_stats[purpose] = {
                'trip_count': len(purpose_trips),
                'mean_distance_km': purpose_trips['route_distance_km'].mean(),
                'destination_charging_required': purpose_trips['requires_destination_charging'].sum(),
                'infeasible_count': purpose_trips.get('infeasible_for_frugal_ev', pd.Series([False]*len(purpose_trips))).sum(),
                'mean_charging_time_minutes': purpose_trips['estimated_charging_time_minutes'].replace([float('inf')], 0).mean()
            }
        
        return purpose_stats
    
    def _analyze_by_distance_bands(self, long_trips: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Analyze long trips by distance bands"""
        distance_bands = {
            '40-60km': (40, 60),
            '60-80km': (60, 80),
            '80-100km': (80, 100),
            '100-150km': (100, 150),
            '150km+': (150, float('inf'))
        }
        
        band_stats = {}
        
        for band_name, (min_dist, max_dist) in distance_bands.items():
            if max_dist == float('inf'):
                band_trips = long_trips[long_trips['route_distance_km'] >= min_dist]
            else:
                band_trips = long_trips[
                    (long_trips['route_distance_km'] >= min_dist) & 
                    (long_trips['route_distance_km'] < max_dist)
                ]
            
            if len(band_trips) > 0:
                band_stats[band_name] = {
                    'trip_count': len(band_trips),
                    'percentage_of_long_trips': len(band_trips) / len(long_trips) * 100,
                    'mean_distance_km': band_trips['route_distance_km'].mean(),
                    'destination_charging_required': band_trips['requires_destination_charging'].sum(),
                    'feasible_for_frugal_ev': (band_trips['route_distance_km'] <= self.max_feasible_distance).sum()
                }
        
        return band_stats
    
    def _analyze_charging_time_requirements(self, long_trips: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Analyze charging time requirements for long trips"""
        # Filter out infinite values (infeasible trips)
        feasible_long_trips = long_trips[
            long_trips['estimated_charging_time_minutes'] != float('inf')
        ]
        
        if len(feasible_long_trips) == 0:
            return {'message': 'No feasible long trips requiring charging time analysis'}
        
        charging_times = feasible_long_trips['estimated_charging_time_minutes']
        
        return {
            'mean_charging_time_minutes': charging_times.mean(),
            'median_charging_time_minutes': charging_times.median(),
            'max_charging_time_minutes': charging_times.max(),
            'trips_requiring_30min_plus': (charging_times >= 30).sum(),
            'trips_requiring_60min_plus': (charging_times >= 60).sum(),
            'charging_time_distribution': {
                '0-30min': ((charging_times >= 0) & (charging_times < 30)).sum(),
                '30-60min': ((charging_times >= 30) & (charging_times < 60)).sum(),
                '60min+': (charging_times >= 60).sum()
            }
        }
    
    def _analyze_trip_volume_impacts(self, long_trips: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Analyze trip volume impacts for long trips"""
        total_long_trip_volume = long_trips['All'].sum()
        
        # Volume by charging scenario
        scenario_volumes = long_trips.groupby('charging_scenario')['All'].sum().to_dict()
        
        # Volume by feasibility
        feasible_volume = long_trips[
            long_trips['route_distance_km'] <= self.max_feasible_distance
        ]['All'].sum()
        
        infeasible_volume = long_trips[
            long_trips['route_distance_km'] > self.max_feasible_distance
        ]['All'].sum()
        
        return {
            'total_long_trip_volume': total_long_trip_volume,
            'volume_by_scenario': scenario_volumes,
            'feasible_trip_volume': feasible_volume,
            'infeasible_trip_volume': infeasible_volume,
            'feasible_percentage': feasible_volume / total_long_trip_volume * 100 if total_long_trip_volume > 0 else 0
        }
    
    def validate_destination_charging_logic(self, routes_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate destination charging evaluation logic for consistency.
        
        Args:
            routes_data: GeoDataFrame with destination charging analysis results
            
        Returns:
            Dict with validation results
        """
        logger.info("Validating destination charging evaluation logic")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        required_cols = ['route_distance_km', 'requires_destination_charging', 'charging_scenario']
        missing_cols = [col for col in required_cols if col not in routes_data.columns]
        
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
            return validation_results
        
        # Validate destination charging logic
        for idx, row in routes_data.iterrows():
            distance = row['route_distance_km']
            requires_charging = row['requires_destination_charging']
            scenario = row['charging_scenario']
            
            # Check destination charging requirement logic
            expected_charging = (
                distance > self.destination_charging_threshold and 
                distance <= self.max_feasible_distance
            )
            
            if requires_charging != expected_charging:
                validation_results['issues'].append(
                    f"Row {idx}: Destination charging requirement inconsistent for {distance:.1f}km"
                )
            
            # Check scenario classification consistency
            if distance <= self.destination_charging_threshold:
                expected_scenario = "home_charging_sufficient"
            elif distance <= self.max_feasible_distance:
                expected_scenario = "destination_charging_required"
            else:
                expected_scenario = "infeasible_for_frugal_ev"
            
            if scenario != expected_scenario:
                validation_results['issues'].append(
                    f"Row {idx}: Charging scenario inconsistent for {distance:.1f}km"
                )
        
        # Calculate validation statistics
        validation_results['statistics'] = {
            'total_routes_validated': len(routes_data),
            'logic_errors_found': len(validation_results['issues']),
            'destination_charging_required': routes_data['requires_destination_charging'].sum(),
            'scenario_distribution': routes_data['charging_scenario'].value_counts().to_dict()
        }
        
        if validation_results['issues']:
            validation_results['is_valid'] = len(validation_results['issues']) <= 3  # Allow minor issues
        
        logger.info(f"Destination charging logic validation {'passed' if validation_results['is_valid'] else 'failed'}")
        
        return validation_results


def validate_range_assessment_accuracy(assessor: RangeFeasibilityAssessor, 
                                     test_distances: List[float]) -> Dict[str, Any]:
    """
    Validate range assessment accuracy using test cases.
    
    Args:
        assessor: RangeFeasibilityAssessor instance to test
        test_distances: List of test distances in kilometers
        
    Returns:
        Dict with validation results
    """
    logger.info("Validating range assessment accuracy with test cases")
    
    validation_results = {
        'is_valid': True,
        'test_results': [],
        'edge_case_results': []
    }
    
    # Test standard cases
    for distance in test_distances:
        assessment = assessor.assess_single_trip_feasibility(distance)
        
        # Validate assessment logic (using effective range, not max range)
        expected_feasible = distance <= assessor.effective_range_km
        expected_category = (
            RangeFeasibilityCategory.FEASIBLE if distance <= assessor.constraints.destination_charging_threshold
            else RangeFeasibilityCategory.CONSTRAINED if distance <= assessor.effective_range_km  # FIXED
            else RangeFeasibilityCategory.INFEASIBLE
        )
        
        test_result = {
            'distance_km': distance,
            'assessment_correct': (
                assessment.one_way_feasible == expected_feasible and
                assessment.feasibility_category == expected_category
            ),
            'expected_category': expected_category.value,
            'actual_category': assessment.feasibility_category.value
        }
        
        validation_results['test_results'].append(test_result)
        
        if not test_result['assessment_correct']:
            validation_results['is_valid'] = False
    
    # Test edge cases
    edge_cases = [
        assessor.constraints.destination_charging_threshold - 0.1,  # Just under threshold
        assessor.constraints.destination_charging_threshold + 0.1,  # Just over threshold
        assessor.effective_range_km - 0.1,                         # Just under effective range
        assessor.effective_range_km + 0.1,                         # Just over effective range
        assessor.constraints.max_range_km - 0.1,                   # Just under max range
        assessor.constraints.max_range_km + 0.1                    # Just over max range
    ]
    
    for distance in edge_cases:
        assessment = assessor.assess_single_trip_feasibility(distance)
        validation_results['edge_case_results'].append({
            'distance_km': distance,
            'category': assessment.feasibility_category.value,
            'one_way_feasible': assessment.one_way_feasible,
            'destination_charging_required': assessment.destination_charging_required
        })
    
    logger.info(f"Range assessment accuracy validation {'passed' if validation_results['is_valid'] else 'failed'}")
    
    return validation_results