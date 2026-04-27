"""
Advanced Charging Infrastructure Analysis Module

Implements sophisticated charging infrastructure assessment methodologies adapted from
Hanig et al. (2025) Nature Communications paper on US EV charging coverage.

Key analyses:
1. Consecutive Coverage Metric - Network-based charging accessibility
2. Trip-Volume-Weighted Infrastructure - Demand-weighted infrastructure metrics
3. Destination-Specific Charging Coverage - Purpose-based destination analysis
4. Infrastructure Gap Classification - Multi-tier bottleneck identification
5. Equity and Accessibility Analysis - Demographic-infrastructure alignment

Author: EV Modeling Project
Date: 2025-11-03
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Dict, List, Tuple, Optional
import warnings


class ConsecutiveCoverageAnalyzer:
    """
    Analyzes consecutive charging coverage using network-based accessibility.
    
    Adapted from Hanig et al. (2025) methodology for Scottish context with
    frugal EV range constraints (40km gap threshold).
    """
    
    def __init__(self, max_gap_km: float = 40.0, search_radius_km: float = 80.0):
        """
        Initialize consecutive coverage analyzer.
        
        Parameters
        ----------
        max_gap_km : float
            Maximum acceptable gap between charging stations (default: 40km,
            half of frugal EV effective range)
        search_radius_km : float
            Maximum search radius from origin (default: 80km, effective range)
        """
        self.max_gap_km = max_gap_km
        self.search_radius_km = search_radius_km
    
    def calculate_area_coverage(
        self,
        origin_areas: gpd.GeoDataFrame,
        charging_stations: gpd.GeoDataFrame,
        od_data: Optional[pd.DataFrame] = None,
        area_id_col: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Calculate consecutive charging coverage from each origin area.
        
        Parameters
        ----------
        origin_areas : gpd.GeoDataFrame
            Geographic areas (Output Areas) as origins
        charging_stations : gpd.GeoDataFrame
            Charging station locations with point geometry
        od_data : pd.DataFrame, optional
            Origin-destination data for trip-volume weighting
        area_id_col : str, optional
            Column name for area identifiers. If None, auto-detects from
            common names: 'area_id', 'geo_code', 'OutputArea', 'OA'
        
        Returns
        -------
        gpd.GeoDataFrame
            Coverage metrics for each origin area
        """
        # Auto-detect area ID column if not provided
        if area_id_col is None:
            for candidate in ['area_id', 'geo_code', 'OutputArea', 'OA', 'code']:
                if candidate in origin_areas.columns:
                    area_id_col = candidate
                    break
            
            if area_id_col is None:
                raise ValueError(
                    "Could not auto-detect area ID column. Please specify area_id_col parameter. "
                    f"Available columns: {origin_areas.columns.tolist()}"
                )
        
        results = []
        
        for idx, origin in origin_areas.iterrows():
            origin_id = origin.get(area_id_col, idx)
            origin_point = origin.geometry.centroid
            
            # Find charging stations within search radius
            stations_in_range = charging_stations[
                charging_stations.distance(origin_point) <= (self.search_radius_km * 1000)
            ].copy()
            
            if len(stations_in_range) == 0:
                coverage_metrics = {
                    'temp_area_id': origin_id,  # Temporary key, will rename to match detected column
                    'consecutive_coverage_score': 0.0,
                    'accessible_stations': 0,
                    'max_charging_gap_km': np.inf,
                    'coverage_category': 'isolated'
                }
            else:
                # Calculate consecutive coverage
                stations_in_range['distance_km'] = stations_in_range.distance(origin_point) / 1000
                stations_sorted = stations_in_range.sort_values('distance_km')
                
                # Find maximum gap between consecutive stations
                distances = stations_sorted['distance_km'].values
                if len(distances) > 1:
                    gaps = np.diff(distances)
                    max_gap = gaps.max()
                else:
                    max_gap = distances[0] if len(distances) > 0 else np.inf
                
                # Calculate coverage score based on gap analysis
                if max_gap <= self.max_gap_km:
                    coverage_score = 1.0
                elif max_gap <= self.max_gap_km * 1.5:
                    coverage_score = 0.7
                elif max_gap <= self.max_gap_km * 2:
                    coverage_score = 0.4
                else:
                    coverage_score = 0.2
                
                # Determine coverage category
                if max_gap <= self.max_gap_km:
                    category = 'excellent'
                elif max_gap <= self.max_gap_km * 1.5:
                    category = 'good'
                elif max_gap <= self.max_gap_km * 2:
                    category = 'moderate'
                else:
                    category = 'poor'
                
                coverage_metrics = {
                    'temp_area_id': origin_id,  # Temporary key, will rename to match detected column
                    'consecutive_coverage_score': coverage_score,
                    'accessible_stations': len(stations_in_range),
                    'max_charging_gap_km': max_gap,
                    'coverage_category': category,
                    'nearest_station_km': distances[0] if len(distances) > 0 else np.inf
                }
            
            results.append(coverage_metrics)
        
        coverage_df = pd.DataFrame(results)
        
        # Rename temp_area_id to match the detected area_id_col for merging
        coverage_df = coverage_df.rename(columns={'temp_area_id': area_id_col})
        
        # Merge with original geometry using detected area_id_col
        result_gdf = origin_areas.merge(coverage_df, on=area_id_col, how='left')
        
        return result_gdf
    
    def calculate_regional_spillover(
        self,
        areas_with_coverage: gpd.GeoDataFrame,
        charging_stations: gpd.GeoDataFrame,
        region_column: str = 'region'
    ) -> pd.DataFrame:
        """
        Calculate infrastructure spillover from neighboring regions.
        
        Parameters
        ----------
        areas_with_coverage : gpd.GeoDataFrame
            Areas with coverage metrics and region identifiers
        charging_stations : gpd.GeoDataFrame
            Charging stations with region identifiers
        region_column : str
            Column name containing region identifiers
        
        Returns
        -------
        pd.DataFrame
            Spillover metrics by region
        """
        spillover_results = []
        
        if region_column not in areas_with_coverage.columns:
            warnings.warn(f"Region column '{region_column}' not found. Skipping spillover analysis.")
            return pd.DataFrame()
        
        for region in areas_with_coverage[region_column].unique():
            region_areas = areas_with_coverage[areas_with_coverage[region_column] == region]
            
            # For each area, determine what fraction of accessible stations are out-of-region
            out_of_region_access = []
            
            for idx, area in region_areas.iterrows():
                area_point = area.geometry.centroid
                nearby_stations = charging_stations[
                    charging_stations.distance(area_point) <= (self.search_radius_km * 1000)
                ]
                
                if len(nearby_stations) > 0:
                    if region_column in nearby_stations.columns:
                        out_of_region_count = len(nearby_stations[nearby_stations[region_column] != region])
                        spillover_ratio = out_of_region_count / len(nearby_stations)
                    else:
                        spillover_ratio = 0.0
                else:
                    spillover_ratio = 0.0
                
                out_of_region_access.append(spillover_ratio)
            
            spillover_results.append({
                'region': region,
                'mean_spillover_ratio': np.mean(out_of_region_access),
                'median_spillover_ratio': np.median(out_of_region_access),
                'areas_with_spillover': sum(1 for x in out_of_region_access if x > 0),
                'total_areas': len(out_of_region_access)
            })
        
        return pd.DataFrame(spillover_results)


class TripVolumeWeightedAnalyzer:
    """
    Weights infrastructure metrics by actual trip demand rather than uniform spatial coverage.
    """
    
    def __init__(self):
        """Initialize trip-volume-weighted analyzer."""
        pass
    
    def calculate_weighted_infrastructure_scores(
        self,
        infrastructure_data: gpd.GeoDataFrame,
        od_data: pd.DataFrame,
        origin_id_col: Optional[str] = None,
        destination_id_col: Optional[str] = None,
        volume_col: Optional[str] = None,
        area_id_col: Optional[str] = None
    ) -> gpd.GeoDataFrame:
        """
        Calculate trip-volume-weighted infrastructure metrics.
        
        Parameters
        ----------
        infrastructure_data : gpd.GeoDataFrame
            Infrastructure metrics by area
        od_data : pd.DataFrame
            Origin-destination trip data
        origin_id_col : str
            Column name for origin area IDs in OD data
        destination_id_col : str
            Column name for destination area IDs in OD data
        volume_col : str
            Column name for trip volumes
        area_id_col : str, optional
            Column name for area IDs in infrastructure_data. If None, auto-detects
        
        Returns
        -------
        gpd.GeoDataFrame
            Infrastructure data with weighted metrics
        """
        # Auto-detect area ID column if not provided
        if area_id_col is None:
            for candidate in ['area_id', 'geo_code', 'OutputArea', 'OA', 'code']:
                if candidate in infrastructure_data.columns:
                    area_id_col = candidate
                    break
            
            if area_id_col is None:
                raise ValueError(
                    "Could not auto-detect area ID column. Please specify area_id_col parameter. "
                    f"Available columns: {infrastructure_data.columns.tolist()}"
                )
        
        # Auto-detect origin column in OD data
        if origin_id_col is None:
            for candidate in ['origin_area_id', 'origin', 'from_area_id', 'o', 'origin_code', 'geo_code_origin', 'geo_code1']:
                if candidate in od_data.columns:
                    origin_id_col = candidate
                    break
            
            if origin_id_col is None or origin_id_col not in od_data.columns:
                raise ValueError(
                    "Could not auto-detect origin column in OD data. "
                    f"Available columns: {od_data.columns.tolist()}. "
                    "Please specify origin_id_col parameter."
                )
        
        # Auto-detect destination column in OD data
        if destination_id_col is None:
            for candidate in ['destination_area_id', 'destination', 'to_area_id', 'd', 'destination_code', 'geo_code_destination', 'geo_code2']:
                if candidate in od_data.columns:
                    destination_id_col = candidate
                    break
            
            if destination_id_col is None or destination_id_col not in od_data.columns:
                raise ValueError(
                    "Could not auto-detect destination column in OD data. "
                    f"Available columns: {od_data.columns.tolist()}. "
                    "Please specify destination_id_col parameter."
                )
        
        # Auto-detect volume column in OD data
        if volume_col is None:
            for candidate in ['trip_volume', 'volume', 'trips', 'count', 'flow', 'n', 'all', 'origin_trips']:
                if candidate in od_data.columns:
                    volume_col = candidate
                    break
            
            if volume_col is None or volume_col not in od_data.columns:
                raise ValueError(
                    "Could not auto-detect volume column in OD data. "
                    f"Available columns: {od_data.columns.tolist()}. "
                    "Please specify volume_col parameter."
                )
        
        # Calculate total trips originating from each area
        origin_volumes = od_data.groupby(origin_id_col)[volume_col].sum().reset_index()
        origin_volumes.columns = [origin_id_col, 'total_origin_trips']
        
        # Calculate total trips destined for each area
        dest_volumes = od_data.groupby(destination_id_col)[volume_col].sum().reset_index()
        dest_volumes.columns = [destination_id_col, 'total_destination_trips']
        
        # Merge trip volumes with infrastructure data
        result = infrastructure_data.copy()
        result = result.merge(origin_volumes, left_on=area_id_col, right_on=origin_id_col, how='left')
        
        # Drop duplicate ID column if created
        if origin_id_col in result.columns and origin_id_col != area_id_col:
            result = result.drop(columns=[origin_id_col])
        
        result = result.merge(dest_volumes, left_on=area_id_col, right_on=destination_id_col, how='left')
        
        # Drop duplicate ID column if created
        if destination_id_col in result.columns and destination_id_col != area_id_col:
            result = result.drop(columns=[destination_id_col])
        
        # Fill NaN with 0 for areas with no trips
        result['total_origin_trips'] = result['total_origin_trips'].fillna(0)
        result['total_destination_trips'] = result['total_destination_trips'].fillna(0)
        result['total_trips'] = result['total_origin_trips'] + result['total_destination_trips']
        
        # Calculate trip volume percentile (relative demand)
        if result['total_trips'].sum() > 0:
            result['trip_demand_percentile'] = result['total_trips'].rank(pct=True)
        else:
            result['trip_demand_percentile'] = 0.0
        
        # Calculate demand-weighted infrastructure score
        # High demand + high infrastructure = optimal
        # High demand + low infrastructure = critical gap
        # Low demand + high infrastructure = overprovisioned
        if 'infrastructure_factor' in result.columns:
            result['demand_weighted_infrastructure'] = (
                result['infrastructure_factor'] * result['trip_demand_percentile']
            )
        
        # Identify critical gaps (high demand, poor infrastructure)
        if 'infrastructure_factor' in result.columns:
            result['demand_infrastructure_gap'] = (
                result['trip_demand_percentile'] - result['infrastructure_factor']
            )
            
            # Critical gaps: top 50% demand but bottom 50% infrastructure
            result['critical_demand_gap'] = (
                (result['trip_demand_percentile'] > 0.5) & 
                (result['infrastructure_factor'] < 0.5)
            )
        
        return result
    
    def calculate_priority_scores(
        self,
        weighted_infrastructure: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Calculate infrastructure investment priority scores.
        
        Parameters
        ----------
        weighted_infrastructure : gpd.GeoDataFrame
            Infrastructure data with weighted metrics
        
        Returns
        -------
        gpd.GeoDataFrame
            Data with priority scores
        """
        result = weighted_infrastructure.copy()
        
        # Priority score: higher for high demand + low infrastructure
        if 'trip_demand_percentile' in result.columns and 'infrastructure_factor' in result.columns:
            result['investment_priority_score'] = (
                result['trip_demand_percentile'] * (1 - result['infrastructure_factor'])
            )
            
            # Categorize priorities
            result['investment_priority'] = pd.cut(
                result['investment_priority_score'],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['low', 'moderate', 'high', 'critical'],
                include_lowest=True
            )
        
        return result


class DestinationChargingAnalyzer:
    """
    Analyzes destination-specific charging coverage by trip purpose.
    """
    
    def __init__(self, proximity_threshold_km: float = 0.5):
        """
        Initialize destination charging analyzer.
        
        Parameters
        ----------
        proximity_threshold_km : float
            Distance threshold for "nearby" charging (default: 0.5km)
        """
        self.proximity_threshold_m = proximity_threshold_km * 1000
    
    def analyze_purpose_specific_coverage(
        self,
        od_data: pd.DataFrame,
        destination_areas: gpd.GeoDataFrame,
        charging_stations: gpd.GeoDataFrame,
        purpose_col: Optional[str] = None,
        volume_col: Optional[str] = None,
        destination_id_col: Optional[str] = None,
        area_id_col: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Analyze charging coverage at destinations by trip purpose.
        
        Parameters
        ----------
        od_data : pd.DataFrame
            Origin-destination trip data with purpose information
        destination_areas : gpd.GeoDataFrame
            Geographic areas as destinations
        charging_stations : gpd.GeoDataFrame
            Charging station locations
        purpose_col : str
            Column name for trip purpose
        volume_col : str
            Column name for trip volumes
        destination_id_col : str
            Column name for destination area IDs in OD data
        area_id_col : str, optional
            Column name for area IDs in destination_areas. If None, auto-detects
        
        Returns
        -------
        Dict[str, Dict]
            Coverage metrics by trip purpose
        """
        # Auto-detect area ID column if not provided
        if area_id_col is None:
            for candidate in ['area_id', 'geo_code', 'OutputArea', 'OA', 'code']:
                if candidate in destination_areas.columns:
                    area_id_col = candidate
                    break
            
            if area_id_col is None:
                raise ValueError(
                    "Could not auto-detect area ID column. Please specify area_id_col parameter. "
                    f"Available columns: {destination_areas.columns.tolist()}"
                )
        
        # Auto-detect purpose column
        if purpose_col is None:
            for candidate in ['purpose', 'trip_purpose', 'Purpose', 'PURPOSE', 'trip_type']:
                if candidate in od_data.columns:
                    purpose_col = candidate
                    break
            
            if purpose_col is None or purpose_col not in od_data.columns:
                raise ValueError(
                    "Could not auto-detect purpose column in OD data. "
                    f"Available columns: {od_data.columns.tolist()}. "
                    "Please specify purpose_col parameter."
                )
        
        # Auto-detect destination column
        if destination_id_col is None:
            for candidate in ['destination_area_id', 'destination', 'to_area_id', 'd', 'destination_code', 'geo_code_destination', 'geo_code2']:
                if candidate in od_data.columns:
                    destination_id_col = candidate
                    break
            
            if destination_id_col is None or destination_id_col not in od_data.columns:
                raise ValueError(
                    "Could not auto-detect destination column in OD data. "
                    f"Available columns: {od_data.columns.tolist()}. "
                    "Please specify destination_id_col parameter."
                )
        
        # Auto-detect volume column
        if volume_col is None:
            for candidate in ['trip_volume', 'volume', 'trips', 'count', 'flow', 'n', 'all', 'origin_trips']:
                if candidate in od_data.columns:
                    volume_col = candidate
                    break
            
            if volume_col is None or volume_col not in od_data.columns:
                raise ValueError(
                    "Could not auto-detect volume column in OD data. "
                    f"Available columns: {od_data.columns.tolist()}. "
                    "Please specify volume_col parameter."
                )
        
        results = {}
        
        # Get unique purposes
        purposes = od_data[purpose_col].unique()
        
        for purpose in purposes:
            # Filter trips for this purpose
            purpose_trips = od_data[od_data[purpose_col] == purpose].copy()
            
            # Get unique destinations for this purpose
            dest_ids = purpose_trips[destination_id_col].unique()
            purpose_destinations = destination_areas[
                destination_areas[area_id_col].isin(dest_ids)
            ].copy()
            
            if len(purpose_destinations) == 0:
                continue
            
            # Check charging availability at each destination
            destinations_with_charging = 0
            total_chargers = 0
            total_trips = purpose_trips[volume_col].sum()
            trips_with_charging = 0
            
            for idx, dest in purpose_destinations.iterrows():
                dest_point = dest.geometry.centroid
                
                # Find nearby charging stations
                nearby_stations = charging_stations[
                    charging_stations.distance(dest_point) <= self.proximity_threshold_m
                ]
                
                has_charging = len(nearby_stations) > 0
                
                if has_charging:
                    destinations_with_charging += 1
                    total_chargers += len(nearby_stations)
                    
                    # Count trips to this destination
                    dest_trips = purpose_trips[
                        purpose_trips[destination_id_col] == dest[area_id_col]
                    ][volume_col].sum()
                    trips_with_charging += dest_trips
            
            # Calculate coverage metrics
            coverage_ratio = destinations_with_charging / len(purpose_destinations) if len(purpose_destinations) > 0 else 0
            avg_chargers = total_chargers / len(purpose_destinations) if len(purpose_destinations) > 0 else 0
            trip_coverage_ratio = trips_with_charging / total_trips if total_trips > 0 else 0
            
            results[purpose] = {
                'total_destinations': len(purpose_destinations),
                'destinations_with_charging': destinations_with_charging,
                'coverage_ratio': coverage_ratio,
                'average_chargers_per_destination': avg_chargers,
                'total_trips': total_trips,
                'trips_with_destination_charging': trips_with_charging,
                'trip_weighted_coverage': trip_coverage_ratio
            }
        
        return results
    
    def identify_critical_destinations(
        self,
        purpose_coverage: Dict[str, Dict],
        min_coverage_threshold: float = 0.5,
        min_trip_volume: float = 1000
    ) -> pd.DataFrame:
        """
        Identify trip purposes with inadequate destination charging.
        
        Parameters
        ----------
        purpose_coverage : Dict[str, Dict]
            Coverage metrics by purpose
        min_coverage_threshold : float
            Minimum acceptable coverage ratio
        min_trip_volume : float
            Minimum trip volume to be considered significant
        
        Returns
        -------
        pd.DataFrame
            Critical purposes needing infrastructure improvements
        """
        critical_purposes = []
        
        for purpose, metrics in purpose_coverage.items():
            if (metrics['total_trips'] >= min_trip_volume and 
                metrics['coverage_ratio'] < min_coverage_threshold):
                
                critical_purposes.append({
                    'purpose': purpose,
                    'coverage_ratio': metrics['coverage_ratio'],
                    'trip_weighted_coverage': metrics['trip_weighted_coverage'],
                    'total_trips': metrics['total_trips'],
                    'coverage_gap': min_coverage_threshold - metrics['coverage_ratio'],
                    'priority': 'high' if metrics['coverage_ratio'] < 0.25 else 'moderate'
                })
        
        # Return empty DataFrame with correct columns if no critical purposes found
        if not critical_purposes:
            return pd.DataFrame(columns=[
                'purpose', 'coverage_ratio', 'trip_weighted_coverage', 
                'total_trips', 'coverage_gap', 'priority'
            ])
        
        return pd.DataFrame(critical_purposes).sort_values('coverage_gap', ascending=False)


class InfrastructureGapClassifier:
    """
    Multi-tier infrastructure gap classification beyond binary bottleneck identification.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize gap classifier.
        
        Parameters
        ----------
        config : Dict, optional
            Configuration with classification thresholds
        """
        if config is None:
            # Default thresholds
            self.thresholds = {
                'excellent': 0.85,
                'good': 0.70,
                'adequate': 0.50,
                'poor': 0.30,
                'critical': 0.0
            }
            self.reliance_threshold = 0.7
        else:
            self.thresholds = config.get('gap_classification_thresholds', {
                'excellent': 0.85,
                'good': 0.70,
                'adequate': 0.50,
                'poor': 0.30,
                'critical': 0.0
            })
            self.reliance_threshold = config.get('charging_infrastructure', {}).get(
                'bottleneck_thresholds', {}
            ).get('high_reliance', 0.7)
    
    def classify_infrastructure_gaps(
        self,
        infrastructure_data: gpd.GeoDataFrame,
        infrastructure_col: str = 'infrastructure_factor',
        reliance_col: str = 'public_charging_reliance'
    ) -> gpd.GeoDataFrame:
        """
        Classify infrastructure gaps into multi-tier categories.
        
        Parameters
        ----------
        infrastructure_data : gpd.GeoDataFrame
            Infrastructure metrics by area
        infrastructure_col : str
            Column with infrastructure factor scores
        reliance_col : str
            Column with public charging reliance scores
        
        Returns
        -------
        gpd.GeoDataFrame
            Data with gap classifications
        """
        result = infrastructure_data.copy()
        
        # Classify infrastructure quality
        conditions = [
            (result[infrastructure_col] >= self.thresholds['excellent']),
            (result[infrastructure_col] >= self.thresholds['good']),
            (result[infrastructure_col] >= self.thresholds['adequate']),
            (result[infrastructure_col] >= self.thresholds['poor']),
        ]
        
        categories = ['excellent', 'good', 'adequate', 'poor']
        result['infrastructure_category'] = np.select(conditions, categories, default='critical')
        
        # Identify critical bottlenecks (poor infrastructure + high reliance)
        if reliance_col in result.columns:
            result['critical_bottleneck'] = (
                (result['infrastructure_category'].isin(['poor', 'critical'])) &
                (result[reliance_col] > self.reliance_threshold)
            )
            
            # Calculate bottleneck severity
            result['bottleneck_severity'] = np.where(
                result['critical_bottleneck'],
                result[reliance_col] * (1 - result[infrastructure_col]),
                0.0
            )
        
        # Gap magnitude (how far below adequate)
        result['infrastructure_gap_magnitude'] = np.maximum(
            0,
            self.thresholds['adequate'] - result[infrastructure_col]
        )
        
        return result
    
    def prioritize_improvements(
        self,
        classified_data: gpd.GeoDataFrame,
        demand_data: Optional[gpd.GeoDataFrame] = None,
        area_id_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prioritize areas for infrastructure improvements.
        
        Parameters
        ----------
        classified_data : gpd.GeoDataFrame
            Data with gap classifications
        demand_data : gpd.GeoDataFrame, optional
            Data with trip demand metrics
        area_id_col : str, optional
            Column name for area IDs. If None, auto-detects
        
        Returns
        -------
        pd.DataFrame
            Prioritized improvement list
        """
        # Auto-detect area ID column if not provided
        if area_id_col is None:
            for candidate in ['area_id', 'geo_code', 'OutputArea', 'OA', 'code']:
                if candidate in classified_data.columns:
                    area_id_col = candidate
                    break
            
            if area_id_col is None:
                # If still not found, use index
                area_id_col = classified_data.index.name or 'index'
        
        # Start with critical bottlenecks
        if 'critical_bottleneck' in classified_data.columns:
            priority_areas = classified_data[classified_data['critical_bottleneck']].copy()
        else:
            priority_areas = classified_data[
                classified_data['infrastructure_category'].isin(['poor', 'critical'])
            ].copy()
        
        if len(priority_areas) == 0:
            return pd.DataFrame()
        
        # If demand data available, incorporate it
        if demand_data is not None and 'total_trips' in demand_data.columns:
            # Detect area_id_col in demand_data if needed
            demand_id_col = area_id_col
            if area_id_col not in demand_data.columns:
                for candidate in ['area_id', 'geo_code', 'OutputArea', 'OA', 'code']:
                    if candidate in demand_data.columns:
                        demand_id_col = candidate
                        break
            
            merge_cols = [demand_id_col, 'total_trips', 'trip_demand_percentile']
            merge_cols = [col for col in merge_cols if col in demand_data.columns]
            
            priority_areas = priority_areas.merge(
                demand_data[merge_cols],
                left_on=area_id_col,
                right_on=demand_id_col,
                how='left'
            )
            
            # Drop duplicate ID column if created
            if demand_id_col in priority_areas.columns and demand_id_col != area_id_col:
                priority_areas = priority_areas.drop(columns=[demand_id_col])
            
            # Calculate composite priority score
            priority_areas['composite_priority'] = (
                priority_areas.get('bottleneck_severity', 0) * 0.5 +
                priority_areas.get('trip_demand_percentile', 0) * 0.5
            )
        else:
            priority_areas['composite_priority'] = priority_areas.get('bottleneck_severity', 0)
        
        # Sort by priority
        priority_list = priority_areas.sort_values('composite_priority', ascending=False)
        
        # Build return columns list
        return_cols = [area_id_col, 'infrastructure_category', 'critical_bottleneck',
                      'bottleneck_severity', 'composite_priority']
        return_cols = [col for col in return_cols if col in priority_list.columns]
        
        return priority_list[return_cols].head(50)


class EquityAccessibilityAnalyzer:
    """
    Analyzes alignment between demographic adoption propensity and infrastructure availability.
    """
    
    def __init__(self):
        """Initialize equity and accessibility analyzer."""
        pass
    
    def analyze_infrastructure_equity(
        self,
        combined_data: gpd.GeoDataFrame,
        adoption_col: Optional[str] = None,
        infrastructure_col: str = 'infrastructure_factor',
        reliance_col: str = 'public_charging_reliance',
        area_id_col: Optional[str] = None
    ) -> Dict:
        """
        Analyze equity between adoption propensity and infrastructure availability.
        
        Parameters
        ----------
        combined_data : gpd.GeoDataFrame
            Combined demographics and infrastructure data
        adoption_col : str, optional
            Column with adoption propensity scores. If None, auto-detects
        infrastructure_col : str
            Column with infrastructure factor scores
        reliance_col : str
            Column with public charging reliance scores
        area_id_col : str, optional
            Column name for area IDs. If None, auto-detects
        
        Returns
        -------
        Dict
            Equity analysis results
        """
        # Auto-detect area ID column if not provided
        if area_id_col is None:
            for candidate in ['area_id', 'geo_code', 'OutputArea', 'OA', 'code']:
                if candidate in combined_data.columns:
                    area_id_col = candidate
                    break
            
            if area_id_col is None:
                # If still not found, use index
                area_id_col = combined_data.index.name or 'index'
        
        # Auto-detect adoption propensity column if not provided
        if adoption_col is None:
            for candidate in ['final_adoption_propensity', 'adoption_propensity', 'propensity_score', 
                             'adoption_score', 'ev_adoption_propensity', 'adoption', 'propensity']:
                if candidate in combined_data.columns:
                    adoption_col = candidate
                    break
            
            if adoption_col is None or adoption_col not in combined_data.columns:
                # Create a synthetic adoption propensity based on available demographic proxies
                print("⚠️  No adoption propensity column found. Creating synthetic score from demographics...")
                adoption_col = self._create_synthetic_adoption_score(combined_data)
        
        data = combined_data.copy()
        
        # Calculate equity gap (difference between need and availability)
        data['equity_gap'] = data[adoption_col] - data[infrastructure_col]
        
        # Identify equity categories
        data['equity_category'] = 'balanced'
        
        # High adoption, poor infrastructure = underserved
        data.loc[
            (data[adoption_col] > 0.7) & (data[infrastructure_col] < 0.5),
            'equity_category'
        ] = 'underserved'
        
        # Low adoption, excellent infrastructure = overprovisioned
        data.loc[
            (data[adoption_col] < 0.3) & (data[infrastructure_col] > 0.8),
            'equity_category'
        ] = 'overprovisioned'
        
        # High adoption, excellent infrastructure = well-served
        data.loc[
            (data[adoption_col] > 0.7) & (data[infrastructure_col] > 0.7),
            'equity_category'
        ] = 'well_served'
        
        # Calculate summary statistics
        equity_summary = {
            'total_areas': len(data),
            'underserved_areas': len(data[data['equity_category'] == 'underserved']),
            'well_served_areas': len(data[data['equity_category'] == 'well_served']),
            'overprovisioned_areas': len(data[data['equity_category'] == 'overprovisioned']),
            'mean_equity_gap': data['equity_gap'].mean(),
            'median_equity_gap': data['equity_gap'].median(),
            'equity_gap_std': data['equity_gap'].std()
        }
        
        # Identify priority areas for intervention
        priority_underserved = data[
            data['equity_category'] == 'underserved'
        ].sort_values('equity_gap', ascending=False)
        
        # Build columns list for priority areas
        priority_cols = [area_id_col, adoption_col, infrastructure_col, 'equity_gap']
        priority_cols = [col for col in priority_cols if col in priority_underserved.columns]
        
        equity_summary['top_priority_areas'] = priority_underserved.head(20)[priority_cols].to_dict('records')
        
        # Calculate correlation between adoption and infrastructure
        if len(data) > 2:
            correlation = data[[adoption_col, infrastructure_col]].corr().iloc[0, 1]
            equity_summary['adoption_infrastructure_correlation'] = correlation
        
        return equity_summary
    
    def analyze_demographic_infrastructure_patterns(
        self,
        combined_data: gpd.GeoDataFrame,
        demographic_cols: List[str],
        infrastructure_col: str = 'infrastructure_factor'
    ) -> pd.DataFrame:
        """
        Analyze how infrastructure availability varies with demographic characteristics.
        
        Parameters
        ----------
        combined_data : gpd.GeoDataFrame
            Combined demographics and infrastructure data
        demographic_cols : List[str]
            List of demographic columns to analyze
        infrastructure_col : str
            Column with infrastructure factor scores
        
        Returns
        -------
        pd.DataFrame
            Correlation analysis results
        """
        correlations = []
        
        for dem_col in demographic_cols:
            if dem_col in combined_data.columns:
                valid_data = combined_data[[dem_col, infrastructure_col]].dropna()
                
                if len(valid_data) > 2:
                    corr = valid_data.corr().iloc[0, 1]
                    
                    correlations.append({
                        'demographic_variable': dem_col,
                        'correlation_with_infrastructure': corr,
                        'interpretation': self._interpret_correlation(corr)
                    })
        
        return pd.DataFrame(correlations).sort_values(
            'correlation_with_infrastructure',
            key=abs,
            ascending=False
        )
    
    @staticmethod
    def _create_synthetic_adoption_score(data: gpd.GeoDataFrame) -> str:
        """
        Create a synthetic adoption propensity score from available demographic proxies.
        
        Uses car ownership, education, and socioeconomic indicators when available.
        
        Parameters
        ----------
        data : gpd.GeoDataFrame
            Data with demographic columns
        
        Returns
        -------
        str
            Name of the created synthetic adoption column
        """
        # Look for demographic proxy columns
        score_components = []
        weights = []
        
        # 1. Car ownership (2+ cars suggests higher adoption potential)
        car_cols = [col for col in data.columns if 'Two.or.more.cars' in col and 'All.people' in col]
        if car_cols:
            # Normalize to 0-1 scale
            score_components.append(
                data[car_cols[0]] / data[car_cols[0]].max()
            )
            weights.append(0.3)
        
        # 2. Education level (Level 4+ suggests higher adoption)
        edu_cols = [col for col in data.columns if 'Level.4.and.above' in col and 'All.people' in col]
        if edu_cols:
            score_components.append(
                data[edu_cols[0]] / data[edu_cols[0]].max()
            )
            weights.append(0.3)
        
        # 3. Social grade AB (higher socioeconomic status)
        ab_cols = [col for col in data.columns if 'AB.Higher.and.intermediate' in col and 'All.people' in col]
        if ab_cols:
            score_components.append(
                data[ab_cols[0]] / data[ab_cols[0]].max()
            )
            weights.append(0.2)
        
        # 4. Home ownership (suggests stability and charging infrastructure)
        owned_cols = [col for col in data.columns if 'Owned..Total' in col and 'All.people' in col]
        if owned_cols:
            score_components.append(
                data[owned_cols[0]] / data[owned_cols[0]].max()
            )
            weights.append(0.2)
        
        # Calculate weighted synthetic score
        if score_components:
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            synthetic_score = sum(
                comp * weight 
                for comp, weight in zip(score_components, normalized_weights)
            )
            
            data['synthetic_adoption_propensity'] = synthetic_score
            print(f"✓ Created synthetic adoption score from {len(score_components)} demographic proxies")
            return 'synthetic_adoption_propensity'
        else:
            # Last resort: use infrastructure factor as a proxy
            print("⚠️  No suitable demographic proxies found. Using uniform adoption propensity.")
            data['synthetic_adoption_propensity'] = 0.5  # Neutral score
            return 'synthetic_adoption_propensity'
    
    @staticmethod
    def _interpret_correlation(corr: float) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(corr)
        direction = "positive" if corr > 0 else "negative"
        
        if abs_corr > 0.7:
            strength = "strong"
        elif abs_corr > 0.4:
            strength = "moderate"
        elif abs_corr > 0.2:
            strength = "weak"
        else:
            strength = "negligible"
        
        return f"{strength} {direction}"

