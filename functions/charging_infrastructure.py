"""
Charging Infrastructure Analysis Module for Frugal EV Analysis

This module provides classes and functions for analyzing charging infrastructure capacity,
accessibility, and demand patterns to support EV adoption feasibility assessment.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
@dataclass
class AccessibilityMetrics:
    """Container for accessibility analysis results"""
    nearest_charger_distance: float
    chargers_within_1km: int
    chargers_within_2km: int
    chargers_within_5km: int
    accessibility_score: float
    public_charging_reliance: float
    chargers_per_capita: float = 0.0  # Chargers per 1000 people
    population_density: float = 0.0   # People per km²


@dataclass
class CapacityAnalysis:
    """Container for charging capacity analysis results"""
    total_charging_points: int
    average_points_per_location: float
    capacity_utilization_estimate: float
    demand_supply_ratio: float
    bottleneck_locations: List[str]


class ChargingInfrastructureAnalyzer:
    """
    Main analysis engine for charging infrastructure accessibility and capacity assessment.
    
    Implements distance calculations from residential areas to charging locations,
    accessibility scoring based on walking distance thresholds, and capacity analysis
    using charging point data.
    """
    
    def __init__(self, 
                 accessibility_thresholds: Optional[Dict[str, float]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize ChargingInfrastructureAnalyzer with accessibility parameters.
        
        Args:
            accessibility_thresholds: Distance thresholds for accessibility scoring
                Default: {'excellent': 0.5, 'good': 1.0, 'fair': 2.0, 'poor': 5.0}
            config: Full configuration dictionary with charging_infrastructure parameters
        """
        # Load from config (all values must be defined in scoring_weights.json)
        if config and 'charging_infrastructure' in config:
            infra_config = config['charging_infrastructure']
            self.accessibility_thresholds = accessibility_thresholds or infra_config['accessibility_thresholds_km']
            self.baseline_reliance_params = infra_config['baseline_reliance_params']
            self.bottleneck_thresholds = infra_config['bottleneck_thresholds']
            self.gap_severity_params = infra_config['gap_severity']
        else:
            self.accessibility_thresholds = accessibility_thresholds or {
                'excellent': 0.5,  # 500m
                'good': 1.0,       # 1km
                'fair': 2.0,       # 2km
                'poor': 5.0        # 5km
            }
            self.baseline_reliance_params = {}
            self.bottleneck_thresholds = {}
            self.gap_severity_params = {}
        
    def load_charging_infrastructure(self, charging_data_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Load charging infrastructure data from GeoJSON file.
        
        Args:
            charging_data_path: Path to charging infrastructure GeoJSON file
            
        Returns:
            GeoDataFrame with charging location data
            
        Raises:
            FileNotFoundError: If charging data file doesn't exist
            ValueError: If required columns are missing
        """
        charging_path = Path(charging_data_path)
        
        if not charging_path.exists():
            raise FileNotFoundError(f"Charging data file not found: {charging_path}")
            
        logger.info(f"Loading charging infrastructure data from {charging_path}")
        
        try:
            # Load GeoJSON file
            charging_gdf = gpd.read_file(charging_path)
            
            # Validate required columns
            required_columns = ['poi_id', 'number_of_points', 'geometry']
            missing_columns = [col for col in required_columns if col not in charging_gdf.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Ensure poi_id is string type
            charging_gdf['poi_id'] = charging_gdf['poi_id'].astype(str)
            
            # Ensure number_of_points is numeric
            charging_gdf['number_of_points'] = pd.to_numeric(
                charging_gdf['number_of_points'], errors='coerce'
            ).fillna(1)  # Default to 1 point if missing
            
            # Ensure geometries are valid
            invalid_geom_count = (~charging_gdf.geometry.is_valid).sum()
            if invalid_geom_count > 0:
                logger.warning(f"Found {invalid_geom_count} invalid geometries in charging data")
                charging_gdf.geometry = charging_gdf.geometry.buffer(0)
                
            # Convert to consistent CRS (British National Grid for accurate distance calculations)
            if charging_gdf.crs != 'EPSG:27700':
                charging_gdf = charging_gdf.to_crs('EPSG:27700')
                
            logger.info(f"Loaded {len(charging_gdf)} charging locations with {charging_gdf['number_of_points'].sum()} total points")
            
            return charging_gdf
            
        except Exception as e:
            logger.error(f"Error loading charging infrastructure data: {e}")
            raise
            
    def calculate_accessibility_scores(self, 
                                     residential_areas: gpd.GeoDataFrame,
                                     charging_locations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate accessibility scores from residential areas to charging locations.
        Memory-efficient implementation using chunked processing.
        
        Args:
            residential_areas: GeoDataFrame with residential area geometries (Output Areas)
            charging_locations: GeoDataFrame with charging point locations
            
        Returns:
            GeoDataFrame with accessibility metrics added to residential areas
        """
        logger.info("Calculating charging infrastructure accessibility scores")
        
        # Ensure both datasets are in the same CRS (British National Grid for accurate distances)
        if residential_areas.crs != 'EPSG:27700':
            residential_areas = residential_areas.to_crs('EPSG:27700')
        if charging_locations.crs != 'EPSG:27700':
            charging_locations = charging_locations.to_crs('EPSG:27700')
            
        # Get centroids of residential areas for distance calculations
        residential_centroids = residential_areas.geometry.centroid
        charging_points = charging_locations.geometry
        
        # Extract coordinates for distance calculations
        charging_coords = np.array([[point.x, point.y] for point in charging_points])
        
        # Process in chunks to avoid memory issues
        chunk_size = min(1000, len(residential_areas))  # Process max 1000 areas at a time
        total_areas = len(residential_areas)
        logger.info(f"Processing {total_areas} residential areas in chunks of {chunk_size}")
        
        # Initialize results
        results = []
        
        # Process residential areas in chunks
        for chunk_start in range(0, total_areas, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_areas)
            chunk_areas = residential_areas.iloc[chunk_start:chunk_end]
            
            logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_areas-1)//chunk_size + 1}: areas {chunk_start}-{chunk_end-1}")
            
            # Get centroids for this chunk
            chunk_centroids = chunk_areas.geometry.centroid
            res_coords = np.array([[point.x, point.y] for point in chunk_centroids])
            
            # Calculate distance matrix for this chunk only
            distance_matrix = np.sqrt(
                ((res_coords[:, np.newaxis, :] - charging_coords[np.newaxis, :, :]) ** 2).sum(axis=2)
            )
            
            # Process each area in the chunk
            for i, (idx, row) in enumerate(chunk_areas.iterrows()):
                distances = distance_matrix[i]
                
                # Calculate accessibility metrics
                nearest_distance = np.min(distances)
                chargers_1km = np.sum(distances <= 1000)
                chargers_2km = np.sum(distances <= 2000)
                chargers_5km = np.sum(distances <= 5000)
                
                # Calculate accessibility score (0-1 scale, higher is better)
                accessibility_score = self._calculate_accessibility_score(nearest_distance)
                
                # Estimate public charging reliance based on housing type and nearest charger distance
                # Pass full dataset for adaptive gamma calculation
                public_charging_reliance = self._estimate_public_charging_reliance(
                    row, nearest_distance, residential_data=residential_areas
                )
                
                # Calculate population density (people per km²)
                area_km2 = row.geometry.area / 1_000_000  # Convert m² to km²
                population = row.get('population', 0) if 'population' in row else 0
                population_density = population / area_km2 if area_km2 > 0 else 0
                
                # Calculate chargers per capita (per 1000 people)
                chargers_per_capita = (chargers_5km / population * 1000) if population > 0 else 0
                
                metrics = AccessibilityMetrics(
                    nearest_charger_distance=nearest_distance,
                    chargers_within_1km=chargers_1km,
                    chargers_within_2km=chargers_2km,
                    chargers_within_5km=chargers_5km,
                    accessibility_score=accessibility_score,
                    public_charging_reliance=public_charging_reliance,
                    chargers_per_capita=chargers_per_capita,
                    population_density=population_density
                )
                
                results.append(metrics)
            
            # Force garbage collection after each chunk to free memory
            import gc
            gc.collect()
            
        # Add results to residential areas dataframe
        result_gdf = residential_areas.copy()
        result_gdf['nearest_charger_distance'] = [r.nearest_charger_distance for r in results]
        result_gdf['chargers_within_1km'] = [r.chargers_within_1km for r in results]
        result_gdf['chargers_within_2km'] = [r.chargers_within_2km for r in results]
        result_gdf['chargers_within_5km'] = [r.chargers_within_5km for r in results]
        result_gdf['accessibility_score'] = [r.accessibility_score for r in results]
        result_gdf['public_charging_reliance'] = [r.public_charging_reliance for r in results]
        result_gdf['chargers_per_capita'] = [r.chargers_per_capita for r in results]
        result_gdf['population_density'] = [r.population_density for r in results]
        
        logger.info(f"Calculated accessibility scores for {len(result_gdf)} residential areas")
        logger.info(f"  Mean chargers per 1000 people: {result_gdf['chargers_per_capita'].mean():.2f}")
        logger.info(f"  Mean population density: {result_gdf['population_density'].mean():.1f} people/km²")
        logger.info(f"  Chargers per capita range: [{result_gdf['chargers_per_capita'].min():.2f}, {result_gdf['chargers_per_capita'].max():.2f}]")
        
        return result_gdf
        
    def _calculate_accessibility_score(self, nearest_distance: float) -> float:
        """
        Calculate accessibility score based on distance to nearest charger.
        Updated thresholds: Excellent ≤100m, Good ≤800m, Fair ≤2km, Poor ≤5km
        
        Args:
            nearest_distance: Distance to nearest charging point in meters
            
        Returns:
            Accessibility score between 0 and 1 (higher is better)
        """
        # Convert distance to km for threshold comparison
        distance_km = nearest_distance / 1000.0
        
        # Score based on accessibility thresholds (updated for realism)
        # Excellent: ≤100m (very walkable, ~1.25 min walk)
        # Good: ≤800m (walkable, ~10 min walk)
        # Fair: ≤2km (requires transport, ~25 min walk)
        # Poor: ≤5km (inaccessible, ~60 min walk)
        if distance_km <= self.accessibility_thresholds['excellent']:
            return 1.0
        elif distance_km <= self.accessibility_thresholds['good']:
            return 0.8
        elif distance_km <= self.accessibility_thresholds['fair']:
            return 0.6
        elif distance_km <= self.accessibility_thresholds['poor']:
            return 0.4
        else:
            return 0.2
            
    def _estimate_public_charging_reliance(self, 
                                         residential_row: pd.Series, 
                                         nearest_distance: float,
                                         residential_data: Optional[pd.DataFrame] = None) -> float:
        """
        Estimate reliance on public charging using paper's baseline formula (@eq-baseline-reliance).
        
        Implements: B_i = α + (β - α) × [(w_f × F_i) / (w_h × H_i + w_f × F_i + γ)]^δ
        Then: R_i = min(1, B_i + 0.3 × min(1, d_i/5))
        
        Args:
            residential_row: Row from residential areas dataframe with housing data
            nearest_distance: Distance to nearest charging point in meters
            residential_data: Full dataset for calculating adaptive gamma (optional)
            
        Returns:
            Public charging reliance score between 0 and 1 (higher means more reliant)
        """
        # Get parameters from config or use defaults
        params = self.baseline_reliance_params if hasattr(self, 'baseline_reliance_params') and self.baseline_reliance_params else {
            'alpha': 0.1,
            'beta': 0.9,
            'w_flat': 0.8,
            'w_house': 0.2,
            'gamma_coefficient': 0.1,
            'delta': 1.0,
            'distance_weight': 0.3,
            'distance_normalization_km': 5
        }
        
        alpha = params['alpha']
        beta = params['beta']
        w_f = params['w_flat']
        w_h = params['w_house']
        c = params['gamma_coefficient']
        delta = params['delta']
        
        # Extract household counts
        house_columns = [col for col in residential_row.index if 'house' in col.lower() or 'bungalow' in col.lower()]
        flat_columns = [col for col in residential_row.index if 'flat' in col.lower() or 'apartment' in col.lower()]
        
        house_households = 0
        flat_households = 0
        
        if house_columns and flat_columns:
            try:
                for col in house_columns:
                    val = pd.to_numeric(residential_row.get(col, 0), errors='coerce')
                    house_households += 0 if pd.isna(val) else val
                    
                for col in flat_columns:
                    val = pd.to_numeric(residential_row.get(col, 0), errors='coerce')
                    flat_households += 0 if pd.isna(val) else val
            except (TypeError, ValueError, AttributeError):
                # Fallback when housing data parsing fails
                # These defaults represent a typical mixed residential area
                house_households = 50  # Typical house household count
                flat_households = 20   # Typical flat household count
                logger.warning("Housing data parsing failed, using fallback defaults: house=50, flat=20")
        
        # Calculate adaptive gamma parameter
        if residential_data is not None and len(residential_data) > 0:
            gamma = self._calculate_adaptive_gamma(residential_data, w_h, w_f, c)
        else:
            # Fallback when full dataset not available
            # γ = c × typical_median, assuming median(w_h×H + w_f×F) ≈ 70 households
            gamma = c * 70  # Fallback: c × typical median household count
            logger.warning(f"Full dataset not available for gamma calculation, using fallback: γ = {c} × 70 = {gamma}")
        
        # Apply paper's baseline reliance equation (@eq-baseline-reliance)
        denominator = w_h * house_households + w_f * flat_households + gamma
        if denominator > 0:
            fraction = (w_f * flat_households) / denominator
            baseline_reliance = alpha + (beta - alpha) * (fraction ** delta)
        else:
            baseline_reliance = (alpha + beta) / 2  # Midpoint fallback
        
        # Distance adjustment (@eq-charging-reliance)
        distance_km = nearest_distance / 1000.0
        distance_factor = min(1.0, distance_km / params['distance_normalization_km'])
        
        # Final reliance score
        reliance_score = baseline_reliance + (distance_factor * params['distance_weight'])
        
        return min(1.0, reliance_score)

    def _calculate_adaptive_gamma(self, 
                                residential_data: pd.DataFrame, 
                                w_h: float, 
                                w_f: float, 
                                c: float) -> float:
        """
        Calculate adaptive smoothing parameter scaled to dataset characteristics.
        
        Args:
            residential_data: Full residential dataset
            w_h: House weight
            w_f: Flat weight  
            c: Scaling coefficient
            
        Returns:
            Adaptive gamma parameter
        """
        try:
            # Calculate weighted household totals for all areas
            household_totals = []
            
            for _, row in residential_data.iterrows():
                house_columns = [col for col in row.index 
                            if 'house' in col.lower() or 'bungalow' in col.lower()]
                flat_columns = [col for col in row.index 
                            if 'flat' in col.lower() or 'apartment' in col.lower()]
                
                house_total = 0
                flat_total = 0
                
                for col in house_columns:
                    val = pd.to_numeric(row.get(col, 0), errors='coerce')
                    house_total += 0 if pd.isna(val) else val
                    
                for col in flat_columns:
                    val = pd.to_numeric(row.get(col, 0), errors='coerce')
                    flat_total += 0 if pd.isna(val) else val
                
                weighted_total = w_h * house_total + w_f * flat_total
                household_totals.append(weighted_total)
            
            # Calculate adaptive gamma as fraction of median
            if household_totals:
                median_households = pd.Series(household_totals).median()
                gamma = c * median_households
                return max(1.0, gamma)  # Ensure minimum smoothing
            else:
                return 10.0  # Fallback if calculation fails
                
        except Exception:
            return 10.0  # Conservative fallback

    def run_sensitivity_analysis(self, 
                            residential_data: pd.DataFrame,
                            charging_points: gpd.GeoDataFrame,
                            param_ranges: dict = None) -> pd.DataFrame:
        """
        Run sensitivity analysis across parameter ranges to assess robustness.
        
        Args:
            residential_data: Residential demographics
            charging_points: Charging infrastructure data
            param_ranges: Parameter ranges for sensitivity testing
            
        Returns:
            DataFrame with sensitivity results
        """
        if param_ranges is None:
            param_ranges = {
                'w_f_w_h_ratio': [2, 4, 6, 8],  # w_f/w_h ratios
                'c': [0.01, 0.05, 0.1, 0.2, 0.5],  # Gamma scaling coefficients
                'delta': [0.6, 0.8, 1.0, 1.2, 1.5]  # Sensitivity curvatures
            }
        
        results = []
        
        for ratio in param_ranges['w_f_w_h_ratio']:
            for c_val in param_ranges['c']:
                for delta_val in param_ranges['delta']:
                    # Normalize weights to sum to 1
                    w_f = ratio / (1 + ratio)
                    w_h = 1 / (1 + ratio)
                    
                    calibration_params = {
                        'w_f': w_f,
                        'w_h': w_h,
                        'c': c_val,
                        'delta': delta_val
                    }
                    
                    # Calculate reliance scores with these parameters
                    # (Implementation would compute scores for sample areas)
                    
                    # Store results for analysis
                    results.append({
                        'w_f_w_h_ratio': ratio,
                        'c': c_val,
                        'delta': delta_val,
                        'w_f': w_f,
                        'w_h': w_h,
                        # Add computed metrics here
                    })
        
        return pd.DataFrame(results)

    def analyze_public_charging_reliance(self, 
                                       demographics: gpd.GeoDataFrame,
                                       charging_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Analyze which households rely on public charging infrastructure.
        
        Args:
            demographics: GeoDataFrame with demographic and housing data
            charging_points: GeoDataFrame with charging point locations
            
        Returns:
            GeoDataFrame with public charging reliance analysis
        """
        logger.info("Analyzing public charging infrastructure reliance")
        
        # Calculate accessibility scores
        result_gdf = self.calculate_accessibility_scores(demographics, charging_points)
        
        # Add additional analysis for public charging reliance
        result_gdf['charging_accessibility_category'] = result_gdf['accessibility_score'].apply(
            self._categorize_accessibility
        )
        
        # Identify areas with high public charging reliance and poor accessibility
        # Use thresholds from config (@eq-bottleneck-indicator) - all must be defined in scoring_weights.json
        thresholds = getattr(self, 'bottleneck_thresholds', None)
        if not thresholds:
            raise ValueError("bottleneck_thresholds must be loaded from scoring_weights.json config")
        if 'high_reliance' not in thresholds:
            raise ValueError("config['charging_infrastructure']['bottleneck_thresholds']['high_reliance'] is required in scoring_weights.json")
        if 'poor_accessibility' not in thresholds:
            raise ValueError("config['charging_infrastructure']['bottleneck_thresholds']['poor_accessibility'] is required in scoring_weights.json")
        
        high_reliance = thresholds['high_reliance']
        poor_accessibility = thresholds['poor_accessibility']
        
        result_gdf['charging_bottleneck'] = (
            (result_gdf['public_charging_reliance'] > high_reliance) & 
            (result_gdf['accessibility_score'] < poor_accessibility)
        )
        
        logger.info(f"Identified {result_gdf['charging_bottleneck'].sum()} areas with charging bottlenecks")
        
        return result_gdf
        
    def _categorize_accessibility(self, score: float) -> str:
        """
        Categorize accessibility score into descriptive categories.
        Updated thresholds for more realistic assessment:
        - Excellent: score ≥0.9 (≤100m away)
        - Good: score ≥0.7 (≤800m away)
        - Fair: score ≥0.5 (≤2km away)
        - Poor: score <0.5 (>2km away)
        """
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'
    
    def calculate_gap_severity_index(self, 
                                     accessibility_results: gpd.GeoDataFrame,
                                     adoption_scores: pd.Series) -> gpd.GeoDataFrame:
        """
        Calculate infrastructure gap severity index (@eq-gap-severity from paper).
        
        Implements: G_i = (d_nearest / d_threshold) × R_i × (S_adoption / S_max)
        
        This provides a continuous prioritization score for infrastructure investment,
        combining:
        - Distance to nearest charger (normalized by threshold)
        - Public charging reliance score
        - Adoption propensity (normalized to max)
        
        Args:
            accessibility_results: GeoDataFrame with accessibility analysis results
                Must include 'nearest_charger_distance' and 'public_charging_reliance'
            adoption_scores: Series of adoption propensity scores indexed by area
            
        Returns:
            GeoDataFrame with 'gap_severity_index' column added
            
        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Calculating infrastructure gap severity index")
        
        # Validate required columns
        required_cols = ['nearest_charger_distance', 'public_charging_reliance']
        missing_cols = [col for col in required_cols if col not in accessibility_results.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Get distance threshold from config or use default
        distance_threshold = 2000  # Default 2000m from paper
        if hasattr(self, 'gap_severity_params') and self.gap_severity_params:
            distance_threshold = self.gap_severity_params.get('distance_threshold_m', 2000)
        
        # Align adoption scores with accessibility results
        result_gdf = accessibility_results.copy()
        
        # Ensure adoption scores align with result indices
        if not adoption_scores.index.equals(result_gdf.index):
            # Try to align by index
            adoption_aligned = adoption_scores.reindex(result_gdf.index, fill_value=0)
        else:
            adoption_aligned = adoption_scores
        
        # Normalize adoption scores to max
        S_max = adoption_aligned.max()
        if S_max > 0:
            adoption_normalized = adoption_aligned / S_max
        else:
            adoption_normalized = pd.Series(0, index=result_gdf.index)
        
        # Calculate gap severity index
        result_gdf['gap_severity_index'] = (
            (result_gdf['nearest_charger_distance'] / distance_threshold) *
            result_gdf['public_charging_reliance'] *
            adoption_normalized
        )
        
        # Add severity category for easier interpretation
        result_gdf['gap_severity_category'] = pd.cut(
            result_gdf['gap_severity_index'],
            bins=[0, 0.25, 0.5, 0.75, float('inf')],
            labels=['low', 'medium', 'high', 'critical']
        )
        
        logger.info(f"Calculated gap severity for {len(result_gdf)} areas")
        logger.info(f"Gap severity range: {result_gdf['gap_severity_index'].min():.3f} to {result_gdf['gap_severity_index'].max():.3f}")
        
        # Identify critical gaps
        critical_count = (result_gdf['gap_severity_category'] == 'critical').sum()
        if critical_count > 0:
            logger.warning(f"Found {critical_count} areas with critical infrastructure gaps")
        
        return result_gdf


class ChargingCapacityAnalyzer:
    """
    Analyzer for charging infrastructure capacity and demand assessment.
    
    Estimates charging point capacity utilization and identifies bottlenecks
    based on demand patterns and infrastructure distribution.
    """
    
    def __init__(self, utilization_assumptions: Optional[Dict[str, float]] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ChargingCapacityAnalyzer with utilization parameters.
        
        Args:
            utilization_assumptions: Assumptions for capacity utilization calculations
            config: Full configuration dictionary with charging_infrastructure parameters
        """
        # Load from config or use defaults
        if config and 'charging_infrastructure' in config:
            infra_config = config['charging_infrastructure']
            capacity_params = infra_config.get('capacity_params', {})
            self.utilization_params = infra_config.get('utilization_params', {})
            self.destination_buffer = infra_config.get('destination_buffer_m', 500)
            self.capacity_categories = infra_config.get('capacity_categories', {})
            
            self.utilization_assumptions = utilization_assumptions or {
                'peak_utilization_rate': 0.8,
                'average_utilization_rate': self.utilization_params.get('base_utilization', 0.45),
                'charging_session_duration': capacity_params.get('session_duration_hours', 2.0),
                'daily_operating_hours': capacity_params.get('daily_operating_hours', 16.0),
            }
        else:
            self.utilization_assumptions = utilization_assumptions or {
                'peak_utilization_rate': 0.8,  # 80% peak utilization
                'average_utilization_rate': 0.45,  # 45% average utilization (paper default)
                'charging_session_duration': 2.0,  # 2 hours average session
                'daily_operating_hours': 16.0,  # 16 hours daily operation
            }
            self.utilization_params = {}
            self.destination_buffer = 500  # Paper default
            self.capacity_categories = {}
        
    def estimate_charging_capacity(self, charging_locations: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Estimate charging capacity and utilization for each location.
        
        Args:
            charging_locations: GeoDataFrame with charging point data
            
        Returns:
            GeoDataFrame with capacity estimates added
        """
        logger.info("Estimating charging point capacity and utilization")
        
        result_gdf = charging_locations.copy()
        
        # Calculate theoretical daily capacity
        session_duration = self.utilization_assumptions['charging_session_duration']
        operating_hours = self.utilization_assumptions['daily_operating_hours']
        
        result_gdf['daily_sessions_capacity'] = (
            result_gdf['number_of_points'] * (operating_hours / session_duration)
        )
        
        # Estimate current utilization (simplified model)
        result_gdf['estimated_utilization'] = self._estimate_current_utilization(result_gdf)
        
        # Calculate available capacity
        result_gdf['available_capacity'] = (
            result_gdf['daily_sessions_capacity'] * 
            (1 - result_gdf['estimated_utilization'])
        )
        
        # Categorize locations by capacity
        result_gdf['capacity_category'] = result_gdf['number_of_points'].apply(
            self._categorize_capacity
        )
        
        logger.info(f"Estimated capacity for {len(result_gdf)} charging locations")
        
        return result_gdf
        
    def _estimate_current_utilization(self, charging_gdf: gpd.GeoDataFrame) -> pd.Series:
        """
        Estimate current utilization using paper's formula (@eq-utilisation-rate).
        
        Implements: U_i = U_base × [1 + α × (N-1)/(N-1+β)] + ε_i
        where β = 0.8 × median(N) and ε_i ~ N(0, 0.08²/√N) (heteroskedastic)
        
        Args:
            charging_gdf: GeoDataFrame with number_of_points column
            
        Returns:
            Series of utilization rates [0,1]
        """
        # Get parameters from config or use paper defaults
        params = self.utilization_params if hasattr(self, 'utilization_params') and self.utilization_params else {
            'base_utilization': 0.45,
            'alpha_hub_premium': 0.7,
            'beta_median_multiplier': 0.8,
            'noise_std': 0.08,
            'heteroskedastic': True
        }
        
        U_base = params['base_utilization']
        alpha = params['alpha_hub_premium']
        beta_multiplier = params['beta_median_multiplier']
        noise_std = params['noise_std']
        
        # Calculate adaptive beta parameter
        median_points = charging_gdf['number_of_points'].median()
        beta = beta_multiplier * median_points
        
        # Apply paper's hub premium formula
        N = charging_gdf['number_of_points']
        hub_premium = 1 + alpha * (N - 1) / (N - 1 + beta)
        base_rates = U_base * hub_premium
        
        # Add heteroskedastic noise (variance = σ²/√N per paper)
        # Paper specifies: ε_i ~ N(0, 0.08²/√N), so std = 0.08/N^(1/4)
        np.random.seed(42)  # For reproducible results
        if params.get('heteroskedastic', True):
            # Noise variance inversely proportional to √N (paper @eq-utilisation-rate)
            noise = np.random.normal(0, 1, len(charging_gdf)) * noise_std / np.power(N, 0.25)
        else:
            noise = np.random.normal(0, noise_std, len(charging_gdf))
        
        utilization = base_rates + noise
        
        # Ensure valid range [0,1] - no hard clipping to preserve model behavior
        return np.clip(utilization, 0.0, 1.0)
        
    def _categorize_capacity(self, num_points: int) -> str:
        """
        Categorize charging locations by capacity using config thresholds.
        
        Reads from config capacity_categories or uses paper defaults:
        - large_hub: ≥10 points
        - medium_hub: ≥5 points  
        - small_hub: ≥3 points
        - basic_station: <3 points
        """
        # Get thresholds from config or use defaults
        categories = getattr(self, 'capacity_categories', {})
        large_threshold = categories.get('large_hub', 10)
        medium_threshold = categories.get('medium_hub', 5)
        small_threshold = categories.get('small_hub', 3)
        
        if num_points >= large_threshold:
            return 'large_hub'
        elif num_points >= medium_threshold:
            return 'medium_hub'
        elif num_points >= small_threshold:
            return 'small_hub'
        else:
            return 'basic_station'
            
    def assess_destination_charging_capacity(self, 
                                           od_data: gpd.GeoDataFrame,
                                           charging_points: gpd.GeoDataFrame,
                                           buffer_distance: Optional[float] = None) -> Dict[str, Any]:
        """
        Assess destination charging capacity at key trip destinations.
        
        Args:
            od_data: Origin-destination trip data
            charging_points: Charging point locations
            buffer_distance: Buffer distance around destinations to search for chargers (meters)
                If None, reads from config (default: 500m per paper @eq-capacity-assessment)
            
        Returns:
            Dictionary with destination charging capacity analysis
        """
        logger.info("Assessing destination charging capacity")
        
        # Get buffer distance from config if not provided (paper default: 500m)
        if buffer_distance is None:
            buffer_distance = getattr(self, 'destination_buffer', 500)
        
        # Ensure consistent CRS
        if od_data.crs != 'EPSG:27700':
            od_data = od_data.to_crs('EPSG:27700')
        if charging_points.crs != 'EPSG:27700':
            charging_points = charging_points.to_crs('EPSG:27700')
            
        # Extract destination points from OD data
        destination_points = od_data.geometry.apply(lambda geom: geom.coords[-1])
        destination_gdf = gpd.GeoDataFrame(
            geometry=[gpd.points_from_xy([coord[0]], [coord[1]])[0] for coord in destination_points],
            crs='EPSG:27700'
        )
        
        # Add trip information (handle both 'All' and 'all' column names)
        if 'All' in od_data.columns:
            destination_gdf['trip_volume'] = od_data['All']
        elif 'all' in od_data.columns:
            destination_gdf['trip_volume'] = od_data['all']
        else:
            destination_gdf['trip_volume'] = 0
            logger.warning("No trip volume column found (expected 'All' or 'all')")
        
        # Handle both 'trip_purpose' and 'purpose' column names
        if 'trip_purpose' in od_data.columns:
            destination_gdf['trip_purpose'] = od_data['trip_purpose']
        elif 'purpose' in od_data.columns:
            destination_gdf['trip_purpose'] = od_data['purpose']
        else:
            destination_gdf['trip_purpose'] = 'unknown'
            logger.warning("No trip purpose column found (expected 'trip_purpose' or 'purpose')")
        
        # Find charging points near destinations
        destination_gdf['nearby_chargers'] = 0
        destination_gdf['nearby_charging_points'] = 0
        
        for idx, dest in destination_gdf.iterrows():
            # Create buffer around destination
            dest_buffer = dest.geometry.buffer(buffer_distance)
            
            # Find charging points within buffer
            nearby_chargers = charging_points[charging_points.geometry.within(dest_buffer)]
            
            destination_gdf.loc[idx, 'nearby_chargers'] = len(nearby_chargers)
            destination_gdf.loc[idx, 'nearby_charging_points'] = nearby_chargers['number_of_points'].sum()
            
        # Analyze capacity by trip purpose
        capacity_by_purpose = {}
        for purpose in destination_gdf['trip_purpose'].unique():
            purpose_data = destination_gdf[destination_gdf['trip_purpose'] == purpose]
            
            capacity_by_purpose[purpose] = {
                'total_destinations': len(purpose_data),
                'destinations_with_charging': (purpose_data['nearby_chargers'] > 0).sum(),
                'average_nearby_points': purpose_data['nearby_charging_points'].mean(),
                'total_trip_volume': purpose_data['trip_volume'].sum(),
                'charging_coverage_ratio': (purpose_data['nearby_chargers'] > 0).mean()
            }
            
        # Overall summary
        summary = {
            'total_destinations_analyzed': len(destination_gdf),
            'destinations_with_charging': (destination_gdf['nearby_chargers'] > 0).sum(),
            'overall_coverage_ratio': (destination_gdf['nearby_chargers'] > 0).mean(),
            'capacity_by_purpose': capacity_by_purpose,
            'average_points_per_destination': destination_gdf['nearby_charging_points'].mean()
        }
        
        logger.info(f"Destination charging analysis complete: {summary['overall_coverage_ratio']:.1%} coverage")
        
        return summary


def validate_charging_infrastructure_analysis(analysis_results: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Validate charging infrastructure analysis results.
    
    Args:
        analysis_results: GeoDataFrame with charging infrastructure analysis results
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'is_valid': True,
        'issues': [],
        'summary_stats': {}
    }
    
    # Check for required columns
    required_columns = [
        'nearest_charger_distance', 'accessibility_score', 
        'public_charging_reliance', 'chargers_within_1km'
    ]
    
    missing_columns = [col for col in required_columns if col not in analysis_results.columns]
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        
    # Check value ranges
    if 'accessibility_score' in analysis_results.columns:
        score_range = analysis_results['accessibility_score'].agg(['min', 'max'])
        if score_range['min'] < 0 or score_range['max'] > 1:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Accessibility scores outside valid range [0,1]")
            
    if 'public_charging_reliance' in analysis_results.columns:
        reliance_range = analysis_results['public_charging_reliance'].agg(['min', 'max'])
        if reliance_range['min'] < 0 or reliance_range['max'] > 1:
            validation_results['is_valid'] = False
            validation_results['issues'].append("Public charging reliance scores outside valid range [0,1]")
            
    # Summary statistics
    if len(validation_results['issues']) == 0:
        validation_results['summary_stats'] = {
            'total_areas': len(analysis_results),
            'average_accessibility_score': analysis_results['accessibility_score'].mean(),
            'areas_with_poor_access': (analysis_results['accessibility_score'] < 0.4).sum(),
            'high_reliance_areas': (analysis_results['public_charging_reliance'] > 0.7).sum(),
            'charging_bottlenecks': analysis_results.get('charging_bottleneck', pd.Series(False)).sum()
        }
        
    return validation_results


# =============================================================================
# 5-Step Replaceability Logic Functions (per eq_revised.qmd)
# =============================================================================

def calculate_home_charging_binary(
    HCF_i: Union[float, pd.Series],
    threshold: float = 0.5
) -> Union[int, pd.Series]:
    """
    Calculate binary home charging indicator H_i per @eq-home-charging-binary.

    H_i = 𝟙{HCF_i ≥ threshold}

    Args:
        HCF_i: Home charging feasibility score [0,1]
        threshold: Binary threshold (default 0.5 from config)

    Returns:
        Binary indicator: 1 if HCF_i ≥ threshold, else 0
    """
    if isinstance(HCF_i, pd.Series):
        return (HCF_i >= threshold).astype(int)
    return 1 if HCF_i >= threshold else 0


def calculate_public_charging_binary(
    distance_to_charger_km: Union[float, pd.Series],
    threshold_km: float = 0.5
) -> Union[int, pd.Series]:
    """
    Calculate binary public charging indicator P_i or P_j per @eq-public-charging-origin/destination.

    P_i = 𝟙{d_i ≤ threshold_km}

    Args:
        distance_to_charger_km: Distance to nearest charger in km
        threshold_km: Maximum distance for public charging availability (default 0.5km)

    Returns:
        Binary indicator: 1 if distance ≤ threshold, else 0
    """
    if isinstance(distance_to_charger_km, pd.Series):
        return (distance_to_charger_km <= threshold_km).astype(int)
    return 1 if distance_to_charger_km <= threshold_km else 0


def calculate_charging_condition(
    H_i: Union[int, pd.Series],
    P_i: Union[int, pd.Series],
    P_j: Union[int, pd.Series],
    distance_km: Union[float, pd.Series],
    config: Optional[Dict[str, Any]] = None
) -> Union[int, pd.Series]:
    """
    Calculate distance-dependent charging condition C_ij per @eq-charging-condition.

    C_ij = {
        H_i ∨ P_i       if d_ij ≤ 40 km (OR logic - origin charging sufficient)
        H_i ∧ P_j       if 40 < d_ij ≤ 80 km (AND logic - destination charging required)
        0               if d_ij > 80 km (infeasible)
    }

    Args:
        H_i: Binary home charging indicator at origin
        P_i: Binary public charging indicator at origin
        P_j: Binary public charging indicator at destination
        distance_km: Trip distance in km
        config: Optional config dict with replaceability_thresholds

    Returns:
        Binary charging condition: 1 if charging available, else 0
    """
    # Get thresholds from config or use defaults
    short_threshold = 40.0
    medium_threshold = 80.0
    if config and 'replaceability_thresholds' in config:
        rt = config['replaceability_thresholds']
        short_threshold = rt.get('short_trip_distance_km', 40.0)
        medium_threshold = rt.get('medium_trip_distance_km', 80.0)

    if isinstance(distance_km, pd.Series):
        # Vectorized implementation
        result = pd.Series(0, index=distance_km.index)
        # Short trips: OR logic
        short_mask = distance_km <= short_threshold
        result.loc[short_mask] = ((H_i | P_i) > 0).astype(int).loc[short_mask]
        # Medium trips: AND logic
        medium_mask = (distance_km > short_threshold) & (distance_km <= medium_threshold)
        result.loc[medium_mask] = ((H_i & P_j) > 0).astype(int).loc[medium_mask]
        # Long trips: 0 (already initialized)
        return result
    else:
        if distance_km <= short_threshold:
            return 1 if (H_i or P_i) else 0
        elif distance_km <= medium_threshold:
            return 1 if (H_i and P_j) else 0
        else:
            return 0


def calculate_energy_required(
    distance_km: Union[float, pd.Series],
    vehicle_type: str = '4-seater',
    config: Optional[Dict[str, Any]] = None
) -> Union[float, pd.Series]:
    """
    Calculate energy required for trip per @eq-energy-required.

    E_ij = d_ij × η_type

    Args:
        distance_km: Trip distance in km
        vehicle_type: '2-seater' or '4-seater'
        config: Optional config with energy_consumption parameters

    Returns:
        Energy required in kWh
    """
    # Get consumption rates from config or use defaults
    eta_2seater = 0.060  # kWh/km
    eta_4seater = 0.065  # kWh/km
    if config and 'energy_consumption' in config:
        ec = config['energy_consumption']
        eta_2seater = ec.get('two_seater_kWh_per_km', 0.060)
        eta_4seater = ec.get('four_seater_kWh_per_km', 0.065)

    eta = eta_2seater if vehicle_type == '2-seater' else eta_4seater
    return distance_km * eta


def calculate_charging_time(
    E_ij: Union[float, pd.Series],
    charger_power_kW: float = 7.0
) -> Union[float, pd.Series]:
    """
    Calculate charging time required per @eq-charging-time.

    t_ij = E_ij / P_charge

    Args:
        E_ij: Energy required in kWh
        charger_power_kW: Charger power in kW (default 7kW)

    Returns:
        Charging time in hours
    """
    return E_ij / charger_power_kW


def calculate_destination_demand(
    N_ij: pd.Series,
    t_ij: pd.Series,
    R_i: pd.Series,
    distance_km: pd.Series,
    config: Optional[Dict[str, Any]] = None
) -> pd.Series:
    """
    Calculate weighted destination charging demand D_j per @eq-destination-demand.

    D_j = Σ N_ij × t_ij × R_i × 𝟙{40 < d_ij ≤ 80 km}

    Only medium-distance trips (40-80km) contribute to destination demand.

    Args:
        N_ij: Converted trips per OD pair
        t_ij: Charging time required (hours)
        R_i: Public charging reliance at origin
        distance_km: Trip distance in km
        config: Optional config with replaceability_thresholds

    Returns:
        Destination demand in charging-hours/day
    """
    short_threshold = 40.0
    medium_threshold = 80.0
    if config and 'replaceability_thresholds' in config:
        rt = config['replaceability_thresholds']
        short_threshold = rt.get('short_trip_distance_km', 40.0)
        medium_threshold = rt.get('medium_trip_distance_km', 80.0)

    # Only medium trips contribute to destination demand
    medium_mask = (distance_km > short_threshold) & (distance_km <= medium_threshold)

    # Calculate weighted demand
    demand = N_ij * t_ij * R_i * medium_mask.astype(int)
    return demand


def calculate_capacity_constraint(
    D_j: Union[float, pd.Series],
    C_avail_hours: Union[float, pd.Series]
) -> Union[int, pd.Series]:
    """
    Calculate capacity constraint indicator Cap_j per @eq-capacity-constraint.

    Cap_j = 𝟙{D_j ≤ C_avail_hours}

    Args:
        D_j: Destination demand in charging-hours
        C_avail_hours: Available capacity in charging-hours

    Returns:
        Binary: 1 if capacity sufficient, else 0
    """
    if isinstance(D_j, pd.Series):
        return (D_j <= C_avail_hours).astype(int)
    return 1 if D_j <= C_avail_hours else 0


def detect_charging_bottleneck(
    D_j: Union[float, pd.Series],
    C_avail_hours: Union[float, pd.Series]
) -> Union[int, pd.Series]:
    """
    Detect charging bottleneck per @eq-bottleneck.

    B_j^bottleneck = 𝟙{D_j > C_avail_hours}

    Args:
        D_j: Destination demand in charging-hours
        C_avail_hours: Available capacity in charging-hours

    Returns:
        Binary: 1 if bottleneck detected, else 0
    """
    if isinstance(D_j, pd.Series):
        return (D_j > C_avail_hours).astype(int)
    return 1 if D_j > C_avail_hours else 0