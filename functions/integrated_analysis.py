"""
Integrated Feasibility Assessment Module for Frugal EV Analysis

This module provides classes and functions for comprehensive evaluation that integrates
range limits, infrastructure availability, and adoption propensity to categorize trips
as feasible, constrained, or infeasible.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

# Import other analysis modules
from .adoption import AdoptionPropensityCalculator, validate_adoption_scores
from .range_feasibility import RangeFeasibilityAssessor, RangeFeasibilityCategory
from .charging_infrastructure import ChargingInfrastructureAnalyzer, validate_charging_infrastructure_analysis
from .data_integration import standardize_od_column_names
from .conversion import TripConversionEstimator, FactorIntegrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeasibilityCategory(Enum):
    """Enumeration of integrated feasibility categories"""
    FEASIBLE = "feasible"                    # All factors support EV adoption
    CONSTRAINED = "constrained"              # Some limiting factors present
    INFEASIBLE = "infeasible"               # Major barriers prevent EV adoption


class ConstraintType(Enum):
    """Enumeration of constraint types that limit EV feasibility"""
    RANGE_CONSTRAINT = "range_constraint"                    # Trip exceeds EV range
    CHARGING_CONSTRAINT = "charging_constraint"              # Insufficient charging infrastructure
    ADOPTION_CONSTRAINT = "adoption_constraint"              # Low demographic adoption propensity
    INFRASTRUCTURE_CONSTRAINT = "infrastructure_constraint"  # Poor charging accessibility
    COMBINED_CONSTRAINT = "combined_constraint"              # Multiple limiting factors


@dataclass
class FeasibilityAssessment:
    """Container for integrated feasibility assessment results"""
    origin_code: str
    destination_code: str
    trip_purpose: str
    feasibility_category: FeasibilityCategory
    limiting_constraints: List[ConstraintType]
    feasibility_score: float  # Overall feasibility score [0,1]
    constraint_scores: Dict[str, float]  # Individual constraint scores
    conversion_potential: float


@dataclass
class IntegratedAnalysisResults:
    """Container for comprehensive analysis results"""
    total_trips_analyzed: int
    feasible_trips: int
    constrained_trips: int
    infeasible_trips: int
    conversion_potential_by_category: Dict[str, float]
    constraint_breakdown: Dict[str, int]
    geographic_summary: Dict[str, Any]
    policy_recommendations: List[str]


class IntegratedAnalysisEngine:
    """
    Main orchestration engine for comprehensive feasibility assessment.
    
    Integrates range limits, infrastructure availability, and adoption propensity
    to provide unified trip categorization and constraint identification.
    """
    
    def __init__(self, 
                 feasibility_thresholds: Optional[Dict[str, float]] = None,
                 constraint_weights: Optional[Dict[str, float]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize IntegratedAnalysisEngine with assessment parameters.
        
        Args:
            feasibility_thresholds: Thresholds for categorizing feasibility
            constraint_weights: Weights for different constraint types
        """
        # Default feasibility thresholds
        self.feasibility_thresholds = feasibility_thresholds or {
            'feasible_threshold': 0.7,      # Score >= 0.7 = feasible
            'constrained_threshold': 0.4,   # Score 0.4-0.7 = constrained
            'infeasible_threshold': 0.4     # Score < 0.4 = infeasible
        }
        
        # Default constraint weights for overall feasibility scoring
        self.constraint_weights = constraint_weights or {
            'range_feasibility': 0.3,       # Range is critical
            'adoption_propensity': 0.25,    # Demographics important
            'charging_accessibility': 0.25, # Infrastructure access key
            'infrastructure_capacity': 0.2  # Capacity constraints
        }
        
        config = config or {}
        charging_config = config['charging_infrastructure']
        integrated_config = config['integrated_analysis']

        bottleneck_thresholds = charging_config['bottleneck_thresholds']
        self.charging_constraint_threshold = bottleneck_thresholds['poor_accessibility']
        self.adoption_constraint_threshold = integrated_config['low_adoption_threshold']
        self.high_adoption_threshold = integrated_config['high_adoption_threshold']
        self.adoption_cap = integrated_config['adoption_cap']
        self.infrastructure_capacity_threshold = bottleneck_thresholds['capacity_threshold']

        # Initialize component analyzers
        self.adoption_calculator = AdoptionPropensityCalculator()
        self.range_assessor = RangeFeasibilityAssessor()
        self.charging_analyzer = ChargingInfrastructureAnalyzer()
        self.conversion_estimator = TripConversionEstimator()
        self.factor_integrator = FactorIntegrator()
        
        logger.info("Initialized IntegratedAnalysisEngine with comprehensive assessment capabilities")
    
    def run_integrated_analysis(self, 
                               od_data: gpd.GeoDataFrame,
                               demographics: gpd.GeoDataFrame,
                               charging_infrastructure: gpd.GeoDataFrame,
                               purpose_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        Run comprehensive integrated feasibility analysis.
        
        Args:
            od_data: Origin-destination trip data with route information
            demographics: Demographic data with adoption propensity scores
            charging_infrastructure: Charging point location data
            purpose_weights: Trip purpose suitability weights
            
        Returns:
            Dictionary with comprehensive analysis results
            
        Raises:
            ValueError: If required data is missing or invalid
        """
        logger.info("Starting integrated feasibility analysis")
        
        # Standardize column names
        od_data = standardize_od_column_names(od_data)
        
        # Validate input data
        self._validate_input_data(od_data, demographics, charging_infrastructure)
        
        # Step 1: Calculate adoption propensity if not already present
        if 'adoption_propensity' not in demographics.columns:
            logger.info("Calculating adoption propensity scores")
            demographics = self.adoption_calculator.calculate_propensity_scores(demographics)
        
        # Step 2: Assess range feasibility
        logger.info("Assessing range feasibility")
        od_with_range = self.range_assessor.assess_range_feasibility(od_data)
        
        # Step 3: Analyze charging infrastructure accessibility
        logger.info("Analyzing charging infrastructure accessibility")
        try:
            demographics_with_charging = self.charging_analyzer.analyze_public_charging_reliance(
                demographics, charging_infrastructure
            )
        except MemoryError as e:
            logger.error(f"Memory error in charging infrastructure analysis: {e}")
            logger.info("Attempting memory-efficient chunked processing...")
            # Fallback to chunked processing if memory error occurs
            chunk_size = min(5000, len(demographics))
            demographics_with_charging = self._process_charging_analysis_chunked(
                demographics, charging_infrastructure, chunk_size
            )
        
        # Step 4: Estimate conversion potential
        logger.info("Estimating conversion potential")
        od_with_conversion = self.conversion_estimator.estimate_od_conversion_potential(
            od_with_range, demographics_with_charging, purpose_weights
        )
        
        # Step 5: Integrate all factors for comprehensive feasibility assessment
        logger.info("Integrating factors for feasibility classification")
        integrated_results = self._integrate_feasibility_factors(
            od_with_conversion, demographics_with_charging
        )
        
        # Step 6: Classify trip feasibility
        logger.info("Classifying trip feasibility")
        classified_results = self.classify_trip_feasibility(integrated_results)
        
        # Step 7: Generate comprehensive analysis summary
        logger.info("Generating analysis summary")
        analysis_summary = self._generate_analysis_summary(classified_results)
        
        logger.info("Integrated feasibility analysis completed")
        
        return {
            'classified_trips': classified_results,
            'analysis_summary': analysis_summary,
            'demographics_with_scores': demographics_with_charging,
            'validation_results': self._validate_integrated_results(classified_results)
        }
    
    def _process_charging_analysis_chunked(self, 
                                         demographics: gpd.GeoDataFrame,
                                         charging_infrastructure: gpd.GeoDataFrame,
                                         chunk_size: int) -> gpd.GeoDataFrame:
        """
        Process charging infrastructure analysis in chunks to avoid memory issues.
        
        Args:
            demographics: Demographic data 
            charging_infrastructure: Charging infrastructure data
            chunk_size: Size of chunks to process
            
        Returns:
            GeoDataFrame with charging analysis results
        """
        logger.info(f"Processing charging analysis in chunks of {chunk_size}")
        
        total_areas = len(demographics)
        results = []
        
        for i in range(0, total_areas, chunk_size):
            end_idx = min(i + chunk_size, total_areas)
            chunk = demographics.iloc[i:end_idx].copy()
            
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(total_areas-1)//chunk_size + 1}: areas {i}-{end_idx-1}")
            
            try:
                chunk_results = self.charging_analyzer.analyze_public_charging_reliance(
                    chunk, charging_infrastructure
                )
                results.append(chunk_results)
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing chunk {i//chunk_size + 1}: {e}")
                # Return chunk with default values if processing fails
                chunk['accessibility_score'] = 0.5
                chunk['public_charging_reliance'] = 0.5
                chunk['charging_accessibility_category'] = 'fair'
                chunk['charging_bottleneck'] = False
                results.append(chunk)
        
        # Combine all chunks
        combined_results = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))
        logger.info(f"Completed chunked charging analysis for {len(combined_results)} areas")
        
        return combined_results
    
    def classify_trip_feasibility(self, integrated_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Classify trips as feasible, constrained, or infeasible based on integrated factors.
        
        Args:
            integrated_data: GeoDataFrame with all feasibility factors integrated
            
        Returns:
            GeoDataFrame with feasibility classification added
        """
        logger.info(f"Classifying feasibility for {len(integrated_data)} trips")
        
        result = integrated_data.copy()

        # Preserve range-based classification if present
        if 'feasibility_category' in result.columns:
            result = result.rename(columns={'feasibility_category': 'range_feasibility_category'})

        # Calculate overall feasibility score
        result['overall_feasibility_score'] = self._calculate_overall_feasibility_score(result)

        if 'range_feasibility_category' in result.columns:
            result['feasibility_category'] = result['range_feasibility_category']
        else:
            # Classify based on feasibility score when range category unavailable
            result['feasibility_category'] = result['overall_feasibility_score'].apply(
                self._categorize_feasibility
            )
        
        # Identify specific limiting constraints for each trip
        result['limiting_constraints'] = result.apply(
            lambda row: self._identify_limiting_constraints(row), axis=1
        )
        
        # Count constraints per trip
        result['constraint_count'] = result['limiting_constraints'].apply(len)
        
        # Calculate constraint severity
        result['constraint_severity'] = result.apply(
            lambda row: self._calculate_constraint_severity(row), axis=1
        )
        
        # Log classification summary
        self._log_classification_summary(result)
        
        return result
    
    def _validate_input_data(self, 
                           od_data: gpd.GeoDataFrame,
                           demographics: gpd.GeoDataFrame,
                           charging_infrastructure: gpd.GeoDataFrame) -> None:
        """Validate that input data has required columns and structure"""
        
        # Validate OD data
        required_od_cols = ['origin_code', 'destination_code', 'geometry']
        missing_od_cols = [col for col in required_od_cols if col not in od_data.columns]
        if missing_od_cols:
            raise ValueError(f"Missing required OD columns: {missing_od_cols}")
        
        # Validate demographics data
        required_demo_cols = ['geo_code', 'geometry']
        missing_demo_cols = [col for col in required_demo_cols if col not in demographics.columns]
        if missing_demo_cols:
            raise ValueError(f"Missing required demographic columns: {missing_demo_cols}")
        
        # Validate charging infrastructure data
        required_charging_cols = ['poi_id', 'geometry']
        missing_charging_cols = [col for col in required_charging_cols if col not in charging_infrastructure.columns]
        if missing_charging_cols:
            raise ValueError(f"Missing required charging infrastructure columns: {missing_charging_cols}")
        
        logger.info("Input data validation passed")
    
    def _integrate_feasibility_factors(self, 
                                     od_data: gpd.GeoDataFrame,
                                     demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Integrate all feasibility factors into a unified dataset"""
        
        result = od_data.copy()
        
        # Attach demographic scores to origins and destinations
        demo_scores = demographics[['geo_code', 'adoption_propensity', 'accessibility_score', 
                                   'public_charging_reliance', 'home_charging_feasibility']].copy()
        demo_scores['geo_code'] = demo_scores['geo_code'].astype(str)
        
        # Ensure OD codes are strings
        result['origin_code'] = result['origin_code'].astype(str)
        result['destination_code'] = result['destination_code'].astype(str)
        
        # Attach origin scores
        origin_scores = demo_scores.rename(columns={
            'geo_code': 'origin_code',
            'adoption_propensity': 'origin_adoption_propensity',
            'accessibility_score': 'origin_accessibility_score',
            'public_charging_reliance': 'origin_charging_reliance',
            'home_charging_feasibility': 'origin_home_charging'
        })
        result = result.merge(origin_scores, on='origin_code', how='left')
        
        # Attach destination scores
        dest_scores = demo_scores.rename(columns={
            'geo_code': 'destination_code',
            'adoption_propensity': 'dest_adoption_propensity',
            'accessibility_score': 'dest_accessibility_score',
            'public_charging_reliance': 'dest_charging_reliance',
            'home_charging_feasibility': 'dest_home_charging'
        })
        result = result.merge(dest_scores, on='destination_code', how='left')
        
        # Calculate combined scores
        result['combined_adoption_propensity'] = (
            result['origin_adoption_propensity'].fillna(0.3) + 
            result['dest_adoption_propensity'].fillna(0.3)
        ) / 2
        
        result['combined_accessibility_score'] = (
            result['origin_accessibility_score'].fillna(0.5) + 
            result['dest_accessibility_score'].fillna(0.5)
        ) / 2
        
        result['combined_charging_reliance'] = (
            result['origin_charging_reliance'].fillna(0.5) + 
            result['dest_charging_reliance'].fillna(0.5)
        ) / 2
        
        return result
    
    def _calculate_overall_feasibility_score(self, data: gpd.GeoDataFrame) -> pd.Series:
        """Calculate overall feasibility score combining all factors"""
        
        # Range feasibility factor (from range assessment)
        range_factor = pd.Series(0.0, index=data.index)
        range_category_col = 'range_feasibility_category' if 'range_feasibility_category' in data.columns else 'feasibility_category'
        if range_category_col in data.columns:
            range_factor = data[range_category_col].map({
                'feasible': 1.0,
                'constrained': 0.6,
                'infeasible': 0.0
            }).fillna(0.5)
        
        # Adoption propensity factor
        adoption_factor = data.get('combined_adoption_propensity', pd.Series(0.5, index=data.index))
        
        # Charging accessibility factor
        accessibility_factor = data.get('combined_accessibility_score', pd.Series(0.5, index=data.index))
        
        # Infrastructure capacity factor (inverse of charging reliance)
        capacity_factor = 1.0 - data.get('combined_charging_reliance', pd.Series(0.5, index=data.index))
        
        # Calculate weighted overall score
        overall_score = (
            range_factor * self.constraint_weights['range_feasibility'] +
            adoption_factor * self.constraint_weights['adoption_propensity'] +
            accessibility_factor * self.constraint_weights['charging_accessibility'] +
            capacity_factor * self.constraint_weights['infrastructure_capacity']
        )
        
        return np.clip(overall_score, 0.0, 1.0)
    
    def _categorize_feasibility(self, feasibility_score: float) -> str:
        """Categorize feasibility based on overall score"""
        if feasibility_score >= self.feasibility_thresholds['feasible_threshold']:
            return FeasibilityCategory.FEASIBLE.value
        elif feasibility_score >= self.feasibility_thresholds['constrained_threshold']:
            return FeasibilityCategory.CONSTRAINED.value
        else:
            return FeasibilityCategory.INFEASIBLE.value
    
    def _identify_limiting_constraints(self, row: pd.Series) -> List[str]:
        """Identify specific constraints limiting EV feasibility for a trip"""
        constraints = []
        
        # Range constraint
        range_category = row.get('range_feasibility_category', row.get('feasibility_category'))
        if range_category == 'infeasible':
            constraints.append(ConstraintType.RANGE_CONSTRAINT.value)
        elif range_category == 'constrained':
            if row.get('destination_charging_required', False):
                constraints.append(ConstraintType.RANGE_CONSTRAINT.value)

        # Adoption constraint
        adoption_score = row.get('combined_adoption_propensity', 0.5)
        if adoption_score < self.adoption_constraint_threshold:
            constraints.append(ConstraintType.ADOPTION_CONSTRAINT.value)

        # Charging accessibility constraint
        accessibility_score = row.get('combined_accessibility_score', 0.5)
        if accessibility_score < self.charging_constraint_threshold:
            constraints.append(ConstraintType.CHARGING_CONSTRAINT.value)

        # Infrastructure capacity constraint
        charging_reliance = row.get('combined_charging_reliance', 0.5)
        if charging_reliance > self.infrastructure_capacity_threshold and accessibility_score < self.charging_constraint_threshold:
            constraints.append(ConstraintType.INFRASTRUCTURE_CONSTRAINT.value)
        
        # Combined constraint if multiple issues
        if len(constraints) > 1:
            constraints.append(ConstraintType.COMBINED_CONSTRAINT.value)
        
        return constraints
    
    def _calculate_constraint_severity(self, row: pd.Series) -> float:
        """Calculate severity of constraints (0=no constraints, 1=severe constraints)"""
        constraints = row.get('limiting_constraints', [])
        
        if not constraints:
            return 0.0
        
        # Weight different constraint types by severity
        severity_weights = {
            ConstraintType.RANGE_CONSTRAINT.value: 0.4,
            ConstraintType.ADOPTION_CONSTRAINT.value: 0.3,
            ConstraintType.CHARGING_CONSTRAINT.value: 0.2,
            ConstraintType.INFRASTRUCTURE_CONSTRAINT.value: 0.1,
            ConstraintType.COMBINED_CONSTRAINT.value: 0.5
        }
        
        total_severity = sum(severity_weights.get(constraint, 0.1) for constraint in constraints)
        return min(1.0, total_severity)
    
    def _log_classification_summary(self, result: gpd.GeoDataFrame) -> None:
        """Log summary statistics of feasibility classification"""
        total_trips = len(result)
        
        # Count by feasibility category
        feasible_count = (result['feasibility_category'] == FeasibilityCategory.FEASIBLE.value).sum()
        constrained_count = (result['feasibility_category'] == FeasibilityCategory.CONSTRAINED.value).sum()
        infeasible_count = (result['feasibility_category'] == FeasibilityCategory.INFEASIBLE.value).sum()
        
        # Constraint analysis
        all_constraints = []
        for constraints_list in result['limiting_constraints']:
            all_constraints.extend(constraints_list)
        
        constraint_counts = pd.Series(all_constraints).value_counts()
        
        logger.info(f"Feasibility classification completed:")
        logger.info(f"  Total trips: {total_trips:,}")
        logger.info(f"  Feasible: {feasible_count:,} ({feasible_count/total_trips*100:.1f}%)")
        logger.info(f"  Constrained: {constrained_count:,} ({constrained_count/total_trips*100:.1f}%)")
        logger.info(f"  Infeasible: {infeasible_count:,} ({infeasible_count/total_trips*100:.1f}%)")
        logger.info(f"  Most common constraints: {dict(constraint_counts.head(3))}")
    
    def _generate_analysis_summary(self, classified_data: gpd.GeoDataFrame) -> IntegratedAnalysisResults:
        """Generate comprehensive analysis summary"""
        
        total_trips = len(classified_data)
        
        # Count by feasibility category
        feasible_count = (classified_data['feasibility_category'] == FeasibilityCategory.FEASIBLE.value).sum()
        constrained_count = (classified_data['feasibility_category'] == FeasibilityCategory.CONSTRAINED.value).sum()
        infeasible_count = (classified_data['feasibility_category'] == FeasibilityCategory.INFEASIBLE.value).sum()
        
        # Conversion potential by category
        conversion_by_category = {}
        for category in [FeasibilityCategory.FEASIBLE.value, FeasibilityCategory.CONSTRAINED.value, FeasibilityCategory.INFEASIBLE.value]:
            category_data = classified_data[classified_data['feasibility_category'] == category]
            conversion_by_category[category] = category_data.get('conversion_potential', pd.Series(0)).sum()
        
        # Constraint breakdown
        all_constraints = []
        for constraints_list in classified_data['limiting_constraints']:
            all_constraints.extend(constraints_list)
        constraint_breakdown = dict(pd.Series(all_constraints).value_counts())
        
        # Geographic summary (simplified)
        geographic_summary = {
            'total_origin_areas': classified_data['origin_code'].nunique(),
            'total_destination_areas': classified_data['destination_code'].nunique(),
            'mean_feasibility_score': classified_data['overall_feasibility_score'].mean(),
            'areas_with_high_constraints': (classified_data['constraint_severity'] > 0.7).sum()
        }
        
        return IntegratedAnalysisResults(
            total_trips_analyzed=total_trips,
            feasible_trips=feasible_count,
            constrained_trips=constrained_count,
            infeasible_trips=infeasible_count,
            conversion_potential_by_category=conversion_by_category,
            constraint_breakdown=constraint_breakdown,
            geographic_summary=geographic_summary,
            policy_recommendations=self._generate_policy_recommendations(classified_data)
        )
    
    def _generate_policy_recommendations(self, classified_data: gpd.GeoDataFrame) -> List[str]:
        """Generate policy recommendations based on analysis results"""
        recommendations = []
        
        # Analyze constraint patterns
        all_constraints = []
        for constraints_list in classified_data['limiting_constraints']:
            all_constraints.extend(constraints_list)
        constraint_counts = pd.Series(all_constraints).value_counts()
        
        # Range constraint recommendations
        if ConstraintType.RANGE_CONSTRAINT.value in constraint_counts.index:
            range_constraint_count = constraint_counts[ConstraintType.RANGE_CONSTRAINT.value]
            if range_constraint_count > len(classified_data) * 0.3:
                recommendations.append(
                    "High prevalence of range constraints suggests need for destination charging infrastructure "
                    "at key trip destinations, particularly for trips 50-100km"
                )
        
        # Adoption constraint recommendations
        if ConstraintType.ADOPTION_CONSTRAINT.value in constraint_counts.index:
            adoption_constraint_count = constraint_counts[ConstraintType.ADOPTION_CONSTRAINT.value]
            if adoption_constraint_count > len(classified_data) * 0.2:
                recommendations.append(
                    "Low adoption propensity in key demographics suggests need for targeted incentives "
                    "and education programs, particularly for lower-income and older populations"
                )
        
        # Charging infrastructure recommendations
        if ConstraintType.CHARGING_CONSTRAINT.value in constraint_counts.index:
            charging_constraint_count = constraint_counts[ConstraintType.CHARGING_CONSTRAINT.value]
            if charging_constraint_count > len(classified_data) * 0.25:
                recommendations.append(
                    "Poor charging accessibility indicates need for strategic placement of public charging "
                    "infrastructure, particularly in areas with high flat/apartment dwelling rates"
                )
        
        # Infrastructure capacity recommendations
        if ConstraintType.INFRASTRUCTURE_CONSTRAINT.value in constraint_counts.index:
            recommendations.append(
                "Infrastructure capacity constraints suggest need for expansion of existing charging hubs "
                "and addition of fast-charging capabilities at high-demand locations"
            )
        
        # General recommendations
        feasible_rate = (classified_data['feasibility_category'] == FeasibilityCategory.FEASIBLE.value).mean()
        if feasible_rate < 0.3:
            recommendations.append(
                "Low overall feasibility rate suggests need for comprehensive policy package combining "
                "infrastructure investment, financial incentives, and regulatory support"
            )
        
        return recommendations
    
    def _validate_integrated_results(self, classified_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Validate integrated analysis results"""
        validation_results = {
            'is_valid': True,
            'issues': [],
            'statistics': {}
        }
        
        # Check required columns
        required_cols = ['feasibility_category', 'overall_feasibility_score', 'limiting_constraints']
        missing_cols = [col for col in required_cols if col not in classified_data.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_cols}")
        
        # Validate feasibility scores
        if 'overall_feasibility_score' in classified_data.columns:
            scores = classified_data['overall_feasibility_score']
            if scores.min() < 0 or scores.max() > 1:
                validation_results['issues'].append("Feasibility scores outside valid range [0,1]")
        
        # Validate category consistency
        if 'feasibility_category' in classified_data.columns and 'overall_feasibility_score' in classified_data.columns:
            inconsistent_count = 0
            for _, row in classified_data.iterrows():
                score = row['overall_feasibility_score']
                category = row['feasibility_category']
                expected_category = self._categorize_feasibility(score)
                if category != expected_category:
                    inconsistent_count += 1
            
            if inconsistent_count > 0:
                validation_results['issues'].append(f"{inconsistent_count} records have inconsistent category/score")
        
        # Calculate statistics
        validation_results['statistics'] = {
            'total_records': len(classified_data),
            'mean_feasibility_score': classified_data.get('overall_feasibility_score', pd.Series(0)).mean(),
            'category_distribution': classified_data.get('feasibility_category', pd.Series()).value_counts().to_dict(),
            'constraint_coverage': len([col for col in classified_data.columns if 'constraint' in col])
        }
        
        if validation_results['issues']:
            validation_results['is_valid'] = False
        
        return validation_results


class FeasibilityClassifier:
    """
    Specialized classifier for constraint identification and bottleneck analysis.
    
    Provides detailed constraint classification and consistency validation
    across all analysis components.
    """
    
    def __init__(self, constraint_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize FeasibilityClassifier with constraint detection parameters.
        
        Args:
            constraint_thresholds: Thresholds for identifying different constraint types
        """
        self.constraint_thresholds = constraint_thresholds or {
            'range_infeasible': 100.0,      # km - trips beyond this are infeasible
            'range_constrained': 50.0,      # km - trips beyond this need destination charging
            'adoption_low': 0.4,            # adoption propensity below this is constraining
            'accessibility_poor': 0.4,      # accessibility score below this is constraining
            'charging_reliance_high': 0.7,  # charging reliance above this is constraining
            'feasibility_low': 0.4          # overall feasibility below this is constrained/infeasible
        }
    
    def classify_constraint_types(self, analysis_data: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Classify specific constraint types and bottlenecks for each trip.
        
        Args:
            analysis_data: GeoDataFrame with integrated analysis results
            
        Returns:
            GeoDataFrame with detailed constraint classification
        """
        logger.info("Classifying constraint types and bottlenecks")
        
        result = analysis_data.copy()
        
        # Classify range constraints
        result['range_constraint_type'] = self._classify_range_constraints(result)
        
        # Classify adoption constraints
        result['adoption_constraint_type'] = self._classify_adoption_constraints(result)
        
        # Classify charging constraints
        result['charging_constraint_type'] = self._classify_charging_constraints(result)
        
        # Identify primary bottleneck for each trip
        result['primary_bottleneck'] = result.apply(
            lambda row: self._identify_primary_bottleneck(row), axis=1
        )
        
        # Calculate bottleneck severity
        result['bottleneck_severity'] = result.apply(
            lambda row: self._calculate_bottleneck_severity(row), axis=1
        )
        
        # Classify intervention priority
        result['intervention_priority'] = self._classify_intervention_priority(result)
        
        logger.info(f"Classified constraints for {len(result)} trips")
        return result
    
    def _classify_range_constraints(self, data: gpd.GeoDataFrame) -> pd.Series:
        """Classify range-related constraints"""
        def classify_range(row):
            distance = row.get('route_distance_km', 0)
            
            if distance > self.constraint_thresholds['range_infeasible']:
                return 'infeasible_range'
            elif distance > self.constraint_thresholds['range_constrained']:
                return 'destination_charging_required'
            else:
                return 'no_range_constraint'
        
        return data.apply(classify_range, axis=1)
    
    def _classify_adoption_constraints(self, data: gpd.GeoDataFrame) -> pd.Series:
        """Classify adoption-related constraints"""
        def classify_adoption(row):
            adoption_score = row.get('combined_adoption_propensity', 0.5)
            
            if adoption_score < 0.2:
                return 'very_low_adoption'
            elif adoption_score < self.constraint_thresholds['adoption_low']:
                return 'low_adoption'
            elif adoption_score < 0.6:
                return 'moderate_adoption'
            else:
                return 'high_adoption'
        
        return data.apply(classify_adoption, axis=1)
    
    def _classify_charging_constraints(self, data: gpd.GeoDataFrame) -> pd.Series:
        """Classify charging infrastructure constraints"""
        def classify_charging(row):
            accessibility = row.get('combined_accessibility_score', 0.5)
            reliance = row.get('combined_charging_reliance', 0.5)
            
            if accessibility < 0.3 and reliance > 0.8:
                return 'severe_charging_constraint'
            elif accessibility < self.constraint_thresholds['accessibility_poor']:
                return 'poor_accessibility'
            elif reliance > self.constraint_thresholds['charging_reliance_high']:
                return 'high_public_reliance'
            else:
                return 'adequate_charging'
        
        return data.apply(classify_charging, axis=1)
    
    def _identify_primary_bottleneck(self, row: pd.Series) -> str:
        """Identify the primary bottleneck limiting EV feasibility"""
        # Score each constraint type (lower score = bigger bottleneck)
        range_score = 1.0 if row.get('range_constraint_type') == 'no_range_constraint' else 0.0
        adoption_score = row.get('combined_adoption_propensity', 0.5)
        charging_score = row.get('combined_accessibility_score', 0.5)
        
        # Find the lowest scoring constraint
        constraint_scores = {
            'range': range_score,
            'adoption': adoption_score,
            'charging': charging_score
        }
        
        primary_bottleneck = min(constraint_scores, key=constraint_scores.get)
        
        # Add severity qualifier
        min_score = constraint_scores[primary_bottleneck]
        if min_score < 0.2:
            return f'severe_{primary_bottleneck}_bottleneck'
        elif min_score < 0.4:
            return f'moderate_{primary_bottleneck}_bottleneck'
        else:
            return f'minor_{primary_bottleneck}_bottleneck'
    
    def _calculate_bottleneck_severity(self, row: pd.Series) -> float:
        """Calculate overall bottleneck severity (0=no bottlenecks, 1=severe bottlenecks)"""
        # Get constraint scores
        range_feasible = 1.0 if row.get('range_constraint_type') == 'no_range_constraint' else 0.3
        adoption_score = row.get('combined_adoption_propensity', 0.5)
        charging_score = row.get('combined_accessibility_score', 0.5)
        
        # Calculate severity as inverse of minimum constraint score
        min_constraint_score = min(range_feasible, adoption_score, charging_score)
        severity = 1.0 - min_constraint_score
        
        return np.clip(severity, 0.0, 1.0)
    
    def _classify_intervention_priority(self, data: gpd.GeoDataFrame) -> pd.Series:
        """Classify intervention priority based on bottleneck severity and trip volume"""
        def classify_priority(row):
            severity = row.get('bottleneck_severity', 0.5)
            trip_volume = row.get('base_trip_volume', 0)
            
            # Normalize trip volume (assuming max volume around 1000)
            volume_score = min(1.0, trip_volume / 1000.0)
            
            # Combined priority score
            priority_score = (severity * 0.7) + (volume_score * 0.3)
            
            if priority_score > 0.8:
                return 'high_priority'
            elif priority_score > 0.6:
                return 'medium_priority'
            elif priority_score > 0.4:
                return 'low_priority'
            else:
                return 'minimal_priority'
        
        return data.apply(classify_priority, axis=1)
    
    def validate_constraint_classification(self, classified_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate constraint classification accuracy and completeness.
        
        Args:
            classified_data: GeoDataFrame with constraint classification results
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Validating constraint classification")
        
        validation_results = {
            'is_valid': True,
            'issues': [],
            'classification_summary': {},
            'consistency_checks': {}
        }
        
        # Check for required classification columns
        required_cols = ['range_constraint_type', 'adoption_constraint_type', 
                        'charging_constraint_type', 'primary_bottleneck']
        missing_cols = [col for col in required_cols if col not in classified_data.columns]
        if missing_cols:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing classification columns: {missing_cols}")
        
        # Validate constraint type consistency
        consistency_issues = 0
        for _, row in classified_data.iterrows():
            # Check range constraint consistency
            distance = row.get('route_distance_km', 0)
            range_constraint = row.get('range_constraint_type', '')
            
            if distance > 100 and range_constraint != 'infeasible_range':
                consistency_issues += 1
            elif distance <= 50 and range_constraint not in ['no_range_constraint']:
                consistency_issues += 1
        
        if consistency_issues > 0:
            validation_results['issues'].append(f"{consistency_issues} constraint classification inconsistencies")
        
        # Generate classification summary
        validation_results['classification_summary'] = {
            'range_constraints': classified_data.get('range_constraint_type', pd.Series()).value_counts().to_dict(),
            'adoption_constraints': classified_data.get('adoption_constraint_type', pd.Series()).value_counts().to_dict(),
            'charging_constraints': classified_data.get('charging_constraint_type', pd.Series()).value_counts().to_dict(),
            'primary_bottlenecks': classified_data.get('primary_bottleneck', pd.Series()).value_counts().to_dict(),
            'intervention_priorities': classified_data.get('intervention_priority', pd.Series()).value_counts().to_dict()
        }
        
        # Consistency checks
        validation_results['consistency_checks'] = {
            'total_records_checked': len(classified_data),
            'consistency_issues_found': consistency_issues,
            'classification_completeness': {
                col: (classified_data[col].notna().sum() / len(classified_data))
                for col in required_cols if col in classified_data.columns
            }
        }
        
        if validation_results['issues']:
            validation_results['is_valid'] = False
        
        logger.info(f"Constraint classification validation {'passed' if validation_results['is_valid'] else 'failed'}")
        
        return validation_results


def validate_integrated_feasibility_assessment(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate complete integrated feasibility assessment pipeline.
    
    Args:
        analysis_results: Dictionary with complete analysis results
        
    Returns:
        Dictionary with comprehensive validation results
    """
    logger.info("Validating integrated feasibility assessment pipeline")
    
    validation_results = {
        'is_valid': True,
        'component_validations': {},
        'pipeline_issues': [],
        'overall_statistics': {}
    }
    
    # Validate classified trips data
    if 'classified_trips' in analysis_results:
        classified_data = analysis_results['classified_trips']
        
        # Basic structure validation
        required_cols = ['feasibility_category', 'overall_feasibility_score', 'limiting_constraints']
        missing_cols = [col for col in required_cols if col not in classified_data.columns]
        if missing_cols:
            validation_results['pipeline_issues'].append(f"Missing required columns in classified trips: {missing_cols}")
        
        # Validate feasibility score distribution
        if 'overall_feasibility_score' in classified_data.columns:
            scores = classified_data['overall_feasibility_score']
            score_stats = {
                'mean': scores.mean(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                'out_of_range_count': ((scores < 0) | (scores > 1)).sum()
            }
            
            if score_stats['out_of_range_count'] > 0:
                validation_results['pipeline_issues'].append("Feasibility scores outside valid range [0,1]")
            
            validation_results['overall_statistics']['feasibility_scores'] = score_stats
    
    # Validate analysis summary
    if 'analysis_summary' in analysis_results:
        summary = analysis_results['analysis_summary']
        
        # Check summary completeness
        required_summary_fields = ['total_trips_analyzed', 'feasible_trips', 'constrained_trips', 'infeasible_trips']
        missing_fields = [field for field in required_summary_fields if not hasattr(summary, field)]
        if missing_fields:
            validation_results['pipeline_issues'].append(f"Missing summary fields: {missing_fields}")
        
        # Validate trip counts consistency
        if hasattr(summary, 'total_trips_analyzed'):
            total_classified = summary.feasible_trips + summary.constrained_trips + summary.infeasible_trips
            if total_classified != summary.total_trips_analyzed:
                validation_results['pipeline_issues'].append("Trip count inconsistency in summary")
    
    # Validate component results
    if 'validation_results' in analysis_results:
        component_validation = analysis_results['validation_results']
        validation_results['component_validations']['integrated_analysis'] = component_validation
        
        if not component_validation.get('is_valid', False):
            validation_results['is_valid'] = False
    
    # Overall pipeline validation
    if validation_results['pipeline_issues']:
        validation_results['is_valid'] = False
    
    logger.info(f"Integrated feasibility assessment validation {'passed' if validation_results['is_valid'] else 'failed'}")

    return validation_results


# =============================================================================
# 5-Step Replaceability Logic Functions (per eq_revised.qmd)
# =============================================================================

def calculate_total_replaceable_trips(
    N_ij: pd.Series,
    Class_ij: pd.Series
) -> float:
    """
    Calculate total replaceable trips R_total per @eq-total-replaceable.

    R_total = Σ N_ij × 𝟙{Class_ij ≠ "infeasible"}

    Args:
        N_ij: Converted trips per OD pair
        Class_ij: Trip classification series

    Returns:
        Total replaceable trips (excludes infeasible)
    """
    feasible_mask = Class_ij != "infeasible"
    return (N_ij * feasible_mask.astype(int)).sum()


def calculate_replacement_rate(
    R_total: float,
    T_total: float
) -> float:
    """
    Calculate replacement rate RR per @eq-replacement-rate.

    RR = R_total / T_total × 100%

    Args:
        R_total: Total replaceable trips
        T_total: Total baseline trips

    Returns:
        Replacement rate as percentage [0-100]
    """
    if T_total <= 0:
        return 0.0
    return (R_total / T_total) * 100.0


def calculate_vehicle_distribution(
    N_ij: pd.Series,
    Class_ij: pd.Series,
    EV_type_i: pd.Series
) -> Dict[str, float]:
    """
    Calculate vehicle type distribution Q_type per @eq-vehicle-distribution.

    Q_type = Σ N_ij × 𝟙{Class_ij ≠ "infeasible" AND EV_type_i = type} / R_total

    Args:
        N_ij: Converted trips per OD pair
        Class_ij: Trip classification series
        EV_type_i: EV type assignment series

    Returns:
        Dictionary with distribution by type (proportions summing to 1)
    """
    feasible_mask = Class_ij != "infeasible"
    R_total = (N_ij * feasible_mask.astype(int)).sum()

    if R_total <= 0:
        return {"2-seater": 0.0, "4-seater": 0.0, "mixed": 0.0}

    distribution = {}
    for ev_type in ["2-seater", "4-seater", "mixed"]:
        type_mask = (EV_type_i == ev_type) & feasible_mask
        type_trips = (N_ij * type_mask.astype(int)).sum()
        distribution[ev_type] = type_trips / R_total

    return distribution


def calculate_class_distribution(
    Class_ij: pd.Series,
    N_ij: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Calculate trip classification distribution for reporting.

    Args:
        Class_ij: Trip classification series
        N_ij: Optional trip volumes for weighted counts

    Returns:
        Dictionary with counts and percentages by class
    """
    if N_ij is not None:
        # Weighted by trip volume
        total = N_ij.sum()
        imm_trips = N_ij[Class_ij == "immediately_feasible"].sum()
        const_trips = N_ij[Class_ij == "constrained"].sum()
        inf_trips = N_ij[Class_ij == "infeasible"].sum()
    else:
        # Unweighted counts
        total = len(Class_ij)
        imm_trips = (Class_ij == "immediately_feasible").sum()
        const_trips = (Class_ij == "constrained").sum()
        inf_trips = (Class_ij == "infeasible").sum()

    return {
        "immediately_feasible": {
            "count": imm_trips,
            "percentage": (imm_trips / total * 100) if total > 0 else 0
        },
        "constrained": {
            "count": const_trips,
            "percentage": (const_trips / total * 100) if total > 0 else 0
        },
        "infeasible": {
            "count": inf_trips,
            "percentage": (inf_trips / total * 100) if total > 0 else 0
        },
        "total": total
    }
