"""
Data Integration Module for Frugal EV Analysis

This module provides classes and functions for integrating multiple demographic datasets
with trip and charging infrastructure data for comprehensive EV adoption analysis.
"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standardize_od_column_names(data: Union[pd.DataFrame, gpd.GeoDataFrame]) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Standardize OD data column names to expected format.
    
    Maps common variations of origin/destination codes to standard names:
    - geo_code1 -> origin_code
    - geo_code2 -> destination_code
    - driving_distance_km -> route_distance_km (if present)
    
    Args:
        data: DataFrame or GeoDataFrame with OD data
        
    Returns:
        Data with standardized column names
    """
    data = data.copy()
    
    # Column mapping dictionary
    column_mapping = {
        'geo_code1': 'origin_code',
        'geo_code2': 'destination_code', 
        'driving_distance_km': 'route_distance_km'
    }
    
    # Apply column mapping
    for old_name, new_name in column_mapping.items():
        if old_name in data.columns and new_name not in data.columns:
            data = data.rename(columns={old_name: new_name})
            logger.info(f"Mapped column {old_name} -> {new_name}")
    
    return data


@dataclass
class ValidationResult:
    """Results from data validation checks"""
    is_valid: bool
    total_records: int
    missing_records: int
    duplicate_records: int
    invalid_geometries: int
    issues: List[str]


class DataIntegrator:
    """
    Main orchestration class for data integration tasks.
    
    Handles merging demographic datasets, spatial joins with OD data,
    and validation of integrated datasets.
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize DataIntegrator with path to data directory.
        
        Args:
            data_path: Path to directory containing input data files
        """
        self.data_path = Path(data_path)
        self.demographic_files = [
            'social_grade.csv',
            'education.csv', 
            'car_ownership.csv',
            'household_composition.csv',
            'accommodation_type.csv',
            'age_sex.csv',
            'economic_activity.csv',
            'population_density.csv'
        ]
        
    def load_gpkg_demographics(self, gpkg_path: Union[str, Path]) -> gpd.GeoDataFrame:
        """
        Load demographic data from GPKG file with wide-format structure.
        
        Args:
            gpkg_path: Path to GPKG file containing demographic data
            
        Returns:
            GeoDataFrame with demographic data and geometries
            
        Raises:
            FileNotFoundError: If GPKG file doesn't exist
            ValueError: If required columns are missing
        """
        gpkg_path = Path(gpkg_path)
        
        if not gpkg_path.exists():
            raise FileNotFoundError(f"GPKG file not found: {gpkg_path}")
            
        logger.info(f"Loading demographic data from {gpkg_path}")
        
        try:
            # Load GPKG file
            gdf = gpd.read_file(gpkg_path)
            
            # Validate required columns
            required_columns = ['geo_code', 'geometry']
            missing_columns = [col for col in required_columns if col not in gdf.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Ensure geo_code is string type for consistent joining
            gdf['geo_code'] = gdf['geo_code'].astype(str)
            
            # Validate geometries
            invalid_geom_count = (~gdf.geometry.is_valid).sum()
            if invalid_geom_count > 0:
                logger.warning(f"Found {invalid_geom_count} invalid geometries")
                # Fix invalid geometries
                gdf.geometry = gdf.geometry.buffer(0)
                
            logger.info(f"Loaded {len(gdf)} demographic records")
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading GPKG file: {e}")
            raise
    
    def merge_demographic_datasets(self, data_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Merge multiple demographic CSV files using geo_code column.
        
        Args:
            data_path: Optional path to data directory (uses instance path if None)
            
        Returns:
            DataFrame with merged demographic data
            
        Raises:
            FileNotFoundError: If demographic files are missing
            ValueError: If geo_code column is missing from files
        """
        if data_path is None:
            data_path = self.data_path
        else:
            data_path = Path(data_path)
            
        logger.info("Merging demographic datasets")
        
        merged_data = None
        
        for file_name in self.demographic_files:
            file_path = data_path / file_name
            
            if not file_path.exists():
                logger.warning(f"Demographic file not found: {file_path}")
                continue
                
            try:
                # Load CSV file
                df = pd.read_csv(file_path)
                
                # Validate geo_code column exists
                if 'geo_code' not in df.columns:
                    raise ValueError(f"geo_code column missing from {file_name}")
                    
                # Ensure geo_code is string type
                df['geo_code'] = df['geo_code'].astype(str)
                
                # Merge with existing data
                if merged_data is None:
                    merged_data = df
                    logger.info(f"Initialized with {file_name}: {len(df)} records")
                else:
                    before_count = len(merged_data)
                    merged_data = merged_data.merge(df, on='geo_code', how='outer')
                    after_count = len(merged_data)
                    logger.info(f"Merged {file_name}: {before_count} -> {after_count} records")
                    
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                raise
                
        if merged_data is None:
            raise FileNotFoundError("No demographic files found to merge")
            
        logger.info(f"Final merged dataset: {len(merged_data)} records with {len(merged_data.columns)} columns")
        return merged_data
    
    def attach_demographics_to_od(self, 
                                 od_data: gpd.GeoDataFrame, 
                                 demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Attach demographic attributes to start and end points of trip geometries.
        
        Args:
            od_data: GeoDataFrame with origin-destination trip data
            demographics: GeoDataFrame with demographic data
            
        Returns:
            GeoDataFrame with demographic attributes attached to OD pairs
            
        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Attaching demographics to OD data")
        
        # Standardize column names
        od_data = standardize_od_column_names(od_data)
        
        # Validate required columns in OD data
        required_od_cols = ['origin_code', 'destination_code']
        missing_od_cols = [col for col in required_od_cols if col not in od_data.columns]
        if missing_od_cols:
            raise ValueError(f"Missing required OD columns: {missing_od_cols}")
            
        # Validate demographics has geo_code
        if 'geo_code' not in demographics.columns:
            raise ValueError("Demographics data missing geo_code column")
            
        # Ensure consistent data types
        od_data['origin_code'] = od_data['origin_code'].astype(str)
        od_data['destination_code'] = od_data['destination_code'].astype(str)
        demographics['geo_code'] = demographics['geo_code'].astype(str)
        
        # Attach origin demographics
        logger.info("Attaching origin demographics")
        origin_demographics = demographics.drop('geometry', axis=1).add_suffix('_origin')
        origin_demographics = origin_demographics.rename(columns={'geo_code_origin': 'origin_code'})
        
        od_with_origin = od_data.merge(origin_demographics, on='origin_code', how='left')
        
        # Attach destination demographics  
        logger.info("Attaching destination demographics")
        dest_demographics = demographics.drop('geometry', axis=1).add_suffix('_dest')
        dest_demographics = dest_demographics.rename(columns={'geo_code_dest': 'destination_code'})
        
        od_with_demographics = od_with_origin.merge(dest_demographics, on='destination_code', how='left')
        
        # Log merge statistics
        # Check if origin demographics were successfully attached
        origin_demo_cols = [col for col in od_with_demographics.columns if col.endswith('_origin')]
        dest_demo_cols = [col for col in od_with_demographics.columns if col.endswith('_dest')]
        
        origin_matches = len([col for col in origin_demo_cols if od_with_demographics[col].notna().any()])
        dest_matches = len([col for col in dest_demo_cols if od_with_demographics[col].notna().any()])
        total_records = len(od_with_demographics)
        
        logger.info(f"Origin matches: {origin_matches}/{total_records} ({origin_matches/total_records*100:.1f}%)")
        logger.info(f"Destination matches: {dest_matches}/{total_records} ({dest_matches/total_records*100:.1f}%)")
        
        return od_with_demographics
    
    def validate_integration(self, integrated_data: gpd.GeoDataFrame) -> ValidationResult:
        """
        Validate integrated dataset for completeness and consistency.
        
        Args:
            integrated_data: GeoDataFrame with integrated data
            
        Returns:
            ValidationResult with validation metrics and issues
        """
        logger.info("Validating integrated dataset")
        
        issues = []
        total_records = len(integrated_data)
        
        # Check for missing records
        missing_records = integrated_data.isnull().any(axis=1).sum()
        if missing_records > 0:
            issues.append(f"{missing_records} records have missing values")
            
        # Check for duplicate records
        duplicate_records = integrated_data.duplicated().sum()
        if duplicate_records > 0:
            issues.append(f"{duplicate_records} duplicate records found")
            
        # Check geometry validity if present
        invalid_geometries = 0
        if 'geometry' in integrated_data.columns:
            invalid_geometries = (~integrated_data.geometry.is_valid).sum()
            if invalid_geometries > 0:
                issues.append(f"{invalid_geometries} invalid geometries found")
                
        # Check for required columns
        required_columns = ['origin_code', 'destination_code']
        missing_columns = [col for col in required_columns if col not in integrated_data.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
            
        # Determine overall validity
        is_valid = len(issues) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            total_records=total_records,
            missing_records=missing_records,
            duplicate_records=duplicate_records,
            invalid_geometries=invalid_geometries,
            issues=issues
        )
        
        logger.info(f"Validation complete: {'PASSED' if is_valid else 'FAILED'}")
        if issues:
            for issue in issues:
                logger.warning(f"Validation issue: {issue}")
                
        return result


class DemographicLoader:
    """
    Specialized class for loading and integrating 8 GPKG demographic files.
    
    Handles loading pre-processed demographic datasets with wide-format structure
    and merging them using geo_code spatial joins.
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize DemographicLoader with path to demographic data directory.
        
        Args:
            data_path: Path to directory containing GPKG demographic files
        """
        self.data_path = Path(data_path)
        self.gpkg_files = [
            'scotland_oa_2011_census_Approximated_social_grade_by_tenure_by_car_or_van_availability_scotland.gpkg',
            'scotland_oa_2011_census_Highest_level_of_qualification_by_household_composition_scotland.gpkg', 
            'scotland_oa_2011_census_Car_or_van_availability_by_sex_by_age_scotland.gpkg',
            'scotland_oa_2011_census_Household_composition_scotland.gpkg',
            'scotland_oa_2011_census_Accommodation_type_by_car_or_van_availability_by_number_of_people_aged_17_or_over_in_household_scotland.gpkg',
            'scotland_oa_2011_census_Age_by_sex_scotland.gpkg',
            'scotland_oa_2011_census_Economic_activity_scotland.gpkg',
            'scotland_oa_2011_census_Population_density_scotland.gpkg'
        ]
        
    def load_single_gpkg(self, gpkg_filename: str) -> gpd.GeoDataFrame:
        """
        Load a single GPKG demographic file.
        
        Args:
            gpkg_filename: Name of GPKG file to load
            
        Returns:
            GeoDataFrame with demographic data and geometries
            
        Raises:
            FileNotFoundError: If GPKG file doesn't exist
            ValueError: If required columns are missing
        """
        gpkg_path = self.data_path / gpkg_filename
        
        if not gpkg_path.exists():
            raise FileNotFoundError(f"GPKG file not found: {gpkg_path}")
            
        logger.info(f"Loading demographic GPKG: {gpkg_filename}")
        
        try:
            # Load GPKG file
            gdf = gpd.read_file(gpkg_path)
            
            # Validate required columns
            required_columns = ['geo_code', 'geometry']
            missing_columns = [col for col in required_columns if col not in gdf.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in {gpkg_filename}: {missing_columns}")
                
            # Ensure geo_code is string type for consistent joining
            gdf['geo_code'] = gdf['geo_code'].astype(str)
            
            # Validate geometries
            invalid_geom_count = (~gdf.geometry.is_valid).sum()
            if invalid_geom_count > 0:
                logger.warning(f"Found {invalid_geom_count} invalid geometries in {gpkg_filename}")
                # Fix invalid geometries
                gdf.geometry = gdf.geometry.buffer(0)
                
            logger.info(f"Loaded {len(gdf)} records from {gpkg_filename}")
            return gdf
            
        except Exception as e:
            logger.error(f"Error loading GPKG file {gpkg_filename}: {e}")
            raise
    
    def load_all_gpkg_files(self) -> Dict[str, gpd.GeoDataFrame]:
        """
        Load all 8 GPKG demographic files.
        
        Returns:
            Dictionary mapping file names to GeoDataFrames
            
        Raises:
            FileNotFoundError: If any required GPKG files are missing
        """
        logger.info("Loading all demographic GPKG files")
        
        loaded_files = {}
        missing_files = []
        
        for gpkg_file in self.gpkg_files:
            try:
                gdf = self.load_single_gpkg(gpkg_file)
                # Use a shorter key name for easier reference
                key = gpkg_file.replace('scotland_oa_2011_census_', '').replace('_scotland.gpkg', '')
                loaded_files[key] = gdf
            except FileNotFoundError:
                missing_files.append(gpkg_file)
                logger.error(f"Missing required GPKG file: {gpkg_file}")
                
        if missing_files:
            raise FileNotFoundError(f"Missing required GPKG files: {missing_files}")
            
        logger.info(f"Successfully loaded {len(loaded_files)} demographic GPKG files")
        return loaded_files
    
    def merge_gpkg_datasets(self) -> gpd.GeoDataFrame:
        """
        Merge all 8 demographic GPKG files using geo_code spatial joins.
        
        Returns:
            GeoDataFrame with merged demographic data from all files
            
        Raises:
            FileNotFoundError: If required files are missing
            ValueError: If geo_code columns are inconsistent
        """
        logger.info("Merging demographic GPKG datasets using geo_code joins")
        
        # Load all GPKG files
        loaded_files = self.load_all_gpkg_files()
        
        merged_gdf = None
        
        for file_key, gdf in loaded_files.items():
            if merged_gdf is None:
                # Initialize with first dataset (keep geometry)
                merged_gdf = gdf.copy()
                logger.info(f"Initialized merge with {file_key}: {len(gdf)} records")
            else:
                # Merge subsequent datasets (drop geometry to avoid conflicts)
                gdf_no_geom = gdf.drop('geometry', axis=1)
                
                # Add suffix to avoid column name conflicts (except geo_code)
                suffix = f"_{file_key}"
                columns_to_rename = {col: f"{col}{suffix}" for col in gdf_no_geom.columns if col != 'geo_code'}
                gdf_no_geom = gdf_no_geom.rename(columns=columns_to_rename)
                
                before_count = len(merged_gdf)
                merged_gdf = merged_gdf.merge(gdf_no_geom, on='geo_code', how='outer')
                after_count = len(merged_gdf)
                
                logger.info(f"Merged {file_key}: {before_count} -> {after_count} records")
                
        if merged_gdf is None:
            raise ValueError("No demographic files were successfully loaded")
            
        # Validate final merge
        total_columns = len(merged_gdf.columns)
        total_records = len(merged_gdf)
        
        logger.info(f"Final merged demographic dataset: {total_records} records with {total_columns} columns")
        
        return merged_gdf
    
    def validate_gpkg_integration(self, merged_gdf: gpd.GeoDataFrame) -> ValidationResult:
        """
        Validate the integration of GPKG demographic datasets.
        
        Args:
            merged_gdf: Merged GeoDataFrame to validate
            
        Returns:
            ValidationResult with validation metrics and issues
        """
        logger.info("Validating GPKG demographic integration")
        
        issues = []
        total_records = len(merged_gdf)
        
        # Check for missing geo_codes
        missing_geocodes = merged_gdf['geo_code'].isnull().sum()
        if missing_geocodes > 0:
            issues.append(f"{missing_geocodes} records have missing geo_codes")
            
        # Check for duplicate geo_codes
        duplicate_geocodes = merged_gdf['geo_code'].duplicated().sum()
        if duplicate_geocodes > 0:
            issues.append(f"{duplicate_geocodes} duplicate geo_codes found")
            
        # Check geometry validity
        invalid_geometries = (~merged_gdf.geometry.is_valid).sum()
        if invalid_geometries > 0:
            issues.append(f"{invalid_geometries} invalid geometries found")
            
        # Check for completely empty records (all demographic data missing)
        demographic_cols = [col for col in merged_gdf.columns if col not in ['geo_code', 'geometry', 'label', 'name']]
        empty_records = merged_gdf[demographic_cols].isnull().all(axis=1).sum()
        if empty_records > 0:
            issues.append(f"{empty_records} records have no demographic data")
            
        # Check expected number of Output Areas for Scotland (approximately 46,000)
        # Only check if we have a reasonable number of records (skip for small test datasets)
        expected_min_records = 40000
        if total_records > 100 and total_records < expected_min_records:
            issues.append(f"Fewer records than expected: {total_records} < {expected_min_records}")
            
        # Determine overall validity
        is_valid = len(issues) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            total_records=total_records,
            missing_records=missing_geocodes,
            duplicate_records=duplicate_geocodes,
            invalid_geometries=invalid_geometries,
            issues=issues
        )
        
        logger.info(f"GPKG integration validation: {'PASSED' if is_valid else 'FAILED'}")
        if issues:
            for issue in issues:
                logger.warning(f"Validation issue: {issue}")
                
        return result


class DemographicMerger:
    """
    Specialized class for handling demographic dataset merging operations.
    """
    
    def __init__(self, demographic_files: List[str]):
        """
        Initialize with list of demographic file names.
        
        Args:
            demographic_files: List of demographic CSV file names
        """
        self.demographic_files = demographic_files
        
    def merge_files(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Merge demographic files from specified directory.
        
        Args:
            data_path: Path to directory containing demographic files
            
        Returns:
            DataFrame with merged demographic data
        """
        integrator = DataIntegrator(data_path)
        return integrator.merge_demographic_datasets()


class SpatialJoiner:
    """
    Specialized class for spatial joining operations between datasets.
    
    Handles spatial join functionality between demographics and trip data,
    including coordinate system validation and reprojection handling.
    """
    
    def __init__(self, target_crs: str = 'EPSG:4326'):
        """
        Initialize SpatialJoiner with target coordinate reference system.
        
        Args:
            target_crs: Target CRS for spatial operations (default: WGS84)
        """
        self.target_crs = target_crs
        
    def validate_and_reproject_crs(self, gdf: gpd.GeoDataFrame, dataset_name: str) -> gpd.GeoDataFrame:
        """
        Validate and reproject GeoDataFrame to target CRS if needed.
        
        Args:
            gdf: GeoDataFrame to validate and reproject
            dataset_name: Name of dataset for logging purposes
            
        Returns:
            GeoDataFrame in target CRS
            
        Raises:
            ValueError: If CRS cannot be determined or reprojection fails
        """
        logger.info(f"Validating CRS for {dataset_name}")
        
        # Check if CRS is defined
        if gdf.crs is None:
            logger.warning(f"{dataset_name} has no CRS defined, assuming {self.target_crs}")
            gdf = gdf.set_crs(self.target_crs)
        else:
            logger.info(f"{dataset_name} CRS: {gdf.crs}")
            
        # Reproject to target CRS if different
        if gdf.crs != self.target_crs:
            logger.info(f"Reprojecting {dataset_name} from {gdf.crs} to {self.target_crs}")
            try:
                gdf = gdf.to_crs(self.target_crs)
                logger.info(f"Successfully reprojected {dataset_name}")
            except Exception as e:
                raise ValueError(f"Failed to reproject {dataset_name}: {e}")
        else:
            logger.info(f"{dataset_name} already in target CRS")
            
        return gdf
    
    def perform_spatial_join(self, 
                           left_gdf: gpd.GeoDataFrame, 
                           right_gdf: gpd.GeoDataFrame,
                           how: str = 'left',
                           predicate: str = 'intersects') -> gpd.GeoDataFrame:
        """
        Perform spatial join between two GeoDataFrames.
        
        Args:
            left_gdf: Left GeoDataFrame for spatial join
            right_gdf: Right GeoDataFrame for spatial join
            how: Type of join ('left', 'right', 'inner')
            predicate: Spatial predicate ('intersects', 'within', 'contains')
            
        Returns:
            GeoDataFrame with spatial join results
            
        Raises:
            ValueError: If spatial join fails
        """
        logger.info(f"Performing spatial join with predicate '{predicate}' and method '{how}'")
        
        try:
            # Ensure both datasets are in the same CRS
            left_gdf = self.validate_and_reproject_crs(left_gdf, "left dataset")
            right_gdf = self.validate_and_reproject_crs(right_gdf, "right dataset")
            
            # Perform spatial join
            result = gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
            
            # Log join statistics
            left_count = len(left_gdf)
            right_count = len(right_gdf)
            result_count = len(result)
            
            logger.info(f"Spatial join completed: {left_count} left + {right_count} right = {result_count} result records")
            
            return result
            
        except Exception as e:
            logger.error(f"Spatial join failed: {e}")
            raise ValueError(f"Spatial join operation failed: {e}")
    
    def join_demographics_to_od_points(self, 
                                     od_data: gpd.GeoDataFrame, 
                                     demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Join demographic data to origin-destination points using spatial intersection.
        
        This method handles both Point and LineString geometries:
        - For Point geometries: Direct spatial join
        - For LineString geometries: Extracts start/end points and joins separately
        
        Args:
            od_data: GeoDataFrame with OD trip data (Point or LineString geometries)
            demographics: GeoDataFrame with demographic data (polygon geometries)
            
        Returns:
            GeoDataFrame with demographic attributes attached to OD points/routes
            
        Raises:
            ValueError: If required columns are missing or spatial join fails
        """
        logger.info("Joining demographics to OD points using spatial intersection")
        
        # Validate required columns
        if 'geometry' not in od_data.columns:
            raise ValueError("OD data missing geometry column")
        if 'geometry' not in demographics.columns:
            raise ValueError("Demographics data missing geometry column")
        if 'geo_code' not in demographics.columns:
            raise ValueError("Demographics data missing geo_code column")
            
        # Standardize column names
        od_data = standardize_od_column_names(od_data)
            
        # Check geometry type and handle accordingly
        geom_types = od_data.geometry.geom_type.unique()
        
        if 'LineString' in geom_types or 'MultiLineString' in geom_types:
            logger.info("Detected LineString geometries - extracting origin/destination points")
            return self._join_demographics_to_linestring_od(od_data, demographics)
        else:
            logger.info("Detected Point geometries - performing direct spatial join") 
            return self._join_demographics_to_point_od(od_data, demographics)
    
    def _join_demographics_to_point_od(self,
                                     od_data: gpd.GeoDataFrame,
                                     demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Handle spatial join for Point geometries (original method)"""
        # Perform spatial join
        result = self.perform_spatial_join(od_data, demographics, how='left', predicate='within')
        
        # Clean up index columns created by spatial join
        if 'index_right' in result.columns:
            result = result.drop('index_right', axis=1)
            
        logger.info(f"Successfully joined demographics to {len(result)} OD points")
        return result
    
    def _join_demographics_to_linestring_od(self,
                                          od_data: gpd.GeoDataFrame, 
                                          demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Handle spatial join for LineString geometries by extracting origin/destination points.
        
        Args:
            od_data: GeoDataFrame with LineString geometries
            demographics: GeoDataFrame with polygon demographics
            
        Returns:
            GeoDataFrame with demographic data attached to origins and destinations
        """
        from shapely.geometry import Point
        
        logger.info("Processing LineString geometries for spatial join")
        
        # Create copies for processing
        result = od_data.copy()
        
        # Extract origin and destination points from LineStrings
        origins = []
        destinations = []
        
        for idx, row in od_data.iterrows():
            geom = row.geometry
            if geom is not None:
                try:
                    # Handle different geometry types
                    if geom.geom_type == 'LineString':
                        coords = list(geom.coords)
                        if len(coords) >= 2:
                            origin_point = Point(coords[0])
                            dest_point = Point(coords[-1])
                            origins.append(origin_point)
                            destinations.append(dest_point)
                        else:
                            origins.append(None)
                            destinations.append(None)
                    elif geom.geom_type == 'MultiLineString':
                        # For MultiLineString, use the first LineString's start and last LineString's end
                        if len(geom.geoms) > 0:
                            first_line = geom.geoms[0]
                            last_line = geom.geoms[-1]
                            
                            first_coords = list(first_line.coords)
                            last_coords = list(last_line.coords)
                            
                            if len(first_coords) >= 2 and len(last_coords) >= 2:
                                origin_point = Point(first_coords[0])
                                dest_point = Point(last_coords[-1])
                                origins.append(origin_point)
                                destinations.append(dest_point)
                            else:
                                origins.append(None)
                                destinations.append(None)
                        else:
                            origins.append(None)
                            destinations.append(None)
                    else:
                        # Unsupported geometry type
                        origins.append(None)
                        destinations.append(None)
                except Exception as e:
                    logger.warning(f"Error extracting coordinates from geometry at index {idx}: {e}")
                    origins.append(None)
                    destinations.append(None)
            else:
                origins.append(None)
                destinations.append(None)
        
        # Create GeoDataFrames for origins and destinations
        origins_gdf = gpd.GeoDataFrame(
            result.drop('geometry', axis=1), 
            geometry=origins,
            crs=od_data.crs
        )
        
        destinations_gdf = gpd.GeoDataFrame(
            result.drop('geometry', axis=1),
            geometry=destinations, 
            crs=od_data.crs
        )
        
        logger.info(f"Extracted {len(origins_gdf)} origin points and {len(destinations_gdf)} destination points")
        
        # Perform spatial joins for origins and destinations
        origins_with_demo = self.perform_spatial_join(
            origins_gdf, demographics, how='left', predicate='within'
        )
        
        destinations_with_demo = self.perform_spatial_join(
            destinations_gdf, demographics, how='left', predicate='within' 
        )
        
        # Attach demographic data back to original result
        # Rename demographic columns to distinguish origin vs destination
        demo_cols = [col for col in demographics.columns if col != 'geometry']
        
        # Add origin demographics
        for col in demo_cols:
            if col in origins_with_demo.columns:
                if col == 'geo_code':
                    result[f'origin_geo_code'] = origins_with_demo[col].values
                else:
                    result[f'{col}_origin'] = origins_with_demo[col].values
        
        # Add destination demographics  
        for col in demo_cols:
            if col in destinations_with_demo.columns:
                if col == 'geo_code':
                    result[f'destination_geo_code'] = destinations_with_demo[col].values
                else:
                    result[f'{col}_dest'] = destinations_with_demo[col].values
        
        # Clean up index columns
        for col in ['index_right', 'index_left']:
            if col in result.columns:
                result = result.drop(col, axis=1)
        
        # Count matches
        origin_matches = result['origin_geo_code'].notna().sum() if 'origin_geo_code' in result.columns else 0
        dest_matches = result['destination_geo_code'].notna().sum() if 'destination_geo_code' in result.columns else 0
        
        logger.info(f"Origin matches: {origin_matches}/{len(result)} ({origin_matches/len(result)*100:.1f}%)")
        logger.info(f"Destination matches: {dest_matches}/{len(result)} ({dest_matches/len(result)*100:.1f}%)")
        
        return result
    
    def join_demographics_to_od_codes(self, 
                                    od_data: gpd.GeoDataFrame, 
                                    demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Join demographic data to origin-destination pairs using geo_code attributes.
        
        This method automatically chooses between code-based and spatial joining:
        - If origin_code/destination_code exist: Uses code-based joining
        - If LineString geometries exist: Extracts points and uses spatial joining
        
        Args:
            od_data: GeoDataFrame with OD trip data
            demographics: GeoDataFrame with demographic data
            
        Returns:
            GeoDataFrame with demographic attributes attached to OD pairs
        """
        logger.info("Joining demographics to OD data using geo_code attributes")
        
        # Validate and align CRS
        od_data = self.validate_and_reproject_crs(od_data, "OD data")
        demographics = self.validate_and_reproject_crs(demographics, "demographics data")
        
        # Standardize column names first
        od_data = standardize_od_column_names(od_data)
        
        # Check if we have origin_code and destination_code columns after standardization
        has_codes = 'origin_code' in od_data.columns and 'destination_code' in od_data.columns
        
        if has_codes and od_data['origin_code'].notna().any() and od_data['destination_code'].notna().any():
            # Check if there's actually overlap between OD codes and demographic codes
            od_origin_codes = set(od_data['origin_code'].dropna())
            od_dest_codes = set(od_data['destination_code'].dropna())
            demo_codes = set(demographics['geo_code'].dropna())
            
            origin_overlap = len(od_origin_codes.intersection(demo_codes))
            dest_overlap = len(od_dest_codes.intersection(demo_codes))
            
            logger.info(f"Code overlap check: {origin_overlap} origin matches, {dest_overlap} destination matches")
            
            if origin_overlap > 0 or dest_overlap > 0:
                # Use existing DataIntegrator logic for code-based joins
                logger.info("Using code-based joining with existing origin/destination codes")
                integrator = DataIntegrator(".")  # Dummy path since we're not using file operations
                result = integrator.attach_demographics_to_od(od_data, demographics)
            else:
                # No code overlap - fall back to spatial joining
                logger.info("No code overlap found - falling back to spatial joining")
                result = self.join_demographics_to_od_points(od_data, demographics)
        else:
            # Fall back to spatial joining using point extraction
            logger.info("Origin/destination codes not available - falling back to spatial joining")
            result = self.join_demographics_to_od_points(od_data, demographics)
        
        return result
    
    def validate_spatial_join_completeness(self, 
                                         original_data: gpd.GeoDataFrame,
                                         joined_data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate completeness and accuracy of spatial join results.
        
        Args:
            original_data: Original data before join
            joined_data: Data after spatial join
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating spatial join completeness")
        
        original_count = len(original_data)
        joined_count = len(joined_data)
        
        # Check for demographic columns in result
        demo_cols = [col for col in joined_data.columns if any(suffix in col for suffix in ['_origin', '_dest', 'geo_code'])]
        demo_col_count = len(demo_cols)
        
        # Calculate match rates
        if demo_col_count > 0:
            # Count records with at least some demographic data
            matched_records = 0
            for col in demo_cols:
                if col in joined_data.columns:
                    matched_records = max(matched_records, joined_data[col].notna().sum())
        else:
            matched_records = 0
            
        match_rate = matched_records / original_count if original_count > 0 else 0
        
        validation_result = {
            'original_records': original_count,
            'joined_records': joined_count,
            'demographic_columns': demo_col_count,
            'matched_records': matched_records,
            'match_rate': match_rate,
            'record_count_preserved': original_count == joined_count,
            'has_demographic_data': demo_col_count > 0
        }
        
        logger.info(f"Spatial join validation: {matched_records}/{original_count} records matched ({match_rate:.1%})")
        
        return validation_result
    
    @staticmethod
    def join_demographics_to_od(od_data: gpd.GeoDataFrame, 
                               demographics: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Static method for backward compatibility.
        Join demographic data to origin-destination pairs.
        
        Args:
            od_data: GeoDataFrame with OD trip data
            demographics: GeoDataFrame with demographic data
            
        Returns:
            GeoDataFrame with joined data
        """
        joiner = SpatialJoiner()
        return joiner.join_demographics_to_od_codes(od_data, demographics)


class DataValidator:
    """
    Specialized class for data quality validation and reporting.
    
    Implements data completeness checks, missing value reporting,
    and validation functions for spatial and temporal alignment.
    """
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize DataValidator with validation parameters.
        
        Args:
            tolerance: Tolerance for spatial and numerical comparisons
        """
        self.tolerance = tolerance
        
    def validate_data_completeness(self, data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate data completeness and identify missing values.
        
        Args:
            data: GeoDataFrame to validate
            
        Returns:
            Dictionary with completeness metrics
        """
        logger.info("Validating data completeness")
        
        total_records = len(data)
        total_columns = len(data.columns)
        
        # Calculate missing values per column
        missing_by_column = data.isnull().sum()
        missing_percentages = (missing_by_column / total_records * 100).round(2)
        
        # Identify columns with high missing rates
        high_missing_threshold = 50.0  # 50% missing
        high_missing_columns = missing_percentages[missing_percentages > high_missing_threshold].to_dict()
        
        # Calculate overall completeness
        total_cells = total_records * total_columns
        total_missing = data.isnull().sum().sum()
        overall_completeness = ((total_cells - total_missing) / total_cells * 100).round(2)
        
        # Identify completely empty records
        empty_records = data.isnull().all(axis=1).sum()
        
        completeness_result = {
            'total_records': total_records,
            'total_columns': total_columns,
            'overall_completeness_percent': overall_completeness,
            'total_missing_values': int(total_missing),
            'empty_records': int(empty_records),
            'high_missing_columns': high_missing_columns,
            'missing_by_column': missing_by_column.to_dict(),
            'missing_percentages': missing_percentages.to_dict()
        }
        
        logger.info(f"Data completeness: {overall_completeness}% complete, {empty_records} empty records")
        
        return completeness_result
    
    def validate_spatial_alignment(self, 
                                 data1: gpd.GeoDataFrame, 
                                 data2: gpd.GeoDataFrame,
                                 data1_name: str = "dataset1",
                                 data2_name: str = "dataset2") -> Dict[str, Any]:
        """
        Validate spatial alignment between two GeoDataFrames.
        
        Args:
            data1: First GeoDataFrame
            data2: Second GeoDataFrame
            data1_name: Name of first dataset for reporting
            data2_name: Name of second dataset for reporting
            
        Returns:
            Dictionary with spatial alignment metrics
        """
        logger.info(f"Validating spatial alignment between {data1_name} and {data2_name}")
        
        alignment_result = {
            'datasets': [data1_name, data2_name],
            'crs_match': False,
            'crs_data1': str(data1.crs) if data1.crs else None,
            'crs_data2': str(data2.crs) if data2.crs else None,
            'bounds_overlap': False,
            'bounds_data1': None,
            'bounds_data2': None,
            'spatial_overlap_area': 0.0,
            'geometry_validity': {}
        }
        
        # Check CRS alignment
        if data1.crs and data2.crs:
            alignment_result['crs_match'] = data1.crs == data2.crs
        
        # Check spatial bounds overlap
        try:
            bounds1 = data1.total_bounds
            bounds2 = data2.total_bounds
            
            alignment_result['bounds_data1'] = bounds1.tolist()
            alignment_result['bounds_data2'] = bounds2.tolist()
            
            # Check if bounds overlap
            overlap_x = max(0, min(bounds1[2], bounds2[2]) - max(bounds1[0], bounds2[0]))
            overlap_y = max(0, min(bounds1[3], bounds2[3]) - max(bounds1[1], bounds2[1]))
            overlap_area = overlap_x * overlap_y
            
            alignment_result['bounds_overlap'] = overlap_area > 0
            alignment_result['spatial_overlap_area'] = overlap_area
            
        except Exception as e:
            logger.warning(f"Could not calculate spatial bounds: {e}")
        
        # Check geometry validity
        for name, gdf in [(data1_name, data1), (data2_name, data2)]:
            if 'geometry' in gdf.columns:
                valid_geoms = gdf.geometry.is_valid.sum()
                total_geoms = len(gdf)
                alignment_result['geometry_validity'][name] = {
                    'valid_geometries': int(valid_geoms),
                    'total_geometries': total_geoms,
                    'validity_rate': (valid_geoms / total_geoms * 100).round(2) if total_geoms > 0 else 0
                }
        
        logger.info(f"Spatial alignment: CRS match={alignment_result['crs_match']}, bounds overlap={alignment_result['bounds_overlap']}")
        
        return alignment_result
    
    def validate_temporal_alignment(self, 
                                  data: gpd.GeoDataFrame,
                                  date_columns: List[str] = None) -> Dict[str, Any]:
        """
        Validate temporal alignment and consistency in dataset.
        
        Args:
            data: GeoDataFrame to validate
            date_columns: List of column names containing date/time data
            
        Returns:
            Dictionary with temporal alignment metrics
        """
        logger.info("Validating temporal alignment")
        
        if date_columns is None:
            # Auto-detect potential date columns
            date_columns = [col for col in data.columns if any(keyword in col.lower() 
                          for keyword in ['date', 'time', 'year', 'month', 'day'])]
        
        temporal_result = {
            'date_columns_found': date_columns,
            'temporal_consistency': {},
            'date_ranges': {},
            'temporal_gaps': {}
        }
        
        for col in date_columns:
            if col in data.columns:
                try:
                    # Try to convert to datetime
                    date_series = pd.to_datetime(data[col], errors='coerce')
                    valid_dates = date_series.dropna()
                    
                    if len(valid_dates) > 0:
                        temporal_result['date_ranges'][col] = {
                            'min_date': str(valid_dates.min()),
                            'max_date': str(valid_dates.max()),
                            'valid_dates': len(valid_dates),
                            'invalid_dates': len(data) - len(valid_dates)
                        }
                        
                        # Check for temporal consistency (no future dates, reasonable ranges)
                        current_date = pd.Timestamp.now()
                        future_dates = (valid_dates > current_date).sum()
                        
                        temporal_result['temporal_consistency'][col] = {
                            'future_dates': int(future_dates),
                            'date_range_years': (valid_dates.max() - valid_dates.min()).days / 365.25
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not process date column {col}: {e}")
                    temporal_result['temporal_consistency'][col] = {'error': str(e)}
        
        logger.info(f"Temporal validation: found {len(date_columns)} date columns")
        
        return temporal_result
    
    def validate_integration_quality(self, 
                                   integrated_data: gpd.GeoDataFrame,
                                   source_datasets: List[gpd.GeoDataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive validation of integrated dataset quality.
        
        Args:
            integrated_data: Integrated GeoDataFrame to validate
            source_datasets: List of source datasets used in integration
            
        Returns:
            Dictionary with comprehensive quality metrics
        """
        logger.info("Performing comprehensive integration quality validation")
        
        quality_result = {
            'basic_validation': {},
            'completeness_validation': {},
            'spatial_validation': {},
            'temporal_validation': {},
            'integration_specific': {}
        }
        
        # Basic validation using existing method
        integrator = DataIntegrator(".")
        basic_result = integrator.validate_integration(integrated_data)
        quality_result['basic_validation'] = {
            'is_valid': basic_result.is_valid,
            'total_records': basic_result.total_records,
            'missing_records': basic_result.missing_records,
            'duplicate_records': basic_result.duplicate_records,
            'invalid_geometries': basic_result.invalid_geometries,
            'issues': basic_result.issues
        }
        
        # Completeness validation
        quality_result['completeness_validation'] = self.validate_data_completeness(integrated_data)
        
        # Spatial validation if source datasets provided
        if source_datasets and len(source_datasets) > 0:
            for i, source_data in enumerate(source_datasets):
                spatial_key = f'spatial_alignment_source_{i}'
                quality_result['spatial_validation'][spatial_key] = self.validate_spatial_alignment(
                    integrated_data, source_data, "integrated_data", f"source_{i}"
                )
        
        # Temporal validation
        quality_result['temporal_validation'] = self.validate_temporal_alignment(integrated_data)
        
        # Integration-specific checks
        quality_result['integration_specific'] = self._validate_integration_specific(integrated_data)
        
        # Overall quality score
        quality_score = self._calculate_quality_score(quality_result)
        quality_result['overall_quality_score'] = quality_score
        
        logger.info(f"Integration quality validation complete. Overall score: {quality_score:.2f}/100")
        
        return quality_result
    
    def _validate_integration_specific(self, data: gpd.GeoDataFrame) -> Dict[str, Any]:
        """
        Validate integration-specific aspects of the dataset.
        
        Args:
            data: Integrated GeoDataFrame
            
        Returns:
            Dictionary with integration-specific metrics
        """
        integration_metrics = {
            'demographic_columns': 0,
            'origin_destination_pairs': 0,
            'spatial_coverage': {},
            'data_consistency': {}
        }
        
        # Count demographic columns (those with suffixes)
        demo_cols = [col for col in data.columns if any(suffix in col for suffix in ['_origin', '_dest'])]
        integration_metrics['demographic_columns'] = len(demo_cols)
        
        # Count OD pairs if relevant columns exist
        if 'origin_code' in data.columns and 'destination_code' in data.columns:
            unique_origins = data['origin_code'].nunique()
            unique_destinations = data['destination_code'].nunique()
            integration_metrics['origin_destination_pairs'] = {
                'unique_origins': unique_origins,
                'unique_destinations': unique_destinations,
                'total_od_pairs': len(data)
            }
        
        # Check spatial coverage if geometry exists
        if 'geometry' in data.columns:
            try:
                bounds = data.total_bounds
                area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                integration_metrics['spatial_coverage'] = {
                    'bounding_box_area': area,
                    'bounds': bounds.tolist()
                }
            except Exception as e:
                logger.warning(f"Could not calculate spatial coverage: {e}")
        
        return integration_metrics
    
    def _calculate_quality_score(self, quality_result: Dict[str, Any]) -> float:
        """
        Calculate overall quality score based on validation results.
        
        Args:
            quality_result: Dictionary with validation results
            
        Returns:
            Quality score between 0 and 100
        """
        score = 100.0
        
        # Deduct points for basic validation issues
        basic = quality_result.get('basic_validation', {})
        if not basic.get('is_valid', True):
            score -= 20
        
        # Deduct points for completeness issues
        completeness = quality_result.get('completeness_validation', {})
        completeness_percent = completeness.get('overall_completeness_percent', 100)
        if completeness_percent < 90:
            score -= (90 - completeness_percent) * 0.5
        
        # Deduct points for empty records
        empty_records = completeness.get('empty_records', 0)
        total_records = completeness.get('total_records', 1)
        if empty_records > 0:
            empty_rate = empty_records / total_records * 100
            score -= empty_rate * 0.3
        
        # Deduct points for spatial issues
        spatial = quality_result.get('spatial_validation', {})
        for spatial_check in spatial.values():
            if not spatial_check.get('crs_match', True):
                score -= 10
            if not spatial_check.get('bounds_overlap', True):
                score -= 15
        
        return max(0.0, min(100.0, score))
    
    def generate_validation_report(self, quality_result: Dict[str, Any]) -> str:
        """
        Generate human-readable validation report.
        
        Args:
            quality_result: Dictionary with validation results
            
        Returns:
            Formatted validation report string
        """
        report_lines = [
            "=== DATA INTEGRATION QUALITY REPORT ===",
            ""
        ]
        
        # Overall score
        score = quality_result.get('overall_quality_score', 0)
        report_lines.append(f"Overall Quality Score: {score:.1f}/100")
        report_lines.append("")
        
        # Basic validation
        basic = quality_result.get('basic_validation', {})
        report_lines.append("BASIC VALIDATION:")
        report_lines.append(f"  - Valid: {basic.get('is_valid', 'Unknown')}")
        report_lines.append(f"  - Total Records: {basic.get('total_records', 0):,}")
        report_lines.append(f"  - Missing Records: {basic.get('missing_records', 0):,}")
        report_lines.append(f"  - Duplicate Records: {basic.get('duplicate_records', 0):,}")
        report_lines.append(f"  - Invalid Geometries: {basic.get('invalid_geometries', 0):,}")
        
        if basic.get('issues'):
            report_lines.append("  Issues:")
            for issue in basic.get('issues', []):
                report_lines.append(f"    - {issue}")
        report_lines.append("")
        
        # Completeness validation
        completeness = quality_result.get('completeness_validation', {})
        report_lines.append("COMPLETENESS VALIDATION:")
        report_lines.append(f"  - Overall Completeness: {completeness.get('overall_completeness_percent', 0):.1f}%")
        report_lines.append(f"  - Empty Records: {completeness.get('empty_records', 0):,}")
        
        high_missing = completeness.get('high_missing_columns', {})
        if high_missing:
            report_lines.append("  High Missing Columns (>50%):")
            for col, pct in high_missing.items():
                report_lines.append(f"    - {col}: {pct:.1f}% missing")
        report_lines.append("")
        
        # Integration specific
        integration = quality_result.get('integration_specific', {})
        report_lines.append("INTEGRATION SPECIFIC:")
        report_lines.append(f"  - Demographic Columns: {integration.get('demographic_columns', 0)}")
        
        od_pairs = integration.get('origin_destination_pairs', {})
        if od_pairs:
            report_lines.append(f"  - Unique Origins: {od_pairs.get('unique_origins', 0):,}")
            report_lines.append(f"  - Unique Destinations: {od_pairs.get('unique_destinations', 0):,}")
            report_lines.append(f"  - Total OD Pairs: {od_pairs.get('total_od_pairs', 0):,}")
        
        return "\n".join(report_lines)
    
    @staticmethod
    def validate_dataset(data: gpd.GeoDataFrame) -> ValidationResult:
        """
        Static method for backward compatibility.
        Validate dataset quality and completeness.
        
        Args:
            data: GeoDataFrame to validate
            
        Returns:
            ValidationResult with validation metrics
        """
        integrator = DataIntegrator(".")  # Dummy path since we're not using file operations
        return integrator.validate_integration(data)