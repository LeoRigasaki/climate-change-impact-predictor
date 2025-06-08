"""
Data processing pipeline orchestrator.
Coordinates data processing across multiple sources and processors.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

from .processors.base_processor import DataQualityChecker
from .processors.air_quality_processor import AirQualityProcessor
from .processors.meteorological_processor import MeteorologicalProcessor
from .data_manager import ClimateDataManager
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

class ClimateDataPipeline:
    """Orchestrates the complete climate data processing pipeline."""
    
    def __init__(self):
        self.data_manager = ClimateDataManager()
        self.air_quality_processor = AirQualityProcessor()
        self.meteorological_processor = MeteorologicalProcessor()
        self.quality_checker = DataQualityChecker()
        
        self.processing_results = {}
        
    def process_location_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        skip_collection: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Process all available data for a specific location.
        
        Args:
            location: Location name (must be in DEFAULT_LOCATIONS)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            skip_collection: If True, skip data collection and use existing files
        
        Returns:
            Dictionary containing processed DataFrames for each data source
        """
        
        logger.info(f"Starting data processing pipeline for {location}")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        results = {}
        
        # Step 1: Data Collection (if not skipping)
        if not skip_collection:
            logger.info("Phase 1: Data Collection")
            try:
                raw_data = self.data_manager.fetch_all_data(
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    save=True
                )
                logger.info("Data collection completed successfully")
            except Exception as e:
                logger.error(f"Data collection failed: {e}")
                raw_data = self._load_existing_data(location, start_date, end_date)
        else:
            logger.info("Skipping data collection, loading existing data...")
            raw_data = self._load_existing_data(location, start_date, end_date)
        
        # Step 2: Data Processing
        logger.info("Phase 2: Data Processing")
        
        # Process Air Quality Data
        if raw_data.get('air_quality'):
            try:
                logger.info("Processing air quality data...")
                air_quality_df = self.air_quality_processor.process(raw_data['air_quality'])
                
                # Quality assessment
                quality_report = self.quality_checker.assess_quality(air_quality_df, 'air_quality')
                
                # Save processed data
                filename = f"air_quality_{location}_{start_date}_{end_date}_processed"
                self.air_quality_processor.save_processed_data(air_quality_df, filename)
                
                results['air_quality'] = {
                    'data': air_quality_df,
                    'quality_report': quality_report
                }
                
                logger.info(f"Air quality processing completed. Quality score: {quality_report['overall_score']}")
                
            except Exception as e:
                logger.error(f"Air quality processing failed: {e}")
                results['air_quality'] = None
        
        # Process Meteorological Data
        if raw_data.get('meteorological'):
            try:
                logger.info("Processing meteorological data...")
                meteorological_df = self.meteorological_processor.process(raw_data['meteorological'])
                
                # Quality assessment
                quality_report = self.quality_checker.assess_quality(meteorological_df, 'meteorological')
                
                # Save processed data
                filename = f"meteorological_{location}_{start_date}_{end_date}_processed"
                self.meteorological_processor.save_processed_data(meteorological_df, filename)
                
                results['meteorological'] = {
                    'data': meteorological_df,
                    'quality_report': quality_report
                }
                
                logger.info(f"Meteorological processing completed. Quality score: {quality_report['overall_score']}")
                
            except Exception as e:
                logger.error(f"Meteorological processing failed: {e}")
                results['meteorological'] = None
        
        # Step 3: Data Integration
        logger.info("Phase 3: Data Integration")
        if results.get('air_quality') and results.get('meteorological'):
            try:
                integrated_df = self._integrate_datasets(
                    results['air_quality']['data'],
                    results['meteorological']['data']
                )
                
                # Save integrated dataset
                filename = f"integrated_{location}_{start_date}_{end_date}"
                filepath = PROCESSED_DATA_DIR / f"{filename}.parquet"
                integrated_df.to_parquet(filepath, index=True)
                
                results['integrated'] = {
                    'data': integrated_df,
                    'quality_report': self.quality_checker.assess_quality(integrated_df, 'integrated')
                }
                
                logger.info(f"Data integration completed: {integrated_df.shape[0]} records, {integrated_df.shape[1]} features")
                
            except Exception as e:
                logger.error(f"Data integration failed: {e}")
                results['integrated'] = None
        
        # Step 4: Generate Processing Report
        self._generate_processing_report(location, start_date, end_date, results)
        
        # Store results
        self.processing_results[f"{location}_{start_date}_{end_date}"] = results
        
        logger.info(f"Pipeline processing completed for {location}")
        return results
    
    def _load_existing_data(self, location: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Load existing raw data files."""
        raw_data = {}
        
        # Define expected file patterns
        file_patterns = {
            'air_quality': f"air_quality_{location}_{start_date}_{end_date}.json",
            'meteorological': f"meteorological_{location}_{start_date}_{end_date}.json"
        }
        
        for data_type, filename in file_patterns.items():
            filepath = RAW_DATA_DIR / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        raw_data[data_type] = json.load(f)
                    logger.info(f"Loaded existing {data_type} data from {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to load {filepath}: {e}")
            else:
                logger.warning(f"Raw data file not found: {filepath}")
        
        return raw_data
    
    def _integrate_datasets(
        self,
        air_quality_df: pd.DataFrame,
        meteorological_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Integrate air quality and meteorological datasets."""
        
        logger.info("Integrating air quality and meteorological datasets...")
        
        # Resample air quality data to daily frequency to match meteorological data
        air_quality_daily = air_quality_df.resample('D').agg({
            # Air quality pollutants - daily averages
            'pm2_5': 'mean',
            'pm10': 'mean',
            'co': 'mean',
            'co2': 'mean',
            'no2': 'mean',
            'so2': 'mean',
            'o3': 'mean',
            'nh3': 'mean',
            'uv_index': 'mean',
            
            # Derived features - daily averages
            'pm_ratio': 'mean',
            'pollution_index': 'mean',
            
            # Moving averages - use last value
            'pm2_5_24h_avg': 'last',
            'pm10_24h_avg': 'last',
            'o3_24h_avg': 'last',
            'no2_24h_avg': 'last',
            
            # AQI values - daily max for health assessment
            'european_aqi': 'max',
            'us_aqi': 'max'
        })
        
        # Merge datasets on datetime index
        integrated_df = meteorological_df.join(air_quality_daily, how='inner', rsuffix='_aq')
        
        # Calculate cross-domain features
        integrated_df = self._calculate_integrated_features(integrated_df)
        
        logger.info(f"Integration completed: {integrated_df.shape[0]} records, {integrated_df.shape[1]} features")
        return integrated_df
    
    def _calculate_integrated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features that depend on both meteorological and air quality data."""
        
        # Temperature-pollution interactions
        if all(col in df.columns for col in ['temperature_2m', 'pm2_5']):
            # High temperature + high pollution risk
            df['heat_pollution_risk'] = (
                (df['temperature_2m'] > df['temperature_2m'].quantile(0.75)) &
                (df['pm2_5'] > df['pm2_5'].quantile(0.75))
            ).astype(int)
        
        # Wind-pollution dispersion
        if all(col in df.columns for col in ['wind_speed_2m', 'pollution_index']):
            # Low wind + high pollution = poor dispersion
            df['poor_dispersion'] = (
                (df['wind_speed_2m'] < df['wind_speed_2m'].quantile(0.25)) &
                (df['pollution_index'] > df['pollution_index'].quantile(0.75))
            ).astype(int)
        
        # Humidity-ozone interaction
        if all(col in df.columns for col in ['relative_humidity', 'o3']):
            # Ozone formation is affected by humidity
            df['ozone_formation_potential'] = df['relative_humidity'] * df['o3']
        
        # Precipitation washing effect
        if all(col in df.columns for col in ['precipitation', 'pm10', 'pm2_5']):
            # Rain can wash out particulates
            df['precip_cleaning_effect'] = df['precipitation'] * (df['pm10'] + df['pm2_5'])
        
        # Overall environmental stress index
        stress_components = []
        weights = {
            'temperature_2m': 0.2,    # Heat stress
            'pollution_index': 0.3,   # Air quality stress
            'uv_index': 0.1,         # UV stress
            'wind_speed_2m': -0.1,   # Wind helps (negative weight)
            'relative_humidity': 0.1, # Humidity stress
            'pm2_5': 0.2,            # Particulate stress
            'o3': 0.1                # Ozone stress
        }
        
        for component, weight in weights.items():
            if component in df.columns:
                # Normalize to 0-1 scale
                normalized = (df[component] - df[component].min()) / (df[component].max() - df[component].min())
                stress_components.append(normalized * weight)
        
        if stress_components:
            df['environmental_stress_index'] = sum(stress_components)
        
        return df
    
    def _generate_processing_report(
        self,
        location: str,
        start_date: str,
        end_date: str,
        results: Dict[str, Any]
    ) -> None:
        """Generate comprehensive processing report."""
        
        report = {
            "processing_summary": {
                "location": location,
                "date_range": f"{start_date} to {end_date}",
                "processing_time": datetime.now().isoformat(),
                "pipeline_version": "1.0.0"
            },
            "data_sources": {},
            "quality_assessment": {},
            "integration_results": {}
        }
        
        # Summarize each data source
        for source, result in results.items():
            if result and 'data' in result:
                df = result['data']
                quality_report = result.get('quality_report', {})
                
                report["data_sources"][source] = {
                    "records": df.shape[0],
                    "features": df.shape[1],
                    "date_range": {
                        "start": df.index.min().isoformat() if hasattr(df.index.min(), 'isoformat') else str(df.index.min()),
                        "end": df.index.max().isoformat() if hasattr(df.index.max(), 'isoformat') else str(df.index.max())
                    },
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
                }
                
                report["quality_assessment"][source] = {
                    "overall_score": quality_report.get('overall_score', 0),
                    "completeness": quality_report.get('completeness', {}).get('missing_percentage', 0),
                    "duplicates": quality_report.get('duplicates', {}).get('duplicate_percentage', 0)
                }
        
        # Integration summary
        if 'integrated' in results and results['integrated']:
            integrated_df = results['integrated']['data']
            report["integration_results"] = {
                "success": True,
                "final_records": integrated_df.shape[0],
                "final_features": integrated_df.shape[1],
                "feature_categories": {
                    "meteorological": len([col for col in integrated_df.columns if any(
                        term in col.lower() for term in ['temp', 'precip', 'wind', 'humidity']
                    )]),
                    "air_quality": len([col for col in integrated_df.columns if any(
                        term in col.lower() for term in ['pm', 'co', 'no', 'so', 'o3', 'aqi']
                    )]),
                    "derived": len([col for col in integrated_df.columns if any(
                        term in col.lower() for term in ['index', 'risk', 'category', 'anomaly']
                    )])
                }
            }
        else:
            report["integration_results"] = {"success": False}
        
        # Save report
        report_filename = f"processing_report_{location}_{start_date}_{end_date}.json"
        report_filepath = PROCESSED_DATA_DIR / report_filename
        
        with open(report_filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Processing report saved to {report_filepath}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of all processing results."""
        if not self.processing_results:
            return {"message": "No processing results available"}
        
        summary = {
            "total_locations_processed": len(self.processing_results),
            "processing_results": {}
        }
        
        for key, results in self.processing_results.items():
            location_summary = {}
            for source, result in results.items():
                if result and 'data' in result:
                    quality_score = result.get('quality_report', {}).get('overall_score', 0)
                    location_summary[source] = {
                        "records": result['data'].shape[0],
                        "features": result['data'].shape[1],
                        "quality_score": quality_score
                    }
            
            summary["processing_results"][key] = location_summary
        
        return summary