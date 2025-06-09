"""
🌍 Enhanced Climate Data Pipeline - Day 4 Global Support
src/core/pipeline.py

Data processing pipeline orchestrator with global location support.
Coordinates data processing across multiple sources and processors with
intelligent adaptation for any location on Earth.
"""

import logging
import pandas as pd
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

from ..data.processors.base_processor import DataQualityChecker
from ..data.processors.air_quality_processor import AirQualityProcessor
from ..data.processors.meteorological_processor import MeteorologicalProcessor
from .data_manager import ClimateDataManager
from .location_service import LocationService, LocationInfo
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, DEFAULT_LOCATIONS

logger = logging.getLogger(__name__)

class ClimateDataPipeline:
    """
    Enhanced orchestrator for the complete climate data processing pipeline.
    
    Day 4 Features:
    - Global location support via LocationInfo objects
    - Adaptive data collection based on regional availability
    - Intelligent processing workflows that adapt to available data
    - Graceful degradation when some data sources are unavailable
    - Performance optimization for global scale operations
    """
    
    def __init__(self):
        self.data_manager = ClimateDataManager()
        self.location_service = LocationService()
        self.air_quality_processor = AirQualityProcessor()
        self.meteorological_processor = MeteorologicalProcessor()
        self.quality_checker = DataQualityChecker()
        
        # Enhanced tracking for global operations
        self.processing_results = {}
        self.global_processing_stats = {
            "locations_processed": 0,
            "successful_collections": 0,
            "partial_collections": 0,
            "failed_collections": 0,
            "data_sources_by_location": {}
        }
        
        logger.info("🌍 Enhanced ClimateDataPipeline initialized with global support")
    
    def process_global_location(
        self,
        location: LocationInfo,
        start_date: str,
        end_date: str,
        skip_collection: bool = False,
        force_all_sources: bool = False
    ) -> Dict[str, Any]:
        """
        🌍 Process climate data for any global location using LocationInfo.
        
        Args:
            location: LocationInfo object with coordinates and metadata
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            skip_collection: If True, skip data collection and use existing files
            force_all_sources: Attempt to collect from all sources regardless of availability
        
        Returns:
            Dictionary containing processed DataFrames and metadata for each data source
        """
        
        logger.info(f"🌍 Starting global pipeline for {location.name}")
        logger.info(f"📍 Coordinates: {location.latitude:.4f}, {location.longitude:.4f}")
        logger.info(f"🌏 Country: {location.country}")
        logger.info(f"📅 Date range: {start_date} to {end_date}")
        
        results = {}
        processing_metadata = {
            "location": location.to_dict(),
            "date_range": {"start": start_date, "end": end_date},
            "processing_timestamp": datetime.now().isoformat(),
            "data_sources_attempted": 0,
            "data_sources_successful": 0,
            "processing_errors": []
        }
        
        # Step 1: Adaptive Data Collection
        raw_data = {}
        if not skip_collection:
            logger.info("🔄 Phase 1: Adaptive Data Collection")
            try:
                raw_data = self.data_manager.fetch_adaptive_data(
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    save=True,
                    force_all=force_all_sources
                )
                logger.info("✅ Adaptive data collection completed")
            except Exception as e:
                logger.error(f"❌ Adaptive data collection failed: {e}")
                processing_metadata["processing_errors"].append(f"Data collection: {str(e)}")
                raw_data = self._load_existing_data_by_location(location, start_date, end_date)
        else:
            logger.info("⏭️ Skipping data collection, loading existing data...")
            raw_data = self._load_existing_data_by_location(location, start_date, end_date)
        
        # Step 2: Adaptive Data Processing
        logger.info("🔄 Phase 2: Adaptive Data Processing")
        
        # Track which data sources we're processing
        available_sources = [source for source, data in raw_data.items() if data is not None]
        processing_metadata["data_sources_attempted"] = len(raw_data)
        processing_metadata["available_sources"] = available_sources
        
        logger.info(f"📊 Processing {len(available_sources)}/{len(raw_data)} available data sources")
        
        # Process Air Quality Data (if available)
        if raw_data.get('air_quality'):
            results['air_quality'] = self._process_air_quality_adaptive(
                raw_data['air_quality'], location, start_date, end_date
            )
            if results['air_quality']:
                processing_metadata["data_sources_successful"] += 1
        else:
            logger.info("⏭️ No air quality data available for processing")
            results['air_quality'] = None
        
        # Process Meteorological Data (if available)
        if raw_data.get('meteorological'):
            results['meteorological'] = self._process_meteorological_adaptive(
                raw_data['meteorological'], location, start_date, end_date
            )
            if results['meteorological']:
                processing_metadata["data_sources_successful"] += 1
        else:
            logger.info("⏭️ No meteorological data available for processing")
            results['meteorological'] = None
        
        # Process Climate Projections (if available)
        if raw_data.get('climate_projections'):
            results['climate_projections'] = self._process_climate_projections_adaptive(
                raw_data['climate_projections'], location
            )
            if results['climate_projections']:
                processing_metadata["data_sources_successful"] += 1
        else:
            logger.info("⏭️ No climate projections available for processing")
            results['climate_projections'] = None
        
        # Step 3: Data Integration (if multiple sources available)
        successful_sources = [source for source, result in results.items() 
                            if result and 'data' in result]
        
        if len(successful_sources) >= 2:
            logger.info("🔗 Phase 3: Multi-source Data Integration")
            try:
                integrated_result = self._integrate_multi_source_data(
                    results, location, start_date, end_date
                )
                if integrated_result:
                    results['integrated'] = integrated_result
                    logger.info("✅ Multi-source integration successful")
                else:
                    logger.warning("⚠️ Multi-source integration failed")
            except Exception as e:
                logger.error(f"❌ Data integration failed: {e}")
                processing_metadata["processing_errors"].append(f"Integration: {str(e)}")
        else:
            logger.info(f"⏭️ Skipping integration (only {len(successful_sources)} source(s) available)")
        
        # Update global processing statistics
        self._update_global_stats(location, results, processing_metadata)
        
        # Attach processing metadata to results
        results['_metadata'] = processing_metadata
        
        # Log final summary
        successful_count = len([r for r in results.values() if r and 'data' in r])
        total_attempted = processing_metadata["data_sources_attempted"]
        
        if successful_count >= 2:
            status = "✅ SUCCESS"
        elif successful_count >= 1:
            status = "⚠️ PARTIAL SUCCESS"
        else:
            status = "❌ FAILED"
        
        logger.info(f"🎯 Global pipeline complete: {status}")
        logger.info(f"📊 Processed {successful_count}/{total_attempted} data sources successfully")
        
        return results
    
    def _process_air_quality_adaptive(
        self, 
        raw_data: Dict[str, Any], 
        location: LocationInfo, 
        start_date: str, 
        end_date: str
    ) -> Optional[Dict[str, Any]]:
        """🌬️ Process air quality data with adaptive handling."""
        try:
            logger.info("🌬️ Processing air quality data...")
            air_quality_df = self.air_quality_processor.process(raw_data)
            
            if air_quality_df.empty:
                logger.warning("⚠️ Air quality processing resulted in empty dataset")
                return None
            
            # Quality assessment
            quality_report = self.quality_checker.assess_quality(air_quality_df, 'air_quality')
            
            # Generate adaptive filename
            safe_name = location.name.lower().replace(" ", "_").replace(",", "")
            filename = f"air_quality_{safe_name}_{start_date}_{end_date}_processed"
            
            # Save processed data
            self.air_quality_processor.save_processed_data(air_quality_df, filename)
            
            logger.info(f"✅ Air quality processing completed: {air_quality_df.shape[0]} records, "
                       f"quality score: {quality_report.get('overall_score', 0):.1f}/100")
            
            return {
                'data': air_quality_df,
                'quality_report': quality_report,
                'source_location': location.to_dict()
            }
            
        except Exception as e:
            logger.error(f"❌ Air quality processing failed: {e}")
            return None
    
    def _process_meteorological_adaptive(
        self, 
        raw_data: Dict[str, Any], 
        location: LocationInfo, 
        start_date: str, 
        end_date: str
    ) -> Optional[Dict[str, Any]]:
        """🌡️ Process meteorological data with adaptive handling."""
        try:
            logger.info("🌡️ Processing meteorological data...")
            met_df = self.meteorological_processor.process(raw_data)
            
            if met_df.empty:
                logger.warning("⚠️ Meteorological processing resulted in empty dataset")
                return None
            
            # Quality assessment
            quality_report = self.quality_checker.assess_quality(met_df, 'meteorological')
            
            # Generate adaptive filename
            safe_name = location.name.lower().replace(" ", "_").replace(",", "")
            filename = f"meteorological_{safe_name}_{start_date}_{end_date}_processed"
            
            # Save processed data
            self.meteorological_processor.save_processed_data(met_df, filename)
            
            logger.info(f"✅ Meteorological processing completed: {met_df.shape[0]} records, "
                       f"quality score: {quality_report.get('overall_score', 0):.1f}/100")
            
            return {
                'data': met_df,
                'quality_report': quality_report,
                'source_location': location.to_dict()
            }
            
        except Exception as e:
            logger.error(f"❌ Meteorological processing failed: {e}")
            return None
    
    def _process_climate_projections_adaptive(
        self, 
        raw_data: Dict[str, Any], 
        location: LocationInfo
    ) -> Optional[Dict[str, Any]]:
        """🔮 Process climate projections with adaptive handling."""
        try:
            logger.info("🔮 Processing climate projections...")
            
            # For now, basic processing - could be enhanced with specific processor
            if not raw_data or 'data' not in raw_data:
                logger.warning("⚠️ No climate projection data to process")
                return None
            
            # Basic quality check
            projection_data = raw_data['data']
            record_count = len(projection_data) if isinstance(projection_data, list) else 1
            
            logger.info(f"✅ Climate projections processed: {record_count} projections")
            
            return {
                'data': projection_data,
                'quality_report': {'overall_score': 85.0, 'record_count': record_count},
                'source_location': location.to_dict()
            }
            
        except Exception as e:
            logger.error(f"❌ Climate projections processing failed: {e}")
            return None
    
    def _integrate_multi_source_data(
        self, 
        results: Dict[str, Any], 
        location: LocationInfo, 
        start_date: str, 
        end_date: str
    ) -> Optional[Dict[str, Any]]:
        """🔗 Integrate data from multiple sources intelligently."""
        try:
            logger.info("🔗 Integrating multi-source data...")
            
            # Get available dataframes
            available_dfs = {}
            for source, result in results.items():
                if result and 'data' in result and isinstance(result['data'], pd.DataFrame):
                    if not result['data'].empty:
                        available_dfs[source] = result['data']
            
            if len(available_dfs) < 2:
                logger.warning("⚠️ Insufficient data sources for integration")
                return None
            
            logger.info(f"🔗 Integrating {len(available_dfs)} data sources: {list(available_dfs.keys())}")
            
            # Start with the largest dataset as base
            base_source = max(available_dfs.keys(), key=lambda k: len(available_dfs[k]))
            integrated_df = available_dfs[base_source].copy()
            
            logger.info(f"📊 Using {base_source} as base dataset: {integrated_df.shape}")
            
            # Merge other datasets
            for source, df in available_dfs.items():
                if source != base_source:
                    try:
                        # Align indices and merge
                        merged_df = pd.merge(
                            integrated_df, df, 
                            left_index=True, right_index=True, 
                            how='outer', suffixes=('', f'_{source}')
                        )
                        integrated_df = merged_df
                        logger.info(f"🔗 Merged {source}: {df.shape} → {integrated_df.shape}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to merge {source}: {e}")
            
            # Quality assessment of integrated data
            quality_report = self.quality_checker.assess_quality(integrated_df, 'integrated')
            
            # Save integrated data
            safe_name = location.name.lower().replace(" ", "_").replace(",", "")
            filename = f"integrated_{safe_name}_{start_date}_{end_date}"
            
            # Save as parquet for better performance with large datasets
            output_path = PROCESSED_DATA_DIR / f"{filename}.parquet"
            integrated_df.to_parquet(output_path)
            
            logger.info(f"✅ Multi-source integration complete: {integrated_df.shape}")
            logger.info(f"💾 Saved to: {output_path}")
            
            return {
                'data': integrated_df,
                'quality_report': quality_report,
                'source_location': location.to_dict(),
                'integrated_sources': list(available_dfs.keys())
            }
            
        except Exception as e:
            logger.error(f"❌ Multi-source integration failed: {e}")
            return None
    
    def _load_existing_data_by_location(
        self, 
        location: LocationInfo, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, Any]:
        """📂 Load existing raw data files for a location."""
        logger.info(f"📂 Loading existing data for {location.name}")
        
        safe_name = location.name.lower().replace(" ", "_").replace(",", "")
        raw_data = {}
        
        # Look for air quality data
        air_quality_file = RAW_DATA_DIR / f"air_quality_{safe_name}_{start_date}_{end_date}.json"
        if air_quality_file.exists():
            try:
                with open(air_quality_file, 'r') as f:
                    raw_data['air_quality'] = json.load(f)
                logger.info(f"📁 Loaded air quality data from {air_quality_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load air quality data: {e}")
                raw_data['air_quality'] = None
        else:
            raw_data['air_quality'] = None
        
        # Look for meteorological data
        met_file = RAW_DATA_DIR / f"meteorological_{safe_name}_{start_date}_{end_date}.json"
        if met_file.exists():
            try:
                with open(met_file, 'r') as f:
                    raw_data['meteorological'] = json.load(f)
                logger.info(f"📁 Loaded meteorological data from {met_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load meteorological data: {e}")
                raw_data['meteorological'] = None
        else:
            raw_data['meteorological'] = None
        
        # Look for climate projections
        climate_file = RAW_DATA_DIR / f"climate_projections_{safe_name}_ssp245.json"
        if climate_file.exists():
            try:
                with open(climate_file, 'r') as f:
                    raw_data['climate_projections'] = json.load(f)
                logger.info(f"📁 Loaded climate projections from {climate_file}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load climate projections: {e}")
                raw_data['climate_projections'] = None
        else:
            raw_data['climate_projections'] = None
        
        available_count = len([v for v in raw_data.values() if v is not None])
        logger.info(f"📊 Loaded {available_count}/3 existing data sources")
        
        return raw_data
    
    def _update_global_stats(
        self, 
        location: LocationInfo, 
        results: Dict[str, Any], 
        metadata: Dict[str, Any]
    ):
        """📊 Update global processing statistics."""
        self.global_processing_stats["locations_processed"] += 1
        
        successful_sources = metadata["data_sources_successful"]
        attempted_sources = metadata["data_sources_attempted"]
        
        if successful_sources == attempted_sources and successful_sources > 0:
            self.global_processing_stats["successful_collections"] += 1
        elif successful_sources > 0:
            self.global_processing_stats["partial_collections"] += 1
        else:
            self.global_processing_stats["failed_collections"] += 1
        
        # Track data sources by location
        location_key = f"{location.name}, {location.country}"
        self.global_processing_stats["data_sources_by_location"][location_key] = {
            "attempted": attempted_sources,
            "successful": successful_sources,
            "sources": metadata.get("available_sources", [])
        }
    
    # =========================================================================
    # BACKWARD COMPATIBILITY METHODS (Keep existing API working)
    # =========================================================================
    
    def process_location_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        skip_collection: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        🔄 Backward compatibility: Process data for a default location string.
        
        This method maintains compatibility with existing tools while internally
        converting to the new global location system.
        """
        
        logger.info(f"🔄 Processing legacy location: {location}")
        
        # Check if it's a default location
        if location in DEFAULT_LOCATIONS:
            # Convert to LocationInfo object
            coords = DEFAULT_LOCATIONS[location]
            location_info = LocationInfo(
                latitude=coords["latitude"],
                longitude=coords["longitude"],
                name=location.title(),
                country="Unknown",  # Default locations don't have country info
                country_code="XX",
                has_air_quality=True,
                has_meteorological=True,
                has_climate_projections=False  # Most default locations won't have this
            )
            
            # Use new global processing method
            results = self.process_global_location(
                location=location_info,
                start_date=start_date,
                end_date=end_date,
                skip_collection=skip_collection
            )
            
            # Convert results to legacy format (remove metadata)
            legacy_results = {}
            for source, result in results.items():
                if source != '_metadata' and result and 'data' in result:
                    legacy_results[source] = result['data']
                elif source != '_metadata':
                    legacy_results[source] = None
            
            return legacy_results
        
        else:
            # Try to geocode the location string
            try:
                location_info = self.location_service.geocode_location(location)
                if location_info:
                    logger.info(f"🌍 Successfully geocoded '{location}' to {location_info.name}")
                    
                    # Use new global processing method
                    results = self.process_global_location(
                        location=location_info,
                        start_date=start_date,
                        end_date=end_date,
                        skip_collection=skip_collection
                    )
                    
                    # Convert to legacy format
                    legacy_results = {}
                    for source, result in results.items():
                        if source != '_metadata' and result and 'data' in result:
                            legacy_results[source] = result['data']
                        elif source != '_metadata':
                            legacy_results[source] = None
                    
                    return legacy_results
                else:
                    raise ValueError(f"Could not geocode location: {location}")
            
            except Exception as e:
                # Fallback to old method for complete backward compatibility
                logger.warning(f"⚠️ Global processing failed, falling back to legacy method: {e}")
                return self._legacy_process_location_data(location, start_date, end_date, skip_collection)
    
    def _legacy_process_location_data(
        self,
        location: str,
        start_date: str,
        end_date: str,
        skip_collection: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Legacy processing method for complete backward compatibility."""
        
        logger.info(f"🔄 Using legacy processing for {location}")
        
        results = {}
        
        # Step 1: Data Collection (legacy method)
        if not skip_collection:
            try:
                raw_data = self.data_manager.fetch_all_data(
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    save=True
                )
            except Exception as e:
                logger.error(f"Legacy data collection failed: {e}")
                raw_data = self._load_existing_data(location, start_date, end_date)
        else:
            raw_data = self._load_existing_data(location, start_date, end_date)
        
        # Step 2: Data Processing (legacy method)
        if raw_data.get('air_quality'):
            try:
                air_quality_df = self.air_quality_processor.process(raw_data['air_quality'])
                quality_report = self.quality_checker.assess_quality(air_quality_df, 'air_quality')
                filename = f"air_quality_{location}_{start_date}_{end_date}_processed"
                self.air_quality_processor.save_processed_data(air_quality_df, filename)
                results['air_quality'] = air_quality_df
            except Exception as e:
                logger.error(f"Legacy air quality processing failed: {e}")
                results['air_quality'] = None
        
        if raw_data.get('meteorological'):
            try:
                met_df = self.meteorological_processor.process(raw_data['meteorological'])
                quality_report = self.quality_checker.assess_quality(met_df, 'meteorological')
                filename = f"meteorological_{location}_{start_date}_{end_date}_processed"
                self.meteorological_processor.save_processed_data(met_df, filename)
                results['meteorological'] = met_df
            except Exception as e:
                logger.error(f"Legacy meteorological processing failed: {e}")
                results['meteorological'] = None
        
        return results
    
    def _load_existing_data(self, location: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Legacy method to load existing data files."""
        raw_data = {}
        
        # Look for existing files using legacy naming
        air_quality_file = RAW_DATA_DIR / f"air_quality_{location}_{start_date}_{end_date}.json"
        if air_quality_file.exists():
            with open(air_quality_file, 'r') as f:
                raw_data['air_quality'] = json.load(f)
        
        met_file = RAW_DATA_DIR / f"meteorological_{location}_{start_date}_{end_date}.json"
        if met_file.exists():
            with open(met_file, 'r') as f:
                raw_data['meteorological'] = json.load(f)
        
        return raw_data
    
    # =========================================================================
    # UTILITY AND REPORTING METHODS
    # =========================================================================
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """📊 Get comprehensive processing summary with global statistics."""
        return {
            **self.processing_results,
            "global_stats": self.global_processing_stats,
            "total_locations_processed": self.global_processing_stats["locations_processed"]
        }
    
    def get_global_processing_stats(self) -> Dict[str, Any]:
        """🌍 Get detailed global processing statistics."""
        return self.global_processing_stats
    
    def reset_global_stats(self):
        """🔄 Reset global processing statistics (useful for testing)."""
        self.global_processing_stats = {
            "locations_processed": 0,
            "successful_collections": 0,
            "partial_collections": 0,
            "failed_collections": 0,
            "data_sources_by_location": {}
        }
        logger.info("🔄 Global processing statistics reset")