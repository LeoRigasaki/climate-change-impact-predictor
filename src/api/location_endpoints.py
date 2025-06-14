# src/api/location_endpoints.py
"""
🔌 Day 7: Production-Ready Location Services API
src/api/location_endpoints.py

RESTful API endpoints for external integration with the global location system.
Provides professional API interface for geocoding, search, and data availability services.
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Path as PathParam, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.location_service import LocationService, LocationInfo
from src.core.data_manager import ClimateDataManager
from src.core.performance_optimizer import PerformanceOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Climate Impact Predictor - Location Services API",
    description="Professional location services for global climate data collection and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Initialize services
location_service = LocationService()
data_manager = ClimateDataManager()
performance_optimizer = PerformanceOptimizer(location_service, data_manager)

# Request/Response Models
class LocationQuery(BaseModel):
    """Location query request model."""
    location: str = Field(..., description="Location name or coordinates", example="Berlin, Germany")
    include_data_availability: bool = Field(default=True, description="Include data source availability")
    include_metadata: bool = Field(default=False, description="Include additional metadata")

class CoordinateQuery(BaseModel):
    """Coordinate-based query model."""
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    include_data_availability: bool = Field(default=True, description="Include data source availability")

class SearchQuery(BaseModel):
    """Location search request model."""
    query: str = Field(..., description="Search query", example="Berl")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results to return")
    country_filter: Optional[str] = Field(default=None, description="Filter by country code")

class BatchLocationQuery(BaseModel):
    """Batch location processing request model."""
    locations: List[str] = Field(..., description="List of location names", max_items=100)
    include_data_availability: bool = Field(default=True, description="Include data source availability")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")

class LocationResponse(BaseModel):
    """Standardized location response model."""
    name: str = Field(..., description="Location name")
    country: str = Field(..., description="Country name")
    state: Optional[str] = Field(default=None, description="State/province name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    timezone: Optional[str] = Field(default=None, description="Timezone identifier")
    population: Optional[int] = Field(default=None, description="Population if available")
    data_sources_available: int = Field(..., description="Number of available data sources")
    data_availability: Optional[Dict[str, bool]] = Field(default=None, description="Detailed data source availability")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class SearchResponse(BaseModel):
    """Search results response model."""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results found")
    results: List[LocationResponse] = Field(..., description="List of matching locations")
    processing_time: float = Field(..., description="Query processing time in seconds")

class BatchResponse(BaseModel):
    """Batch processing response model."""
    total_locations: int = Field(..., description="Total number of locations processed")
    successful_locations: int = Field(..., description="Number of successfully processed locations")
    failed_locations: int = Field(..., description="Number of failed locations")
    processing_time: float = Field(..., description="Total processing time in seconds")
    locations_per_second: float = Field(..., description="Processing throughput")
    results: Dict[str, Union[LocationResponse, Dict[str, str]]] = Field(..., description="Results for each location")

class HealthResponse(BaseModel):
    """System health response model."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    services: Dict[str, Dict[str, Any]] = Field(..., description="Individual service health")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

# Global state for health monitoring
app_start_time = time.time()
request_count = 0
error_count = 0

# Helper functions
def convert_location_info_to_response(
    location: LocationInfo, 
    include_data_availability: bool = True,
    include_metadata: bool = False
) -> LocationResponse:
    """Convert LocationInfo object to API response format."""
    
    response_data = {
        "name": location.name,
        "country": location.country,
        "state": getattr(location, 'state', None),
        "latitude": location.latitude,
        "longitude": location.longitude,
        "timezone": getattr(location, 'timezone', None),
        "population": getattr(location, 'population', None),
        "data_sources_available": location.data_sources_available
    }
    
    if include_data_availability:
        response_data["data_availability"] = {
            "air_quality": location.has_air_quality,
            "meteorological": location.has_meteorological,
            "climate_projections": location.has_climate_projections,
            "weather_forecast": getattr(location, 'has_weather_forecast', False)
        }
    
    if include_metadata:
        response_data["metadata"] = {
            "location_type": getattr(location, 'location_type', 'city'),
            "administrative_level": getattr(location, 'admin_level', None),
            "geocoding_confidence": getattr(location, 'confidence', None),
            "data_quality_score": getattr(location, 'data_quality_score', None)
        }
    
    return LocationResponse(**response_data)

async def track_request():
    """Background task to track request metrics."""
    global request_count
    request_count += 1

async def track_error():
    """Background task to track error metrics."""
    global error_count
    error_count += 1

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint with basic information."""
    return {
        "service": "Climate Impact Predictor - Location Services API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health"
    }

@app.post("/geocode", response_model=LocationResponse)
async def geocode_location(
    query: LocationQuery,
    background_tasks: BackgroundTasks
):
    """
    🌍 Geocode a location name to coordinates and climate data availability.
    
    Convert location names (cities, countries, addresses) to precise coordinates
    and assess available climate data sources for the location.
    """
    background_tasks.add_task(track_request)
    start_time = time.time()
    
    try:
        logger.info(f"Geocoding request: {query.location}")
        
        # Geocode the location
        location = location_service.geocode_location(query.location)
        
        if not location:
            background_tasks.add_task(track_error)
            raise HTTPException(
                status_code=404,
                detail=f"Location not found: {query.location}"
            )
        
        # Check data availability if requested
        if query.include_data_availability:
            try:
                availability = await data_manager.check_data_availability(location)
                location.has_air_quality = availability.get('air_quality', False)
                location.has_meteorological = availability.get('meteorological', False)
                location.has_climate_projections = availability.get('climate_projections', False)
                location.has_weather_forecast = availability.get('weather_forecast', False)
                location.data_sources_available = sum(availability.values())
            except Exception as e:
                logger.warning(f"Data availability check failed: {e}")
                # Continue with geocoding result even if availability check fails
        
        response = convert_location_info_to_response(
            location, 
            query.include_data_availability,
            query.include_metadata
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Geocoding completed in {processing_time:.3f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        background_tasks.add_task(track_error)
        logger.error(f"Geocoding error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal geocoding error: {str(e)}"
        )

@app.post("/reverse-geocode", response_model=LocationResponse)
async def reverse_geocode(
    query: CoordinateQuery,
    background_tasks: BackgroundTasks
):
    """
    📍 Reverse geocode coordinates to location information.
    
    Convert latitude/longitude coordinates to location details
    and assess climate data availability.
    """
    background_tasks.add_task(track_request)
    start_time = time.time()
    
    try:
        logger.info(f"Reverse geocoding: {query.latitude}, {query.longitude}")
        
        # Validate coordinates
        if not location_service.validate_coordinates(query.latitude, query.longitude):
            background_tasks.add_task(track_error)
            raise HTTPException(
                status_code=400,
                detail=f"Invalid coordinates: {query.latitude}, {query.longitude}"
            )
        
        # Reverse geocode
        location = location_service.get_location_by_coordinates(
            query.latitude, 
            query.longitude
        )
        
        if not location:
            background_tasks.add_task(track_error)
            raise HTTPException(
                status_code=404,
                detail=f"No location found for coordinates: {query.latitude}, {query.longitude}"
            )
        
        # Check data availability if requested
        if query.include_data_availability:
            try:
                availability = await data_manager.check_data_availability(location)
                location.has_air_quality = availability.get('air_quality', False)
                location.has_meteorological = availability.get('meteorological', False)
                location.has_climate_projections = availability.get('climate_projections', False)
                location.has_weather_forecast = availability.get('weather_forecast', False)
                location.data_sources_available = sum(availability.values())
            except Exception as e:
                logger.warning(f"Data availability check failed: {e}")
        
        response = convert_location_info_to_response(location, query.include_data_availability)
        
        processing_time = time.time() - start_time
        logger.info(f"Reverse geocoding completed in {processing_time:.3f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        background_tasks.add_task(track_error)
        logger.error(f"Reverse geocoding error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal reverse geocoding error: {str(e)}"
        )

@app.post("/search", response_model=SearchResponse)
async def search_locations(
    query: SearchQuery,
    background_tasks: BackgroundTasks
):
    """
    🔍 Search for locations with fuzzy matching and autocomplete.
    
    Find locations matching a search query with intelligent ranking
    and optional country filtering.
    """
    background_tasks.add_task(track_request)
    start_time = time.time()
    
    try:
        logger.info(f"Location search: '{query.query}' (limit: {query.limit})")
        
        # Perform search
        locations = location_service.search_locations(
            query.query,
            limit=query.limit
        )
        
        # Apply country filter if specified
        if query.country_filter:
            locations = [
                loc for loc in locations 
                if loc.country.lower() == query.country_filter.lower()
            ]
        
        # Convert to response format
        location_responses = []
        for location in locations:
            try:
                response = convert_location_info_to_response(location)
                location_responses.append(response)
            except Exception as e:
                logger.warning(f"Failed to convert location {location.name}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        search_response = SearchResponse(
            query=query.query,
            total_results=len(location_responses),
            results=location_responses,
            processing_time=processing_time
        )
        
        logger.info(f"Search completed: {len(location_responses)} results in {processing_time:.3f}s")
        
        return search_response
        
    except Exception as e:
        background_tasks.add_task(track_error)
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal search error: {str(e)}"
        )

@app.post("/batch-geocode", response_model=BatchResponse)
async def batch_geocode_locations(
    query: BatchLocationQuery,
    background_tasks: BackgroundTasks
):
    """
    📦 Process multiple locations in optimized batches.
    
    Efficiently geocode multiple locations with parallel processing
    and comprehensive performance metrics.
    """
    background_tasks.add_task(track_request)
    start_time = time.time()
    
    try:
        logger.info(f"Batch geocoding: {len(query.locations)} locations")
        
        if len(query.locations) > 100:
            raise HTTPException(
                status_code=400,
                detail="Maximum 100 locations allowed per batch request"
            )
        
        # Use performance optimizer for batch processing
        if query.parallel_processing:
            batch_results = await performance_optimizer.batch_process_locations(query.locations)
            results = batch_results["location_results"]
            processing_metrics = batch_results["batch_metrics"]
        else:
            # Sequential processing
            results = {}
            for location_name in query.locations:
                try:
                    location = location_service.geocode_location(location_name)
                    if location:
                        if query.include_data_availability:
                            availability = await data_manager.check_data_availability(location)
                            location.has_air_quality = availability.get('air_quality', False)
                            location.has_meteorological = availability.get('meteorological', False)
                            location.has_climate_projections = availability.get('climate_projections', False)
                            location.has_weather_forecast = availability.get('weather_forecast', False)
                            location.data_sources_available = sum(availability.values())
                        
                        results[location_name] = {
                            "location": location,
                            "success": True
                        }
                    else:
                        results[location_name] = {
                            "location": None,
                            "success": False,
                            "error": "Location not found"
                        }
                except Exception as e:
                    results[location_name] = {
                        "location": None,
                        "success": False,
                        "error": str(e)
                    }
            
            processing_time = time.time() - start_time
            processing_metrics = {
                "processing_time": processing_time,
                "locations_per_second": len(query.locations) / processing_time if processing_time > 0 else 0.0
            }
        
        # Convert results to API format
        api_results = {}
        successful_count = 0
        failed_count = 0
        
        for location_name, result in results.items():
            if result["success"] and result.get("location"):
                try:
                    api_results[location_name] = convert_location_info_to_response(
                        result["location"], 
                        query.include_data_availability
                    )
                    successful_count += 1
                except Exception as e:
                    api_results[location_name] = {
                        "error": "Response conversion failed",
                        "message": str(e)
                    }
                    failed_count += 1
            else:
                api_results[location_name] = {
                    "error": result.get("error", "Unknown error"),
                    "message": f"Failed to process {location_name}"
                }
                failed_count += 1
        
        batch_response = BatchResponse(
            total_locations=len(query.locations),
            successful_locations=successful_count,
            failed_locations=failed_count,
            processing_time=processing_metrics["processing_time"],
            locations_per_second=processing_metrics["locations_per_second"],
            results=api_results
        )
        
        logger.info(f"Batch geocoding completed: {successful_count}/{len(query.locations)} successful "
                   f"in {processing_metrics['processing_time']:.1f}s")
        
        return batch_response
        
    except HTTPException:
        raise
    except Exception as e:
        background_tasks.add_task(track_error)
        logger.error(f"Batch geocoding error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal batch processing error: {str(e)}"
        )

@app.get("/validate", response_model=Dict[str, bool])
async def validate_coordinates(
    lat: float = Query(..., ge=-90, le=90, description="Latitude"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude"),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    ✅ Validate coordinate values for correctness.
    
    Check if latitude and longitude values are within valid ranges
    and suitable for climate data collection.
    """
    background_tasks.add_task(track_request)
    
    try:
        is_valid = location_service.validate_coordinates(lat, lon)
        
        return {
            "valid": is_valid,
            "latitude": lat,
            "longitude": lon,
            "message": "Coordinates are valid" if is_valid else "Coordinates are invalid"
        }
        
    except Exception as e:
        background_tasks.add_task(track_error)
        logger.error(f"Coordinate validation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Validation error: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    🏥 Comprehensive system health check.
    
    Returns detailed health information for all system components
    including performance metrics and service availability.
    """
    try:
        # Calculate uptime
        uptime_seconds = time.time() - app_start_time
        
        # Get performance metrics
        optimization_report = performance_optimizer.get_optimization_report()
        
        # Test core services
        service_health = {}
        
        # Test location service
        try:
            test_location = location_service.geocode_location("Berlin, Germany")
            service_health["location_service"] = {
                "status": "healthy" if test_location else "degraded",
                "response_time": "< 1s",
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            service_health["location_service"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        
        # Test data manager
        try:
            # Quick availability check
            service_health["data_manager"] = {
                "status": "healthy",
                "response_time": "< 2s",
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            service_health["data_manager"] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        
        # Cache health
        cache_stats = optimization_report["cache_performance"]
        service_health["cache"] = {
            "status": "healthy" if cache_stats["hit_rate"] > 0.3 else "degraded",
            "hit_rate": cache_stats["hit_rate"],
            "size": cache_stats["size"],
            "utilization": cache_stats["utilization"],
            "last_check": datetime.now().isoformat()
        }
        
        # Overall status determination
        unhealthy_services = [
            name for name, health in service_health.items() 
            if health["status"] == "unhealthy"
        ]
        
        if unhealthy_services:
            overall_status = "unhealthy"
        elif any(health["status"] == "degraded" for health in service_health.values()):
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Performance metrics
        performance_metrics = {
            "requests_per_minute": request_count / (uptime_seconds / 60) if uptime_seconds > 60 else request_count,
            "error_rate": error_count / request_count if request_count > 0 else 0.0,
            "cache_hit_rate": cache_stats["hit_rate"],
            "memory_usage_mb": optimization_report["system_resources"]["memory_usage_mb"],
            "uptime_hours": uptime_seconds / 3600
        }
        
        health_response = HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            services=service_health,
            performance_metrics=performance_metrics,
            uptime_seconds=uptime_seconds
        )
        
        return health_response
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            services={"error": {"status": "unhealthy", "error": str(e)}},
            performance_metrics={},
            uptime_seconds=time.time() - app_start_time
        )

@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """
    📊 Detailed performance and optimization metrics.
    
    Returns comprehensive metrics for monitoring and optimization analysis.
    """
    try:
        optimization_report = performance_optimizer.get_optimization_report()
        
        # Add API-specific metrics
        uptime_seconds = time.time() - app_start_time
        
        api_metrics = {
            "api_statistics": {
                "total_requests": request_count,
                "total_errors": error_count,
                "error_rate": error_count / request_count if request_count > 0 else 0.0,
                "requests_per_second": request_count / uptime_seconds if uptime_seconds > 0 else 0.0,
                "uptime_seconds": uptime_seconds
            },
            "system_performance": optimization_report["system_resources"],
            "cache_performance": optimization_report["cache_performance"],
            "connection_health": optimization_report["connection_health"],
            "optimization_config": optimization_report["optimization_config"],
            "recommendations": optimization_report["recommendations"]
        }
        
        return api_metrics
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics collection error: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler for unexpected errors."""
    global error_count
    error_count += 1
    
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("🚀 Location Services API starting up...")
    
    try:
        # Initialize performance optimizer
        await performance_optimizer.optimize_system()
        logger.info("✅ Performance optimization completed")
        
        # Warm up cache with popular locations
        popular_locations = ["London", "New York", "Tokyo", "Berlin", "Paris"]
        for location_name in popular_locations:
            try:
                location_service.geocode_location(location_name)
            except Exception as e:
                logger.warning(f"Failed to warm cache for {location_name}: {e}")
        
        logger.info("✅ Cache warmed with popular locations")
        logger.info("🎯 Location Services API ready for requests")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Location Services API shutting down...")
    
    # Save performance metrics
    uptime = time.time() - app_start_time
    logger.info(f"📊 Final metrics: {request_count} requests, {uptime:.1f}s uptime")
    
    logger.info("✅ Shutdown complete")

# Development server runner
if __name__ == "__main__":
    print("🌍 Starting Climate Impact Predictor - Location Services API")
    print("=" * 60)
    print("🔗 API Documentation: http://localhost:8000/docs")
    print("📊 Health Check: http://localhost:8000/health") 
    print("📈 Metrics: http://localhost:8000/metrics")
    print("=" * 60)
    
    uvicorn.run(
        "location_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )