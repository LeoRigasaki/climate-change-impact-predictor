"""
🌍 Enhanced Global Location Service - Day 4 Adaptive Pipeline
src/core/location_service.py

Core location service supporting any coordinates globally with enhanced validation,
geocoding, and intelligent suggestions for climate predictions.
"""

import json
import logging
import sqlite3
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class LocationInfo:
    """Comprehensive location information for climate predictions."""
    
    # Core coordinates
    latitude: float
    longitude: float
    
    # Location identification
    name: str
    country: str
    country_code: str
    
    # Administrative details
    state: Optional[str] = None
    city: Optional[str] = None
    region: Optional[str] = None
    
    # Climate zone classification
    climate_zone: Optional[str] = None
    timezone: Optional[str] = None
    
    # Data availability flags
    has_air_quality: bool = True
    has_meteorological: bool = True
    has_climate_projections: bool = True
    
    # Quality metrics
    confidence_score: float = 1.0
    data_sources: List[str] = None
    
    def __post_init__(self):
        """Initialize default values and validate coordinates."""
        if self.data_sources is None:
            self.data_sources = []
            
        # Enhanced coordinate validation - THIS FIXES THE TEST FAILURE
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}. Must be between -90 and 90.")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}. Must be between -180 and 180.")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LocationInfo':
        """Create LocationInfo from dictionary."""
        return cls(**data)
    
    def cache_key(self) -> str:
        """Generate unique cache key for this location."""
        coords = f"{self.latitude:.4f},{self.longitude:.4f}"
        return hashlib.md5(coords.encode()).hexdigest()
    
    def get_location_type(self) -> str:
        """Determine location type for smart API selection."""
        # Ocean detection (simplified)
        if self.country == "Unknown":
            # Check if it's likely ocean based on coordinates
            if (abs(self.latitude) < 60 and 
                (self.name.lower().find('ocean') >= 0 or 
                 self.name.lower().find('sea') >= 0 or
                 'ocean' in self.name.lower())):
                return "ocean"
            elif abs(self.latitude) > 70:
                return "polar"
            else:
                return "remote"
        
        # Urban vs rural detection (basic)
        if self.city and len(self.city) > 0:
            return "urban"
        else:
            return "rural"
    
    def get_available_apis(self) -> List[str]:
        """Determine which APIs should be available for this location."""
        location_type = self.get_location_type()
        available = []
        
        # NASA POWER - works globally
        available.append("nasa_power")
        
        # Open-Meteo Air Quality - varies by region
        if location_type in ["urban", "rural"]:
            available.append("open_meteo")
        elif location_type not in ["ocean", "polar"]:
            available.append("open_meteo_limited")
        
        # World Bank - requires valid country
        if self.country and self.country != "Unknown":
            available.append("world_bank")
        
        return available


class LocationService:
    """
    Enhanced global location service for dynamic climate predictions.
    
    Features:
    - Geocoding and reverse geocoding with enhanced validation
    - Location validation and suggestions
    - Climate zone classification
    - Data availability checking
    - Intelligent caching system with performance improvements
    - Smart API selection logic
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize enhanced location service with improved caching."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize geocoder with custom user agent
        self.geocoder = Nominatim(user_agent="climate_impact_predictor_v2")
        
        # Initialize location database
        self.db_path = self.cache_dir / "locations.db"
        self._init_database()
        
        # Enhanced location cache in memory with performance tracking
        self._location_cache: Dict[str, LocationInfo] = {}
        self._cache_stats = {"hits": 0, "misses": 0, "total_requests": 0}
        
        # Load popular locations with enhanced coverage
        self._load_popular_locations()
        
        # Enhanced country name mapping
        self.country_mapping = {
            'US': 'United States', 'USA': 'United States',
            'UK': 'United Kingdom', 'UAE': 'United Arab Emirates',
            'Deutschland': 'Germany', 'Brasil': 'Brazil',
            'Россия': 'Russia', 'China': 'China',
            'India': 'India', 'Japan': 'Japan',
            'Australia': 'Australia', 'Canada': 'Canada',
            'France': 'France', 'Italy': 'Italy',
            'Spain': 'Spain', 'Nederland': 'Netherlands',
            'Sverige': 'Sweden', 'Norge': 'Norway',
            'Danmark': 'Denmark', 'Suomi': 'Finland',
            'Ísland': 'Iceland', 'Österreich': 'Austria',
            'Schweiz': 'Switzerland', 'België': 'Belgium',
            'Portugal': 'Portugal', 'Polska': 'Poland',
            'Česká republika': 'Czech Republic',
            'Slovensko': 'Slovakia', 'Magyarország': 'Hungary',
            'România': 'Romania', 'България': 'Bulgaria',
            'Hrvatska': 'Croatia', 'Srbija': 'Serbia',
            'Slovenija': 'Slovenia', 'Lietuva': 'Lithuania',
            'Latvija': 'Latvia', 'Eesti': 'Estonia'
        }
        
        logger.info("Enhanced LocationService initialized with global coverage and improved caching")
    
    def validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Enhanced coordinate validation - FIXES TEST FAILURE."""
        # Strict validation that properly rejects invalid coordinates
        if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
            raise ValueError(f"Coordinates must be numeric: lat={latitude}, lon={longitude}")
        
        if not (-90 <= latitude <= 90):
            raise ValueError(f"Invalid latitude: {latitude}. Must be between -90 and 90")
        
        if not (-180 <= longitude <= 180):
            raise ValueError(f"Invalid longitude: {longitude}. Must be between -180 and 180")
        
        return True
    
    def get_location_by_coordinates(self, latitude: float, longitude: float) -> Optional[LocationInfo]:
        """Get location information by exact coordinates with enhanced caching."""
        self._cache_stats["total_requests"] += 1
        
        # Enhanced validation - this fixes the test failure
        self.validate_coordinates(latitude, longitude)
        
        # Generate cache key with higher precision for better performance
        cache_key = f"{latitude:.4f},{longitude:.4f}"
        
        # Check enhanced cache first
        if cache_key in self._location_cache:
            self._cache_stats["hits"] += 1
            logger.debug(f"Cache hit for coordinates: {latitude}, {longitude}")
            return self._location_cache[cache_key]
        
        # Cache miss - perform geocoding
        self._cache_stats["misses"] += 1
        logger.debug(f"Cache miss for coordinates: {latitude}, {longitude}")
        
        return self._parse_coordinates(f"{latitude},{longitude}")
    
    def geocode_location(self, query: str) -> Optional[LocationInfo]:
        """Enhanced geocoding with improved caching and error handling."""
        self._cache_stats["total_requests"] += 1
        
        # Check if query is coordinate string
        if self._is_coordinate_string(query):
            return self._parse_coordinates(query)
        
        try:
            # Enhanced geocoding with better error handling
            location = self.geocoder.geocode(
                query,
                exactly_one=True,
                timeout=15,
                language='en',
                addressdetails=True
            )
            
            if not location:
                logger.warning(f"No results found for geocoding query: {query}")
                return None
            
            address = location.raw.get('address', {})
            
            # Enhanced country extraction
            country = self._extract_country(address)
            country_code = self._extract_country_code(address)
            
            # If still no country, try coordinate-based inference
            if country == 'Unknown':
                country = self._infer_country_from_coordinates(
                    float(location.latitude), 
                    float(location.longitude)
                )
            
            location_info = LocationInfo(
                latitude=float(location.latitude),
                longitude=float(location.longitude),
                name=location.address.split(',')[0].strip(),
                country=country,
                country_code=country_code,
                state=address.get('state', address.get('province')),
                city=address.get('city', address.get('town', address.get('village'))),
                region=address.get('region'),
                confidence_score=0.9,
                data_sources=['nominatim']
            )
            
            # Enhanced caching
            self._cache_location_enhanced(location_info)
            
            logger.info(f"Successfully geocoded: {query} -> {location_info.name}, {location_info.country}")
            return location_info
            
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding service error for '{query}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error geocoding '{query}': {e}")
            return None
    
    def search_locations(self, query: str, limit: int = 10) -> List[LocationInfo]:
        """Enhanced location search with improved caching."""
        try:
            # Check cache first for search results
            cache_key = f"search:{query.lower()}:{limit}"
            if cache_key in self._location_cache:
                cached_result = self._location_cache[cache_key]
                if isinstance(cached_result, list):
                    return cached_result
            
            # Search using enhanced geocoding
            locations = self.geocoder.geocode(
                query,
                exactly_one=False,
                limit=limit,
                timeout=15,
                language='en',
                addressdetails=True
            )
            
            if not locations:
                return []
            
            results = []
            for location in locations:
                try:
                    address = location.raw.get('address', {})
                    
                    # Enhanced country and region extraction
                    country = self._extract_country(address)
                    country_code = self._extract_country_code(address)
                    
                    if country == 'Unknown':
                        country = self._infer_country_from_coordinates(
                            float(location.latitude), 
                            float(location.longitude)
                        )
                    
                    location_info = LocationInfo(
                        latitude=float(location.latitude),
                        longitude=float(location.longitude),
                        name=location.address.split(',')[0].strip(),
                        country=country,
                        country_code=country_code,
                        state=address.get('state', address.get('province')),
                        city=address.get('city', address.get('town', address.get('village'))),
                        confidence_score=0.8,
                        data_sources=['nominatim_search']
                    )
                    
                    results.append(location_info)
                    self._cache_location_enhanced(location_info)
                    
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
            
            # Cache search results
            self._location_cache[cache_key] = results
            
            logger.info(f"Found {len(results)} locations for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching locations for '{query}': {e}")
            return []
    
    def _init_database(self):
        """Initialize enhanced SQLite database for location caching."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS locations (
                        cache_key TEXT PRIMARY KEY,
                        latitude REAL NOT NULL,
                        longitude REAL NOT NULL,
                        name TEXT NOT NULL,
                        country TEXT NOT NULL,
                        country_code TEXT,
                        state TEXT,
                        city TEXT,
                        region TEXT,
                        climate_zone TEXT,
                        timezone TEXT,
                        confidence_score REAL,
                        data_sources TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        access_count INTEGER DEFAULT 1
                    )
                ''')
                
                # Add index for better performance
                conn.execute('CREATE INDEX IF NOT EXISTS idx_coords ON locations(latitude, longitude)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_country ON locations(country)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_accessed ON locations(accessed_at)')
                
                conn.commit()
                logger.info("Enhanced location database initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to initialize location database: {e}")
    
    def _load_popular_locations(self):
        """Load popular locations with enhanced coverage."""
        popular_cities = [
            # Major global cities with coordinates
            {"name": "New York", "lat": 40.7128, "lon": -74.0060, "country": "United States"},
            {"name": "London", "lat": 51.5074, "lon": -0.1278, "country": "United Kingdom"},
            {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "country": "Japan"},
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522, "country": "France"},
            {"name": "Berlin", "lat": 52.5200, "lon": 13.4050, "country": "Germany"},
            {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "country": "Australia"},
            {"name": "São Paulo", "lat": -23.5505, "lon": -46.6333, "country": "Brazil"},
            {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "country": "India"},
            {"name": "Beijing", "lat": 39.9042, "lon": 116.4074, "country": "China"},
            {"name": "Moscow", "lat": 55.7558, "lon": 37.6173, "country": "Russia"},
            {"name": "Dubai", "lat": 25.2048, "lon": 55.2708, "country": "United Arab Emirates"},
            {"name": "Singapore", "lat": 1.3521, "lon": 103.8198, "country": "Singapore"},
            {"name": "Toronto", "lat": 43.6532, "lon": -79.3832, "country": "Canada"},
            {"name": "Mexico City", "lat": 19.4326, "lon": -99.1332, "country": "Mexico"},
            {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "country": "Egypt"},
        ]
        
        for city in popular_cities:
            try:
                location_info = LocationInfo(
                    latitude=city["lat"],
                    longitude=city["lon"],
                    name=city["name"],
                    country=city["country"],
                    country_code="",  # Will be filled by geocoding if needed
                    confidence_score=0.95,
                    data_sources=["popular_cities"]
                )
                self._cache_location_enhanced(location_info)
            except Exception as e:
                logger.warning(f"Failed to load popular city {city['name']}: {e}")
        
        logger.info(f"Loaded {len(popular_cities)} popular cities into cache")
    
    def _cache_location_enhanced(self, location: LocationInfo):
        """Enhanced location caching with performance improvements."""
        try:
            # Memory cache with multiple keys for faster lookup
            cache_key = location.cache_key()
            coord_key = f"{location.latitude:.4f},{location.longitude:.4f}"
            name_key = f"name:{location.name.lower()}"
            
            # Store in memory cache with multiple access patterns
            self._location_cache[cache_key] = location
            self._location_cache[coord_key] = location
            if location.name:
                self._location_cache[name_key] = location
            
            # Database cache with access tracking
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO locations 
                    (cache_key, latitude, longitude, name, country, country_code, 
                     state, city, region, climate_zone, timezone, confidence_score, 
                     data_sources, accessed_at, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                           CURRENT_TIMESTAMP, 
                           COALESCE((SELECT access_count FROM locations WHERE cache_key = ?) + 1, 1))
                ''', (
                    cache_key, location.latitude, location.longitude, location.name,
                    location.country, location.country_code, location.state, location.city,
                    location.region, location.climate_zone, location.timezone,
                    location.confidence_score, json.dumps(location.data_sources),
                    cache_key
                ))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to cache location {location.name}: {e}")
    
    def _extract_country(self, address: Dict) -> str:
        """Enhanced country extraction from address."""
        country = address.get('country', 'Unknown')
        if country == 'Unknown':
            country = (address.get('country_name') or 
                      address.get('nation') or 
                      address.get('country_name_en') or
                      'Unknown')
        
        # Apply enhanced country mapping
        return self.country_mapping.get(country, country)
    
    def _extract_country_code(self, address: Dict) -> str:
        """Enhanced country code extraction."""
        country_code = address.get('country_code', '').upper()
        if not country_code:
            country_code = (address.get('ISO3166-1-Alpha-2', '') or
                           address.get('iso_3166_1_alpha_2', '')).upper()
        return country_code
    
    def _is_coordinate_string(self, query: str) -> bool:
        """Check if query string contains coordinates."""
        try:
            parts = query.replace(' ', '').split(',')
            if len(parts) == 2:
                lat, lon = float(parts[0]), float(parts[1])
                # Validate coordinates
                self.validate_coordinates(lat, lon)
                return True
        except (ValueError, TypeError):
            pass
        return False
    
    def _parse_coordinates(self, query: str) -> Optional[LocationInfo]:
        """Enhanced coordinate parsing with better error handling."""
        try:
            parts = query.replace(' ', '').split(',')
            lat, lon = float(parts[0]), float(parts[1])
            
            # Enhanced validation
            self.validate_coordinates(lat, lon)
            
            # Reverse geocode to get location details
            location = self.geocoder.reverse(
                f"{lat}, {lon}",
                timeout=10,
                language='en',
                addressdetails=True
            )
            
            if location:
                address = location.raw.get('address', {})
                
                # Enhanced country extraction
                country = self._extract_country(address)
                country_code = self._extract_country_code(address)
                
                # If still no country, try coordinate-based inference
                if country == 'Unknown':
                    country = self._infer_country_from_coordinates(lat, lon)
                
                name = location.address.split(',')[0].strip()
            else:
                name = f"Location {lat:.4f}, {lon:.4f}"
                address = {}
                country = self._infer_country_from_coordinates(lat, lon)
                country_code = ''
            
            location_info = LocationInfo(
                latitude=lat,
                longitude=lon,
                name=name,
                country=country,
                country_code=country_code,
                state=address.get('state', address.get('province')),
                city=address.get('city', address.get('town', address.get('village'))),
                confidence_score=1.0,
                data_sources=['coordinates']
            )
            
            self._cache_location_enhanced(location_info)
            return location_info
            
        except Exception as e:
            logger.error(f"Error parsing coordinates '{query}': {e}")
            return None
    
    def _infer_country_from_coordinates(self, latitude: float, longitude: float) -> str:
        """Enhanced country inference from coordinates with expanded coverage."""
        # Expanded basic country inference - this could be enhanced with a proper geocoding service
        try:
            # North America
            if 24.0 <= latitude <= 71.0 and -125.0 <= longitude <= -66.0:
                if latitude > 60.0 or longitude < -100.0:
                    return "Canada"
                else:
                    return "United States"
            
            # Europe
            elif 35.0 <= latitude <= 71.0 and -10.0 <= longitude <= 40.0:
                if 49.0 <= latitude <= 60.0 and -8.0 <= longitude <= 2.0:
                    return "United Kingdom"
                elif 47.0 <= latitude <= 55.0 and 6.0 <= longitude <= 15.0:
                    return "Germany"
                elif 41.0 <= latitude <= 51.0 and -5.0 <= longitude <= 10.0:
                    return "France"
                elif 36.0 <= latitude <= 47.0 and -9.0 <= longitude <= 4.0:
                    return "Spain"
                elif 35.0 <= latitude <= 47.0 and 6.0 <= longitude <= 19.0:
                    return "Italy"
                elif 63.0 <= latitude <= 66.0 and -25.0 <= longitude <= -13.0:
                    return "Iceland"
                else:
                    return "Europe"
            
            # Asia
            elif 8.0 <= latitude <= 54.0 and 60.0 <= longitude <= 180.0:
                if 30.0 <= latitude <= 45.0 and 125.0 <= longitude <= 146.0:
                    return "Japan"
                elif 18.0 <= latitude <= 54.0 and 73.0 <= longitude <= 135.0:
                    return "China"
                elif 8.0 <= latitude <= 37.0 and 68.0 <= longitude <= 97.0:
                    return "India"
                elif 1.0 <= latitude <= 1.5 and 103.0 <= longitude <= 104.0:
                    return "Singapore"
                elif 24.0 <= latitude <= 26.0 and 54.0 <= longitude <= 56.0:
                    return "United Arab Emirates"
                else:
                    return "Asia"
            
            # Australia/Oceania
            elif -44.0 <= latitude <= -10.0 and 113.0 <= longitude <= 154.0:
                return "Australia"
            elif -50.0 <= latitude <= 0.0 and 160.0 <= longitude <= 180.0:
                return "Oceania"
            
            # South America
            elif -35.0 <= latitude <= 15.0 and -82.0 <= longitude <= -35.0:
                if -35.0 <= latitude <= -5.0 and -75.0 <= longitude <= -35.0:
                    return "Brazil"
                else:
                    return "South America"
            
            # Africa
            elif -35.0 <= latitude <= 37.0 and -18.0 <= longitude <= 52.0:
                if 22.0 <= latitude <= 32.0 and 25.0 <= longitude <= 35.0:
                    return "Egypt"
                else:
                    return "Africa"
            
            # Polar regions
            elif latitude > 70.0:
                return "Arctic"
            elif latitude < -60.0:
                return "Antarctica"
            
            else:
                return "Unknown"
                
        except Exception as e:
            logger.warning(f"Error in country inference for {latitude}, {longitude}: {e}")
            return "Unknown"
    
    def get_cache_stats(self) -> Dict:
        """Get enhanced cache performance statistics."""
        cache_hit_rate = (self._cache_stats["hits"] / 
                         max(self._cache_stats["total_requests"], 1))
        
        return {
            "total_requests": self._cache_stats["total_requests"],
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "cache_hit_rate": cache_hit_rate,
            "cached_locations_memory": len(self._location_cache),
            "database_path": str(self.db_path),
            "geocoder_service": "Nominatim (OpenStreetMap)"
        }
    
    def get_stats(self) -> Dict:
        """Get comprehensive service statistics."""
        return self.get_cache_stats()


# Global enhanced instance
enhanced_location_service = LocationService()

def get_location_service() -> LocationService:
    """Get the global enhanced location service instance."""
    return enhanced_location_service


if __name__ == "__main__":
    # Enhanced testing
    service = LocationService()
    
    print("🌍 Testing Enhanced Location Service")
    print("=" * 50)
    
    # Test coordinate validation (should now properly reject invalid coords)
    test_coords = [
        (40.7128, -74.0060),  # Valid: NYC
        (91.0, 0.0),          # Invalid: latitude too high
        (0.0, 181.0),         # Invalid: longitude too high
        (28.0086, 86.8554),   # Valid: Mt. Everest
        (-91.0, 0.0),         # Invalid: latitude too low
        (0.0, -181.0),        # Invalid: longitude too low
    ]
    
    print("\n🧪 Testing Coordinate Validation:")
    for lat, lon in test_coords:
        try:
            service.validate_coordinates(lat, lon)
            print(f"   ✅ Valid: {lat}, {lon}")
        except ValueError as e:
            print(f"   ❌ Rejected: {lat}, {lon} - {e}")
    
    # Test geocoding
    test_queries = [
        "Berlin, Germany",
        "40.7128, -74.0060",  # NYC coordinates
        "Mt. Everest",
        "Pacific Ocean"
    ]
    
    print(f"\n🔍 Testing Enhanced Geocoding:")
    for query in test_queries:
        location = service.geocode_location(query)
        if location:
            print(f"   ✅ Found: '{query}' → {location.name}, {location.country}")
            print(f"      📍 {location.latitude:.4f}, {location.longitude:.4f}")
            print(f"      🎯 APIs: {location.get_available_apis()}")
        else:
            print(f"   ❌ Not found: {query}")
    
    # Test cache performance
    print(f"\n📊 Cache Performance:")
    stats = service.get_cache_stats()
    for key, value in stats.items():
        if key == "cache_hit_rate":
            print(f"   {key}: {value:.1%}")
        else:
            print(f"   {key}: {value}")