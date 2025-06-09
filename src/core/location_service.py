"""
🌍 Dynamic Climate Impact Predictor - Global Location Service
src/core/location_service.py

Core location service supporting any coordinates globally with validation,
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
            
        # Validate coordinates
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


class LocationService:
    """
    Global location service for dynamic climate predictions.
    
    Features:
    - Geocoding and reverse geocoding
    - Location validation and suggestions
    - Climate zone classification
    - Data availability checking
    - Intelligent caching system
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        """Initialize location service with caching."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize geocoder with custom user agent
        self.geocoder = Nominatim(user_agent="climate_impact_predictor")
        
        # Initialize location database
        self.db_path = self.cache_dir / "locations.db"
        self._init_database()
        
        # Location cache in memory
        self._location_cache: Dict[str, LocationInfo] = {}
        
        # Load popular locations
        self._load_popular_locations()
        
        # Country name mapping for better recognition
        self.country_mapping = {
            'US': 'United States',
            'UK': 'United Kingdom', 
            'UAE': 'United Arab Emirates',
            'USA': 'United States',
            'Deutschland': 'Germany',
            'Brasil': 'Brazil',
            'Россия': 'Russia',
            'China': 'China',
            'India': 'India',
            'Japan': 'Japan',
            'Australia': 'Australia',
            'Canada': 'Canada',
            'France': 'France',
            'Italy': 'Italy',
            'Spain': 'Spain',
            'Nederland': 'Netherlands',
            'Sverige': 'Sweden',
            'Norge': 'Norway',
            'Danmark': 'Denmark',
            'Suomi': 'Finland',
            'Ísland': 'Iceland'
        }
        
        logger.info("LocationService initialized with global coverage")
    
    def _init_database(self):
        """Initialize SQLite database for location caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id TEXT PRIMARY KEY,
                latitude REAL,
                longitude REAL,
                name TEXT,
                country TEXT,
                country_code TEXT,
                state TEXT,
                city TEXT,
                region TEXT,
                climate_zone TEXT,
                timezone TEXT,
                has_air_quality BOOLEAN,
                has_meteorological BOOLEAN,
                has_climate_projections BOOLEAN,
                confidence_score REAL,
                data_sources TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_coordinates 
            ON locations (latitude, longitude)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_name 
            ON locations (name, country)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("Location database initialized")
    
    def _load_popular_locations(self):
        """Load popular locations into cache for quick access."""
        popular_cities = [
            {"name": "New York", "country": "United States", "lat": 40.7128, "lon": -74.0060},
            {"name": "London", "country": "United Kingdom", "lat": 51.5074, "lon": -0.1278},
            {"name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
            {"name": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050},
            {"name": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522},
            {"name": "Sydney", "country": "Australia", "lat": -33.8688, "lon": 151.2093},
            {"name": "São Paulo", "country": "Brazil", "lat": -23.5558, "lon": -46.6396},
            {"name": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
            {"name": "Beijing", "country": "China", "lat": 39.9042, "lon": 116.4074},
            {"name": "Cairo", "country": "Egypt", "lat": 30.0444, "lon": 31.2357},
        ]
        
        for city in popular_cities:
            try:
                location = LocationInfo(
                    latitude=city["lat"],
                    longitude=city["lon"],
                    name=city["name"],
                    country=city["country"],
                    country_code="",  # Will be filled by geocoding
                    city=city["name"],
                    confidence_score=1.0,
                    data_sources=["popular_cities"]
                )
                self._location_cache[location.cache_key()] = location
            except Exception as e:
                logger.warning(f"Failed to load popular city {city['name']}: {e}")
    
    def geocode_location(self, query: str, timeout: int = 10) -> Optional[LocationInfo]:
        """
        Geocode a location query into coordinates and details.
        
        Args:
            query: Location query (city, address, coordinates)
            timeout: Geocoding timeout in seconds
            
        Returns:
            LocationInfo object or None if not found
        """
        try:
            # Try to parse as coordinates first
            if self._is_coordinate_string(query):
                return self._parse_coordinates(query)
            
            # Use geocoding service
            location = self.geocoder.geocode(
                query, 
                timeout=timeout,
                exactly_one=True,
                language='en',
                addressdetails=True  # Request detailed address information
            )
            
            if not location:
                logger.warning(f"No location found for query: {query}")
                return None
            
            # Extract location details
            address = location.raw.get('address', {})
            
            # Better country extraction
            country = address.get('country', 'Unknown')
            if country == 'Unknown':
                # Try alternative fields
                country = (address.get('country_name') or 
                          address.get('nation') or 
                          'Unknown')
            
            # Apply country mapping
            country = self.country_mapping.get(country, country)
            
            # Extract country code
            country_code = address.get('country_code', '').upper()
            if not country_code:
                country_code = address.get('ISO3166-1-Alpha-2', '').upper()
            
            # If still no country, try to infer from coordinates
            if country == 'Unknown':
                country = self._infer_country_from_coordinates(float(location.latitude), float(location.longitude))
            
            location_info = LocationInfo(
                latitude=float(location.latitude),
                longitude=float(location.longitude),
                name=location.address.split(',')[0].strip(),
                country=country,
                country_code=country_code,
                state=address.get('state', address.get('province')),
                city=address.get('city', address.get('town', address.get('village'))),
                region=address.get('region'),
                confidence_score=0.9,  # High confidence for geocoded results
                data_sources=['nominatim']
            )
            
            # Cache the result
            self._cache_location(location_info)
            
            logger.info(f"Successfully geocoded: {query} -> {location_info.name}, {location_info.country}")
            return location_info
            
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding service error for '{query}': {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error geocoding '{query}': {e}")
            return None
    
    def _is_coordinate_string(self, query: str) -> bool:
        """Check if query string contains coordinates."""
        try:
            parts = query.replace(' ', '').split(',')
            if len(parts) == 2:
                float(parts[0])
                float(parts[1])
                return True
        except ValueError:
            pass
        return False
    
    def _parse_coordinates(self, query: str) -> Optional[LocationInfo]:
        """Parse coordinate string into LocationInfo."""
        try:
            parts = query.replace(' ', '').split(',')
            lat, lon = float(parts[0]), float(parts[1])
            
            # Reverse geocode to get location details
            location = self.geocoder.reverse(
                f"{lat}, {lon}",
                timeout=10,
                language='en',
                addressdetails=True  # Request detailed address information
            )
            
            if location:
                address = location.raw.get('address', {})
                
                # Better country extraction
                country = address.get('country', 'Unknown')
                if country == 'Unknown':
                    country = (address.get('country_name') or 
                              address.get('nation') or 
                              'Unknown')
                
                # Apply country mapping
                country = self.country_mapping.get(country, country)
                
                # Extract country code
                country_code = address.get('country_code', '').upper()
                if not country_code:
                    country_code = address.get('ISO3166-1-Alpha-2', '').upper()
                
                # If still no country, try to infer from coordinates
                if country == 'Unknown':
                    country = self._infer_country_from_coordinates(lat, lon)
                
                name = location.address.split(',')[0].strip()
            else:
                name = f"Location {lat:.4f}, {lon:.4f}"
                address = {}
                country = 'Unknown'
                country_code = ''
            
            location_info = LocationInfo(
                latitude=lat,
                longitude=lon,
                name=name,
                country=country,
                country_code=country_code,
                state=address.get('state', address.get('province')),
                city=address.get('city', address.get('town', address.get('village'))),
                confidence_score=1.0,  # High confidence for exact coordinates
                data_sources=['coordinates']
            )
            
            self._cache_location(location_info)
            return location_info
            
        except Exception as e:
            logger.error(f"Error parsing coordinates '{query}': {e}")
            return None
    
    def search_locations(self, query: str, limit: int = 10) -> List[LocationInfo]:
        """
        Search for multiple location matches.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of LocationInfo objects
        """
        try:
            # Check cache first
            cached_results = self._search_cache(query, limit)
            if cached_results:
                return cached_results
            
            # Search using geocoding
            locations = self.geocoder.geocode(
                query,
                exactly_one=False,
                limit=limit,
                timeout=15,
                language='en',
                addressdetails=True  # Request detailed address information
            )
            
            if not locations:
                return []
            
            results = []
            for location in locations:
                try:
                    address = location.raw.get('address', {})
                    
                    # Better country extraction
                    country = address.get('country', 'Unknown')
                    if country == 'Unknown':
                        country = (address.get('country_name') or 
                                  address.get('nation') or 
                                  'Unknown')
                    
                    # Apply country mapping
                    country = self.country_mapping.get(country, country)
                    
                    # Extract country code
                    country_code = address.get('country_code', '').upper()
                    if not country_code:
                        country_code = address.get('ISO3166-1-Alpha-2', '').upper()
                    
                    # If still no country, try to infer from coordinates
                    if country == 'Unknown':
                        country = self._infer_country_from_coordinates(float(location.latitude), float(location.longitude))
                    
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
                    self._cache_location(location_info)
                    
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
            
            logger.info(f"Found {len(results)} locations for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching locations for '{query}': {e}")
            return []
    
    def _search_cache(self, query: str, limit: int) -> List[LocationInfo]:
        """Search cached locations."""
        query_lower = query.lower()
        matches = []
        
        for location in self._location_cache.values():
            if (query_lower in location.name.lower() or 
                query_lower in location.country.lower() or
                (location.city and query_lower in location.city.lower())):
                matches.append(location)
                
            if len(matches) >= limit:
                break
        
        return matches
    
    def validate_coordinates(self, latitude: float, longitude: float) -> bool:
        """Validate coordinate ranges."""
        return (-90 <= latitude <= 90) and (-180 <= longitude <= 180)
    
    def get_location_by_coordinates(self, latitude: float, longitude: float) -> Optional[LocationInfo]:
        """Get location information by exact coordinates."""
        if not self.validate_coordinates(latitude, longitude):
            raise ValueError(f"Invalid coordinates: {latitude}, {longitude}")
        
        # Check cache first
        temp_location = LocationInfo(
            latitude=latitude, 
            longitude=longitude, 
            name="", 
            country="", 
            country_code=""  # Add required parameter
        )
        cache_key = temp_location.cache_key()
        
        if cache_key in self._location_cache:
            return self._location_cache[cache_key]
        
        # Use coordinate parsing
        return self._parse_coordinates(f"{latitude},{longitude}")
    
    def _cache_location(self, location: LocationInfo):
        """Cache location in memory and database."""
        cache_key = location.cache_key()
        self._location_cache[cache_key] = location
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO locations (
                    id, latitude, longitude, name, country, country_code,
                    state, city, region, climate_zone, timezone,
                    has_air_quality, has_meteorological, has_climate_projections,
                    confidence_score, data_sources
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key,
                location.latitude,
                location.longitude,
                location.name,
                location.country,
                location.country_code,
                location.state,
                location.city,
                location.region,
                location.climate_zone,
                location.timezone,
                location.has_air_quality,
                location.has_meteorological,
                location.has_climate_projections,
                location.confidence_score,
                json.dumps(location.data_sources)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error caching location: {e}")
    
    def suggest_locations(self, partial_query: str, limit: int = 5) -> List[str]:
        """
        Get location suggestions for autocomplete.
        
        Args:
            partial_query: Partial location name
            limit: Maximum suggestions
            
        Returns:
            List of location suggestion strings
        """
        suggestions = []
        query_lower = partial_query.lower()
        
        # Search cached locations
        for location in self._location_cache.values():
            location_str = f"{location.name}, {location.country}"
            if query_lower in location_str.lower():
                suggestions.append(location_str)
                
            if len(suggestions) >= limit:
                break
        
        return suggestions
    
    def _infer_country_from_coordinates(self, latitude: float, longitude: float) -> str:
        """Infer country from coordinates using rough geographic bounds."""
        # Very basic country inference - could be expanded
        if 24.0 <= latitude <= 71.0 and -125.0 <= longitude <= -66.0:
            return "United States"
        elif 49.0 <= latitude <= 60.0 and -8.0 <= longitude <= 2.0:
            return "United Kingdom"
        elif 47.0 <= latitude <= 55.0 and 6.0 <= longitude <= 15.0:
            return "Germany"
        elif 30.0 <= latitude <= 45.0 and 125.0 <= longitude <= 146.0:
            return "Japan"
        elif -44.0 <= latitude <= -10.0 and 113.0 <= longitude <= 154.0:
            return "Australia"
        elif 1.0 <= latitude <= 1.5 and 103.0 <= longitude <= 104.0:
            return "Singapore"
        elif 24.0 <= latitude <= 26.0 and 54.0 <= longitude <= 56.0:
            return "United Arab Emirates"
        elif 63.0 <= latitude <= 66.0 and -25.0 <= longitude <= -13.0:
            return "Iceland"
        elif -35.0 <= latitude <= -5.0 and -75.0 <= longitude <= -35.0:
            return "Brazil"
        else:
            return "Unknown"
    
    def get_stats(self) -> Dict:
        """Get location service statistics."""
        return {
            "cached_locations": len(self._location_cache),
            "database_path": str(self.db_path),
            "geocoder_service": "Nominatim (OpenStreetMap)",
            "popular_cities_loaded": len([l for l in self._location_cache.values() 
                                        if "popular_cities" in l.data_sources])
        }


# Global instance for easy import
location_service = LocationService()


def get_location_service() -> LocationService:
    """Get the global location service instance."""
    return location_service


if __name__ == "__main__":
    # Quick test
    service = LocationService()
    
    # Test geocoding
    print("🌍 Testing Location Service")
    print("=" * 50)
    
    test_queries = [
        "Berlin, Germany",
        "40.7128, -74.0060",  # New York coordinates
        "Sydney",
        "São Paulo, Brazil"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Testing: {query}")
        location = service.geocode_location(query)
        if location:
            print(f"✅ Found: {location.name}, {location.country}")
            print(f"📍 Coordinates: {location.latitude:.4f}, {location.longitude:.4f}")
        else:
            print(f"❌ Not found: {query}")
    
    print(f"\n📊 Service Stats: {service.get_stats()}")