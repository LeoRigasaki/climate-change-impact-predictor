#!/usr/bin/env python3
"""
🏗️ Location Database Builder
tools/build_location_db.py

Build comprehensive location database for fast autocomplete and suggestions.
Includes major cities, countries, and climate zones worldwide.
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.location_service import LocationService, LocationInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocationDatabaseBuilder:
    """Build comprehensive location database for global coverage."""
    
    def __init__(self):
        """Initialize database builder."""
        self.location_service = LocationService()
        self.built_locations: List[LocationInfo] = []
        
        # Major world cities by population and climate importance
        self.world_cities = [
            # Mega Cities (10M+ population)
            {"name": "Tokyo", "country": "Japan", "priority": 1},
            {"name": "Delhi", "country": "India", "priority": 1},
            {"name": "Shanghai", "country": "China", "priority": 1},
            {"name": "São Paulo", "country": "Brazil", "priority": 1},
            {"name": "Mexico City", "country": "Mexico", "priority": 1},
            {"name": "Cairo", "country": "Egypt", "priority": 1},
            {"name": "Beijing", "country": "China", "priority": 1},
            {"name": "Mumbai", "country": "India", "priority": 1},
            {"name": "Osaka", "country": "Japan", "priority": 1},
            {"name": "New York", "country": "United States", "priority": 1},
            {"name": "Karachi", "country": "Pakistan", "priority": 1},
            {"name": "Buenos Aires", "country": "Argentina", "priority": 1},
            {"name": "Chongqing", "country": "China", "priority": 1},
            {"name": "Istanbul", "country": "Turkey", "priority": 1},
            {"name": "Kolkata", "country": "India", "priority": 1},
            {"name": "Manila", "country": "Philippines", "priority": 1},
            {"name": "Lagos", "country": "Nigeria", "priority": 1},
            {"name": "Rio de Janeiro", "country": "Brazil", "priority": 1},
            {"name": "Tianjin", "country": "China", "priority": 1},
            {"name": "Kinshasa", "country": "Democratic Republic of the Congo", "priority": 1},
            {"name": "Guangzhou", "country": "China", "priority": 1},
            {"name": "Los Angeles", "country": "United States", "priority": 1},
            {"name": "Moscow", "country": "Russia", "priority": 1},
            {"name": "Shenzhen", "country": "China", "priority": 1},
            {"name": "Lahore", "country": "Pakistan", "priority": 1},
            {"name": "Bangalore", "country": "India", "priority": 1},
            {"name": "Paris", "country": "France", "priority": 1},
            {"name": "Bogotá", "country": "Colombia", "priority": 1},
            {"name": "Jakarta", "country": "Indonesia", "priority": 1},
            {"name": "Chennai", "country": "India", "priority": 1},
            {"name": "Lima", "country": "Peru", "priority": 1},
            {"name": "Bangkok", "country": "Thailand", "priority": 1},
            {"name": "Seoul", "country": "South Korea", "priority": 1},
            {"name": "Nagoya", "country": "Japan", "priority": 1},
            {"name": "Hyderabad", "country": "India", "priority": 1},
            {"name": "London", "country": "United Kingdom", "priority": 1},
            {"name": "Tehran", "country": "Iran", "priority": 1},
            {"name": "Chicago", "country": "United States", "priority": 1},
            {"name": "Chengdu", "country": "China", "priority": 1},
            {"name": "Nanjing", "country": "China", "priority": 1},
            {"name": "Wuhan", "country": "China", "priority": 1},
            {"name": "Ho Chi Minh City", "country": "Vietnam", "priority": 1},
            {"name": "Luanda", "country": "Angola", "priority": 1},
            {"name": "Ahmedabad", "country": "India", "priority": 1},
            {"name": "Kuala Lumpur", "country": "Malaysia", "priority": 1},
            {"name": "Xi'an", "country": "China", "priority": 1},
            {"name": "Hong Kong", "country": "China", "priority": 1},
            {"name": "Dongguan", "country": "China", "priority": 1},
            {"name": "Hangzhou", "country": "China", "priority": 1},
            {"name": "Foshan", "country": "China", "priority": 1},
            {"name": "Shenyang", "country": "China", "priority": 1},
            
            # Major Regional Centers (High Climate Importance)
            {"name": "Berlin", "country": "Germany", "priority": 2},
            {"name": "Madrid", "country": "Spain", "priority": 2},
            {"name": "Rome", "country": "Italy", "priority": 2},
            {"name": "Amsterdam", "country": "Netherlands", "priority": 2},
            {"name": "Vienna", "country": "Austria", "priority": 2},
            {"name": "Stockholm", "country": "Sweden", "priority": 2},
            {"name": "Copenhagen", "country": "Denmark", "priority": 2},
            {"name": "Oslo", "country": "Norway", "priority": 2},
            {"name": "Helsinki", "country": "Finland", "priority": 2},
            {"name": "Warsaw", "country": "Poland", "priority": 2},
            {"name": "Prague", "country": "Czech Republic", "priority": 2},
            {"name": "Budapest", "country": "Hungary", "priority": 2},
            {"name": "Zurich", "country": "Switzerland", "priority": 2},
            {"name": "Brussels", "country": "Belgium", "priority": 2},
            {"name": "Dublin", "country": "Ireland", "priority": 2},
            {"name": "Lisbon", "country": "Portugal", "priority": 2},
            {"name": "Athens", "country": "Greece", "priority": 2},
            {"name": "Kiev", "country": "Ukraine", "priority": 2},
            {"name": "Bucharest", "country": "Romania", "priority": 2},
            {"name": "Sofia", "country": "Bulgaria", "priority": 2},
            
            # North America
            {"name": "Toronto", "country": "Canada", "priority": 2},
            {"name": "Vancouver", "country": "Canada", "priority": 2},
            {"name": "Montreal", "country": "Canada", "priority": 2},
            {"name": "Houston", "country": "United States", "priority": 2},
            {"name": "Philadelphia", "country": "United States", "priority": 2},
            {"name": "Phoenix", "country": "United States", "priority": 2},
            {"name": "San Antonio", "country": "United States", "priority": 2},
            {"name": "San Diego", "country": "United States", "priority": 2},
            {"name": "Dallas", "country": "United States", "priority": 2},
            {"name": "San Jose", "country": "United States", "priority": 2},
            {"name": "Austin", "country": "United States", "priority": 2},
            {"name": "Jacksonville", "country": "United States", "priority": 2},
            {"name": "San Francisco", "country": "United States", "priority": 2},
            {"name": "Indianapolis", "country": "United States", "priority": 2},
            {"name": "Columbus", "country": "United States", "priority": 2},
            {"name": "Fort Worth", "country": "United States", "priority": 2},
            {"name": "Charlotte", "country": "United States", "priority": 2},
            {"name": "Detroit", "country": "United States", "priority": 2},
            {"name": "El Paso", "country": "United States", "priority": 2},
            {"name": "Memphis", "country": "United States", "priority": 2},
            {"name": "Seattle", "country": "United States", "priority": 2},
            {"name": "Denver", "country": "United States", "priority": 2},
            {"name": "Washington", "country": "United States", "priority": 2},
            {"name": "Boston", "country": "United States", "priority": 2},
            {"name": "Nashville", "country": "United States", "priority": 2},
            {"name": "Baltimore", "country": "United States", "priority": 2},
            {"name": "Oklahoma City", "country": "United States", "priority": 2},
            {"name": "Portland", "country": "United States", "priority": 2},
            {"name": "Las Vegas", "country": "United States", "priority": 2},
            {"name": "Milwaukee", "country": "United States", "priority": 2},
            {"name": "Albuquerque", "country": "United States", "priority": 2},
            {"name": "Tucson", "country": "United States", "priority": 2},
            {"name": "Fresno", "country": "United States", "priority": 2},
            {"name": "Sacramento", "country": "United States", "priority": 2},
            {"name": "Kansas City", "country": "United States", "priority": 2},
            {"name": "Mesa", "country": "United States", "priority": 2},
            {"name": "Atlanta", "country": "United States", "priority": 2},
            {"name": "Omaha", "country": "United States", "priority": 2},
            {"name": "Colorado Springs", "country": "United States", "priority": 2},
            {"name": "Raleigh", "country": "United States", "priority": 2},
            {"name": "Miami", "country": "United States", "priority": 2},
            {"name": "Long Beach", "country": "United States", "priority": 2},
            {"name": "Virginia Beach", "country": "United States", "priority": 2},
            {"name": "Oakland", "country": "United States", "priority": 2},
            {"name": "Minneapolis", "country": "United States", "priority": 2},
            {"name": "Tampa", "country": "United States", "priority": 2},
            {"name": "Tulsa", "country": "United States", "priority": 2},
            {"name": "Arlington", "country": "United States", "priority": 2},
            {"name": "New Orleans", "country": "United States", "priority": 2},
            {"name": "Wichita", "country": "United States", "priority": 2},
            {"name": "Cleveland", "country": "United States", "priority": 2},
            {"name": "Bakersfield", "country": "United States", "priority": 2},
            {"name": "Tampa", "country": "United States", "priority": 2},
            {"name": "Aurora", "country": "United States", "priority": 2},
            {"name": "Anaheim", "country": "United States", "priority": 2},
            {"name": "Honolulu", "country": "United States", "priority": 2},
            {"name": "Santa Ana", "country": "United States", "priority": 2},
            {"name": "Corpus Christi", "country": "United States", "priority": 2},
            {"name": "Riverside", "country": "United States", "priority": 2},
            {"name": "St. Louis", "country": "United States", "priority": 2},
            {"name": "Lexington", "country": "United States", "priority": 2},
            {"name": "Stockton", "country": "United States", "priority": 2},
            {"name": "Pittsburgh", "country": "United States", "priority": 2},
            {"name": "Saint Paul", "country": "United States", "priority": 2},
            {"name": "Cincinnati", "country": "United States", "priority": 2},
            {"name": "Anchorage", "country": "United States", "priority": 2},
            {"name": "Henderson", "country": "United States", "priority": 2},
            {"name": "Greensboro", "country": "United States", "priority": 2},
            {"name": "Plano", "country": "United States", "priority": 2},
            {"name": "Newark", "country": "United States", "priority": 2},
            {"name": "Toledo", "country": "United States", "priority": 2},
            {"name": "Lincoln", "country": "United States", "priority": 2},
            {"name": "Orlando", "country": "United States", "priority": 2},
            {"name": "Chula Vista", "country": "United States", "priority": 2},
            {"name": "Jersey City", "country": "United States", "priority": 2},
            {"name": "Chandler", "country": "United States", "priority": 2},
            {"name": "Laredo", "country": "United States", "priority": 2},
            {"name": "Madison", "country": "United States", "priority": 2},
            {"name": "Buffalo", "country": "United States", "priority": 2},
            {"name": "Lubbock", "country": "United States", "priority": 2},
            {"name": "Scottsdale", "country": "United States", "priority": 2},
            {"name": "Reno", "country": "United States", "priority": 2},
            {"name": "Glendale", "country": "United States", "priority": 2},
            {"name": "Gilbert", "country": "United States", "priority": 2},
            {"name": "Winston-Salem", "country": "United States", "priority": 2},
            {"name": "North Las Vegas", "country": "United States", "priority": 2},
            {"name": "Norfolk", "country": "United States", "priority": 2},
            {"name": "Chesapeake", "country": "United States", "priority": 2},
            {"name": "Garland", "country": "United States", "priority": 2},
            {"name": "Irving", "country": "United States", "priority": 2},
            {"name": "Hialeah", "country": "United States", "priority": 2},
            {"name": "Fremont", "country": "United States", "priority": 2},
            {"name": "Boise", "country": "United States", "priority": 2},
            {"name": "Richmond", "country": "United States", "priority": 2},
            {"name": "Baton Rouge", "country": "United States", "priority": 2},
            {"name": "Des Moines", "country": "United States", "priority": 2},
            {"name": "Spokane", "country": "United States", "priority": 2},
            {"name": "Modesto", "country": "United States", "priority": 2},
            {"name": "Fayetteville", "country": "United States", "priority": 2},
            {"name": "Tacoma", "country": "United States", "priority": 2},
            {"name": "Oxnard", "country": "United States", "priority": 2},
            {"name": "Fontana", "country": "United States", "priority": 2},
            {"name": "Columbus", "country": "United States", "priority": 2},
            {"name": "Montgomery", "country": "United States", "priority": 2},
            {"name": "Moreno Valley", "country": "United States", "priority": 2},
            {"name": "Shreveport", "country": "United States", "priority": 2},
            {"name": "Aurora", "country": "United States", "priority": 2},
            {"name": "Yonkers", "country": "United States", "priority": 2},
            {"name": "Akron", "country": "United States", "priority": 2},
            {"name": "Huntington Beach", "country": "United States", "priority": 2},
            {"name": "Little Rock", "country": "United States", "priority": 2},
            {"name": "Augusta", "country": "United States", "priority": 2},
            {"name": "Amarillo", "country": "United States", "priority": 2},
            {"name": "Glendale", "country": "United States", "priority": 2},
            {"name": "Mobile", "country": "United States", "priority": 2},
            {"name": "Grand Rapids", "country": "United States", "priority": 2},
            {"name": "Salt Lake City", "country": "United States", "priority": 2},
            {"name": "Tallahassee", "country": "United States", "priority": 2},
            {"name": "Huntsville", "country": "United States", "priority": 2},
            {"name": "Grand Prairie", "country": "United States", "priority": 2},
            {"name": "Knoxville", "country": "United States", "priority": 2},
            {"name": "Worcester", "country": "United States", "priority": 2},
            {"name": "Newport News", "country": "United States", "priority": 2},
            {"name": "Brownsville", "country": "United States", "priority": 2},
            {"name": "Overland Park", "country": "United States", "priority": 2},
            {"name": "Santa Clarita", "country": "United States", "priority": 2},
            {"name": "Providence", "country": "United States", "priority": 2},
            {"name": "Garden Grove", "country": "United States", "priority": 2},
            {"name": "Chattanooga", "country": "United States", "priority": 2},
            {"name": "Oceanside", "country": "United States", "priority": 2},
            {"name": "Jackson", "country": "United States", "priority": 2},
            {"name": "Fort Lauderdale", "country": "United States", "priority": 2},
            {"name": "Santa Rosa", "country": "United States", "priority": 2},
            {"name": "Rancho Cucamonga", "country": "United States", "priority": 2},
            {"name": "Port St. Lucie", "country": "United States", "priority": 2},
            {"name": "Tempe", "country": "United States", "priority": 2},
            {"name": "Ontario", "country": "United States", "priority": 2},
            {"name": "Vancouver", "country": "United States", "priority": 2},
            {"name": "Cape Coral", "country": "United States", "priority": 2},
            {"name": "Sioux Falls", "country": "United States", "priority": 2},
            {"name": "Springfield", "country": "United States", "priority": 2},
            {"name": "Peoria", "country": "United States", "priority": 2},
            {"name": "Pembroke Pines", "country": "United States", "priority": 2},
            {"name": "Elk Grove", "country": "United States", "priority": 2},
            {"name": "Salem", "country": "United States", "priority": 2},
            {"name": "Lancaster", "country": "United States", "priority": 2},
            {"name": "Corona", "country": "United States", "priority": 2},
            {"name": "Eugene", "country": "United States", "priority": 2},
            {"name": "Palmdale", "country": "United States", "priority": 2},
            {"name": "Salinas", "country": "United States", "priority": 2},
            {"name": "Springfield", "country": "United States", "priority": 2},
            {"name": "Pasadena", "country": "United States", "priority": 2},
            {"name": "Fort Collins", "country": "United States", "priority": 2},
            {"name": "Hayward", "country": "United States", "priority": 2},
            {"name": "Pomona", "country": "United States", "priority": 2},
            {"name": "Cary", "country": "United States", "priority": 2},
            {"name": "Rockford", "country": "United States", "priority": 2},
            {"name": "Alexandria", "country": "United States", "priority": 2},
            {"name": "Escondido", "country": "United States", "priority": 2},
            {"name": "McKinney", "country": "United States", "priority": 2},
            {"name": "Kansas City", "country": "United States", "priority": 2},
            {"name": "Joliet", "country": "United States", "priority": 2},
            {"name": "Sunnyvale", "country": "United States", "priority": 2},
            
            # Australia & Oceania
            {"name": "Sydney", "country": "Australia", "priority": 2},
            {"name": "Melbourne", "country": "Australia", "priority": 2},
            {"name": "Brisbane", "country": "Australia", "priority": 2},
            {"name": "Perth", "country": "Australia", "priority": 2},
            {"name": "Adelaide", "country": "Australia", "priority": 2},
            {"name": "Gold Coast", "country": "Australia", "priority": 2},
            {"name": "Newcastle", "country": "Australia", "priority": 2},
            {"name": "Canberra", "country": "Australia", "priority": 2},
            {"name": "Auckland", "country": "New Zealand", "priority": 2},
            {"name": "Wellington", "country": "New Zealand", "priority": 2},
            {"name": "Christchurch", "country": "New Zealand", "priority": 2},
            {"name": "Suva", "country": "Fiji", "priority": 3},
            {"name": "Port Moresby", "country": "Papua New Guinea", "priority": 3},
            
            # Africa
            {"name": "Cape Town", "country": "South Africa", "priority": 2},
            {"name": "Johannesburg", "country": "South Africa", "priority": 2},
            {"name": "Durban", "country": "South Africa", "priority": 2},
            {"name": "Nairobi", "country": "Kenya", "priority": 2},
            {"name": "Addis Ababa", "country": "Ethiopia", "priority": 2},
            {"name": "Accra", "country": "Ghana", "priority": 2},
            {"name": "Dakar", "country": "Senegal", "priority": 2},
            {"name": "Casablanca", "country": "Morocco", "priority": 2},
            {"name": "Tunis", "country": "Tunisia", "priority": 2},
            {"name": "Algiers", "country": "Algeria", "priority": 2},
            {"name": "Khartoum", "country": "Sudan", "priority": 3},
            {"name": "Kampala", "country": "Uganda", "priority": 3},
            {"name": "Dar es Salaam", "country": "Tanzania", "priority": 3},
            {"name": "Maputo", "country": "Mozambique", "priority": 3},
            {"name": "Harare", "country": "Zimbabwe", "priority": 3},
            {"name": "Lusaka", "country": "Zambia", "priority": 3},
            {"name": "Antananarivo", "country": "Madagascar", "priority": 3},
            
            # Middle East
            {"name": "Dubai", "country": "United Arab Emirates", "priority": 2},
            {"name": "Riyadh", "country": "Saudi Arabia", "priority": 2},
            {"name": "Tel Aviv", "country": "Israel", "priority": 2},
            {"name": "Jerusalem", "country": "Israel", "priority": 2},
            {"name": "Amman", "country": "Jordan", "priority": 2},
            {"name": "Kuwait City", "country": "Kuwait", "priority": 2},
            {"name": "Doha", "country": "Qatar", "priority": 2},
            {"name": "Manama", "country": "Bahrain", "priority": 3},
            {"name": "Muscat", "country": "Oman", "priority": 3},
            {"name": "Baghdad", "country": "Iraq", "priority": 3},
            {"name": "Damascus", "country": "Syria", "priority": 3},
            {"name": "Beirut", "country": "Lebanon", "priority": 3},
            
            # South America
            {"name": "Caracas", "country": "Venezuela", "priority": 2},
            {"name": "Quito", "country": "Ecuador", "priority": 2},
            {"name": "La Paz", "country": "Bolivia", "priority": 2},
            {"name": "Asunción", "country": "Paraguay", "priority": 3},
            {"name": "Montevideo", "country": "Uruguay", "priority": 3},
            {"name": "Santiago", "country": "Chile", "priority": 2},
            {"name": "Brasília", "country": "Brazil", "priority": 2},
            {"name": "Salvador", "country": "Brazil", "priority": 2},
            {"name": "Fortaleza", "country": "Brazil", "priority": 2},
            {"name": "Belo Horizonte", "country": "Brazil", "priority": 2},
            {"name": "Manaus", "country": "Brazil", "priority": 2},
            {"name": "Curitiba", "country": "Brazil", "priority": 2},
            {"name": "Recife", "country": "Brazil", "priority": 2},
            {"name": "Porto Alegre", "country": "Brazil", "priority": 2},
            {"name": "Belém", "country": "Brazil", "priority": 3},
            {"name": "Goiânia", "country": "Brazil", "priority": 3},
            {"name": "Guarulhos", "country": "Brazil", "priority": 3},
            
            # Central America & Caribbean
            {"name": "Guatemala City", "country": "Guatemala", "priority": 3},
            {"name": "San Salvador", "country": "El Salvador", "priority": 3},
            {"name": "Tegucigalpa", "country": "Honduras", "priority": 3},
            {"name": "Managua", "country": "Nicaragua", "priority": 3},
            {"name": "San José", "country": "Costa Rica", "priority": 3},
            {"name": "Panama City", "country": "Panama", "priority": 3},
            {"name": "Havana", "country": "Cuba", "priority": 2},
            {"name": "Santo Domingo", "country": "Dominican Republic", "priority": 3},
            {"name": "Port-au-Prince", "country": "Haiti", "priority": 3},
            {"name": "Kingston", "country": "Jamaica", "priority": 3},
            {"name": "San Juan", "country": "Puerto Rico", "priority": 3},
            
            # Asia-Pacific
            {"name": "Singapore", "country": "Singapore", "priority": 2},
            {"name": "Hanoi", "country": "Vietnam", "priority": 2},
            {"name": "Yangon", "country": "Myanmar", "priority": 3},
            {"name": "Phnom Penh", "country": "Cambodia", "priority": 3},
            {"name": "Vientiane", "country": "Laos", "priority": 3},
            {"name": "Ulaanbaatar", "country": "Mongolia", "priority": 3},
            {"name": "Almaty", "country": "Kazakhstan", "priority": 3},
            {"name": "Tashkent", "country": "Uzbekistan", "priority": 3},
            {"name": "Bishkek", "country": "Kyrgyzstan", "priority": 3},
            {"name": "Dushanbe", "country": "Tajikistan", "priority": 3},
            {"name": "Ashgabat", "country": "Turkmenistan", "priority": 3},
            {"name": "Kabul", "country": "Afghanistan", "priority": 3},
            {"name": "Islamabad", "country": "Pakistan", "priority": 2},
            {"name": "Kathmandu", "country": "Nepal", "priority": 3},
            {"name": "Thimphu", "country": "Bhutan", "priority": 3},
            {"name": "Colombo", "country": "Sri Lanka", "priority": 3},
            {"name": "Malé", "country": "Maldives", "priority": 3},
            
            # Climate-Critical Locations
            {"name": "Reykjavik", "country": "Iceland", "priority": 2},
            {"name": "Nuuk", "country": "Greenland", "priority": 3},
            {"name": "Longyearbyen", "country": "Norway", "priority": 3},
            {"name": "Fairbanks", "country": "United States", "priority": 3},
            {"name": "Tromsø", "country": "Norway", "priority": 3},
            {"name": "Murmansk", "country": "Russia", "priority": 3},
            {"name": "Iqaluit", "country": "Canada", "priority": 3},
            {"name": "Yellowknife", "country": "Canada", "priority": 3},
            {"name": "Whitehorse", "country": "Canada", "priority": 3},
            {"name": "Ushuaia", "country": "Argentina", "priority": 3},
            {"name": "Punta Arenas", "country": "Chile", "priority": 3},
            {"name": "McMurdo Station", "country": "Antarctica", "priority": 3},
            {"name": "Alice Springs", "country": "Australia", "priority": 3},
            {"name": "Darwin", "country": "Australia", "priority": 3},
            {"name": "Lhasa", "country": "Tibet", "priority": 3},
            {"name": "Kathmandu", "country": "Nepal", "priority": 3},
            {"name": "La Rinconada", "country": "Peru", "priority": 3},
        ]
        
        # Climate zones mapping
        self.climate_zones = {
            "tropical": ["Singapore", "Jakarta", "Mumbai", "Bangkok", "Manila", "Ho Chi Minh City", "Kuala Lumpur"],
            "desert": ["Dubai", "Riyadh", "Phoenix", "Las Vegas", "Alice Springs", "Cairo"],
            "mediterranean": ["Los Angeles", "San Francisco", "Madrid", "Rome", "Athens", "Tel Aviv"],
            "temperate": ["London", "Paris", "Berlin", "New York", "Tokyo", "Seoul"],
            "continental": ["Moscow", "Beijing", "Chicago", "Toronto", "Warsaw"],
            "subarctic": ["Fairbanks", "Yellowknife", "Murmansk", "Tromsø"],
            "arctic": ["Reykjavik", "Nuuk", "Longyearbyen", "Iqaluit"],
            "polar": ["McMurdo Station"],
            "mountain": ["Denver", "Lhasa", "Kathmandu", "La Rinconada"],
            "coastal": ["Sydney", "Cape Town", "Vancouver", "Miami", "Lisbon"]
        }
        
        logger.info(f"LocationDatabaseBuilder initialized with {len(self.world_cities)} cities")
    
    async def build_database(self, priority_filter: int = 3, batch_size: int = 10) -> Dict[str, Any]:
        """
        Build comprehensive location database.
        
        Args:
            priority_filter: Include cities with priority <= this value (1=highest)
            batch_size: Number of cities to process per batch
            
        Returns:
            Build summary statistics
        """
        logger.info("🏗️ Starting Location Database Build")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        # Filter cities by priority
        cities_to_process = [
            city for city in self.world_cities 
            if city.get("priority", 3) <= priority_filter
        ]
        
        logger.info(f"Processing {len(cities_to_process)} cities (priority <= {priority_filter})")
        
        # Process in batches
        successful = 0
        failed = 0
        batch_count = 0
        
        for i in range(0, len(cities_to_process), batch_size):
            batch = cities_to_process[i:i + batch_size]
            batch_count += 1
            
            logger.info(f"\n🔄 Processing Batch {batch_count} ({len(batch)} cities)")
            
            for city in batch:
                try:
                    query = f"{city['name']}, {city['country']}"
                    location = self.location_service.geocode_location(query)
                    
                    if location:
                        # Add climate zone if known
                        self._add_climate_zone(location, city['name'])
                        
                        # Add priority and build metadata
                        location.data_sources.append(f"world_cities_p{city['priority']}")
                        
                        self.built_locations.append(location)
                        successful += 1
                        
                        logger.info(f"✅ {location.name}, {location.country}")
                    else:
                        failed += 1
                        logger.warning(f"❌ Failed: {query}")
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"❌ Error processing {city['name']}: {e}")
                
                # Brief pause between requests
                await asyncio.sleep(0.5)
            
            # Longer pause between batches
            if i + batch_size < len(cities_to_process):
                logger.info(f"⏸️ Batch complete. Pausing 2 seconds...")
                await asyncio.sleep(2)
        
        # Build summary
        summary = {
            "total_processed": len(cities_to_process),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(cities_to_process) if cities_to_process else 0,
            "priority_filter": priority_filter,
            "processing_time": time.time() - start_time,
            "locations_built": len(self.built_locations),
            "climate_zones_assigned": len([l for l in self.built_locations if l.climate_zone]),
            "continents_covered": len(set(self._get_continent(l) for l in self.built_locations)),
        }
        
        return summary
    
    def _add_climate_zone(self, location: LocationInfo, city_name: str):
        """Add climate zone information to location."""
        for zone, cities in self.climate_zones.items():
            if city_name in cities:
                location.climate_zone = zone
                break
        
        # Auto-detect based on coordinates if not found
        if not location.climate_zone:
            location.climate_zone = self._auto_detect_climate_zone(location)
    
    def _auto_detect_climate_zone(self, location: LocationInfo) -> str:
        """Auto-detect climate zone based on coordinates."""
        lat = abs(location.latitude)
        
        if lat >= 66.5:
            return "arctic"
        elif lat >= 60:
            return "subarctic"
        elif lat >= 50:
            return "continental"
        elif lat >= 30:
            return "temperate"
        elif lat >= 23.5:
            return "subtropical"
        else:
            return "tropical"
    
    def _get_continent(self, location: LocationInfo) -> str:
        """Determine continent from location."""
        continent_mapping = {
            # North America
            "United States": "North America", "Canada": "North America", "Mexico": "North America",
            "Guatemala": "North America", "Belize": "North America", "El Salvador": "North America",
            "Honduras": "North America", "Nicaragua": "North America", "Costa Rica": "North America",
            "Panama": "North America", "Cuba": "North America", "Jamaica": "North America",
            "Haiti": "North America", "Dominican Republic": "North America", "Bahamas": "North America",
            "Puerto Rico": "North America",
            
            # South America
            "Brazil": "South America", "Argentina": "South America", "Chile": "South America",
            "Peru": "South America", "Colombia": "South America", "Venezuela": "South America",
            "Ecuador": "South America", "Bolivia": "South America", "Paraguay": "South America",
            "Uruguay": "South America", "Guyana": "South America", "Suriname": "South America",
            "French Guiana": "South America",
            
            # Europe
            "United Kingdom": "Europe", "France": "Europe", "Germany": "Europe", "Italy": "Europe",
            "Spain": "Europe", "Portugal": "Europe", "Netherlands": "Europe", "Belgium": "Europe",
            "Switzerland": "Europe", "Austria": "Europe", "Poland": "Europe", "Czech Republic": "Europe",
            "Hungary": "Europe", "Romania": "Europe", "Bulgaria": "Europe", "Greece": "Europe",
            "Sweden": "Europe", "Norway": "Europe", "Denmark": "Europe", "Finland": "Europe",
            "Iceland": "Europe", "Ireland": "Europe", "Lithuania": "Europe", "Latvia": "Europe",
            "Estonia": "Europe", "Slovakia": "Europe", "Slovenia": "Europe", "Croatia": "Europe",
            "Serbia": "Europe", "Bosnia and Herzegovina": "Europe", "Montenegro": "Europe",
            "Albania": "Europe", "North Macedonia": "Europe", "Moldova": "Europe", "Ukraine": "Europe",
            "Belarus": "Europe", "Russia": "Europe",
            
            # Asia
            "China": "Asia", "India": "Asia", "Japan": "Asia", "South Korea": "Asia",
            "Indonesia": "Asia", "Thailand": "Asia", "Vietnam": "Asia", "Philippines": "Asia",
            "Malaysia": "Asia", "Singapore": "Asia", "Myanmar": "Asia", "Cambodia": "Asia",
            "Laos": "Asia", "Bangladesh": "Asia", "Pakistan": "Asia", "Afghanistan": "Asia",
            "Iran": "Asia", "Iraq": "Asia", "Turkey": "Asia", "Saudi Arabia": "Asia",
            "United Arab Emirates": "Asia", "Qatar": "Asia", "Kuwait": "Asia", "Bahrain": "Asia",
            "Oman": "Asia", "Yemen": "Asia", "Jordan": "Asia", "Lebanon": "Asia", "Syria": "Asia",
            "Israel": "Asia", "Palestine": "Asia", "Kazakhstan": "Asia", "Uzbekistan": "Asia",
            "Turkmenistan": "Asia", "Kyrgyzstan": "Asia", "Tajikistan": "Asia", "Mongolia": "Asia",
            "North Korea": "Asia", "Nepal": "Asia", "Bhutan": "Asia", "Sri Lanka": "Asia",
            "Maldives": "Asia",
            
            # Africa
            "Nigeria": "Africa", "Egypt": "Africa", "South Africa": "Africa", "Kenya": "Africa",
            "Ghana": "Africa", "Morocco": "Africa", "Algeria": "Africa", "Tunisia": "Africa",
            "Libya": "Africa", "Sudan": "Africa", "Ethiopia": "Africa", "Tanzania": "Africa",
            "Uganda": "Africa", "Angola": "Africa", "Mozambique": "Africa", "Madagascar": "Africa",
            "Cameroon": "Africa", "Ivory Coast": "Africa", "Niger": "Africa", "Burkina Faso": "Africa",
            "Mali": "Africa", "Malawi": "Africa", "Zambia": "Africa", "Senegal": "Africa",
            "Somalia": "Africa", "Chad": "Africa", "Guinea": "Africa", "Rwanda": "Africa",
            "Benin": "Africa", "Burundi": "Africa", "Tunisia": "Africa", "Togo": "Africa",
            "Sierra Leone": "Africa", "Liberia": "Africa", "Central African Republic": "Africa",
            "Mauritania": "Africa", "Eritrea": "Africa", "Gambia": "Africa", "Botswana": "Africa",
            "Gabon": "Africa", "Lesotho": "Africa", "Guinea-Bissau": "Africa", "Equatorial Guinea": "Africa",
            "Mauritius": "Africa", "Eswatini": "Africa", "Djibouti": "Africa", "Comoros": "Africa",
            "Cape Verde": "Africa", "São Tomé and Príncipe": "Africa", "Seychelles": "Africa",
            "Democratic Republic of the Congo": "Africa", "Republic of the Congo": "Africa",
            "Zimbabwe": "Africa", "Namibia": "Africa",
            
            # Oceania
            "Australia": "Oceania", "New Zealand": "Oceania", "Papua New Guinea": "Oceania",
            "Fiji": "Oceania", "Solomon Islands": "Oceania", "Vanuatu": "Oceania",
            "Samoa": "Oceania", "Kiribati": "Oceania", "Tonga": "Oceania", "Micronesia": "Oceania",
            "Palau": "Oceania", "Marshall Islands": "Oceania", "Nauru": "Oceania", "Tuvalu": "Oceania",
            
            # Special cases
            "Antarctica": "Antarctica", "Greenland": "North America", "Tibet": "Asia"
        }
        
        return continent_mapping.get(location.country, "Unknown")
    
    def save_database(self, filename: str = None) -> Path:
        """Save built database to JSON file."""
        if filename is None:
            filename = f"location_database_{int(time.time())}.json"
        
        output_path = Path("data/cache") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        database = {
            "metadata": {
                "version": "1.0",
                "build_timestamp": time.time(),
                "total_locations": len(self.built_locations),
                "climate_zones": list(self.climate_zones.keys()),
                "data_sources": list(set(
                    source for location in self.built_locations 
                    for source in location.data_sources
                ))
            },
            "locations": [location.to_dict() for location in self.built_locations],
            "climate_zones": self.climate_zones,
            "statistics": {
                "by_continent": self._get_continent_stats(),
                "by_climate_zone": self._get_climate_zone_stats(),
                "by_priority": self._get_priority_stats()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(database, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Database saved to: {output_path}")
        return output_path
    
    def _get_continent_stats(self) -> Dict[str, int]:
        """Get location count by continent."""
        stats = {}
        for location in self.built_locations:
            continent = self._get_continent(location)
            stats[continent] = stats.get(continent, 0) + 1
        return stats
    
    def _get_climate_zone_stats(self) -> Dict[str, int]:
        """Get location count by climate zone."""
        stats = {}
        for location in self.built_locations:
            zone = location.climate_zone or "unknown"
            stats[zone] = stats.get(zone, 0) + 1
        return stats
    
    def _get_priority_stats(self) -> Dict[str, int]:
        """Get location count by priority level."""
        stats = {}
        for location in self.built_locations:
            # Extract priority from data sources
            priority = 3  # default
            for source in location.data_sources:
                if "world_cities_p" in source:
                    try:
                        priority = int(source.split("p")[1])
                        break
                    except (IndexError, ValueError):
                        pass
            
            priority_label = f"priority_{priority}"
            stats[priority_label] = stats.get(priority_label, 0) + 1
        return stats
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print formatted build summary."""
        print("\n" + "=" * 60)
        print("🏗️ LOCATION DATABASE BUILD SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 PROCESSING RESULTS:")
        print(f"   Total Processed: {summary['total_processed']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Processing Time: {summary['processing_time']:.1f} seconds")
        
        print(f"\n🌍 DATABASE CONTENTS:")
        print(f"   Locations Built: {summary['locations_built']}")
        print(f"   Climate Zones Assigned: {summary['climate_zones_assigned']}")
        print(f"   Continents Covered: {summary['continents_covered']}")
        print(f"   Priority Filter: <= {summary['priority_filter']}")
        
        # Additional statistics
        if self.built_locations:
            continent_stats = self._get_continent_stats()
            climate_stats = self._get_climate_zone_stats()
            
            print(f"\n🌍 BY CONTINENT:")
            for continent, count in sorted(continent_stats.items()):
                print(f"   {continent}: {count} locations")
            
            print(f"\n🌡️ BY CLIMATE ZONE:")
            for zone, count in sorted(climate_stats.items()):
                print(f"   {zone}: {count} locations")
        
        print("=" * 60)


async def main():
    """Main database building function."""
    print("🏗️ Location Database Builder")
    print("Building comprehensive global location database...")
    
    builder = LocationDatabaseBuilder()
    
    try:
        # Build database with priority 2 and below (major cities + regional centers)
        summary = await builder.build_database(priority_filter=2, batch_size=5)
        
        # Print summary
        builder.print_summary(summary)
        
        # Save database
        db_path = builder.save_database()
        
        # Success criteria
        if summary['success_rate'] >= 0.8 and summary['locations_built'] >= 100:
            print(f"\n✅ DATABASE BUILD SUCCESSFUL!")
            print(f"   {summary['locations_built']} locations ready for global climate predictions")
            return True
        else:
            print(f"\n⚠️ DATABASE BUILD NEEDS IMPROVEMENT")
            print(f"   Target: 80% success rate, 100+ locations")
            print(f"   Actual: {summary['success_rate']:.1%} success rate, {summary['locations_built']} locations")
            return False
            
    except KeyboardInterrupt:
        print("\n🛑 Database build interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Database build failed: {e}")
        logger.error(f"Main build error: {e}")
        return False


def build_priority_subset(priority: int):
    """Build database for specific priority level."""
    print(f"🏗️ Building Priority {priority} Location Database")
    
    async def build():
        builder = LocationDatabaseBuilder()
        summary = await builder.build_database(priority_filter=priority, batch_size=10)
        builder.print_summary(summary)
        builder.save_database(f"location_db_priority_{priority}.json")
        return summary
    
    return asyncio.run(build())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build global location database")
    parser.add_argument("--priority", type=int, default=2, 
                       help="Maximum priority level to include (1=highest, 3=lowest)")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Number of locations to process per batch")
    parser.add_argument("--quick", action="store_true",
                       help="Build priority 1 only (mega cities)")
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick build - mega cities only
        result = build_priority_subset(1)
        sys.exit(0 if result['success_rate'] >= 0.8 else 1)
    else:
        # Full build
        result = asyncio.run(main())
        sys.exit(0 if result else 1)