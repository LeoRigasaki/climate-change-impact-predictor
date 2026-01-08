# app/location_picker.py
#!/usr/bin/env python3
"""
Climate Impact Predictor - Professional ML Prediction Interface
app/location_picker.py

Clean, scientific interface for climate prediction with ML integration.
"""

import sys
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import asyncio
import time
import requests
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.location_service import LocationService, LocationInfo
from src.core.data_manager import ClimateDataManager
from src.core.pipeline import ClimateDataPipeline
from src.features.universal_engine import UniversalFeatureEngine

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Climate Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PROFESSIONAL STYLING - Clean Scientific Design
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Root Variables */
    :root {
        --bg-primary: #0a0f1a;
        --bg-secondary: #111827;
        --bg-card: #1a2234;
        --bg-elevated: #243044;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-primary: #06b6d4;
        --accent-secondary: #0891b2;
        --accent-success: #10b981;
        --accent-warning: #f59e0b;
        --accent-error: #ef4444;
        --border-subtle: #1e293b;
        --border-default: #334155;
    }

    /* Base Typography */
    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Page Header */
    .page-header {
        text-align: center;
        padding: 2rem 0 3rem;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 2rem;
    }

    .page-header h1 {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.025em;
        margin: 0 0 0.5rem;
    }

    .page-header p {
        font-size: 1rem;
        color: var(--text-secondary);
        margin: 0;
    }

    /* Section Headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin: 2rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* Cards */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .card-header {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* Data Display */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 1rem;
    }

    .metric-item {
        background: var(--bg-elevated);
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }

    .metric-label {
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        color: var(--text-primary);
    }

    .metric-value.large {
        font-size: 2rem;
    }

    .metric-unit {
        font-size: 0.875rem;
        color: var(--text-secondary);
    }

    /* Location Card */
    .location-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.75rem 0;
    }

    .location-card h4 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.5rem;
    }

    .location-card p {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin: 0.25rem 0;
    }

    .location-card .coord {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.25rem 0.625rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .status-online {
        background: rgba(16, 185, 129, 0.15);
        color: var(--accent-success);
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: var(--accent-error);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .status-available {
        background: rgba(6, 182, 212, 0.15);
        color: var(--accent-primary);
        border: 1px solid rgba(6, 182, 212, 0.3);
    }

    /* Data Source Indicators */
    .source-indicator {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.125rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .source-available {
        background: rgba(16, 185, 129, 0.1);
        color: var(--accent-success);
        border: 1px solid rgba(16, 185, 129, 0.2);
    }

    .source-unavailable {
        background: rgba(239, 68, 68, 0.1);
        color: var(--accent-error);
        border: 1px solid rgba(239, 68, 68, 0.2);
    }

    /* Controls Panel */
    .controls-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }

    .controls-panel h3 {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 1rem;
    }

    /* Results Panel */
    .results-panel {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-elevated) 100%);
        border: 1px solid var(--border-default);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }

    .results-panel h2 {
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 1.25rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    /* Prediction Results */
    .prediction-result {
        background: var(--bg-secondary);
        border-radius: 6px;
        padding: 1.25rem;
        margin: 1rem 0;
    }

    .prediction-title {
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }

    /* Table Styling */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }

    .data-table th {
        text-align: left;
        padding: 0.75rem;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        border-bottom: 1px solid var(--border-subtle);
    }

    .data-table td {
        padding: 0.75rem;
        color: var(--text-secondary);
        border-bottom: 1px solid var(--border-subtle);
    }

    .data-table td.mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-subtle);
    }

    section[data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem;
    }

    .sidebar-section {
        margin-bottom: 1.5rem;
    }

    .sidebar-title {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }

    /* Button Overrides */
    .stButton > button {
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        border-radius: 6px;
        transition: all 0.15s ease;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent-primary);
        border: none;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--accent-secondary);
    }

    /* Input Styling */
    .stTextInput > div > div > input {
        font-family: 'DM Sans', sans-serif;
        border-radius: 6px;
        border: 1px solid var(--border-default);
        background: var(--bg-elevated);
    }

    .stSelectbox > div > div {
        border-radius: 6px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        margin-top: 3rem;
        border-top: 1px solid var(--border-subtle);
        font-size: 0.8rem;
        color: var(--text-muted);
    }

    /* Plotly Chart Containers */
    .js-plotly-plot {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Welcome Panel */
    .welcome-panel {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
    }

    .welcome-panel h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.75rem;
    }

    .welcome-panel p {
        color: var(--text-secondary);
        margin: 0.5rem 0;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin-top: 1.5rem;
    }

    .feature-item {
        background: var(--bg-elevated);
        border-radius: 6px;
        padding: 1rem;
        font-size: 0.875rem;
        color: var(--text-secondary);
    }

    /* Streamlit specific overrides */
    .stMarkdown {
        color: var(--text-secondary);
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }

    .stAlert {
        border-radius: 6px;
    }

    /* Hide empty elements */
    .element-container:empty {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SERVICE CLASSES
# =============================================================================

class MLPredictionService:
    """Service for ML prediction API integration."""

    def __init__(self, api_base_url: str = None):
        self.api_base_url = api_base_url or os.getenv("API_BASE_URL", "http://localhost:8000")

    def predict_basic_climate(self, city: str) -> Dict[str, Any]:
        """Get basic climate prediction from API."""
        try:
            response = requests.post(
                f"{self.api_base_url}/predict/basic",
                json={"city": city},
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict_lstm_forecast(self, city: str, days: int = 7) -> Dict[str, Any]:
        """Get LSTM forecast from API."""
        try:
            response = requests.post(
                f"{self.api_base_url}/predict/forecast",
                json={"city": city, "days": days},
                timeout=30
            )
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_api_health(self) -> bool:
        """Check if API is available."""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            return response.status_code == 200
        except:
            return False


class WorldCapitalsService:
    """Service for world capitals comparison data."""

    def __init__(self):
        self.capitals_data = {
            "Washington, D.C.": {"country": "United States", "lat": 38.9072, "lon": -77.0369, "continent": "North America"},
            "London": {"country": "United Kingdom", "lat": 51.5074, "lon": -0.1278, "continent": "Europe"},
            "Berlin": {"country": "Germany", "lat": 52.5200, "lon": 13.4050, "continent": "Europe"},
            "Paris": {"country": "France", "lat": 48.8566, "lon": 2.3522, "continent": "Europe"},
            "Tokyo": {"country": "Japan", "lat": 35.6762, "lon": 139.6503, "continent": "Asia"},
            "Beijing": {"country": "China", "lat": 39.9042, "lon": 116.4074, "continent": "Asia"},
            "New Delhi": {"country": "India", "lat": 28.6139, "lon": 77.2090, "continent": "Asia"},
            "Moscow": {"country": "Russia", "lat": 55.7558, "lon": 37.6176, "continent": "Europe"},
            "Cairo": {"country": "Egypt", "lat": 30.0444, "lon": 31.2357, "continent": "Africa"},
            "Cape Town": {"country": "South Africa", "lat": -33.9249, "lon": 18.4241, "continent": "Africa"},
            "Seoul": {"country": "South Korea", "lat": 37.5665, "lon": 126.9780, "continent": "Asia"},
            "Bangkok": {"country": "Thailand", "lat": 13.7563, "lon": 100.5018, "continent": "Asia"},
            "Singapore": {"country": "Singapore", "lat": 1.3521, "lon": 103.8198, "continent": "Asia"},
            "Dubai": {"country": "United Arab Emirates", "lat": 25.2048, "lon": 55.2708, "continent": "Asia"},
            "Sydney": {"country": "Australia", "lat": -33.8688, "lon": 151.2093, "continent": "Oceania"},
        }

    def get_closest_capitals(self, location_info: LocationInfo, limit: int = 5) -> List[Dict]:
        """Get closest world capitals to a location."""
        target_lat = location_info.latitude
        target_lon = location_info.longitude

        distances = []
        for capital, data in self.capitals_data.items():
            distance = self._calculate_distance(target_lat, target_lon, data['lat'], data['lon'])
            distances.append({
                'name': capital,
                'country': data['country'],
                'continent': data['continent'],
                'latitude': data['lat'],
                'longitude': data['lon'],
                'distance_km': distance
            })

        distances.sort(key=lambda x: x['distance_km'])
        return distances[:limit]

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates using Haversine formula."""
        R = 6371  # Earth's radius in km

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)

        a = (np.sin(delta_lat/2)**2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

        return R * c


# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class ClimatePredictor:
    """Professional climate prediction application."""

    def __init__(self):
        self._init_session_state()
        self._init_services()

    def _init_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'selected_location': None,
            'location_history': [],
            'search_results': [],
            'data_availability': {},
            'ml_predictions': {},
            'prediction_history': [],
            'collection_status': {},
            'processing_results': {},
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _init_services(self):
        """Initialize backend services."""
        if 'services_initialized' not in st.session_state:
            st.session_state.location_service = LocationService()
            st.session_state.data_manager = ClimateDataManager()
            st.session_state.pipeline = ClimateDataPipeline()
            st.session_state.capitals_service = WorldCapitalsService()
            st.session_state.ml_service = MLPredictionService()
            st.session_state.feature_engine = UniversalFeatureEngine()
            st.session_state.services_initialized = True

        self.location_service = st.session_state.location_service
        self.data_manager = st.session_state.data_manager
        self.pipeline = st.session_state.pipeline
        self.capitals_service = st.session_state.capitals_service
        self.ml_service = st.session_state.ml_service

    # =========================================================================
    # HEADER & NAVIGATION
    # =========================================================================

    def render_header(self):
        """Render page header."""
        st.markdown("""
        <div class="page-header">
            <h1>Climate Impact Predictor</h1>
            <p>ML-powered climate analysis with LSTM forecasting</p>
        </div>
        """, unsafe_allow_html=True)

    def render_sidebar(self):
        """Render sidebar with controls and status."""
        with st.sidebar:
            st.markdown('<div class="sidebar-title">Quick Access</div>', unsafe_allow_html=True)

            # Popular Locations
            locations = ["New York, USA", "London, UK", "Tokyo, Japan", "Berlin, Germany",
                        "Sydney, Australia", "Singapore", "Dubai, UAE", "Paris, France"]

            for loc in locations:
                if st.button(loc, use_container_width=True, key=f"quick_{loc}"):
                    result = self.location_service.geocode_location(loc)
                    if result:
                        self._select_location(result)

            st.markdown("---")

            # Prediction History
            if st.session_state.prediction_history:
                st.markdown('<div class="sidebar-title">Recent Predictions</div>', unsafe_allow_html=True)
                for entry in st.session_state.prediction_history[:3]:
                    timestamp = entry["timestamp"].strftime("%H:%M")
                    st.markdown(f"**{entry['location']}** ({timestamp})")

            st.markdown("---")

            # System Status
            st.markdown('<div class="sidebar-title">System Status</div>', unsafe_allow_html=True)

            api_status = self.ml_service.check_api_health()
            if api_status:
                st.markdown('<span class="status-badge status-online">ML API Online</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-offline">ML API Offline</span>', unsafe_allow_html=True)

            st.markdown('<span class="status-badge status-online">Location Service</span>', unsafe_allow_html=True)
            st.markdown('<span class="status-badge status-online">Data Manager</span>', unsafe_allow_html=True)

    # =========================================================================
    # LOCATION SEARCH
    # =========================================================================

    def render_location_search(self):
        """Render location search interface."""
        st.markdown('<div class="section-header">Location Search</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])

        with col1:
            search_query = st.text_input(
                "Search location",
                placeholder="Enter city, country, or coordinates...",
                label_visibility="collapsed",
                key="location_search"
            )

        with col2:
            search_clicked = st.button("Search", type="primary", use_container_width=True)

        if search_clicked and search_query:
            self._perform_search(search_query)

        if st.session_state.search_results:
            self._render_search_results()

    def _perform_search(self, query: str):
        """Execute location search."""
        try:
            results = self.location_service.search_locations(query, limit=5)
            if results:
                st.session_state.search_results = results
                for location in results:
                    self._check_data_availability(location)
            else:
                st.session_state.search_results = []
                st.warning(f"No locations found for '{query}'")
        except Exception as e:
            st.error(f"Search error: {str(e)}")

    def _check_data_availability(self, location: LocationInfo):
        """Check data availability for a location."""
        try:
            availability = self.data_manager.check_data_availability_sync(location)
            key = f"{location.latitude:.4f},{location.longitude:.4f}"
            st.session_state.data_availability[key] = availability
        except:
            key = f"{location.latitude:.4f},{location.longitude:.4f}"
            st.session_state.data_availability[key] = {
                "air_quality": True, "meteorological": True, "climate_projections": True
            }

    def _render_search_results(self):
        """Render search results."""
        for i, location in enumerate(st.session_state.search_results):
            key = f"{location.latitude:.4f},{location.longitude:.4f}"
            availability = st.session_state.data_availability.get(key, {})

            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                st.markdown(f"""
                <div class="location-card">
                    <h4>{location.name}</h4>
                    <p>Country: {location.country}</p>
                    <p class="coord">{location.latitude:.4f}, {location.longitude:.4f}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                self._render_availability_badges(availability)

            with col3:
                if st.button("Select", key=f"select_{i}", type="primary", use_container_width=True):
                    self._select_location(location)

    def _render_availability_badges(self, availability: Dict):
        """Render data source availability badges."""
        sources = [
            ("Air Quality", availability.get("air_quality", False)),
            ("Weather", availability.get("meteorological", False)),
            ("Projections", availability.get("climate_projections", False))
        ]

        badges_html = ""
        for name, available in sources:
            css_class = "source-available" if available else "source-unavailable"
            badges_html += f'<span class="source-indicator {css_class}">{name}</span>'

        st.markdown(badges_html, unsafe_allow_html=True)

    def _select_location(self, location: LocationInfo):
        """Select a location for analysis."""
        st.session_state.selected_location = location
        if location.name not in [l.name for l in st.session_state.location_history]:
            st.session_state.location_history.insert(0, location)
            st.session_state.location_history = st.session_state.location_history[:10]
        st.rerun()

    # =========================================================================
    # SELECTED LOCATION VIEW
    # =========================================================================

    def render_selected_location(self):
        """Render the selected location analysis view."""
        location = st.session_state.selected_location

        st.markdown(f'<div class="section-header">Analysis: {location.name}, {location.country}</div>',
                   unsafe_allow_html=True)

        # Location Info Card
        col1, col2 = st.columns([2, 1])

        with col1:
            confidence = location.confidence_score * 100 if location.confidence_score <= 1.0 else location.confidence_score
            st.markdown(f"""
            <div class="card">
                <div class="card-header">Location Details</div>
                <div class="metric-grid">
                    <div class="metric-item">
                        <div class="metric-label">Latitude</div>
                        <div class="metric-value" style="font-size: 1.1rem;">{location.latitude:.4f}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Longitude</div>
                        <div class="metric-value" style="font-size: 1.1rem;">{location.longitude:.4f}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Confidence</div>
                        <div class="metric-value" style="font-size: 1.1rem;">{confidence:.0f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            key = f"{location.latitude:.4f},{location.longitude:.4f}"
            availability = st.session_state.data_availability.get(key, {})

            st.markdown('<div class="card"><div class="card-header">Data Sources</div>', unsafe_allow_html=True)
            self._render_availability_badges(availability)
            st.markdown('</div>', unsafe_allow_html=True)

        # Map
        self._render_map(location)

        # ML Prediction Controls
        self._render_prediction_controls(location)

        # Data Pipeline Controls
        self._render_pipeline_controls(location)

        # Results
        if st.session_state.ml_predictions.get(location.name):
            self._render_prediction_results(location)

        # World Comparison
        self._render_world_comparison(location)

    def _render_map(self, location: LocationInfo):
        """Render location map."""
        try:
            map_data = pd.DataFrame({
                'lat': [location.latitude],
                'lon': [location.longitude],
                'name': [location.name]
            })

            fig = px.scatter_map(
                map_data, lat='lat', lon='lon',
                hover_name='name', zoom=7, height=300
            )

            fig.update_traces(marker=dict(size=12, color='#06b6d4'))
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"Map unavailable. Coordinates: {location.latitude:.4f}, {location.longitude:.4f}")

    # =========================================================================
    # ML PREDICTION CONTROLS
    # =========================================================================

    def _render_prediction_controls(self, location: LocationInfo):
        """Render ML prediction controls."""
        st.markdown("""
        <div class="controls-panel">
            <h3>ML Prediction</h3>
        </div>
        """, unsafe_allow_html=True)

        api_healthy = self.ml_service.check_api_health()

        if not api_healthy:
            st.error("ML API is offline. Start the server with: python app/api_server.py")
            return

        col1, col2, col3 = st.columns([2, 2, 1])

        with col1:
            prediction_type = st.selectbox(
                "Model",
                ["Basic Climate", "LSTM Forecast", "Both Models"],
                label_visibility="collapsed"
            )

        with col2:
            forecast_days = st.slider("Forecast Days", 1, 30, 7, label_visibility="collapsed")

        with col3:
            if st.button("Run Prediction", type="primary", use_container_width=True):
                self._run_predictions(location, prediction_type, forecast_days)

    def _run_predictions(self, location: LocationInfo, prediction_type: str, forecast_days: int):
        """Execute ML predictions."""
        results = {}
        success_count = 0
        total = 0

        with st.spinner("Running ML predictions..."):
            if prediction_type in ["Basic Climate", "Both Models"]:
                total += 1
                basic_result = self.ml_service.predict_basic_climate(location.name)
                results['basic'] = basic_result
                if basic_result.get('success', False) or 'prediction' in basic_result:
                    success_count += 1

            if prediction_type in ["LSTM Forecast", "Both Models"]:
                total += 1
                lstm_result = self.ml_service.predict_lstm_forecast(location.name, forecast_days)
                results['lstm'] = lstm_result
                if lstm_result.get('success', False) or 'forecast' in lstm_result:
                    success_count += 1

        st.session_state.ml_predictions[location.name] = results
        st.session_state.prediction_history.insert(0, {
            'location': location.name,
            'timestamp': datetime.now(),
            'prediction_type': prediction_type,
            'success_count': success_count,
            'total_predictions': total
        })

        st.rerun()

    def _render_prediction_results(self, location: LocationInfo):
        """Render prediction results."""
        results = st.session_state.ml_predictions.get(location.name, {})

        st.markdown("""
        <div class="results-panel">
            <h2>Prediction Results</h2>
        </div>
        """, unsafe_allow_html=True)

        # Basic Climate Results
        if 'basic' in results:
            basic = results['basic']
            prediction = basic.get('prediction', {})
            # Handle nested prediction structure from API
            if isinstance(prediction.get('prediction'), dict):
                prediction = prediction['prediction']

            st.markdown('<div class="prediction-title">Basic Climate Model</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)

            with col1:
                # Try multiple possible key names for temperature
                temp = prediction.get('temperature_avg') or prediction.get('temperature', 'N/A')
                if isinstance(temp, (int, float)):
                    st.metric("Temperature", f"{temp:.1f} C")
                else:
                    st.metric("Temperature", str(temp))

            with col2:
                # Try multiple possible key names for AQI
                aqi = prediction.get('aqi_prediction') or prediction.get('air_quality_index', 'N/A')
                if isinstance(aqi, (int, float)):
                    st.metric("Air Quality Index", f"{aqi:.0f}")
                else:
                    st.metric("Air Quality Index", str(aqi))

            with col3:
                st.metric("Model", basic.get('model_used', 'Base Model'))

        # LSTM Forecast Results
        if 'lstm' in results:
            lstm = results['lstm']
            forecast_data = lstm.get('forecast', [])

            st.markdown('<div class="prediction-title">LSTM Weather Forecast</div>', unsafe_allow_html=True)

            # Handle forecast as list of objects with 'temperature' and 'date' keys
            # Also handle old format with 'temperatures' array
            temps = None
            dates = None

            if isinstance(forecast_data, list) and len(forecast_data) > 0:
                temps = [f.get('temperature', 0) for f in forecast_data]
                dates = [f.get('date', f"Day {i+1}") for i, f in enumerate(forecast_data)]
            elif isinstance(forecast_data, dict) and 'temperatures' in forecast_data:
                temps = forecast_data['temperatures']
                dates = forecast_data.get('dates', list(range(len(temps))))

            if temps and len(temps) > 0:
                # Create forecast chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates, y=temps,
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='#06b6d4', width=2),
                    marker=dict(size=6)
                ))

                fig.update_layout(
                    height=300,
                    margin=dict(l=40, r=20, t=20, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(26,34,52,1)',
                    font=dict(color='#94a3b8'),
                    xaxis=dict(gridcolor='#334155', title='Date'),
                    yaxis=dict(gridcolor='#334155', title='Temperature (C)')
                )

                st.plotly_chart(fig, use_container_width=True)

                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Average", f"{np.mean(temps):.1f} C")
                with col2:
                    st.metric("Minimum", f"{np.min(temps):.1f} C")
                with col3:
                    st.metric("Maximum", f"{np.max(temps):.1f} C")
                with col4:
                    st.metric("Range", f"{np.max(temps) - np.min(temps):.1f} C")

    # =========================================================================
    # DATA PIPELINE CONTROLS
    # =========================================================================

    def _render_pipeline_controls(self, location: LocationInfo):
        """Render data collection pipeline controls."""
        st.markdown('<div class="section-header">Data Pipeline</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Check Availability", use_container_width=True):
                self._check_data_availability(location)
                st.rerun()

        with col2:
            if st.button("Collect Data", type="primary", use_container_width=True):
                self._collect_data(location)

        with col3:
            if st.button("Process Data", use_container_width=True):
                self._process_data(location)

        with col4:
            if st.button("Full Workflow", use_container_width=True):
                self._collect_data(location)
                if st.session_state.collection_status.get("successful_sources", 0) > 0:
                    self._process_data(location)

    def _collect_data(self, location: LocationInfo):
        """Collect climate data for location."""
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        with st.spinner(f"Collecting data for {location.name}..."):
            try:
                results = self.data_manager.fetch_adaptive_data_sync(
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    forecast_days=7,
                    save=True,
                    force_all=False
                )

                successful = len([r for r in results.values() if r is not None])
                st.session_state.collection_status = {
                    "status": "completed",
                    "results": results,
                    "successful_sources": successful,
                    "total_sources": len(results)
                }

                if successful > 0:
                    st.success(f"Data collected: {successful}/{len(results)} sources")
                else:
                    st.warning("No data was retrieved")

            except Exception as e:
                st.error(f"Collection failed: {e}")

        st.rerun()

    def _process_data(self, location: LocationInfo):
        """Process collected data."""
        if not st.session_state.collection_status.get("results"):
            st.warning("No collected data. Run data collection first.")
            return

        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        with st.spinner(f"Processing data for {location.name}..."):
            try:
                async def run_processing():
                    return await self.pipeline.process_global_location(
                        location=location,
                        start_date=start_date,
                        end_date=end_date,
                        skip_collection=False
                    )

                try:
                    loop = asyncio.get_event_loop()
                    results = loop.run_until_complete(run_processing())
                except RuntimeError:
                    results = asyncio.run(run_processing())

                st.session_state.processing_results = results
                st.success("Data processing completed")

            except Exception as e:
                st.error(f"Processing failed: {e}")

        st.rerun()

    # =========================================================================
    # WORLD COMPARISON
    # =========================================================================

    def _render_world_comparison(self, location: LocationInfo):
        """Render world capitals comparison."""
        st.markdown('<div class="section-header">Regional Comparison</div>', unsafe_allow_html=True)

        closest = self.capitals_service.get_closest_capitals(location, limit=5)

        if closest:
            data = []
            for cap in closest:
                data.append({
                    "City": cap['name'],
                    "Country": cap['country'],
                    "Distance": f"{cap['distance_km']:.0f} km",
                    "Continent": cap['continent']
                })

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.info(f"Nearest major city: {closest[0]['name']}, {closest[0]['country']} ({closest[0]['distance_km']:.0f} km)")

    # =========================================================================
    # WELCOME VIEW
    # =========================================================================

    def render_welcome(self):
        """Render welcome panel when no location selected."""
        st.markdown("""
        <div class="welcome-panel">
            <h3>Climate Analysis Platform</h3>
            <p>Search for any location worldwide to begin climate analysis.</p>
            <div class="feature-grid">
                <div class="feature-item">ML Prediction API</div>
                <div class="feature-item">LSTM Forecasting</div>
                <div class="feature-item">Historical Analysis</div>
                <div class="feature-item">Global Comparison</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Getting Started</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Quick Start:**
            1. Search for a location
            2. Select from results
            3. Run ML prediction
            4. Analyze results
            """)

        with col2:
            st.markdown("""
            **Suggested Locations:**
            - Tokyo, Japan
            - London, UK
            - New York, USA
            - Sydney, Australia
            """)

    # =========================================================================
    # MAIN RUN METHOD
    # =========================================================================

    def run(self):
        """Main application entry point."""
        self.render_header()
        self.render_sidebar()
        self.render_location_search()

        if st.session_state.selected_location:
            self.render_selected_location()
        else:
            self.render_welcome()

        # Footer
        st.markdown("""
        <div class="footer">
            Climate Impact Predictor | ML-Powered Analysis
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    try:
        app = ClimatePredictor()
        app.run()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page or check your configuration.")
        with st.expander("Debug Information"):
            st.code(str(e))


if __name__ == "__main__":
    main()
