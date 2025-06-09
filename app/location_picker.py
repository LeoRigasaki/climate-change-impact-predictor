#!/usr/bin/env python3
"""
🌍 Dynamic Climate Impact Predictor - Location Picker UI
app/location_picker.py

Interactive Streamlit interface for global location selection and 
climate prediction preview. Day 3 prototype.
"""

import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import asyncio
import time
from typing import Optional, List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.location_service import LocationService, LocationInfo

# Simple DataManager for UI compatibility
class DataManager:
    """Simple data manager for UI prototype."""
    def __init__(self):
        pass
    
    def get_stats(self):
        return {"status": "Day 3 Prototype"}

# Configure Streamlit page
st.set_page_config(
    page_title="🌍 Climate Impact Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .location-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)


class LocationPickerApp:
    """Streamlit app for interactive location selection and climate preview."""
    
    def __init__(self):
        """Initialize the location picker app."""
        # Initialize services
        if 'location_service' not in st.session_state:
            with st.spinner("🌍 Initializing global location service..."):
                st.session_state.location_service = LocationService()
                st.session_state.data_manager = DataManager()
        
        self.location_service = st.session_state.location_service
        self.data_manager = st.session_state.data_manager
        
        # Session state for selected location
        if 'selected_location' not in st.session_state:
            st.session_state.selected_location = None
        if 'location_history' not in st.session_state:
            st.session_state.location_history = []
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
    
    def render_header(self):
        """Render the main header section."""
        st.markdown('<h1 class="main-header">🌍 Dynamic Climate Impact Predictor</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Predict climate impacts for <strong>any location on Earth</strong> using real-time data from professional APIs
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_location_search(self):
        """Render the location search interface."""
        st.subheader("🔍 Find Your Location")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Search input
            search_query = st.text_input(
                "Search for any city, country, or coordinates",
                placeholder="e.g., Berlin, Germany or 52.5200, 13.4050",
                help="Try: 'New York', 'Tokyo, Japan', 'São Paulo', or coordinates like '40.7128, -74.0060'"
            )
        
        with col2:
            # Search button
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Handle search
        if search_clicked and search_query:
            with st.spinner(f"🌍 Searching for '{search_query}'..."):
                self.perform_location_search(search_query)
        
        # Show search results
        if st.session_state.search_results:
            self.render_search_results()
    
    def perform_location_search(self, query: str):
        """Perform location search and update session state."""
        try:
            # Search for multiple matches
            results = self.location_service.search_locations(query, limit=5)
            
            if results:
                st.session_state.search_results = results
                st.success(f"✅ Found {len(results)} location(s) for '{query}'")
            else:
                st.session_state.search_results = []
                st.warning(f"❌ No locations found for '{query}'. Try a different search term.")
                
        except Exception as e:
            st.error(f"❌ Search error: {str(e)}")
            st.session_state.search_results = []
    
    def render_search_results(self):
        """Render search results with selection options."""
        st.subheader("📍 Search Results")
        
        for i, location in enumerate(st.session_state.search_results):
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.markdown(f"""
                    <div class="location-card">
                        <h4>📍 {location.name}</h4>
                        <p><strong>Country:</strong> {location.country}</p>
                        <p><strong>Coordinates:</strong> {location.latitude:.4f}, {location.longitude:.4f}</p>
                        {f'<p><strong>Climate Zone:</strong> {location.climate_zone}</p>' if location.climate_zone else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Confidence and data availability
                    confidence_color = "green" if location.confidence_score > 0.8 else "orange" if location.confidence_score > 0.5 else "red"
                    st.markdown(f"**Confidence:** <span style='color: {confidence_color}'>{location.confidence_score:.1%}</span>", 
                               unsafe_allow_html=True)
                    
                    if location.has_air_quality:
                        st.markdown("✅ Air Quality Data")
                    if location.has_meteorological:
                        st.markdown("✅ Weather Data")
                    if location.has_climate_projections:
                        st.markdown("✅ Climate Projections")
                
                with col3:
                    if st.button(f"Select", key=f"select_{i}", type="primary"):
                        self.select_location(location)
                
                st.divider()
    
    def select_location(self, location: LocationInfo):
        """Select a location and add to history."""
        st.session_state.selected_location = location
        
        # Add to history (avoid duplicates)
        history_keys = [f"{loc.latitude:.4f},{loc.longitude:.4f}" for loc in st.session_state.location_history]
        current_key = f"{location.latitude:.4f},{location.longitude:.4f}"
        
        if current_key not in history_keys:
            st.session_state.location_history.insert(0, location)
            # Keep only last 10 locations
            st.session_state.location_history = st.session_state.location_history[:10]
        
        # Clear search results
        st.session_state.search_results = []
        
        st.success(f"✅ Selected: {location.name}, {location.country}")
        st.rerun()
    
    def render_selected_location(self):
        """Render details for the currently selected location."""
        if not st.session_state.selected_location:
            return
        
        location = st.session_state.selected_location
        
        st.subheader(f"🎯 Selected Location: {location.name}")
        
        # Location details in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📍 Location</h3>
                <p><strong>{location.name}</strong></p>
                <p>{location.country}</p>
                <p>{location.latitude:.4f}°, {location.longitude:.4f}°</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            climate_zone = location.climate_zone or "Unknown"
            zone_emoji = {
                "tropical": "🌴", "desert": "🏜️", "temperate": "🌳", 
                "arctic": "🧊", "subarctic": "❄️", "mountain": "🏔️",
                "coastal": "🌊", "continental": "🏞️", "mediterranean": "🌺"
            }.get(climate_zone, "🌍")
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{zone_emoji} Climate</h3>
                <p><strong>{climate_zone.title()}</strong></p>
                <p>Zone Classification</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            data_score = sum([location.has_air_quality, location.has_meteorological, location.has_climate_projections])
            data_emoji = "🟢" if data_score == 3 else "🟡" if data_score == 2 else "🔴"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{data_emoji} Data Quality</h3>
                <p><strong>{data_score}/3 Sources</strong></p>
                <p>Available APIs</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Render location map
        self.render_location_map(location)
        
        # Show prediction preview
        self.render_prediction_preview(location)
    
    def render_location_map(self, location: LocationInfo):
        """Render interactive map for the selected location."""
        st.subheader("🗺️ Location Map")
        
        # Create map data
        map_data = pd.DataFrame({
            'lat': [location.latitude],
            'lon': [location.longitude],
            'name': [location.name],
            'country': [location.country]
        })
        
        # Create Plotly map
        fig = px.scatter_map(
            map_data,
            lat='lat',
            lon='lon',
            hover_name='name',
            hover_data={'country': True, 'lat': ':.4f', 'lon': ':.4f'},
            zoom=8,
            height=400,
            map_style="open-street-map"
        )
        
        fig.update_layout(
            title=f"📍 {location.name}, {location.country}",
            font=dict(size=14),
            showlegend=False
        )
        
        # Add marker styling
        fig.update_traces(
            marker=dict(size=15, color='red'),
            hovertemplate='<b>%{hovertext}</b><br>' +
                         'Country: %{customdata[0]}<br>' +
                         'Latitude: %{customdata[1]}<br>' +
                         'Longitude: %{customdata[2]}<extra></extra>'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_prediction_preview(self, location: LocationInfo):
        """Render climate prediction preview for selected location."""
        st.subheader("🔮 Climate Prediction Preview")
        
        # Show what predictions would be available
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌡️ Available Predictions")
            
            if location.has_meteorological:
                st.markdown("✅ **Temperature Extremes**")
                st.markdown("   • Heat wave probability")
                st.markdown("   • Cold snap risk")
                st.markdown("   • Seasonal temperature trends")
            
            if location.has_air_quality:
                st.markdown("✅ **Air Quality Health Impact**")
                st.markdown("   • PM2.5 exposure risk")
                st.markdown("   • Respiratory health index")
                st.markdown("   • Air quality forecasting")
            
            if location.has_climate_projections:
                st.markdown("✅ **Long-term Climate Change**")
                st.markdown("   • Temperature projections to 2100")
                st.markdown("   • Precipitation changes")
                st.markdown("   • Extreme weather frequency")
        
        with col2:
            st.markdown("### 📊 Prediction Capabilities")
            
            # Sample prediction data (mock for Day 3)
            prediction_types = ["Temperature Risk", "Air Quality Index", "Climate Change Impact"]
            availability = [100, 90, 85] if all([location.has_meteorological, location.has_air_quality, location.has_climate_projections]) else [60, 40, 30]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=prediction_types,
                    y=availability,
                    marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                    text=[f"{v}%" for v in availability],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Data Availability by Prediction Type",
                xaxis_title="Prediction Type",
                yaxis_title="Availability (%)",
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Run Full Prediction", type="primary", use_container_width=True):
                self.run_prediction_demo(location)
        
        with col2:
            if st.button("📊 View Historical Data", use_container_width=True):
                st.info("📊 Historical data analysis coming in Day 4-5!")
        
        with col3:
            if st.button("📈 Compare Locations", use_container_width=True):
                st.info("📈 Location comparison feature coming in Day 6-7!")
    
    def run_prediction_demo(self, location: LocationInfo):
        """Run a demo prediction for the selected location."""
        st.subheader("🔄 Running Climate Prediction...")
        
        # Progress bar simulation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        steps = [
            ("🔍 Validating location coordinates", 0.2),
            ("📡 Fetching real-time air quality data", 0.4),
            ("🌤️ Collecting meteorological data", 0.6),
            ("🌍 Loading climate projections", 0.8),
            ("🧠 Processing prediction models", 1.0)
        ]
        
        for step_text, progress in steps:
            status_text.text(step_text)
            progress_bar.progress(progress)
            time.sleep(1)  # Simulate processing time
        
        status_text.text("✅ Prediction complete!")
        
        # Show mock results
        st.markdown("""
        <div class="success-message">
            <h4>🎉 Prediction Results for {}</h4>
            <p><strong>🌡️ Temperature Risk:</strong> Moderate (65% confidence)</p>
            <p><strong>🌫️ Air Quality Index:</strong> Good (AQI: 42)</p>
            <p><strong>📈 Climate Trend:</strong> Warming (+1.2°C by 2050)</p>
            <p><em>Full implementation coming in Days 8-14!</em></p>
        </div>
        """.format(location.name), unsafe_allow_html=True)
        
        st.balloons()  # Celebration effect
    
    def render_location_history(self):
        """Render location history in sidebar."""
        if not st.session_state.location_history:
            return
        
        st.sidebar.subheader("📚 Recent Locations")
        
        for i, location in enumerate(st.session_state.location_history):
            if st.sidebar.button(
                f"📍 {location.name}, {location.country}",
                key=f"history_{i}",
                use_container_width=True
            ):
                self.select_location(location)
    
    def render_quick_locations(self):
        """Render quick access to popular locations."""
        st.sidebar.subheader("🌍 Popular Locations")
        
        popular_locations = [
            "New York, USA",
            "London, UK", 
            "Tokyo, Japan",
            "Berlin, Germany",
            "Sydney, Australia",
            "São Paulo, Brazil"
        ]
        
        for location_query in popular_locations:
            if st.sidebar.button(location_query, use_container_width=True):
                with st.spinner(f"Loading {location_query}..."):
                    result = self.location_service.geocode_location(location_query)
                    if result:
                        self.select_location(result)
                    else:
                        st.sidebar.error(f"Could not load {location_query}")
    
    def render_service_stats(self):
        """Render location service statistics."""
        st.sidebar.subheader("📊 Service Stats")
        
        stats = self.location_service.get_stats()
        
        st.sidebar.metric("Cached Locations", stats["cached_locations"])
        st.sidebar.metric("Geocoder Service", "Nominatim (OSM)")
        
        if st.sidebar.button("🔄 Refresh Stats"):
            st.rerun()
    
    def run(self):
        """Main app execution."""
        # Render header
        self.render_header()
        
        # Sidebar content
        with st.sidebar:
            st.title("🎛️ Controls")
            self.render_quick_locations()
            self.render_location_history()
            self.render_service_stats()
        
        # Main content
        self.render_location_search()
        
        if st.session_state.selected_location:
            self.render_selected_location()
        else:
            # Show welcome message
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background-color: #f8f9fa; border-radius: 10px; margin: 2rem 0;">
                <h3>🌍 Welcome to Dynamic Climate Prediction!</h3>
                <p>Search for any location above to get started with real-time climate impact predictions.</p>
                <p><em>Try searching for your city, or use the popular locations in the sidebar!</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>🌍 Dynamic Climate Impact Predictor | Day 3 Prototype | Built with Streamlit & Professional APIs</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    try:
        app = LocationPickerApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the problem persists.")


if __name__ == "__main__":
    main()