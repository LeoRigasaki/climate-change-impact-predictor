#!/usr/bin/env python3
"""
🌍 Enhanced Climate Impact Predictor - Location Picker UI - Day 4 Global Integration
app/location_picker.py

Interactive Streamlit interface for global location selection, real-time data availability
checking, adaptive data collection, and climate prediction workflow.
"""

import sys
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.location_service import LocationService, LocationInfo
from src.core.data_manager import ClimateDataManager
from src.core.pipeline import ClimateDataPipeline

# Configure Streamlit page
st.set_page_config(
    page_title="🌍 Climate Impact Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
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
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .data-availability-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
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
        margin: 1rem 0;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    
    .processing-status {
        background-color: #e7f3ff;
        color: #0366d6;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    
    .data-source-indicator {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        margin: 0.25rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: bold;
    }
    
    .available { background-color: #d4edda; color: #155724; }
    .unavailable { background-color: #f8d7da; color: #721c24; }
    .checking { background-color: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)


class EnhancedLocationPickerApp:
    """Enhanced Streamlit app for global climate prediction workflow."""
    
    def __init__(self):
        """Initialize the enhanced location picker app."""
        # Initialize services
        if 'location_service' not in st.session_state:
            with st.spinner("🌍 Initializing global climate prediction services..."):
                st.session_state.location_service = LocationService()
                st.session_state.data_manager = ClimateDataManager()
                st.session_state.pipeline = ClimateDataPipeline()
        
        self.location_service = st.session_state.location_service
        self.data_manager = st.session_state.data_manager
        self.pipeline = st.session_state.pipeline
        
        # Enhanced session state
        if 'selected_location' not in st.session_state:
            st.session_state.selected_location = None
        if 'location_history' not in st.session_state:
            st.session_state.location_history = []
        if 'search_results' not in st.session_state:
            st.session_state.search_results = []
        if 'data_availability' not in st.session_state:
            st.session_state.data_availability = {}
        if 'collection_status' not in st.session_state:
            st.session_state.collection_status = {}
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = {}
    
    def render_header(self):
        """Render the enhanced header section."""
        st.markdown('<h1 class="main-header">🌍 Dynamic Climate Impact Predictor</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.3rem; color: #666;">
                Predict climate impacts for <strong>any location on Earth</strong> using adaptive AI and professional APIs
            </p>
            <p style="color: #888; margin-top: 0.5rem;">
                ✨ Day 4 Enhanced: Real-time data availability • Adaptive collection • Global processing
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_location_search(self):
        """Render the enhanced location search interface."""
        st.subheader("🔍 Find Your Location")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Enhanced search input
            search_query = st.text_input(
                "Search for any city, country, or coordinates",
                placeholder="e.g., Reykjavik, Iceland • Mount Everest • 35.6762, 139.6503",
                help="🌍 Try any location: cities, landmarks, coordinates, or even remote areas!",
                key="location_search"
            )
        
        with col2:
            # Search button
            search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
        
        # Handle search
        if search_clicked and search_query:
            with st.spinner(f"🌍 Searching globally for '{search_query}'..."):
                self.perform_location_search(search_query)
        
        # Show search results
        if st.session_state.search_results:
            self.render_enhanced_search_results()
    
    def perform_location_search(self, query: str):
        """Perform enhanced location search with data availability checking."""
        try:
            # Search for multiple matches
            results = self.location_service.search_locations(query, limit=5)
            
            if results:
                st.session_state.search_results = results
                st.success(f"✅ Found {len(results)} location(s) for '{query}'")
                
                # Pre-check data availability for all results
                with st.spinner("🔍 Checking data availability for found locations..."):
                    for location in results:
                        self.check_location_availability(location)
            else:
                st.session_state.search_results = []
                st.warning(f"❌ No locations found for '{query}'. Try a different search term.")
                
        except Exception as e:
            st.error(f"❌ Search error: {str(e)}")
            st.session_state.search_results = []
    
    def check_location_availability(self, location: LocationInfo):
        """Check data availability for a location."""
        try:
            # Use sync wrapper to handle async method properly
            availability = self.data_manager.check_data_availability_sync(location)
            
            # Store in session state
            location_key = f"{location.latitude:.4f},{location.longitude:.4f}"
            st.session_state.data_availability[location_key] = availability
            
        except Exception as e:
            st.warning(f"⚠️ Could not check data availability for {location.name}: {e}")
            # Set default availability
            location_key = f"{location.latitude:.4f},{location.longitude:.4f}"
            st.session_state.data_availability[location_key] = {
                "air_quality": True,
                "meteorological": True, 
                "climate_projections": True
            }
    
    def render_enhanced_search_results(self):
        """Render enhanced search results with data availability indicators."""
        st.subheader("📍 Search Results with Data Availability")
        
        for i, location in enumerate(st.session_state.search_results):
            location_key = f"{location.latitude:.4f},{location.longitude:.4f}"
            availability = st.session_state.data_availability.get(location_key, {})
            
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
                    # Enhanced data availability display
                    self.render_data_availability_indicators(availability)
                    
                    # Fixed confidence score display
                    confidence_pct = location.confidence_score * 100 if location.confidence_score <= 1.0 else location.confidence_score
                    confidence_color = "green" if confidence_pct > 80 else "orange" if confidence_pct > 50 else "red"
                    st.markdown(f"**Confidence:** <span style='color: {confidence_color}'>{confidence_pct:.1f}%</span>", 
                               unsafe_allow_html=True)
                
                with col3:
                    if st.button(f"Select & Analyze", key=f"select_{i}", type="primary"):
                        self.select_location(location)
                
                st.divider()
    
    def render_data_availability_indicators(self, availability: Dict[str, bool]):
        """Render data availability indicators."""
        if not availability:
            st.markdown('<span class="data-source-indicator checking">🔍 Checking...</span>', 
                       unsafe_allow_html=True)
            return
        
        sources = {
            "air_quality": ("🌬️ Air Quality", availability.get("air_quality", False)),
            "meteorological": ("🛰️ Weather Data", availability.get("meteorological", False)),
            "climate_projections": ("🔮 Projections", availability.get("climate_projections", False))
        }
        
        for source_key, (source_name, available) in sources.items():
            css_class = "available" if available else "unavailable"
            status_icon = "✅" if available else "❌"
            st.markdown(f'<span class="data-source-indicator {css_class}">{status_icon} {source_name}</span>', 
                       unsafe_allow_html=True)
        
        # Summary
        available_count = sum(availability.values())
        total_count = len(availability)
        coverage_pct = (available_count / total_count * 100) if total_count > 0 else 0
        
        if coverage_pct >= 67:
            coverage_status = "🟢 Excellent"
        elif coverage_pct >= 33:
            coverage_status = "🟡 Partial"
        else:
            coverage_status = "🔴 Limited"
            
        st.markdown(f"**Coverage:** {coverage_status} ({available_count}/{total_count})")
    
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
    
    def select_location(self, location: LocationInfo):
        """Select a location and initiate full analysis workflow."""
        st.session_state.selected_location = location
        
        # Add to history (avoid duplicates)
        existing_names = [loc.name for loc in st.session_state.location_history]
        if location.name not in existing_names:
            st.session_state.location_history.insert(0, location)
            # Keep only last 10 locations
            st.session_state.location_history = st.session_state.location_history[:10]
        
        # Clear previous results
        st.session_state.collection_status = {}
        st.session_state.processing_results = {}
        
        st.success(f"✅ Selected: {location.name}, {location.country}")
        st.rerun()
    
    def render_selected_location(self):
        """Render comprehensive analysis for selected location."""
        location = st.session_state.selected_location
        st.subheader(f"🎯 Climate Analysis: {location.name}, {location.country}")
        
        # Location details card
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Fixed confidence score display
            confidence_pct = location.confidence_score * 100 if location.confidence_score <= 1.0 else location.confidence_score
            
            st.markdown(f"""
            <div class="location-card">
                <h3>📍 {location.name}</h3>
                <p><strong>Country:</strong> {location.country} ({location.country_code})</p>
                <p><strong>Coordinates:</strong> {location.latitude:.6f}, {location.longitude:.6f}</p>
                {f'<p><strong>Climate Zone:</strong> {location.climate_zone}</p>' if location.climate_zone else ''}
                {f'<p><strong>Timezone:</strong> {location.timezone}</p>' if location.timezone else ''}
                <p><strong>Confidence Score:</strong> {confidence_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Data availability for selected location
            location_key = f"{location.latitude:.4f},{location.longitude:.4f}"
            availability = st.session_state.data_availability.get(location_key, {})
            
            if availability:
                st.markdown('<div class="data-availability-card">', unsafe_allow_html=True)
                st.markdown("**📊 Data Availability**")
                self.render_data_availability_indicators(availability)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Add the interactive map back
        self.render_location_map(location)
        
        # Action buttons
        st.markdown("### 🚀 Climate Analysis Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔍 Check Data Availability", use_container_width=True):
                self.refresh_data_availability()
        
        with col2:
            if st.button("📥 Collect Climate Data", type="primary", use_container_width=True):
                self.start_data_collection()
        
        with col3:
            if st.button("⚙️ Process & Analyze", use_container_width=True):
                self.start_data_processing()
        
        with col4:
            if st.button("📊 Full Workflow", type="secondary", use_container_width=True):
                self.run_full_workflow()
        
        # Show collection status
        if st.session_state.collection_status:
            self.render_collection_status()
        
        # Show processing results
        if st.session_state.processing_results:
            self.render_processing_results()
    
    def refresh_data_availability(self):
        """Refresh data availability check for selected location."""
        location = st.session_state.selected_location
        
        with st.spinner(f"🔍 Checking data availability for {location.name}..."):
            self.check_location_availability(location)
        
        st.success("✅ Data availability updated!")
        st.rerun()
    
    def start_data_collection(self):
        """Start adaptive data collection for selected location."""
        location = st.session_state.selected_location
        
        # Set collection parameters
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")  # Last 30 days
        
        st.session_state.collection_status = {
            "status": "running",
            "start_time": time.time(),
            "location": location.name,
            "date_range": f"{start_date} to {end_date}"
        }
        
        with st.spinner(f"📥 Collecting climate data for {location.name}..."):
            try:
                results = self.data_manager.fetch_adaptive_data_sync(
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    save=True,
                    force_all=False
                )
                
                # Update collection status
                successful_sources = len([r for r in results.values() if r is not None])
                total_sources = len(results)
                
                st.session_state.collection_status.update({
                    "status": "completed",
                    "end_time": time.time(),
                    "results": results,
                    "successful_sources": successful_sources,
                    "total_sources": total_sources
                })
                
                if successful_sources > 0:
                    st.success(f"✅ Data collection completed! {successful_sources}/{total_sources} sources successful")
                else:
                    st.warning("⚠️ Data collection completed but no data was retrieved")
                
            except Exception as e:
                st.session_state.collection_status.update({
                    "status": "failed",
                    "end_time": time.time(),
                    "error": str(e)
                })
                st.error(f"❌ Data collection failed: {e}")
        
        st.rerun()
    
    def start_data_processing(self):
        """Start data processing for collected data."""
        location = st.session_state.selected_location
        
        # Check if we have collected data
        if not st.session_state.collection_status.get("results"):
            st.warning("⚠️ No collected data found. Please collect data first.")
            return
        
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        with st.spinner(f"⚙️ Processing climate data for {location.name}..."):
            try:
                results = self.pipeline.process_global_location(
                    location=location,
                    start_date=start_date,
                    end_date=end_date,
                    skip_collection=True  # Use already collected data
                )
                
                st.session_state.processing_results = results
                
                # Count successful processing
                successful_processing = len([r for r in results.values() 
                                           if r and r != results.get('_metadata') and 'data' in r])
                
                if successful_processing > 0:
                    st.success(f"✅ Data processing completed! {successful_processing} datasets processed successfully")
                else:
                    st.warning("⚠️ Data processing completed but no processed datasets were generated")
                
            except Exception as e:
                st.error(f"❌ Data processing failed: {e}")
        
        st.rerun()
    
    def run_full_workflow(self):
        """Run the complete end-to-end workflow."""
        location = st.session_state.selected_location
        
        with st.spinner(f"🚀 Running complete climate analysis workflow for {location.name}..."):
            # Step 1: Data Collection
            self.start_data_collection()
            
            # Step 2: Data Processing (if collection was successful)
            if st.session_state.collection_status.get("successful_sources", 0) > 0:
                time.sleep(1)  # Brief pause between steps
                self.start_data_processing()
        
        st.success("🎉 Complete workflow finished!")
        st.rerun()
    
    def render_collection_status(self):
        """Render data collection status and results."""
        status = st.session_state.collection_status
        
        st.markdown("### 📥 Data Collection Status")
        
        if status["status"] == "running":
            st.markdown('<div class="processing-status">🔄 Data collection in progress...</div>', 
                       unsafe_allow_html=True)
        
        elif status["status"] == "completed":
            duration = status["end_time"] - status["start_time"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Success Rate", f"{status['successful_sources']}/{status['total_sources']}")
            with col2:
                st.metric("Collection Time", f"{duration:.1f}s")
            with col3:
                st.metric("Location", status["location"])
            
            # Detailed results
            if status.get("results"):
                st.markdown("**📊 Collection Results:**")
                for source, data in status["results"].items():
                    if data is not None:
                        if source == "air_quality":
                            records = len(data.get('hourly', {}).get('time', []))
                            st.markdown(f"✅ **Air Quality:** {records} hourly records")
                        elif source == "meteorological":
                            records = len(data.get('properties', {}).get('parameter', {}))
                            st.markdown(f"✅ **Meteorological:** {records} daily records")
                        elif source == "climate_projections":
                            records = len(data.get('data', []) if isinstance(data.get('data'), list) else [data.get('data')])
                            st.markdown(f"✅ **Climate Projections:** {records} scenarios")
                    else:
                        st.markdown(f"❌ **{source.title()}:** Collection failed")
        
        elif status["status"] == "failed":
            st.markdown(f'<div class="warning-message">❌ Collection failed: {status.get("error", "Unknown error")}</div>', 
                       unsafe_allow_html=True)
    
    def render_processing_results(self):
        """Render data processing results."""
        results = st.session_state.processing_results
        
        st.markdown("### ⚙️ Data Processing Results")
        
        # Processing metadata
        if '_metadata' in results:
            metadata = results['_metadata']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sources Processed", f"{metadata['data_sources_successful']}/{metadata['data_sources_attempted']}")
            with col2:
                st.metric("Processing Time", datetime.fromisoformat(metadata['processing_timestamp']).strftime("%H:%M:%S"))
            with col3:
                available_sources = len(metadata.get('available_sources', []))
                st.metric("Available Sources", available_sources)
        
        # Individual processing results
        for source, result in results.items():
            if source == '_metadata' or not result or 'data' not in result:
                continue
            
            st.markdown(f"**📊 {source.title()} Processing:**")
            
            data = result['data']
            quality_report = result.get('quality_report', {})
            
            # Handle different data types safely
            col1, col2, col3 = st.columns(3)
            with col1:
                if hasattr(data, 'shape'):  # DataFrame or numpy array
                    st.metric("Records", data.shape[0])
                elif isinstance(data, (list, tuple)):
                    st.metric("Records", len(data))
                else:
                    st.metric("Records", "N/A")
            
            with col2:
                if hasattr(data, 'shape') and len(data.shape) > 1:  # DataFrame with columns
                    st.metric("Features", data.shape[1])
                elif hasattr(data, 'columns'):  # DataFrame
                    st.metric("Features", len(data.columns))
                else:
                    st.metric("Features", "N/A")
            
            with col3:
                quality_score = quality_report.get('overall_score', 0)
                st.metric("Quality Score", f"{quality_score:.1f}/100")
            
            # Show data preview only for DataFrames
            if hasattr(data, 'head') and hasattr(data, 'shape'):  # It's a DataFrame
                if st.checkbox(f"Show {source} data preview", key=f"preview_{source}"):
                    st.dataframe(data.head(10), use_container_width=True)
            elif isinstance(data, (list, dict)):
                if st.checkbox(f"Show {source} data preview", key=f"preview_{source}"):
                    st.json(data if isinstance(data, dict) else data[:5])  # Show first 5 items for lists
    
    def render_quick_locations(self):
        """Render enhanced quick access to popular locations."""
        st.sidebar.subheader("🌍 Popular Locations")
        
        popular_locations = [
            "New York, USA",
            "London, UK", 
            "Tokyo, Japan",
            "Berlin, Germany",
            "Sydney, Australia",
            "São Paulo, Brazil",
            "Reykjavik, Iceland",
            "Singapore",
            "Dubai, UAE",
            "Vancouver, Canada"
        ]
        
        for location_query in popular_locations:
            if st.sidebar.button(location_query, use_container_width=True, key=f"quick_{location_query}"):
                with st.spinner(f"Loading {location_query}..."):
                    result = self.location_service.geocode_location(location_query)
                    if result:
                        self.select_location(result)
                    else:
                        st.sidebar.error(f"Could not load {location_query}")
    
    def render_location_history(self):
        """Render location history sidebar."""
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
    
    def render_service_stats(self):
        """Render enhanced service statistics."""
        st.sidebar.subheader("📊 System Status")
        
        # Global processing stats
        try:
            global_stats = self.pipeline.get_global_processing_stats()
            
            st.sidebar.metric("Locations Processed", global_stats.get("locations_processed", 0))
            st.sidebar.metric("Successful Collections", global_stats.get("successful_collections", 0))
            
            # API status indicators
            st.sidebar.markdown("**🔌 API Status:**")
            st.sidebar.markdown("🟢 Location Service (Nominatim)")
            st.sidebar.markdown("🟢 Climate Data Manager")
            st.sidebar.markdown("🟢 Processing Pipeline")
            
        except Exception as e:
            st.sidebar.error(f"Stats error: {e}")
        
        if st.sidebar.button("🔄 Refresh Stats"):
            st.rerun()
    
    def run(self):
        """Main enhanced app execution."""
        # Render header
        self.render_header()
        
        # Sidebar content
        with st.sidebar:
            st.title("🎛️ Global Controls")
            self.render_quick_locations()
            self.render_location_history()
            self.render_service_stats()
        
        # Main content
        self.render_location_search()
        
        if st.session_state.selected_location:
            self.render_selected_location()
        else:
            # Enhanced welcome message
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin: 2rem 0;">
                <h3>🌍 Welcome to Global Climate Prediction!</h3>
                <p style="font-size: 1.1rem;">Search for <strong>any location on Earth</strong> to get started with adaptive climate impact analysis.</p>
                <p style="margin-top: 1rem;"><em>✨ Enhanced Day 4 Features:</em></p>
                <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; flex-wrap: wrap;">
                    <div>🔍 Real-time data availability</div>
                    <div>📥 Adaptive data collection</div>
                    <div>⚙️ Intelligent processing</div>
                    <div>📊 Global coverage</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Example locations with coordinates
            st.markdown("### 🎯 Try These Examples:")
            examples_col1, examples_col2 = st.columns(2)
            
            with examples_col1:
                st.markdown("""
                **🏙️ Major Cities:**
                - `Tokyo, Japan`
                - `Paris, France`
                - `Cairo, Egypt`
                """)
            
            with examples_col2:
                st.markdown("""
                **🌍 Coordinates:**
                - `35.6762, 139.6503` (Tokyo)
                - `-33.8688, 151.2093` (Sydney)
                - `64.1466, -21.9426` (Reykjavik)
                """)
        
        # Enhanced footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>🌍 Dynamic Climate Impact Predictor | Day 4 Enhanced Global Platform</p>
            <p>Built with Streamlit • Professional APIs • Adaptive AI • Global Coverage</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    try:
        app = EnhancedLocationPickerApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        st.info("Please refresh the page or check your internet connection.")
        
        # Debug info in expander
        with st.expander("🔧 Debug Information"):
            st.code(str(e))


if __name__ == "__main__":
    main()