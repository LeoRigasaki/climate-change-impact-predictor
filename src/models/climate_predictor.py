#!/usr/bin/env python3
"""
🌍 Global Climate Predictor - Final Production Version
src/models/climate_predictor.py

Real-time climate predictions for any city using fixed neural network.
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class GlobalClimatePredictor:
    """Global climate prediction system using trained neural network."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.geocoder = Nominatim(user_agent="climate_predictor")
        self._load_model_components()
    
    def _load_model_components(self):
        """Load trained model and preprocessing components."""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required")
        
        # Load model (try .keras first, fallback to .h5)
        keras_path = self.model_dir / "global_climate_model.keras"
        h5_path = self.model_dir / "global_climate_model.h5"
        
        if keras_path.exists():
            self.model = tf.keras.models.load_model(str(keras_path))
        elif h5_path.exists():
            self.model = tf.keras.models.load_model(str(h5_path), compile=False)
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        else:
            raise FileNotFoundError("No model file found")
        
        # Load preprocessing components
        with open(self.model_dir / "feature_scaler.pkl", 'rb') as f:
            self.feature_scaler = pickle.load(f)
        with open(self.model_dir / "target_scaler.pkl", 'rb') as f:
            self.target_scaler = pickle.load(f)
        with open(self.model_dir / "climate_encoder.pkl", 'rb') as f:
            self.climate_encoder = pickle.load(f)
        with open(self.model_dir / "model_metadata_final.json", 'r') as f:
            self.metadata = json.load(f)
    
    def geocode_city(self, city_name: str) -> Optional[Tuple[float, float, str]]:
        """Get coordinates and country for city."""
        try:
            location = self.geocoder.geocode(city_name, timeout=10)
            if location:
                address_parts = location.address.split(', ')
                country = address_parts[-1] if address_parts else "Unknown"
                return location.latitude, location.longitude, country
            return None
        except:
            return None
    
    def determine_climate_zone(self, latitude: float) -> str:
        """Determine climate zone from latitude."""
        lat_abs = abs(latitude)
        if lat_abs > 66.5:
            return "polar"
        elif lat_abs > 35:
            return "temperate"
        elif lat_abs > 23.5:
            return "subtropical"
        else:
            return "tropical"
    
    def _estimate_features(self, latitude: float, climate_zone: str) -> Tuple[float, float]:
        """Estimate PM2.5 and wind based on location (using same logic as training)."""
        
        # PM2.5 estimates by climate zone
        pm25_by_zone = {"tropical": 15.0, "subtropical": 12.0, "temperate": 8.0, "polar": 5.0}
        pm25 = pm25_by_zone.get(climate_zone, 10.0)
        
        # Wind estimates by climate zone (matching training fix)
        wind_by_zone = {"tropical": 3.2, "subtropical": 3.8, "temperate": 4.1, "polar": 5.2}
        wind = wind_by_zone.get(climate_zone, 3.5)
        
        return pm25, wind
    
    def _estimate_historical_temp(self, latitude: float) -> float:
        """Estimate historical temperature (matching training fix logic)."""
        lat_abs = abs(latitude)
        if lat_abs > 66.5:
            return -5.0
        elif lat_abs > 45:
            return 12.0
        elif lat_abs > 23.5:
            return 22.0
        else:
            return 28.0
    
    def predict_climate(self, city_name: str) -> Dict:
        """Generate climate predictions for a city."""
        
        # Geocode
        geocode_result = self.geocode_city(city_name)
        if not geocode_result:
            return {"success": False, "error": f"Could not find '{city_name}'"}
        
        latitude, longitude, country = geocode_result
        climate_zone = self.determine_climate_zone(latitude)
        
        try:
            # Encode climate zone
            climate_zone_encoded = self.climate_encoder.transform([climate_zone])[0]
            
            # Estimate features (using same logic as training data fix)
            pm25, wind = self._estimate_features(latitude, climate_zone)
            historical_temp = self._estimate_historical_temp(latitude)
            
            # Create feature vector
            features = np.array([[
                latitude, longitude, climate_zone_encoded, pm25, historical_temp, wind
            ]])
            
            # Scale and predict
            features_scaled = self.feature_scaler.transform(features)
            pred_scaled = self.model.predict(features_scaled, verbose=0)
            pred_original = self.target_scaler.inverse_transform(pred_scaled)[0]
            
            # Extract predictions
            temp_max = pred_original[:3]   # 3 days max temp
            temp_min = pred_original[3:6]  # 3 days min temp
            aqi = pred_original[6]         # AQI
            uv = pred_original[7:10]       # 3 days UV
            precip = pred_original[10:13]  # 3 days precipitation
            
            # Calculate derived values
            heat_index = max(temp_max[0], temp_max[0] + (65 - 40) * 0.5) if temp_max[0] > 26 else temp_max[0]
            
            return {
                "success": True,
                "location": {
                    "city": city_name,
                    "country": country,
                    "coordinates": f"({latitude:.1f}°, {longitude:.1f}°)",
                    "climate_zone": climate_zone.title()
                },
                "forecasts": {
                    "temperature_3day": [
                        {"day": i+1, "min": temp_min[i], "max": temp_max[i]}
                        for i in range(3)
                    ],
                    "uv_index_3day": [uv[i] for i in range(3)],
                    "precipitation_3day": [precip[i] for i in range(3)]
                },
                "current": {
                    "aqi": aqi,
                    "heat_index": heat_index,
                    "aqi_category": self._get_aqi_category(aqi),
                    "heat_category": self._get_heat_category(heat_index)
                }
            }
            
        except Exception as e:
            return {"success": False, "error": f"Prediction failed: {str(e)}"}
    
    def _get_aqi_category(self, aqi: float) -> str:
        """Convert AQI to health category."""
        if aqi <= 12: return "Good"
        elif aqi <= 35: return "Moderate"
        elif aqi <= 55: return "Unhealthy for Sensitive"
        elif aqi <= 150: return "Unhealthy"
        else: return "Very Unhealthy"
    
    def _get_heat_category(self, heat_index: float) -> str:
        """Convert heat index to safety category."""
        if heat_index < 27: return "Safe"
        elif heat_index < 32: return "Caution"
        elif heat_index < 41: return "Extreme Caution"
        elif heat_index < 54: return "Danger"
        else: return "Extreme Danger"


def format_prediction(forecast: Dict) -> str:
    """Format prediction for display."""
    
    if not forecast["success"]:
        return f"❌ {forecast['error']}"
    
    loc = forecast["location"]
    temps = forecast["forecasts"]["temperature_3day"]
    current = forecast["current"]
    uv = forecast["forecasts"]["uv_index_3day"]
    precip = forecast["forecasts"]["precipitation_3day"]
    
    output = f"\n📍 {loc['city']}, {loc['country']} {loc['coordinates']}\n"
    output += f"🌍 Climate Zone: {loc['climate_zone']}\n\n"
    
    output += f"🌡️ Temperature Forecast:\n"
    for temp in temps:
        output += f"   Day {temp['day']}: {temp['min']:.1f}°C - {temp['max']:.1f}°C\n"
    
    output += f"\n💨 Air Quality: AQI {current['aqi']:.0f} ({current['aqi_category']})\n"
    output += f"🔥 Heat Index: {current['heat_index']:.1f}°C ({current['heat_category']})\n"
    output += f"☀️ UV Index: {uv[0]:.1f} today\n"
    output += f"🌧️ Rain Chance: {precip[0]:.0f}% today\n"
    
    return output


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode
        city = " ".join(sys.argv[1:])
        predictor = GlobalClimatePredictor()
        result = predictor.predict_climate(city)
        print(format_prediction(result))
    else:
        # Interactive mode
        predictor = GlobalClimatePredictor()
        
        while True:
            try:
                city = input("\n🌍 Enter city name (or 'quit'): ").strip()
                if city.lower() in ['quit', 'exit', 'q']:
                    break
                if city:
                    result = predictor.predict_climate(city)
                    print(format_prediction(result))
            except KeyboardInterrupt:
                break