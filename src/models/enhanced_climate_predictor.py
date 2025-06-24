#!/usr/bin/env python3
"""
src/models/enhanced_climate_predictor.py
🌍 Enhanced Global Climate Predictor - Day 10 Advanced ML Integration

Combines Day 8 base model with Day 10 advanced ML techniques:
- LSTM long-term forecasting (7-30 days)
- Multi-output comprehensive predictions
- Original climate prediction capabilities
"""

import sys
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.location_service import LocationService
from src.core.data_manager import ClimateDataManager


class EnhancedGlobalClimatePredictor:
    """
    🌍 Enhanced Global Climate Predictor with Advanced ML
    
    Integrates multiple ML models for comprehensive climate prediction:
    - Day 8: Base climate prediction model
    - Day 10: LSTM long-term forecasting
    - Day 10: Multi-output comprehensive prediction
    """
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.geocoder = Nominatim(user_agent="enhanced_climate_predictor")
        self.location_service = LocationService()
        self.data_manager = ClimateDataManager()
        
        # Model availability flags
        self.base_model_available = False
        self.lstm_model_available = False
        self.multi_output_model_available = False
        
        self._load_all_models()
    
    def _load_all_models(self):
        """Load all available models and preprocessing components."""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for enhanced climate prediction")
        
        # Load Day 8 Base Model
        self._load_base_model()
        
        # Load Day 10 Advanced Models
        self._load_lstm_model()
        self._load_multi_output_model()
        
        # Load metadata
        self._load_metadata()
    
    def _load_base_model(self):
        """Load Day 8 base climate model."""
        try:
            # Load base model
            keras_path = self.model_dir / "global_climate_model.keras"
            h5_path = self.model_dir / "global_climate_model.h5"
            
            if keras_path.exists():
                self.base_model = tf.keras.models.load_model(str(keras_path))
            elif h5_path.exists():
                self.base_model = tf.keras.models.load_model(str(h5_path), compile=False)
                self.base_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            else:
                raise FileNotFoundError("Base model not found")
            
            # Load base preprocessing
            with open(self.model_dir / "feature_scaler.pkl", 'rb') as f:
                self.base_feature_scaler = pickle.load(f)
            with open(self.model_dir / "target_scaler.pkl", 'rb') as f:
                self.base_target_scaler = pickle.load(f)
            with open(self.model_dir / "climate_encoder.pkl", 'rb') as f:
                self.base_climate_encoder = pickle.load(f)
            
            self.base_model_available = True
            
        except Exception as e:
            print(f"⚠️ Base model loading failed: {e}")
            self.base_model_available = False
    
    def _load_lstm_model(self):
        """Load Day 10 LSTM forecasting model."""
        try:
            # Load LSTM model
            lstm_path = self.model_dir / "advanced_lstm_forecaster.keras"
            if not lstm_path.exists():
                raise FileNotFoundError("LSTM model not found")
            
            self.lstm_model = tf.keras.models.load_model(str(lstm_path))
            
            # Load LSTM preprocessing
            with open(self.model_dir / "advanced_lstm_scaler_X.pkl", 'rb') as f:
                self.lstm_scaler_X = pickle.load(f)
            with open(self.model_dir / "advanced_lstm_scaler_y.pkl", 'rb') as f:
                self.lstm_scaler_y = pickle.load(f)
            
            self.lstm_model_available = True
            
        except Exception as e:
            print(f"⚠️ LSTM model loading failed: {e}")
            self.lstm_model_available = False
    
    def _load_multi_output_model(self):
        """Load Day 10 multi-output prediction model."""
        try:
            # Load multi-output model
            multi_path = self.model_dir / "production_multi_output_predictor.keras"
            if not multi_path.exists():
                raise FileNotFoundError("Multi-output model not found")
            
            self.multi_output_model = tf.keras.models.load_model(str(multi_path))
            
            # Load multi-output preprocessing
            with open(self.model_dir / "production_multi_scaler_X.pkl", 'rb') as f:
                self.multi_scaler_X = pickle.load(f)
            with open(self.model_dir / "production_multi_scalers_y.pkl", 'rb') as f:
                self.multi_scalers_y = pickle.load(f)
            with open(self.model_dir / "production_climate_encoder.pkl", 'rb') as f:
                self.production_climate_encoder = pickle.load(f)
            
            self.multi_output_model_available = True
            
        except Exception as e:
            print(f"⚠️ Multi-output model loading failed: {e}")
            self.multi_output_model_available = False
    
    def _load_metadata(self):
        """Load model metadata."""
        try:
            # Load Day 8 metadata
            with open(self.model_dir / "model_metadata_final.json", 'r') as f:
                self.base_metadata = json.load(f)
        except:
            self.base_metadata = {}
        
        try:
            # Load Day 10 metadata
            with open(self.model_dir / "day10_enhanced_metadata.json", 'r') as f:
                self.advanced_metadata = json.load(f)
        except:
            self.advanced_metadata = {}
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all available models."""
        return {
            "base_model": self.base_model_available,
            "lstm_forecaster": self.lstm_model_available,
            "multi_output_predictor": self.multi_output_model_available,
            "fully_enhanced": all([
                self.base_model_available,
                self.lstm_model_available, 
                self.multi_output_model_available
            ])
        }
    
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
        abs_lat = abs(latitude)
        if abs_lat <= 23.5:
            return "tropical"
        elif abs_lat <= 35:
            return "subtropical"
        elif abs_lat <= 50:
            return "temperate"
        elif abs_lat <= 66.5:
            return "continental"
        else:
            return "polar"
    
    def prepare_features_for_prediction(self, city_name: str) -> Optional[Dict[str, Any]]:
        """Prepare features for all prediction models."""
        
        # Geocode city
        geo_result = self.geocode_city(city_name)
        if not geo_result:
            return None
        
        latitude, longitude, country = geo_result
        climate_zone = self.determine_climate_zone(latitude)
        
        # Get real-time climate data
        try:
            from src.core.location_service import LocationInfo
            
            location = LocationInfo(
                name=city_name,
                latitude=latitude,
                longitude=longitude,
                country=country
            )
            
            # Collect current climate data
            import asyncio
            from datetime import datetime, timedelta
            
            # Use small date range for current conditions
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            
            # This would normally be an async call, but for demo we'll use mock data
            current_data = {
                "current_pm25": 15.0,  # Mock current PM2.5
                "historical_temp_avg": 20.0,  # Mock historical temperature
                "wind_avg": 3.5,  # Mock wind speed
                "aqi_avg": 25.0  # Mock AQI
            }
            
        except:
            # Use mock data if real data unavailable
            current_data = {
                "current_pm25": 15.0,
                "historical_temp_avg": 20.0,
                "wind_avg": 3.5,
                "aqi_avg": 25.0
            }
        
        return {
            "location": {
                "city": city_name,
                "latitude": latitude,
                "longitude": longitude,
                "country": country,
                "climate_zone": climate_zone
            },
            "current_data": current_data
        }
    
    def predict_climate(self, city_name: str) -> Dict[str, Any]:
        """Original Day 8 climate prediction (backward compatibility)."""
        
        if not self.base_model_available:
            return {
                "success": False,
                "error": "Base model not available",
                "city": city_name
            }
        
        # Prepare features
        feature_data = self.prepare_features_for_prediction(city_name)
        if not feature_data:
            return {
                "success": False,
                "error": "Could not geocode city",
                "city": city_name
            }
        
        try:
            location = feature_data["location"]
            current = feature_data["current_data"]
            
            # Prepare input features (matching Day 8 model)
            features = np.array([[
                current["current_pm25"],
                current["historical_temp_avg"], 
                current["wind_avg"],
                location["latitude"],
                location["longitude"],
                self.base_climate_encoder.transform([location["climate_zone"]])[0]
            ]])
            
            # Scale features
            features_scaled = self.base_feature_scaler.transform(features)
            
            # Make prediction
            prediction_scaled = self.base_model.predict(features_scaled, verbose=0)
            
            # Inverse transform prediction
            prediction = self.base_target_scaler.inverse_transform(prediction_scaled)[0]
            
            return {
                "success": True,
                "city": city_name,
                "location": location,
                "current": current,
                "prediction": {
                    "temperature_avg": float(prediction[0]),
                    "aqi_prediction": float(prediction[1]) if len(prediction) > 1 else current["aqi_avg"]
                },
                "model_used": "Day 8 Base Model"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
                "city": city_name
            }
    
    def predict_long_term(self, city_name: str, days: int = 7) -> Dict[str, Any]:
        """LSTM-based long-term weather forecasting."""
        
        if not self.lstm_model_available:
            return {
                "success": False,
                "error": "LSTM model not available",
                "city": city_name
            }
        
        # Prepare features
        feature_data = self.prepare_features_for_prediction(city_name)
        if not feature_data:
            return {
                "success": False,
                "error": "Could not geocode city",
                "city": city_name
            }
        
        try:
            location = feature_data["location"]
            current = feature_data["current_data"]
            
            # Prepare LSTM input features
            lstm_features = np.array([[
                current["current_pm25"],
                current["historical_temp_avg"],
                current["wind_avg"],
                location["latitude"],
                location["longitude"],
                self.production_climate_encoder.transform([location["climate_zone"]])[0]
            ]])
            
            # Scale features
            lstm_features_scaled = self.lstm_scaler_X.transform(lstm_features)
            
            # Make LSTM prediction
            lstm_prediction_scaled = self.lstm_model.predict(lstm_features_scaled, verbose=0)
            
            # Inverse transform to get actual temperatures
            lstm_prediction = self.lstm_scaler_y.inverse_transform(lstm_prediction_scaled)[0]
            
            # Create forecast sequence
            forecast_days = min(days, 7)  # LSTM trained for 7-day sequences
            forecast = []
            
            for i in range(forecast_days):
                forecast.append({
                    "day": i + 1,
                    "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                    "temperature": float(lstm_prediction[i]),
                    "confidence": "high" if i < 3 else "medium"
                })
            
            return {
                "success": True,
                "city": city_name,
                "location": location,
                "forecast_type": "LSTM Long-term",
                "forecast_days": forecast_days,
                "forecast": forecast,
                "model_used": "Day 10 LSTM Forecaster"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"LSTM prediction failed: {str(e)}",
                "city": city_name
            }
    
    def predict_comprehensive(self, city_name: str) -> Dict[str, Any]:
        """Multi-output comprehensive climate prediction."""
        
        if not self.multi_output_model_available:
            return {
                "success": False,
                "error": "Multi-output model not available",
                "city": city_name
            }
        
        # Prepare features
        feature_data = self.prepare_features_for_prediction(city_name)
        if not feature_data:
            return {
                "success": False,
                "error": "Could not geocode city",
                "city": city_name
            }
        
        try:
            location = feature_data["location"]
            current = feature_data["current_data"]
            
            # Prepare multi-output input features
            multi_features = np.array([[
                current["current_pm25"],
                current["historical_temp_avg"],
                current["wind_avg"],
                location["latitude"],
                location["longitude"],
                self.production_climate_encoder.transform([location["climate_zone"]])[0]
            ]])
            
            # Scale features
            multi_features_scaled = self.multi_scaler_X.transform(multi_features)
            
            # Make multi-output prediction
            multi_predictions = self.multi_output_model.predict(multi_features_scaled, verbose=0)
            
            # Process multiple outputs
            predictions = {}
            target_names = list(self.multi_scalers_y.keys())
            
            for i, target_name in enumerate(target_names):
                if i < len(multi_predictions):
                    # Inverse transform prediction
                    pred_scaled = multi_predictions[i].reshape(1, -1)
                    pred_actual = self.multi_scalers_y[target_name].inverse_transform(pred_scaled)[0]
                    
                    if target_name == "temperature":
                        predictions["temperature_3day"] = [
                            {"day": j+1, "temperature": float(pred_actual[j])} 
                            for j in range(len(pred_actual))
                        ]
                    elif target_name == "temperature_min":
                        predictions["temperature_min_3day"] = [
                            {"day": j+1, "temperature_min": float(pred_actual[j])} 
                            for j in range(len(pred_actual))
                        ]
                    elif target_name == "precipitation":
                        predictions["precipitation_3day"] = [
                            {"day": j+1, "precipitation_prob": float(pred_actual[j])} 
                            for j in range(len(pred_actual))
                        ]
                    elif target_name == "uv_index":
                        predictions["uv_index_3day"] = [
                            {"day": j+1, "uv_index": float(pred_actual[j])} 
                            for j in range(len(pred_actual))
                        ]
                    elif target_name == "air_quality":
                        predictions["air_quality"] = float(pred_actual[0])
            
            return {
                "success": True,
                "city": city_name,
                "location": location,
                "current": current,
                "comprehensive_predictions": predictions,
                "model_used": "Day 10 Multi-Output Predictor"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Multi-output prediction failed: {str(e)}",
                "city": city_name
            }
    
    def predict_climate_advanced(self, city_name: str) -> Dict[str, Any]:
        """Ultimate climate prediction combining all models."""
        
        results = {
            "city": city_name,
            "enhanced_prediction": True,
            "models_used": [],
            "predictions": {}
        }
        
        # Get basic prediction
        if self.base_model_available:
            base_result = self.predict_climate(city_name)
            if base_result["success"]:
                results["predictions"]["base"] = base_result
                results["models_used"].append("Base Model")
                results["location"] = base_result["location"]
                results["current"] = base_result["current"]
        
        # Get LSTM forecast
        if self.lstm_model_available:
            lstm_result = self.predict_long_term(city_name, 7)
            if lstm_result["success"]:
                results["predictions"]["lstm_forecast"] = lstm_result
                results["models_used"].append("LSTM Forecaster")
        
        # Get comprehensive prediction
        if self.multi_output_model_available:
            multi_result = self.predict_comprehensive(city_name)
            if multi_result["success"]:
                results["predictions"]["comprehensive"] = multi_result
                results["models_used"].append("Multi-Output Predictor")
        
        # Check if any predictions succeeded
        if results["predictions"]:
            results["success"] = True
            results["model_count"] = len(results["models_used"])
        else:
            results["success"] = False
            results["error"] = "No models available or all predictions failed"
        
        return results


# Import for backward compatibility
from datetime import datetime, timedelta