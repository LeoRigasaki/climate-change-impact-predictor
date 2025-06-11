#!/usr/bin/env python3
"""
🎯 Uncertainty Quantifier - Day 6 Confidence & Uncertainty System
src/validation/uncertainty_quantifier.py

Advanced uncertainty quantification system that calculates confidence intervals,
measurement uncertainties, and reliability bounds for climate data and predictions.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class UncertaintyQuantifier:
    """
    🎯 Advanced Uncertainty Quantification System
    
    Provides comprehensive uncertainty analysis for climate data including:
    - Measurement uncertainty from API data sources
    - Geographic uncertainty based on location characteristics
    - Temporal uncertainty from data age and collection methods
    - Model uncertainty from prediction algorithms
    - Cross-source uncertainty from data integration
    """
    
    def __init__(self):
        """Initialize Uncertainty Quantifier."""
        
        # API measurement uncertainties (typical ranges)
        self.api_uncertainties = {
            "Open-Meteo": {
                "temperature_2m": {"absolute": 1.5, "relative": 0.05},  # ±1.5°C or 5%
                "relative_humidity": {"absolute": 5.0, "relative": 0.08},  # ±5% or 8%
                "precipitation": {"absolute": 0.5, "relative": 0.15},  # ±0.5mm or 15%
                "wind_speed_2m": {"absolute": 1.0, "relative": 0.10},  # ±1.0 m/s or 10%
                "pm2_5": {"absolute": 2.0, "relative": 0.20},  # ±2.0 μg/m³ or 20%
                "pm10": {"absolute": 3.0, "relative": 0.20},  # ±3.0 μg/m³ or 20%
                "ozone": {"absolute": 5.0, "relative": 0.15}   # ±5.0 μg/m³ or 15%
            },
            "NASA-POWER": {
                "temperature_2m": {"absolute": 1.0, "relative": 0.03},  # Higher accuracy
                "precipitation": {"absolute": 0.3, "relative": 0.12},
                "relative_humidity": {"absolute": 3.0, "relative": 0.05},
                "wind_speed_2m": {"absolute": 0.8, "relative": 0.08}
            },
            "World-Bank-CCKP": {
                "temperature_projections": {"absolute": 2.0, "relative": 0.10},  # Future projections
                "precipitation_projections": {"absolute": 1.0, "relative": 0.25}
            }
        }
        
        # Geographic uncertainty factors by region
        self.geographic_uncertainty_factors = {
            "north_america": {"factor": 0.9, "reason": "Dense monitoring network"},
            "europe": {"factor": 0.85, "reason": "Excellent monitoring coverage"},
            "asia": {"factor": 1.15, "reason": "Variable monitoring density"},
            "africa": {"factor": 1.4, "reason": "Sparse monitoring network"},
            "south_america": {"factor": 1.25, "reason": "Limited coverage in remote areas"},
            "oceania": {"factor": 1.2, "reason": "Island geography challenges"},
            "antarctica": {"factor": 1.8, "reason": "Extreme conditions, minimal monitoring"}
        }
        
        # Climate zone uncertainty modifiers
        self.climate_zone_modifiers = {
            "tropical": {"stability": 0.9, "predictability": 0.85},
            "arid": {"stability": 1.1, "predictability": 0.90},
            "temperate": {"stability": 1.0, "predictability": 0.95},
            "continental": {"stability": 1.3, "predictability": 0.80},
            "polar": {"stability": 1.5, "predictability": 0.70}
        }
        
        # Temporal uncertainty factors
        self.temporal_uncertainty = {
            "real_time": 1.0,      # Current data
            "hours_1_6": 1.1,      # 1-6 hours old
            "hours_6_24": 1.2,     # 6-24 hours old
            "days_1_7": 1.4,       # 1-7 days old
            "days_7_30": 1.6,      # 1-4 weeks old
            "months_1_6": 2.0,     # 1-6 months old
            "months_6_plus": 2.5   # 6+ months old
        }
        
        self.uncertainty_cache = {}
        self.calculation_stats = {
            "total_calculations": 0,
            "cache_hits": 0,
            "avg_calculation_time": 0.0
        }
        
        logger.info("🎯 Uncertainty Quantifier initialized")
    
    def quantify_measurement_uncertainty(
        self, 
        data: pd.DataFrame, 
        data_sources: List[str],
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Quantify measurement uncertainty from API data sources.
        
        Args:
            data: Climate data DataFrame
            data_sources: List of data sources used
            variables: Specific variables to analyze (optional)
            
        Returns:
            Comprehensive measurement uncertainty analysis
        """
        calculation_start = datetime.now()
        
        try:
            if variables is None:
                variables = list(data.columns)
            
            uncertainty_analysis = {
                "variable_uncertainties": {},
                "source_reliabilities": {},
                "confidence_intervals": {},
                "overall_uncertainty_score": 0.0,
                "metadata": {
                    "calculation_timestamp": calculation_start.isoformat(),
                    "data_sources": data_sources,
                    "variables_analyzed": variables
                }
            }
            
            total_uncertainty = 0.0
            variables_processed = 0
            
            for variable in variables:
                if variable not in data.columns:
                    continue
                
                # Skip non-numeric variables
                if not pd.api.types.is_numeric_dtype(data[variable]):
                    logger.debug(f"Skipping non-numeric variable: {variable}")
                    continue
                
                # Get variable data
                var_data = data[variable].dropna()
                if len(var_data) == 0:
                    continue
                
                # Calculate measurement uncertainty for this variable
                var_uncertainty = self._calculate_variable_measurement_uncertainty(
                    var_data, variable, data_sources
                )
                
                uncertainty_analysis["variable_uncertainties"][variable] = var_uncertainty
                
                # Calculate confidence intervals
                confidence_intervals = self._calculate_confidence_intervals(var_data, var_uncertainty)
                uncertainty_analysis["confidence_intervals"][variable] = confidence_intervals
                
                total_uncertainty += var_uncertainty["combined_uncertainty"]
                variables_processed += 1
            
            # Calculate overall uncertainty score
            if variables_processed > 0:
                avg_uncertainty = total_uncertainty / variables_processed
                uncertainty_analysis["overall_uncertainty_score"] = min(100, avg_uncertainty * 100)
            
            # Source reliability assessment
            uncertainty_analysis["source_reliabilities"] = self._assess_source_reliabilities(data_sources)
            
            # Update calculation stats
            calculation_time = (datetime.now() - calculation_start).total_seconds()
            self._update_calculation_stats(calculation_time)
            uncertainty_analysis["metadata"]["calculation_time"] = calculation_time
            
            logger.info(f"✅ Measurement uncertainty calculated: {uncertainty_analysis['overall_uncertainty_score']:.1f}% avg uncertainty")
            
            return uncertainty_analysis
            
        except Exception as e:
            logger.error(f"❌ Measurement uncertainty calculation failed: {e}")
            return self._create_fallback_uncertainty_analysis(data, str(e))
    
    def calculate_regional_uncertainty(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate uncertainty factors based on geographic region and climate characteristics.
        
        Args:
            data: Climate data DataFrame
            location_info: Location metadata (lat, lon, climate_zone, etc.)
            
        Returns:
            Regional uncertainty analysis
        """
        try:
            # Determine geographic and climate context
            geographic_region = self._determine_geographic_region(location_info)
            climate_zone = self._determine_climate_zone(location_info)
            
            # Get regional uncertainty factors
            geo_factors = self.geographic_uncertainty_factors.get(
                geographic_region, 
                {"factor": 1.3, "reason": "Unknown region"}
            )
            
            climate_modifiers = self.climate_zone_modifiers.get(
                climate_zone,
                {"stability": 1.2, "predictability": 0.80}
            )
            
            # Calculate regional uncertainty multiplier
            regional_multiplier = geo_factors["factor"] * climate_modifiers["stability"]
            
            # Assess location-specific uncertainty factors
            location_factors = self._assess_location_specific_factors(location_info)
            
            # Calculate adjusted uncertainties for each variable
            adjusted_uncertainties = {}
            for column in data.columns:
                if pd.api.types.is_numeric_dtype(data[column]):
                    base_std = data[column].std()
                    regional_uncertainty = base_std * regional_multiplier
                    
                    adjusted_uncertainties[column] = {
                        "base_std": float(base_std),
                        "regional_multiplier": regional_multiplier,
                        "adjusted_uncertainty": float(regional_uncertainty),
                        "confidence_95": float(regional_uncertainty * 1.96)
                    }
            
            return {
                "regional_context": {
                    "geographic_region": geographic_region,
                    "climate_zone": climate_zone,
                    "regional_multiplier": regional_multiplier
                },
                "uncertainty_factors": {
                    "geographic_factor": geo_factors["factor"],
                    "geographic_reason": geo_factors["reason"],
                    "climate_stability": climate_modifiers["stability"],
                    "climate_predictability": climate_modifiers["predictability"]
                },
                "location_factors": location_factors,
                "adjusted_uncertainties": adjusted_uncertainties,
                "overall_regional_uncertainty": regional_multiplier,
                "uncertainty_category": self._categorize_uncertainty_level(regional_multiplier)
            }
            
        except Exception as e:
            logger.error(f"❌ Regional uncertainty calculation failed: {e}")
            return {"error": str(e), "overall_regional_uncertainty": 1.3}
    
    def estimate_temporal_uncertainty(
        self, 
        data: pd.DataFrame, 
        data_collection_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty based on data age and temporal characteristics.
        
        Args:
            data: Climate data DataFrame
            data_collection_timestamp: When data was collected (optional)
            
        Returns:
            Temporal uncertainty analysis
        """
        try:
            current_time = datetime.now()
            
            if data_collection_timestamp is None:
                # Estimate from data index if available
                if isinstance(data.index, pd.DatetimeIndex):
                    data_collection_timestamp = data.index.max().to_pydatetime()
                else:
                    data_collection_timestamp = current_time - timedelta(hours=1)  # Assume recent
            
            # Calculate data age
            data_age = current_time - data_collection_timestamp
            age_category = self._categorize_data_age(data_age)
            temporal_multiplier = self.temporal_uncertainty.get(age_category, 2.0)
            
            # Assess temporal consistency
            temporal_consistency = self._assess_temporal_consistency(data)
            
            # Calculate temporal variability
            temporal_variability = self._calculate_temporal_variability(data)
            
            # Adjust uncertainty based on temporal patterns
            pattern_adjustment = self._assess_temporal_patterns(data)
            
            # Final temporal uncertainty factor
            final_temporal_factor = temporal_multiplier * pattern_adjustment
            
            return {
                "data_age": {
                    "age_timedelta": str(data_age),
                    "age_category": age_category,
                    "base_multiplier": temporal_multiplier
                },
                "temporal_consistency": temporal_consistency,
                "temporal_variability": temporal_variability,
                "pattern_adjustment": pattern_adjustment,
                "final_temporal_factor": final_temporal_factor,
                "uncertainty_increase": f"{(final_temporal_factor - 1.0) * 100:.1f}%",
                "reliability_category": self._categorize_temporal_reliability(final_temporal_factor)
            }
            
        except Exception as e:
            logger.error(f"❌ Temporal uncertainty estimation failed: {e}")
            return {"error": str(e), "final_temporal_factor": 1.5}
    
    def compute_uncertainty_bounds(
        self, 
        predictions: Union[pd.Series, np.ndarray, float], 
        uncertainty_components: Dict[str, float],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compute comprehensive uncertainty bounds for predictions.
        
        Args:
            predictions: Prediction values
            uncertainty_components: Dict of uncertainty factors
            confidence_level: Confidence level (0.95 = 95%)
            
        Returns:
            Uncertainty bounds and confidence intervals
        """
        try:
            # Convert predictions to numpy array for consistent handling
            if isinstance(predictions, pd.Series):
                pred_values = predictions.values
            elif isinstance(predictions, (int, float)):
                pred_values = np.array([predictions])
            else:
                pred_values = np.array(predictions)
            
            # Combine uncertainty components
            combined_uncertainty = self._combine_uncertainty_components(uncertainty_components)
            
            # Calculate standard error
            standard_error = combined_uncertainty["total_uncertainty"]
            
            # Calculate confidence intervals
            confidence_multiplier = stats.norm.ppf((1 + confidence_level) / 2)
            margin_of_error = standard_error * confidence_multiplier
            
            # Calculate bounds
            lower_bounds = pred_values - margin_of_error
            upper_bounds = pred_values + margin_of_error
            
            # Calculate prediction intervals (wider than confidence intervals)
            prediction_multiplier = confidence_multiplier * 1.2  # Account for future uncertainty
            pred_margin = standard_error * prediction_multiplier
            pred_lower = pred_values - pred_margin
            pred_upper = pred_values + pred_margin
            
            # Quality assessment
            uncertainty_quality = self._assess_uncertainty_quality(
                combined_uncertainty, 
                len(pred_values)
            )
            
            return {
                "predictions": pred_values.tolist() if len(pred_values) > 1 else float(pred_values[0]),
                "uncertainty_components": uncertainty_components,
                "combined_uncertainty": combined_uncertainty,
                "confidence_intervals": {
                    "confidence_level": confidence_level,
                    "lower_bounds": lower_bounds.tolist() if len(lower_bounds) > 1 else float(lower_bounds[0]),
                    "upper_bounds": upper_bounds.tolist() if len(upper_bounds) > 1 else float(upper_bounds[0]),
                    "margin_of_error": float(margin_of_error)
                },
                "prediction_intervals": {
                    "lower_bounds": pred_lower.tolist() if len(pred_lower) > 1 else float(pred_lower[0]),
                    "upper_bounds": pred_upper.tolist() if len(pred_upper) > 1 else float(pred_upper[0]),
                    "margin_of_error": float(pred_margin)
                },
                "uncertainty_quality": uncertainty_quality,
                "interpretation": self._generate_uncertainty_interpretation(
                    combined_uncertainty, confidence_level
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Uncertainty bounds calculation failed: {e}")
            return self._create_fallback_bounds(predictions, str(e))
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get comprehensive uncertainty quantification system summary."""
        return {
            "calculation_stats": self.calculation_stats,
            "api_uncertainties": self.api_uncertainties,
            "geographic_factors": self.geographic_uncertainty_factors,
            "climate_modifiers": self.climate_zone_modifiers,
            "temporal_factors": self.temporal_uncertainty,
            "capabilities": {
                "measurement_uncertainty": True,
                "regional_uncertainty": True,
                "temporal_uncertainty": True,
                "confidence_intervals": True,
                "prediction_bounds": True,
                "uncertainty_propagation": True
            }
        }
    
    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================
    
    def _calculate_variable_measurement_uncertainty(
        self, 
        var_data: pd.Series, 
        variable: str, 
        data_sources: List[str]
    ) -> Dict[str, Any]:
        """Calculate measurement uncertainty for a specific variable."""
        try:
            uncertainties = []
            
            # Get uncertainties from each relevant source
            for source in data_sources:
                source_key = self._map_source_to_api_key(source)
                if source_key in self.api_uncertainties:
                    api_uncertainties = self.api_uncertainties[source_key]
                    if variable in api_uncertainties:
                        var_uncertainty = api_uncertainties[variable]
                        
                        # Calculate absolute and relative uncertainties
                        absolute_unc = var_uncertainty["absolute"]
                        relative_unc = var_uncertainty["relative"] * abs(var_data.mean())
                        
                        # Use the larger of absolute or relative uncertainty
                        combined_unc = max(absolute_unc, relative_unc)
                        uncertainties.append(combined_unc)
            
            # If no specific uncertainty found, estimate from data variability
            if not uncertainties:
                data_std = var_data.std()
                estimated_uncertainty = data_std * 0.1  # Assume 10% of standard deviation
                uncertainties.append(estimated_uncertainty)
            
            # Combine uncertainties (root sum of squares for independent sources)
            combined_uncertainty = np.sqrt(sum(u**2 for u in uncertainties)) / len(uncertainties)
            
            return {
                "individual_uncertainties": uncertainties,
                "combined_uncertainty": float(combined_uncertainty),
                "relative_uncertainty": float(combined_uncertainty / abs(var_data.mean())) if var_data.mean() != 0 else 0.0,
                "data_mean": float(var_data.mean()),
                "data_std": float(var_data.std())
            }
            
        except Exception as e:
            logger.warning(f"Variable uncertainty calculation failed for {variable}: {e}")
            return {"combined_uncertainty": var_data.std() * 0.1 if len(var_data) > 0 else 1.0}
    
    def _calculate_confidence_intervals(
        self, 
        var_data: pd.Series, 
        uncertainty_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate confidence intervals for a variable."""
        try:
            mean_val = var_data.mean()
            uncertainty = uncertainty_info["combined_uncertainty"]
            
            # 95% confidence interval
            ci_95_margin = uncertainty * 1.96
            ci_95_lower = mean_val - ci_95_margin
            ci_95_upper = mean_val + ci_95_margin
            
            # 99% confidence interval
            ci_99_margin = uncertainty * 2.576
            ci_99_lower = mean_val - ci_99_margin
            ci_99_upper = mean_val + ci_99_margin
            
            return {
                "mean": float(mean_val),
                "uncertainty": float(uncertainty),
                "ci_95": {
                    "lower": float(ci_95_lower),
                    "upper": float(ci_95_upper),
                    "margin": float(ci_95_margin)
                },
                "ci_99": {
                    "lower": float(ci_99_lower),
                    "upper": float(ci_99_upper),
                    "margin": float(ci_99_margin)
                }
            }
            
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return {"error": str(e)}
    
    def _assess_source_reliabilities(self, data_sources: List[str]) -> Dict[str, float]:
        """Assess reliability of data sources."""
        reliabilities = {}
        
        source_reliability_map = {
            "NASA-POWER": 0.95,
            "Open-Meteo": 0.88,
            "World-Bank-CCKP": 0.85,
            "unknown": 0.75
        }
        
        for source in data_sources:
            source_key = self._map_source_to_api_key(source)
            reliability = source_reliability_map.get(source_key, 0.75)
            reliabilities[source] = reliability
        
        return reliabilities
    
    def _determine_geographic_region(self, location_info: Dict[str, Any]) -> str:
        """Determine geographic region from coordinates."""
        lat = location_info.get("latitude", 0)
        lon = location_info.get("longitude", 0)
        
        # Simple continent classification
        if -180 <= lon <= -30:
            return "north_america" if lat >= 15 else "south_america"
        elif -30 < lon <= 60:
            return "europe" if lat >= 35 else "africa"
        elif 60 < lon <= 180:
            return "asia" if lat >= -10 else "oceania"
        else:
            return "unknown"
    
    def _determine_climate_zone(self, location_info: Dict[str, Any]) -> str:
        """Determine climate zone from location info."""
        if "climate_zone" in location_info:
            return location_info["climate_zone"]
        
        lat = abs(location_info.get("latitude", 0))
        
        if lat >= 66.5:
            return "polar"
        elif lat >= 45:
            return "continental"
        elif lat >= 23.5:
            return "temperate"
        else:
            return "tropical"
    
    def _assess_location_specific_factors(self, location_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess location-specific uncertainty factors."""
        factors = {
            "coastal_proximity": "unknown",
            "elevation_factor": 1.0,
            "urban_factor": 1.0
        }
        
        # Coastal proximity (if coordinates near coast, more stable)
        lat = location_info.get("latitude", 0)
        lon = location_info.get("longitude", 0)
        
        # Simple elevation estimation (very rough)
        if "elevation" in location_info:
            elevation = location_info["elevation"]
            if elevation > 2000:  # High elevation
                factors["elevation_factor"] = 1.3
            elif elevation > 500:  # Moderate elevation
                factors["elevation_factor"] = 1.1
        
        return factors
    
    def _categorize_data_age(self, data_age: timedelta) -> str:
        """Categorize data age for uncertainty estimation."""
        hours = data_age.total_seconds() / 3600
        
        if hours <= 1:
            return "real_time"
        elif hours <= 6:
            return "hours_1_6"
        elif hours <= 24:
            return "hours_6_24"
        elif hours <= 168:  # 1 week
            return "days_1_7"
        elif hours <= 720:  # 1 month
            return "days_7_30"
        elif hours <= 4320:  # 6 months
            return "months_1_6"
        else:
            return "months_6_plus"
    
    def _assess_temporal_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess temporal consistency of the data."""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return {"consistency_score": 0.8, "note": "Non-temporal index"}
            
            # Check for regular intervals
            time_diffs = data.index.to_series().diff().dropna()
            if len(time_diffs) == 0:
                return {"consistency_score": 0.5, "note": "Insufficient temporal data"}
            
            # Calculate consistency
            median_diff = time_diffs.median()
            std_diff = time_diffs.std()
            
            # Consistency score based on regularity
            if std_diff == pd.Timedelta(0):
                consistency_score = 1.0
            else:
                # Penalize irregular intervals
                irregularity = std_diff / median_diff
                consistency_score = max(0.3, 1.0 - irregularity.total_seconds() / 3600)  # Normalize by hours
            
            return {
                "consistency_score": float(consistency_score),
                "median_interval": str(median_diff),
                "interval_std": str(std_diff),
                "total_points": len(data)
            }
            
        except Exception as e:
            return {"consistency_score": 0.7, "error": str(e)}
    
    def _calculate_temporal_variability(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate temporal variability metrics."""
        try:
            variability_metrics = {}
            
            for column in data.select_dtypes(include=[np.number]).columns:
                var_data = data[column].dropna()
                if len(var_data) > 1:
                    # Calculate various variability measures
                    std_dev = var_data.std()
                    coefficient_of_variation = std_dev / abs(var_data.mean()) if var_data.mean() != 0 else np.inf
                    
                    variability_metrics[column] = {
                        "std_dev": float(std_dev),
                        "coefficient_of_variation": float(coefficient_of_variation),
                        "min_value": float(var_data.min()),
                        "max_value": float(var_data.max()),
                        "range": float(var_data.max() - var_data.min())
                    }
            
            # Overall variability score
            cv_values = [m["coefficient_of_variation"] for m in variability_metrics.values() 
                        if not np.isinf(m["coefficient_of_variation"])]
            
            avg_cv = np.mean(cv_values) if cv_values else 0.3
            variability_score = min(2.0, 1.0 + avg_cv)  # Scale variability to uncertainty multiplier
            
            return {
                "variable_metrics": variability_metrics,
                "average_coefficient_of_variation": float(avg_cv),
                "variability_score": float(variability_score)
            }
            
        except Exception as e:
            return {"variability_score": 1.2, "error": str(e)}
    
    def _assess_temporal_patterns(self, data: pd.DataFrame) -> float:
        """Assess temporal patterns for uncertainty adjustment."""
        try:
            # Look for seasonal patterns, trends, etc.
            if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 10:
                return 1.0  # No adjustment
            
            # Simple trend detection
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return 1.0
            
            # Calculate trend strength for first numeric column
            first_col = numeric_columns[0]
            values = data[first_col].dropna()
            
            if len(values) < 5:
                return 1.0
            
            # Linear trend
            x = np.arange(len(values))
            try:
                slope, _, r_value, _, _ = stats.linregress(x, values)
                trend_strength = abs(r_value)
                
                # Strong trends reduce uncertainty, weak trends increase it
                if trend_strength > 0.7:
                    return 0.9  # Reduce uncertainty
                elif trend_strength < 0.3:
                    return 1.1  # Increase uncertainty
                else:
                    return 1.0  # No change
            except:
                return 1.0
            
        except Exception as e:
            logger.debug(f"Pattern assessment failed: {e}")
            return 1.0
    
    def _combine_uncertainty_components(self, uncertainty_components: Dict[str, float]) -> Dict[str, Any]:
        """Combine multiple uncertainty components."""
        try:
            # Root sum of squares for independent uncertainties
            total_variance = sum(u**2 for u in uncertainty_components.values())
            total_uncertainty = np.sqrt(total_variance)
            
            # Individual contributions
            contributions = {}
            for component, uncertainty in uncertainty_components.items():
                contribution_percentage = (uncertainty**2 / total_variance) * 100 if total_variance > 0 else 0
                contributions[component] = {
                    "uncertainty": float(uncertainty),
                    "contribution_percentage": float(contribution_percentage)
                }
            
            return {
                "individual_components": uncertainty_components,
                "contributions": contributions,
                "total_uncertainty": float(total_uncertainty),
                "combination_method": "root_sum_of_squares"
            }
            
        except Exception as e:
            return {
                "total_uncertainty": 1.0,
                "error": str(e)
            }
    
    def _assess_uncertainty_quality(self, combined_uncertainty: Dict[str, Any], n_predictions: int) -> Dict[str, Any]:
        """Assess the quality of uncertainty estimates."""
        total_unc = combined_uncertainty.get("total_uncertainty", 1.0)
        
        # Quality based on uncertainty magnitude and sample size
        if total_unc < 0.1:
            quality = "High"
        elif total_unc < 0.3:
            quality = "Good"
        elif total_unc < 0.6:
            quality = "Moderate"
        else:
            quality = "Low"
        
        # Adjust for sample size
        if n_predictions < 5:
            quality = "Low"  # Small sample reduces quality
        
        confidence_level = "High" if total_unc < 0.2 else "Moderate" if total_unc < 0.5 else "Low"
        
        return {
            "uncertainty_quality": quality,
            "confidence_level": confidence_level,
            "total_uncertainty": float(total_unc),
            "sample_size": n_predictions,
            "reliability_note": f"Uncertainty estimates are {quality.lower()} quality with {confidence_level.lower()} confidence"
        }
    
    def _generate_uncertainty_interpretation(
        self, 
        combined_uncertainty: Dict[str, Any], 
        confidence_level: float
    ) -> str:
        """Generate human-readable interpretation of uncertainty."""
        total_unc = combined_uncertainty.get("total_uncertainty", 1.0)
        confidence_pct = int(confidence_level * 100)
        
        if total_unc < 0.1:
            return f"Very high confidence predictions with {confidence_pct}% certainty within ±{total_unc:.2f} units"
        elif total_unc < 0.3:
            return f"High confidence predictions with {confidence_pct}% certainty within ±{total_unc:.2f} units"
        elif total_unc < 0.6:
            return f"Moderate confidence predictions with {confidence_pct}% certainty within ±{total_unc:.2f} units"
        else:
            return f"Lower confidence predictions with {confidence_pct}% certainty within ±{total_unc:.2f} units"
    
    def _categorize_uncertainty_level(self, uncertainty_factor: float) -> str:
        """Categorize uncertainty level."""
        if uncertainty_factor < 1.1:
            return "Low"
        elif uncertainty_factor < 1.3:
            return "Moderate"
        elif uncertainty_factor < 1.6:
            return "High"
        else:
            return "Very High"
    
    def _categorize_temporal_reliability(self, temporal_factor: float) -> str:
        """Categorize temporal reliability."""
        if temporal_factor < 1.2:
            return "Excellent"
        elif temporal_factor < 1.5:
            return "Good"
        elif temporal_factor < 2.0:
            return "Fair"
        else:
            return "Poor"
    
    def _map_source_to_api_key(self, source: str) -> str:
        """Map data source string to API uncertainty key."""
        source_lower = source.lower()
        
        if "open" in source_lower or "meteo" in source_lower:
            return "Open-Meteo"
        elif "nasa" in source_lower or "power" in source_lower:
            return "NASA-POWER"
        elif "world" in source_lower or "bank" in source_lower or "cckp" in source_lower:
            return "World-Bank-CCKP"
        else:
            return "unknown"
    
    def _update_calculation_stats(self, calculation_time: float):
        """Update internal calculation statistics."""
        self.calculation_stats["total_calculations"] += 1
        
        # Update average calculation time
        total_calcs = self.calculation_stats["total_calculations"]
        current_avg = self.calculation_stats["avg_calculation_time"]
        
        self.calculation_stats["avg_calculation_time"] = (
            (current_avg * (total_calcs - 1) + calculation_time) / total_calcs
        )
    
    def _create_fallback_uncertainty_analysis(self, data: pd.DataFrame, error_msg: str) -> Dict[str, Any]:
        """Create fallback uncertainty analysis when calculation fails."""
        return {
            "overall_uncertainty_score": 50.0,
            "fallback": True,
            "error": error_msg,
            "basic_uncertainty_estimate": {
                "estimated_uncertainty": data.std().mean() * 0.2 if len(data.select_dtypes(include=[np.number]).columns) > 0 else 1.0,
                "confidence_level": "Low",
                "note": "Fallback uncertainty estimate based on data variability"
            }
        }
    
    def _create_fallback_bounds(self, predictions: Union[pd.Series, np.ndarray, float], error_msg: str) -> Dict[str, Any]:
        """Create fallback uncertainty bounds when calculation fails."""
        if isinstance(predictions, (int, float)):
            pred_val = float(predictions)
            fallback_margin = abs(pred_val) * 0.2  # 20% margin
        else:
            pred_array = np.array(predictions)
            pred_val = pred_array.tolist() if len(pred_array) > 1 else float(pred_array[0])
            fallback_margin = np.std(pred_array) * 1.96 if len(pred_array) > 1 else abs(float(pred_array[0])) * 0.2
        
        return {
            "predictions": pred_val,
            "fallback": True,
            "error": error_msg,
            "fallback_bounds": {
                "margin_of_error": float(fallback_margin),
                "confidence_level": 0.95,
                "note": "Fallback uncertainty bounds - use with caution"
            }
        }


# Convenience functions for common uncertainty calculations
def calculate_prediction_uncertainty(
    predictions: Union[pd.Series, np.ndarray, float],
    data_sources: List[str],
    location_info: Dict[str, Any],
    data_age_hours: float = 1.0
) -> Dict[str, Any]:
    """
    Convenience function for calculating comprehensive prediction uncertainty.
    
    Args:
        predictions: Prediction values
        data_sources: List of data sources used
        location_info: Location metadata
        data_age_hours: Age of data in hours
        
    Returns:
        Comprehensive uncertainty analysis with bounds
    """
    quantifier = UncertaintyQuantifier()
    
    # Create dummy data for uncertainty calculation if needed
    if isinstance(predictions, (int, float)):
        dummy_data = pd.DataFrame({"prediction": [predictions]})
    else:
        dummy_data = pd.DataFrame({"prediction": predictions})
    
    # Calculate uncertainty components
    measurement_unc = quantifier.quantify_measurement_uncertainty(dummy_data, data_sources)
    regional_unc = quantifier.calculate_regional_uncertainty(dummy_data, location_info)
    temporal_unc = quantifier.estimate_temporal_uncertainty(
        dummy_data, 
        datetime.now() - timedelta(hours=data_age_hours)
    )
    
    # Combine uncertainties
    uncertainty_components = {
        "measurement": measurement_unc.get("overall_uncertainty_score", 10) / 100,
        "regional": regional_unc.get("overall_regional_uncertainty", 1.2) - 1.0,
        "temporal": temporal_unc.get("final_temporal_factor", 1.1) - 1.0
    }
    
    # Calculate bounds
    bounds = quantifier.compute_uncertainty_bounds(predictions, uncertainty_components)
    
    return {
        "uncertainty_components": uncertainty_components,
        "bounds": bounds,
        "component_analyses": {
            "measurement": measurement_unc,
            "regional": regional_unc,
            "temporal": temporal_unc
        }
    }


def get_simple_confidence_interval(
    value: float, 
    uncertainty_percentage: float = 15.0, 
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Simple confidence interval calculation for quick estimates.
    
    Args:
        value: Central value
        uncertainty_percentage: Uncertainty as percentage (default 15%)
        confidence_level: Confidence level (default 0.95)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    uncertainty = abs(value) * (uncertainty_percentage / 100.0)
    confidence_multiplier = stats.norm.ppf((1 + confidence_level) / 2)
    margin = uncertainty * confidence_multiplier
    
    return (value - margin, value + margin)


if __name__ == "__main__":
    # Example usage and testing
    print("🎯 Uncertainty Quantifier - Day 6")
    print("=" * 50)
    
    # Create test data
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    test_data = pd.DataFrame({
        "temperature_2m": np.random.normal(20, 5, 100),
        "relative_humidity": np.random.uniform(40, 90, 100),
        "precipitation": np.random.exponential(2, 100),
        "pm2_5": np.random.exponential(15, 100)
    }, index=dates)
    
    # Test location
    test_location = {
        "name": "Berlin",
        "country": "Germany",
        "latitude": 52.52,
        "longitude": 13.41,
        "climate_zone": "temperate"
    }
    
    # Test uncertainty quantifier
    quantifier = UncertaintyQuantifier()
    
    # Test measurement uncertainty
    measurement_unc = quantifier.quantify_measurement_uncertainty(
        test_data, 
        ["Open-Meteo", "NASA-POWER"]
    )
    
    # Test regional uncertainty
    regional_unc = quantifier.calculate_regional_uncertainty(test_data, test_location)
    
    # Test temporal uncertainty
    temporal_unc = quantifier.estimate_temporal_uncertainty(test_data)
    
    # Test uncertainty bounds
    test_predictions = np.array([20.5, 22.1, 19.8])
    uncertainty_components = {
        "measurement": 0.15,
        "regional": 0.10,
        "temporal": 0.05
    }
    bounds = quantifier.compute_uncertainty_bounds(test_predictions, uncertainty_components)
    
    print(f"✅ Uncertainty Quantifier tests complete!")
    print(f"📊 Measurement uncertainty: {measurement_unc['overall_uncertainty_score']:.1f}%")
    print(f"🌍 Regional uncertainty factor: {regional_unc['overall_regional_uncertainty']:.2f}x")
    print(f"🕐 Temporal uncertainty factor: {temporal_unc['final_temporal_factor']:.2f}x")
    print(f"🎯 Prediction bounds: ±{bounds['confidence_intervals']['margin_of_error']:.2f}")
    
    # Test convenience function
    simple_uncertainty = calculate_prediction_uncertainty(
        25.0, 
        ["Open-Meteo"], 
        test_location, 
        2.0
    )
    print(f"🔧 Simple uncertainty analysis complete")
    
    # Test simple confidence interval
    lower, upper = get_simple_confidence_interval(25.0, 12.0)
    print(f"📈 Simple 95% CI: [{lower:.1f}, {upper:.1f}]")