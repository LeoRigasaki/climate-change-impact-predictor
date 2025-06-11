#!/usr/bin/env python3
"""
🌍 Global Baseline Comparator - Day 6 Climate Reference System
src/validation/baseline_comparator.py

Advanced global climate baseline comparison system that provides historical context,
worldwide reference comparisons, and climate change acceleration analysis.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)

class GlobalBaselineComparator:
    """
    🌍 Global Climate Baseline Comparison System
    
    Provides comprehensive baseline comparisons including:
    - Global climate reference data (1991-2020 WMO standards)
    - Historical climate normal comparisons
    - Regional climate baseline analysis
    - Climate change acceleration metrics
    - Seasonal baseline adjustments
    - Multi-variable baseline integration
    """
    
    def __init__(self, baseline_data_path: Optional[Path] = None):
        """Initialize Global Baseline Comparator."""
        
        # Global climate baselines (WMO 1991-2020 standard period approximations)
        self.global_baselines = {
            "temperature_2m": {
                "global_mean": 14.0,  # °C global average
                "global_std": 8.5,    # Global temperature variability
                "seasonal_variation": 4.2,
                "trend_per_decade": 0.18,  # °C per decade warming
                "reference_period": "1991-2020"
            },
            "precipitation": {
                "global_mean": 2.74,  # mm/day global average
                "global_std": 3.1,    # High variability
                "seasonal_variation": 1.8,
                "trend_per_decade": 0.04,  # mm/day per decade
                "reference_period": "1991-2020"
            },
            "relative_humidity": {
                "global_mean": 65.0,  # % global average
                "global_std": 20.0,   # Regional variations
                "seasonal_variation": 8.0,
                "trend_per_decade": -0.5,  # Slight decrease trend
                "reference_period": "1991-2020"
            },
            "wind_speed_2m": {
                "global_mean": 3.8,   # m/s global average
                "global_std": 2.1,    # Wind variability
                "seasonal_variation": 1.2,
                "trend_per_decade": -0.02,  # Slight stilling trend
                "reference_period": "1991-2020"
            },
            "pm2_5": {
                "global_mean": 15.0,  # μg/m³ global average
                "global_std": 12.0,   # High regional variability
                "seasonal_variation": 5.0,
                "trend_per_decade": 0.8,  # Increasing in many regions
                "reference_period": "2010-2020"  # Shorter record for air quality
            }
        }
        
        # Regional climate baselines by latitude zones
        self.regional_baselines = {
            "tropical": {  # 0-23.5°
                "temperature_2m": {"mean": 26.5, "std": 3.2, "seasonal_var": 1.5},
                "precipitation": {"mean": 4.2, "std": 2.8, "seasonal_var": 2.5},
                "relative_humidity": {"mean": 78.0, "std": 12.0, "seasonal_var": 8.0},
                "climate_stability": 0.9
            },
            "subtropical": {  # 23.5-35°
                "temperature_2m": {"mean": 20.5, "std": 8.1, "seasonal_var": 6.2},
                "precipitation": {"mean": 2.1, "std": 1.9, "seasonal_var": 1.4},
                "relative_humidity": {"mean": 62.0, "std": 18.0, "seasonal_var": 12.0},
                "climate_stability": 0.7
            },
            "temperate": {  # 35-60°
                "temperature_2m": {"mean": 12.8, "std": 12.5, "seasonal_var": 8.5},
                "precipitation": {"mean": 2.8, "std": 2.1, "seasonal_var": 1.2},
                "relative_humidity": {"mean": 68.0, "std": 15.0, "seasonal_var": 10.0},
                "climate_stability": 0.6
            },
            "boreal": {  # 60-70°
                "temperature_2m": {"mean": 1.2, "std": 18.2, "seasonal_var": 15.8},
                "precipitation": {"mean": 1.8, "std": 1.2, "seasonal_var": 0.8},
                "relative_humidity": {"mean": 72.0, "std": 12.0, "seasonal_var": 8.0},
                "climate_stability": 0.4
            },
            "polar": {  # >70°
                "temperature_2m": {"mean": -12.5, "std": 22.1, "seasonal_var": 20.5},
                "precipitation": {"mean": 0.8, "std": 0.6, "seasonal_var": 0.4},
                "relative_humidity": {"mean": 75.0, "std": 15.0, "seasonal_var": 12.0},
                "climate_stability": 0.3
            }
        }
        
        # Historical periods for trend analysis
        self.historical_periods = {
            "pre_industrial": {"start": 1880, "end": 1920, "temp_anomaly": -0.2},
            "mid_century": {"start": 1951, "end": 1980, "temp_anomaly": 0.0},
            "modern_baseline": {"start": 1991, "end": 2020, "temp_anomaly": 0.85},
            "current_decade": {"start": 2020, "end": 2030, "temp_anomaly": 1.15}
        }
        
        # Climate change acceleration factors
        self.acceleration_factors = {
            "arctic": 2.5,      # Arctic amplification
            "high_latitude": 1.8,
            "mid_latitude": 1.0,
            "tropical": 0.7,
            "oceanic": 0.8,
            "continental": 1.3
        }
        
        self.comparison_cache = {}
        self.baseline_stats = {
            "comparisons_performed": 0,
            "cache_hits": 0,
            "avg_comparison_time": 0.0
        }
        
        logger.info("🌍 Global Baseline Comparator initialized")
        logger.info(f"📊 Global baselines: {len(self.global_baselines)} variables")
        logger.info(f"🗺️ Regional baselines: {len(self.regional_baselines)} zones")
    
    def compare_to_global_baseline(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any],
        variables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare location data to global climate baselines.
        
        Args:
            data: Climate data DataFrame
            location_info: Location metadata (lat, lon, etc.)
            variables: Variables to compare (optional)
            
        Returns:
            Comprehensive global baseline comparison
        """
        comparison_start = datetime.now()
        
        try:
            if variables is None:
                variables = [col for col in data.columns if col in self.global_baselines]
            
            comparison_analysis = {
                "global_comparisons": {},
                "baseline_deviations": {},
                "percentile_rankings": {},
                "climate_anomalies": {},
                "historical_context": {},
                "metadata": {
                    "comparison_timestamp": comparison_start.isoformat(),
                    "location": location_info,
                    "variables_compared": variables,
                    "baseline_period": "1991-2020"
                }
            }
            
            location_name = f"{location_info.get('name', 'Unknown')}, {location_info.get('country', 'Unknown')}"
            
            for variable in variables:
                if variable not in data.columns or variable not in self.global_baselines:
                    continue
                
                var_data = data[variable].dropna()
                if len(var_data) == 0:
                    continue
                
                baseline_info = self.global_baselines[variable]
                
                # Calculate global comparison metrics
                global_comparison = self._calculate_global_comparison_metrics(
                    var_data, variable, baseline_info, location_info
                )
                comparison_analysis["global_comparisons"][variable] = global_comparison
                
                # Calculate baseline deviations
                baseline_deviation = self._calculate_baseline_deviations(
                    var_data, baseline_info
                )
                comparison_analysis["baseline_deviations"][variable] = baseline_deviation
                
                # Calculate global percentile rankings
                percentile_ranking = self._calculate_global_percentile_ranking(
                    var_data, baseline_info, location_info
                )
                comparison_analysis["percentile_rankings"][variable] = percentile_ranking
                
                # Detect climate anomalies
                climate_anomaly = self._detect_climate_anomalies(
                    var_data, baseline_info, variable
                )
                comparison_analysis["climate_anomalies"][variable] = climate_anomaly
                
                # Historical context analysis
                historical_context = self._analyze_historical_context(
                    var_data, variable, location_info
                )
                comparison_analysis["historical_context"][variable] = historical_context
            
            # Overall baseline comparison summary
            comparison_analysis["summary"] = self._generate_baseline_comparison_summary(
                comparison_analysis, location_info
            )
            
            # Update statistics
            comparison_time = (datetime.now() - comparison_start).total_seconds()
            self._update_baseline_stats(comparison_time)
            comparison_analysis["metadata"]["comparison_time"] = comparison_time
            
            logger.info(f"✅ Global baseline comparison complete for {location_name}")
            
            return comparison_analysis
            
        except Exception as e:
            logger.error(f"❌ Global baseline comparison failed: {e}")
            return self._create_fallback_comparison(data, location_info, str(e))
    
    def calculate_global_percentiles(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate global percentile rankings for location data.
        
        Args:
            data: Climate data DataFrame
            location_info: Location metadata
            
        Returns:
            Global percentile analysis
        """
        try:
            percentile_analysis = {
                "variable_percentiles": {},
                "overall_ranking": {},
                "regional_context": {},
                "extremes_analysis": {}
            }
            
            # Determine regional context
            regional_zone = self._determine_regional_zone(location_info)
            regional_baselines = self.regional_baselines.get(regional_zone, {})
            
            for variable in data.columns:
                if variable not in self.global_baselines:
                    continue
                
                var_data = data[variable].dropna()
                if len(var_data) == 0:
                    continue
                
                baseline = self.global_baselines[variable]
                regional_baseline = regional_baselines.get(variable, {})
                
                # Calculate global percentiles
                mean_value = var_data.mean()
                global_mean = baseline["global_mean"]
                global_std = baseline["global_std"]
                
                # Global percentile (assumes normal distribution)
                z_score = (mean_value - global_mean) / global_std
                global_percentile = stats.norm.cdf(z_score) * 100
                
                # Regional percentile if regional baseline available
                regional_percentile = None
                if regional_baseline:
                    regional_mean = regional_baseline.get("mean", global_mean)
                    regional_std = regional_baseline.get("std", global_std)
                    regional_z = (mean_value - regional_mean) / regional_std
                    regional_percentile = stats.norm.cdf(regional_z) * 100
                
                # Extremes analysis
                extremes = self._analyze_variable_extremes(var_data, baseline)
                
                percentile_analysis["variable_percentiles"][variable] = {
                    "global_percentile": float(global_percentile),
                    "regional_percentile": float(regional_percentile) if regional_percentile else None,
                    "z_score_global": float(z_score),
                    "value_mean": float(mean_value),
                    "global_baseline": float(global_mean),
                    "regional_baseline": float(regional_baseline.get("mean", global_mean)) if regional_baseline else None
                }
                
                percentile_analysis["extremes_analysis"][variable] = extremes
            
            # Overall ranking
            all_percentiles = [p["global_percentile"] for p in percentile_analysis["variable_percentiles"].values()]
            if all_percentiles:
                overall_percentile = np.mean(all_percentiles)
                percentile_analysis["overall_ranking"] = {
                    "overall_global_percentile": float(overall_percentile),
                    "ranking_category": self._categorize_percentile_ranking(overall_percentile),
                    "variables_analyzed": len(all_percentiles)
                }
            
            # Regional context
            percentile_analysis["regional_context"] = {
                "regional_zone": regional_zone,
                "climate_stability": regional_baselines.get("climate_stability", 0.7),
                "regional_comparison_available": bool(regional_baselines)
            }
            
            return percentile_analysis
            
        except Exception as e:
            logger.error(f"❌ Global percentile calculation failed: {e}")
            return {"error": str(e)}
    
    def assess_climate_change_acceleration(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any],
        reference_period: str = "1991-2020"
    ) -> Dict[str, Any]:
        """
        Assess climate change acceleration relative to global trends.
        
        Args:
            data: Climate data DataFrame
            location_info: Location metadata
            reference_period: Reference period for comparison
            
        Returns:
            Climate change acceleration analysis
        """
        try:
            acceleration_analysis = {
                "temperature_acceleration": {},
                "precipitation_trends": {},
                "regional_acceleration_factor": 1.0,
                "projected_changes": {},
                "acceleration_category": "moderate"
            }
            
            # Determine regional acceleration factor
            regional_zone = self._determine_regional_zone(location_info)
            latitude = abs(location_info.get("latitude", 45))
            
            # Get acceleration factor based on location characteristics
            if latitude > 70:
                base_acceleration = self.acceleration_factors["arctic"]
            elif latitude > 60:
                base_acceleration = self.acceleration_factors["high_latitude"]
            elif latitude > 30:
                base_acceleration = self.acceleration_factors["mid_latitude"]
            else:
                base_acceleration = self.acceleration_factors["tropical"]
            
            # Adjust for continental vs oceanic (simplified)
            # Assume more continental if longitude suggests interior location
            is_continental = self._estimate_continental_influence(location_info)
            if is_continental:
                base_acceleration *= 1.2
            else:
                base_acceleration *= 0.9
            
            acceleration_analysis["regional_acceleration_factor"] = base_acceleration
            
            # Temperature acceleration analysis
            if "temperature_2m" in data.columns:
                temp_data = data["temperature_2m"].dropna()
                if len(temp_data) > 10:  # Need sufficient data for trend
                    temp_acceleration = self._calculate_temperature_acceleration(
                        temp_data, base_acceleration
                    )
                    acceleration_analysis["temperature_acceleration"] = temp_acceleration
            
            # Precipitation trend analysis
            if "precipitation" in data.columns:
                precip_data = data["precipitation"].dropna()
                if len(precip_data) > 10:
                    precip_trends = self._calculate_precipitation_trends(
                        precip_data, base_acceleration
                    )
                    acceleration_analysis["precipitation_trends"] = precip_trends
            
            # Project future changes
            current_decade = self.historical_periods["current_decade"]
            projected_changes = self._project_future_climate_changes(
                location_info, base_acceleration, current_decade
            )
            acceleration_analysis["projected_changes"] = projected_changes
            
            # Categorize acceleration
            acceleration_analysis["acceleration_category"] = self._categorize_acceleration(
                base_acceleration
            )
            
            return acceleration_analysis
            
        except Exception as e:
            logger.error(f"❌ Climate change acceleration assessment failed: {e}")
            return {"error": str(e), "regional_acceleration_factor": 1.0}
    
    def build_global_baseline_comparisons(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build comprehensive global baseline comparison report.
        
        Args:
            data: Climate data DataFrame
            location_info: Location metadata
            
        Returns:
            Comprehensive baseline comparison report
        """
        try:
            comprehensive_report = {
                "executive_summary": {},
                "global_baseline_comparison": {},
                "global_percentiles": {},
                "climate_change_acceleration": {},
                "historical_context": {},
                "seasonal_baselines": {},
                "recommendations": [],
                "metadata": {
                    "report_timestamp": datetime.now().isoformat(),
                    "location": location_info,
                    "data_period": self._determine_data_period(data),
                    "baseline_standards": "WMO 1991-2020"
                }
            }
            
            location_name = f"{location_info.get('name', 'Unknown')}, {location_info.get('country', 'Unknown')}"
            
            # Global baseline comparison
            comprehensive_report["global_baseline_comparison"] = self.compare_to_global_baseline(
                data, location_info
            )
            
            # Global percentiles
            comprehensive_report["global_percentiles"] = self.calculate_global_percentiles(
                data, location_info
            )
            
            # Climate change acceleration
            comprehensive_report["climate_change_acceleration"] = self.assess_climate_change_acceleration(
                data, location_info
            )
            
            # Seasonal baseline analysis
            if isinstance(data.index, pd.DatetimeIndex):
                seasonal_analysis = self._analyze_seasonal_baselines(data, location_info)
                comprehensive_report["seasonal_baselines"] = seasonal_analysis
            
            # Historical context
            historical_context = self._build_historical_context_analysis(
                comprehensive_report, location_info
            )
            comprehensive_report["historical_context"] = historical_context
            
            # Executive summary
            executive_summary = self._generate_executive_summary(
                comprehensive_report, location_info
            )
            comprehensive_report["executive_summary"] = executive_summary
            
            # Generate recommendations
            recommendations = self._generate_baseline_recommendations(
                comprehensive_report, location_info
            )
            comprehensive_report["recommendations"] = recommendations
            
            logger.info(f"✅ Comprehensive baseline report complete for {location_name}")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"❌ Comprehensive baseline comparison failed: {e}")
            return {"error": str(e)}
    
    def get_baseline_summary(self) -> Dict[str, Any]:
        """Get comprehensive baseline comparator system summary."""
        return {
            "baseline_stats": self.baseline_stats,
            "global_baselines": self.global_baselines,
            "regional_baselines": self.regional_baselines,
            "historical_periods": self.historical_periods,
            "acceleration_factors": self.acceleration_factors,
            "capabilities": {
                "global_baseline_comparison": True,
                "global_percentile_ranking": True,
                "climate_change_acceleration": True,
                "historical_context_analysis": True,
                "seasonal_baseline_analysis": True,
                "comprehensive_reporting": True
            }
        }
    
    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================
    
    def _calculate_global_comparison_metrics(
        self, 
        var_data: pd.Series, 
        variable: str, 
        baseline_info: Dict[str, Any],
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comprehensive global comparison metrics."""
        try:
            mean_value = var_data.mean()
            global_mean = baseline_info["global_mean"]
            global_std = baseline_info["global_std"]
            
            # Basic comparisons
            absolute_difference = mean_value - global_mean
            relative_difference = (absolute_difference / global_mean) * 100 if global_mean != 0 else 0
            standard_deviations = absolute_difference / global_std if global_std != 0 else 0
            
            # Variability comparison
            local_std = var_data.std()
            variability_ratio = local_std / global_std if global_std != 0 else 1.0
            
            # Trend comparison (if enough data)
            trend_comparison = None
            if len(var_data) > 30:  # Need sufficient data for trend
                trend_comparison = self._compare_local_vs_global_trend(
                    var_data, baseline_info.get("trend_per_decade", 0)
                )
            
            return {
                "mean_value": float(mean_value),
                "global_baseline": float(global_mean),
                "absolute_difference": float(absolute_difference),
                "relative_difference_percent": float(relative_difference),
                "standard_deviations_from_global": float(standard_deviations),
                "local_variability": float(local_std),
                "global_variability": float(global_std),
                "variability_ratio": float(variability_ratio),
                "trend_comparison": trend_comparison,
                "comparison_category": self._categorize_comparison(standard_deviations)
            }
            
        except Exception as e:
            logger.warning(f"Global comparison calculation failed for {variable}: {e}")
            return {"error": str(e)}
    
    def _calculate_baseline_deviations(
        self, 
        var_data: pd.Series, 
        baseline_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate deviations from baseline with statistical significance."""
        try:
            mean_value = var_data.mean()
            global_mean = baseline_info["global_mean"]
            global_std = baseline_info["global_std"]
            
            # Calculate deviation
            deviation = mean_value - global_mean
            deviation_magnitude = abs(deviation)
            
            # Statistical significance (simple t-test approximation)
            n = len(var_data)
            standard_error = var_data.std() / np.sqrt(n) if n > 1 else global_std
            t_statistic = deviation / standard_error if standard_error != 0 else 0
            
            # Approximate p-value (for large samples)
            p_value = 2 * (1 - stats.norm.cdf(abs(t_statistic))) if n > 30 else None
            
            # Confidence interval for the difference
            confidence_95 = 1.96 * standard_error
            ci_lower = deviation - confidence_95
            ci_upper = deviation + confidence_95
            
            return {
                "deviation": float(deviation),
                "deviation_magnitude": float(deviation_magnitude),
                "standard_deviations": float(deviation / global_std) if global_std != 0 else 0,
                "statistical_significance": {
                    "t_statistic": float(t_statistic),
                    "p_value": float(p_value) if p_value else None,
                    "significant_at_95": bool(p_value and p_value < 0.05) if p_value else None
                },
                "confidence_interval_95": {
                    "lower": float(ci_lower),
                    "upper": float(ci_upper)
                },
                "deviation_category": self._categorize_deviation(deviation_magnitude / global_std if global_std != 0 else 0)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_global_percentile_ranking(
        self, 
        var_data: pd.Series, 
        baseline_info: Dict[str, Any],
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate global percentile ranking with confidence bounds."""
        try:
            mean_value = var_data.mean()
            global_mean = baseline_info["global_mean"]
            global_std = baseline_info["global_std"]
            
            # Global percentile
            z_score = (mean_value - global_mean) / global_std
            global_percentile = stats.norm.cdf(z_score) * 100
            
            # Uncertainty in percentile (due to sample size and measurement uncertainty)
            n = len(var_data)
            percentile_uncertainty = self._estimate_percentile_uncertainty(z_score, n)
            
            # Percentile confidence bounds
            percentile_lower = max(0, global_percentile - percentile_uncertainty)
            percentile_upper = min(100, global_percentile + percentile_uncertainty)
            
            return {
                "global_percentile": float(global_percentile),
                "z_score": float(z_score),
                "percentile_confidence_bounds": {
                    "lower": float(percentile_lower),
                    "upper": float(percentile_upper),
                    "uncertainty": float(percentile_uncertainty)
                },
                "ranking_category": self._categorize_percentile_ranking(global_percentile),
                "interpretation": self._interpret_percentile_ranking(global_percentile)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_climate_anomalies(
        self, 
        var_data: pd.Series, 
        baseline_info: Dict[str, Any],
        variable: str
    ) -> Dict[str, Any]:
        """Detect climate anomalies relative to global baseline."""
        try:
            mean_value = var_data.mean()
            global_mean = baseline_info["global_mean"]
            global_std = baseline_info["global_std"]
            
            # Anomaly detection thresholds
            moderate_threshold = 1.0  # 1 standard deviation
            strong_threshold = 2.0    # 2 standard deviations
            extreme_threshold = 3.0   # 3 standard deviations
            
            deviation_magnitude = abs(mean_value - global_mean) / global_std
            
            # Classify anomaly
            if deviation_magnitude >= extreme_threshold:
                anomaly_level = "extreme"
                severity = "Very High"
            elif deviation_magnitude >= strong_threshold:
                anomaly_level = "strong"
                severity = "High"
            elif deviation_magnitude >= moderate_threshold:
                anomaly_level = "moderate"
                severity = "Medium"
            else:
                anomaly_level = "normal"
                severity = "Low"
            
            # Direction of anomaly
            if mean_value > global_mean:
                direction = "above_normal"
                direction_desc = f"Above normal {variable}"
            else:
                direction = "below_normal"
                direction_desc = f"Below normal {variable}"
            
            return {
                "anomaly_detected": anomaly_level != "normal",
                "anomaly_level": anomaly_level,
                "anomaly_severity": severity,
                "anomaly_direction": direction,
                "deviation_magnitude": float(deviation_magnitude),
                "description": direction_desc if anomaly_level != "normal" else f"Normal {variable}",
                "requires_attention": anomaly_level in ["strong", "extreme"]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_historical_context(
        self, 
        var_data: pd.Series, 
        variable: str,
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze historical context for the variable."""
        try:
            mean_value = var_data.mean()
            
            historical_context = {
                "current_vs_periods": {},
                "trend_context": {},
                "climate_era": "modern"
            }
            
            # Compare to different historical periods
            baseline_info = self.global_baselines.get(variable, {})
            global_mean = baseline_info.get("global_mean", mean_value)
            
            for period_name, period_info in self.historical_periods.items():
                if variable == "temperature_2m":
                    # Adjust temperature for historical period
                    period_adjustment = period_info.get("temp_anomaly", 0)
                    period_baseline = global_mean + period_adjustment
                    
                    difference = mean_value - period_baseline
                    
                    historical_context["current_vs_periods"][period_name] = {
                        "period_baseline": float(period_baseline),
                        "current_value": float(mean_value),
                        "difference": float(difference),
                        "period_years": f"{period_info['start']}-{period_info['end']}"
                    }
            
            # Determine climate era
            if variable == "temperature_2m":
                temp_anomaly = mean_value - global_mean
                if temp_anomaly > 1.5:
                    historical_context["climate_era"] = "anthropocene_warming"
                elif temp_anomaly > 0.5:
                    historical_context["climate_era"] = "modern_warming"
                else:
                    historical_context["climate_era"] = "baseline_period"
            
            return historical_context
            
        except Exception as e:
            return {"error": str(e)}
    
    def _determine_regional_zone(self, location_info: Dict[str, Any]) -> str:
        """Determine regional climate zone from location."""
        latitude = abs(location_info.get("latitude", 45))
        
        if latitude >= 70:
            return "polar"
        elif latitude >= 60:
            return "boreal"
        elif latitude >= 35:
            return "temperate"
        elif latitude >= 23.5:
            return "subtropical"
        else:
            return "tropical"
    
    def _analyze_variable_extremes(
        self, 
        var_data: pd.Series, 
        baseline_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze variable extremes relative to global baseline."""
        try:
            global_mean = baseline_info["global_mean"]
            global_std = baseline_info["global_std"]
            
            # Calculate extreme values
            min_value = var_data.min()
            max_value = var_data.max()
            
            # Z-scores for extremes
            min_z_score = (min_value - global_mean) / global_std if global_std != 0 else 0
            max_z_score = (max_value - global_mean) / global_std if global_std != 0 else 0
            
            # Categorize extremes
            def categorize_extreme(z_score):
                abs_z = abs(z_score)
                if abs_z >= 4:
                    return "unprecedented"
                elif abs_z >= 3:
                    return "extreme"
                elif abs_z >= 2:
                    return "very_unusual"
                elif abs_z >= 1.5:
                    return "unusual"
                else:
                    return "normal"
            
            return {
                "minimum": {
                    "value": float(min_value),
                    "z_score": float(min_z_score),
                    "category": categorize_extreme(min_z_score)
                },
                "maximum": {
                    "value": float(max_value),
                    "z_score": float(max_z_score),
                    "category": categorize_extreme(max_z_score)
                },
                "range": float(max_value - min_value),
                "extreme_events_detected": any(abs(z) >= 2 for z in [min_z_score, max_z_score])
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _categorize_percentile_ranking(self, percentile: float) -> str:
        """Categorize percentile ranking."""
        if percentile >= 95:
            return "top_5_percent"
        elif percentile >= 90:
            return "top_10_percent"
        elif percentile >= 75:
            return "upper_quartile"
        elif percentile >= 25:
            return "middle_range"
        elif percentile >= 10:
            return "lower_quartile"
        elif percentile >= 5:
            return "bottom_10_percent"
        else:
            return "bottom_5_percent"
    
    def _interpret_percentile_ranking(self, percentile: float) -> str:
        """Generate human-readable interpretation of percentile ranking."""
        if percentile >= 99:
            return "Among the highest globally (top 1%)"
        elif percentile >= 95:
            return "Very high globally (top 5%)"
        elif percentile >= 90:
            return "High globally (top 10%)"
        elif percentile >= 75:
            return "Above average globally (top 25%)"
        elif percentile >= 50:
            return "Above global median"
        elif percentile >= 25:
            return "Below global median"
        elif percentile >= 10:
            return "Low globally (bottom 25%)"
        elif percentile >= 5:
            return "Very low globally (bottom 10%)"
        else:
            return "Among the lowest globally (bottom 5%)"
    
    def _calculate_temperature_acceleration(
        self, 
        temp_data: pd.Series, 
        base_acceleration: float
    ) -> Dict[str, Any]:
        """Calculate temperature acceleration metrics."""
        try:
            # Estimate local trend if possible
            if isinstance(temp_data.index, pd.DatetimeIndex) and len(temp_data) > 20:
                # Calculate trend over time
                time_numeric = pd.to_numeric(temp_data.index) / 1e9 / 365.25 / 24 / 3600  # Convert to years
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, temp_data)
                
                # Convert slope to °C per decade
                local_trend_per_decade = slope * 10
                
                # Compare to global trend
                global_trend = self.global_baselines["temperature_2m"]["trend_per_decade"]
                acceleration_ratio = local_trend_per_decade / global_trend if global_trend != 0 else 1.0
                
                return {
                    "local_trend_per_decade": float(local_trend_per_decade),
                    "global_trend_per_decade": float(global_trend),
                    "acceleration_ratio": float(acceleration_ratio),
                    "expected_acceleration": float(base_acceleration),
                    "trend_significance": {
                        "r_squared": float(r_value**2),
                        "p_value": float(p_value),
                        "significant": bool(p_value < 0.05)
                    },
                    "acceleration_category": self._categorize_acceleration(acceleration_ratio)
                }
            else:
                # Use expected acceleration based on location
                return {
                    "expected_acceleration": float(base_acceleration),
                    "note": "Insufficient temporal data for trend calculation",
                    "acceleration_category": self._categorize_acceleration(base_acceleration)
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_precipitation_trends(
        self, 
        precip_data: pd.Series, 
        base_acceleration: float
    ) -> Dict[str, Any]:
        """Calculate precipitation trend analysis."""
        try:
            # Precipitation trends are more complex and variable
            mean_precip = precip_data.mean()
            std_precip = precip_data.std()
            cv_precip = std_precip / mean_precip if mean_precip != 0 else 0
            
            # Estimate trend if temporal data available
            trend_analysis = {}
            if isinstance(precip_data.index, pd.DatetimeIndex) and len(precip_data) > 20:
                time_numeric = pd.to_numeric(precip_data.index) / 1e9 / 365.25 / 24 / 3600
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, precip_data)
                
                # Convert to mm/day per decade
                local_trend_per_decade = slope * 10
                global_trend = self.global_baselines["precipitation"]["trend_per_decade"]
                
                trend_analysis = {
                    "local_trend_per_decade": float(local_trend_per_decade),
                    "global_trend_per_decade": float(global_trend),
                    "trend_significance": float(p_value),
                    "trend_direction": "increasing" if local_trend_per_decade > 0 else "decreasing"
                }
            
            return {
                "mean_precipitation": float(mean_precip),
                "precipitation_variability": float(cv_precip),
                "variability_category": "high" if cv_precip > 1.0 else "moderate" if cv_precip > 0.5 else "low",
                "trend_analysis": trend_analysis
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _project_future_climate_changes(
        self, 
        location_info: Dict[str, Any], 
        acceleration_factor: float,
        current_period: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Project future climate changes based on location and acceleration."""
        try:
            # Project temperature changes for different time horizons
            global_warming_rate = 0.18  # °C per decade (IPCC estimates)
            local_warming_rate = global_warming_rate * acceleration_factor
            
            projections = {
                "2030": {
                    "temperature_change": float(local_warming_rate * 1),  # 1 decade
                    "uncertainty_range": float(local_warming_rate * 0.3)
                },
                "2050": {
                    "temperature_change": float(local_warming_rate * 3),  # 3 decades
                    "uncertainty_range": float(local_warming_rate * 0.5)
                },
                "2100": {
                    "temperature_change": float(local_warming_rate * 8),  # 8 decades
                    "uncertainty_range": float(local_warming_rate * 1.0)
                }
            }
            
            # Regional context
            regional_zone = self._determine_regional_zone(location_info)
            
            return {
                "temperature_projections": projections,
                "local_warming_rate_per_decade": float(local_warming_rate),
                "global_warming_rate_per_decade": float(global_warming_rate),
                "acceleration_factor": float(acceleration_factor),
                "regional_context": regional_zone,
                "projection_confidence": "medium",  # Could be enhanced with more sophisticated modeling
                "note": "Projections based on linear extrapolation with regional acceleration factors"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _categorize_acceleration(self, acceleration_factor: float) -> str:
        """Categorize climate change acceleration."""
        if acceleration_factor >= 2.0:
            return "very_high"
        elif acceleration_factor >= 1.5:
            return "high"
        elif acceleration_factor >= 1.2:
            return "moderate_high"
        elif acceleration_factor >= 0.8:
            return "moderate"
        elif acceleration_factor >= 0.5:
            return "low"
        else:
            return "very_low"
    
    def _analyze_seasonal_baselines(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze seasonal variations against baseline expectations."""
        try:
            seasonal_analysis = {}
            
            # Determine hemisphere for correct seasons
            latitude = location_info.get("latitude", 0)
            is_northern = latitude >= 0
            
            # Group data by season
            data_with_season = data.copy()
            if is_northern:
                season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                            3: "Spring", 4: "Spring", 5: "Spring",
                            6: "Summer", 7: "Summer", 8: "Summer",
                            9: "Fall", 10: "Fall", 11: "Fall"}
            else:
                # Southern hemisphere - seasons are reversed
                season_map = {12: "Summer", 1: "Summer", 2: "Summer",
                            3: "Fall", 4: "Fall", 5: "Fall",
                            6: "Winter", 7: "Winter", 8: "Winter",
                            9: "Spring", 10: "Spring", 11: "Spring"}
            
            data_with_season['season'] = data_with_season.index.month.map(season_map)
            
            for variable in data.columns:
                if variable not in self.global_baselines:
                    continue
                
                baseline_info = self.global_baselines[variable]
                seasonal_var = baseline_info.get("seasonal_variation", 0)
                
                variable_seasonal = {}
                
                for season in ["Winter", "Spring", "Summer", "Fall"]:
                    season_data = data_with_season[data_with_season['season'] == season][variable].dropna()
                    
                    if len(season_data) > 0:
                        season_mean = season_data.mean()
                        season_std = season_data.std()
                        
                        # Compare to expected seasonal variation
                        expected_variation = seasonal_var
                        actual_variation = season_std
                        
                        variable_seasonal[season] = {
                            "mean": float(season_mean),
                            "std": float(season_std),
                            "sample_size": len(season_data),
                            "expected_variation": float(expected_variation),
                            "actual_variation": float(actual_variation),
                            "variation_ratio": float(actual_variation / expected_variation) if expected_variation != 0 else 1.0
                        }
                
                seasonal_analysis[variable] = variable_seasonal
            
            return {
                "seasonal_patterns": seasonal_analysis,
                "hemisphere": "Northern" if is_northern else "Southern",
                "seasonal_baseline_reference": "WMO global seasonal normals"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _build_historical_context_analysis(
        self, 
        comprehensive_report: Dict[str, Any], 
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build comprehensive historical context analysis."""
        try:
            historical_analysis = {
                "climate_era_assessment": {},
                "long_term_trends": {},
                "historical_significance": {}
            }
            
            # Extract temperature information for historical context
            global_comparison = comprehensive_report.get("global_baseline_comparison", {})
            temp_comparison = global_comparison.get("global_comparisons", {}).get("temperature_2m", {})
            
            if temp_comparison:
                mean_temp = temp_comparison.get("mean_value", 14.0)
                global_baseline = temp_comparison.get("global_baseline", 14.0)
                temp_anomaly = mean_temp - global_baseline
                
                # Assess climate era
                if temp_anomaly > 2.0:
                    climate_era = "dangerous_warming"
                    era_description = "Dangerous anthropogenic warming period"
                elif temp_anomaly > 1.5:
                    climate_era = "significant_warming"
                    era_description = "Significant anthropogenic warming period"
                elif temp_anomaly > 0.5:
                    climate_era = "modern_warming"
                    era_description = "Modern warming period"
                elif temp_anomaly > -0.5:
                    climate_era = "baseline_period"
                    era_description = "Near baseline climate period"
                else:
                    climate_era = "cooler_period"
                    era_description = "Cooler than baseline period"
                
                historical_analysis["climate_era_assessment"] = {
                    "era": climate_era,
                    "description": era_description,
                    "temperature_anomaly": float(temp_anomaly),
                    "significance": "High" if abs(temp_anomaly) > 1.0 else "Moderate" if abs(temp_anomaly) > 0.5 else "Low"
                }
            
            # Acceleration context
            acceleration_data = comprehensive_report.get("climate_change_acceleration", {})
            acceleration_factor = acceleration_data.get("regional_acceleration_factor", 1.0)
            
            historical_analysis["long_term_trends"] = {
                "warming_acceleration": float(acceleration_factor),
                "acceleration_context": self._categorize_acceleration(acceleration_factor),
                "projected_warming_2050": float(acceleration_factor * 0.18 * 3),  # 3 decades * warming rate
                "trend_confidence": "Medium"
            }
            
            return historical_analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_executive_summary(
        self, 
        comprehensive_report: Dict[str, Any], 
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary of baseline comparison."""
        try:
            location_name = f"{location_info.get('name', 'Unknown')}, {location_info.get('country', 'Unknown')}"
            
            # Extract key metrics
            percentiles = comprehensive_report.get("global_percentiles", {})
            overall_ranking = percentiles.get("overall_ranking", {})
            overall_percentile = overall_ranking.get("overall_global_percentile", 50)
            
            acceleration_data = comprehensive_report.get("climate_change_acceleration", {})
            acceleration_factor = acceleration_data.get("regional_acceleration_factor", 1.0)
            
            # Key findings
            key_findings = []
            
            # Climate ranking
            ranking_category = self._categorize_percentile_ranking(overall_percentile)
            if ranking_category in ["top_5_percent", "top_10_percent"]:
                key_findings.append(f"Climate conditions rank in the {ranking_category.replace('_', ' ')} globally")
            elif ranking_category in ["bottom_5_percent", "bottom_10_percent"]:
                key_findings.append(f"Climate conditions rank in the {ranking_category.replace('_', ' ')} globally")
            
            # Acceleration assessment
            acceleration_category = self._categorize_acceleration(acceleration_factor)
            if acceleration_category in ["very_high", "high"]:
                key_findings.append(f"Experiencing {acceleration_category.replace('_', ' ')} climate change acceleration")
            
            # Climate anomalies
            global_comparison = comprehensive_report.get("global_baseline_comparison", {})
            anomalies = global_comparison.get("climate_anomalies", {})
            extreme_anomalies = [var for var, anom in anomalies.items() 
                               if isinstance(anom, dict) and anom.get("anomaly_level") in ["strong", "extreme"]]
            
            if extreme_anomalies:
                key_findings.append(f"Extreme climate anomalies detected in: {', '.join(extreme_anomalies)}")
            
            return {
                "location": location_name,
                "overall_climate_percentile": float(overall_percentile),
                "climate_ranking_category": ranking_category,
                "acceleration_factor": float(acceleration_factor),
                "acceleration_category": acceleration_category,
                "key_findings": key_findings,
                "summary_assessment": self._generate_summary_assessment(overall_percentile, acceleration_factor),
                "report_confidence": "Medium",
                "baseline_reference": "WMO 1991-2020 Climate Normals"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_baseline_recommendations(
        self, 
        comprehensive_report: Dict[str, Any], 
        location_info: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on baseline comparison."""
        recommendations = []
        
        try:
            # Extract key data
            percentiles = comprehensive_report.get("global_percentiles", {})
            overall_percentile = percentiles.get("overall_ranking", {}).get("overall_global_percentile", 50)
            
            acceleration_data = comprehensive_report.get("climate_change_acceleration", {})
            acceleration_factor = acceleration_data.get("regional_acceleration_factor", 1.0)
            
            global_comparison = comprehensive_report.get("global_baseline_comparison", {})
            anomalies = global_comparison.get("climate_anomalies", {})
            
            # Climate ranking recommendations
            if overall_percentile > 90:
                recommendations.append("🌟 Excellent climate conditions - ideal for outdoor activities and agriculture")
            elif overall_percentile < 10:
                recommendations.append("⚠️ Challenging climate conditions - consider adaptation strategies")
            
            # Acceleration recommendations
            if acceleration_factor > 1.5:
                recommendations.append("🔥 High climate change acceleration - prioritize adaptation and mitigation")
                recommendations.append("📊 Monitor climate trends closely for rapid changes")
            elif acceleration_factor > 1.2:
                recommendations.append("📈 Moderate climate acceleration - develop climate resilience plans")
            
            # Anomaly-specific recommendations
            for variable, anomaly_data in anomalies.items():
                if isinstance(anomaly_data, dict) and anomaly_data.get("requires_attention"):
                    if variable == "temperature_2m":
                        recommendations.append("🌡️ Extreme temperature anomalies - enhance heat/cold protection measures")
                    elif variable == "precipitation":
                        recommendations.append("💧 Extreme precipitation anomalies - improve water management systems")
                    elif variable == "pm2_5":
                        recommendations.append("💨 Air quality concerns - implement pollution monitoring and protection")
            
            # General recommendations
            recommendations.append("📋 Regular climate monitoring recommended for trend detection")
            recommendations.append("🔍 Compare to regional peers for local context")
            
            return recommendations
            
        except Exception as e:
            return ["❌ Unable to generate recommendations due to analysis error"]
    
    def _generate_summary_assessment(self, overall_percentile: float, acceleration_factor: float) -> str:
        """Generate overall summary assessment."""
        # Climate quality assessment
        if overall_percentile > 80:
            climate_quality = "excellent"
        elif overall_percentile > 60:
            climate_quality = "good"
        elif overall_percentile > 40:
            climate_quality = "moderate"
        elif overall_percentile > 20:
            climate_quality = "challenging"
        else:
            climate_quality = "difficult"
        
        # Change rate assessment
        if acceleration_factor > 1.8:
            change_rate = "very rapid climate change"
        elif acceleration_factor > 1.3:
            change_rate = "rapid climate change"
        elif acceleration_factor > 1.1:
            change_rate = "moderate climate change"
        else:
            change_rate = "slow climate change"
        
        return f"Location has {climate_quality} climate conditions with {change_rate} relative to global patterns."
    
    # Additional helper methods continued...
    
    def _compare_local_vs_global_trend(self, var_data: pd.Series, global_trend: float) -> Dict[str, Any]:
        """Compare local trend to global trend."""
        try:
            if len(var_data) < 10:
                return {"error": "Insufficient data for trend analysis"}
            
            # Simple linear trend
            x = np.arange(len(var_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, var_data)
            
            # Convert to same units as global trend (per decade)
            # This is approximate - would need actual time series for accurate conversion
            local_trend_relative = slope * 10  # Assume daily data approximation
            
            return {
                "local_trend": float(local_trend_relative),
                "global_trend": float(global_trend),
                "trend_ratio": float(local_trend_relative / global_trend) if global_trend != 0 else 1.0,
                "trend_significance": float(p_value),
                "r_squared": float(r_value**2)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _categorize_comparison(self, standard_deviations: float) -> str:
        """Categorize comparison based on standard deviations."""
        abs_std = abs(standard_deviations)
        if abs_std >= 3:
            return "extreme_difference"
        elif abs_std >= 2:
            return "large_difference"
        elif abs_std >= 1:
            return "moderate_difference"
        elif abs_std >= 0.5:
            return "small_difference"
        else:
            return "similar_to_global"
    
    def _categorize_deviation(self, deviation_std: float) -> str:
        """Categorize deviation magnitude."""
        if deviation_std >= 3:
            return "extreme"
        elif deviation_std >= 2:
            return "large"
        elif deviation_std >= 1:
            return "moderate"
        elif deviation_std >= 0.5:
            return "small"
        else:
            return "minimal"
    
    def _estimate_percentile_uncertainty(self, z_score: float, sample_size: int) -> float:
        """Estimate uncertainty in percentile calculation."""
        # Simple approximation based on sample size and z-score magnitude
        base_uncertainty = 100 / np.sqrt(sample_size) if sample_size > 0 else 20
        z_penalty = abs(z_score) * 2  # Higher uncertainty for extreme values
        
        return min(25, base_uncertainty + z_penalty)  # Cap at 25%
    
    def _generate_baseline_comparison_summary(
        self, 
        comparison_analysis: Dict[str, Any], 
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of baseline comparison analysis."""
        try:
            summary = {
                "variables_analyzed": len(comparison_analysis.get("global_comparisons", {})),
                "anomalies_detected": 0,
                "significant_deviations": 0,
                "overall_assessment": "normal"
            }
            
            # Count anomalies and significant deviations
            anomalies = comparison_analysis.get("climate_anomalies", {})
            deviations = comparison_analysis.get("baseline_deviations", {})
            
            for variable, anomaly_data in anomalies.items():
                if isinstance(anomaly_data, dict):
                    if anomaly_data.get("anomaly_detected", False):
                        summary["anomalies_detected"] += 1
                    if anomaly_data.get("requires_attention", False):
                        summary["significant_deviations"] += 1
            
            # Overall assessment
            if summary["significant_deviations"] > 2:
                summary["overall_assessment"] = "concerning"
            elif summary["anomalies_detected"] > 1:
                summary["overall_assessment"] = "notable"
            else:
                summary["overall_assessment"] = "normal"
            
            return summary
            
        except Exception as e:
            return {"error": str(e), "overall_assessment": "unknown"}

    def _estimate_continental_influence(self, location_info: Dict[str, Any]) -> bool:
        """Estimate if location has continental climate influence."""
        # Simple heuristic - could be enhanced with actual distance-to-coast data
        longitude = location_info.get("longitude", 0)
        latitude = location_info.get("latitude", 0)
        
        # Very rough continental estimates
        if abs(latitude) > 50:  # High latitude
            return abs(longitude) > 20  # Away from major coasts
        else:
            return abs(longitude - 90) < 45 or abs(longitude + 90) < 45  # Continental interiors
    
    def _determine_data_period(self, data: pd.DataFrame) -> str:
        """Determine the time period covered by the data."""
        try:
            if isinstance(data.index, pd.DatetimeIndex):
                start_date = data.index.min().strftime("%Y-%m-%d")
                end_date = data.index.max().strftime("%Y-%m-%d")
                return f"{start_date} to {end_date}"
            else:
                return f"{len(data)} data points"
        except:
            return "Unknown period"
    
    def _update_baseline_stats(self, comparison_time: float):
        """Update internal baseline statistics."""
        self.baseline_stats["comparisons_performed"] += 1
        
        # Update average comparison time
        total_comparisons = self.baseline_stats["comparisons_performed"]
        current_avg = self.baseline_stats["avg_comparison_time"]
        
        self.baseline_stats["avg_comparison_time"] = (
            (current_avg * (total_comparisons - 1) + comparison_time) / total_comparisons
        )
    
    def _create_fallback_comparison(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any], 
        error_msg: str
    ) -> Dict[str, Any]:
        """Create fallback comparison when analysis fails."""
        return {
            "fallback": True,
            "error": error_msg,
            "basic_stats": {
                "data_shape": data.shape,
                "numeric_columns": len(data.select_dtypes(include=[np.number]).columns),
                "location": location_info.get("name", "Unknown")
            },
            "fallback_assessment": "Unable to perform detailed baseline comparison - data processing error"
        }


# Convenience functions for common baseline operations
def compare_location_to_global_baseline(
    data: pd.DataFrame,
    location_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function for quick global baseline comparison.
    
    Args:
        data: Climate data DataFrame
        location_info: Location metadata
        
    Returns:
        Global baseline comparison results
    """
    comparator = GlobalBaselineComparator()
    return comparator.compare_to_global_baseline(data, location_info)


def get_climate_change_acceleration(
    data: pd.DataFrame,
    location_info: Dict[str, Any]
) -> float:
    """
    Get climate change acceleration factor for a location.
    
    Args:
        data: Climate data DataFrame
        location_info: Location metadata
        
    Returns:
        Acceleration factor (1.0 = global average)
    """
    comparator = GlobalBaselineComparator()
    acceleration_analysis = comparator.assess_climate_change_acceleration(data, location_info)
    return acceleration_analysis.get("regional_acceleration_factor", 1.0)


if __name__ == "__main__":
    # Example usage and testing
    print("🌍 Global Baseline Comparator - Day 6")
    print("=" * 50)
    
    # Create test data
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    test_data = pd.DataFrame({
        "temperature_2m": np.random.normal(15, 8, 365),  # Slightly above global average
        "precipitation": np.random.exponential(2.5, 365),
        "relative_humidity": np.random.uniform(50, 85, 365),
        "pm2_5": np.random.exponential(18, 365)  # Slightly above global average
    }, index=dates)
    
    # Test location (Berlin)
    test_location = {
        "name": "Berlin",
        "country": "Germany",
        "latitude": 52.52,
        "longitude": 13.41
    }
    
    # Test baseline comparator
    comparator = GlobalBaselineComparator()
    
    # Test global baseline comparison
    baseline_comparison = comparator.compare_to_global_baseline(test_data, test_location)
    
    # Test global percentiles
    global_percentiles = comparator.calculate_global_percentiles(test_data, test_location)
    
    # Test climate change acceleration
    acceleration_analysis = comparator.assess_climate_change_acceleration(test_data, test_location)
    
    # Test comprehensive report
    comprehensive_report = comparator.build_global_baseline_comparisons(test_data, test_location)
    
    print(f"✅ Baseline Comparator tests complete!")
    print(f"🌍 Global percentile: {global_percentiles.get('overall_ranking', {}).get('overall_global_percentile', 50):.1f}%")
    print(f"🔥 Climate acceleration: {acceleration_analysis.get('regional_acceleration_factor', 1.0):.2f}x")
    print(f"📊 Variables compared: {len(baseline_comparison.get('global_comparisons', {}))}")
    print(f"📋 Recommendations: {len(comprehensive_report.get('recommendations', []))}")
    
    # Test convenience functions
    quick_comparison = compare_location_to_global_baseline(test_data, test_location)
    acceleration_factor = get_climate_change_acceleration(test_data, test_location)
    
    print(f"🔧 Quick comparison complete")
    print(f"⚡ Acceleration factor: {acceleration_factor:.2f}x")