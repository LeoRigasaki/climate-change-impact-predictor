#!/usr/bin/env python3
"""
🌍 Global Data Validator - Day 6 Core Validation System
src/validation/global_validator.py

Advanced data quality validation system that assesses climate data quality
based on regional characteristics, data source reliability, and global context.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class GlobalDataValidator:
    """
    🌍 Global Data Quality Validation System
    
    Provides comprehensive data quality assessment tailored to:
    - Regional climate characteristics and expectations
    - Data source reliability by geographic region
    - Temporal consistency and completeness
    - Cross-source validation and agreement
    - Global baseline conformity checks
    """
    
    def __init__(self):
        """Initialize Global Data Validator."""
        
        # Regional data quality expectations
        self.regional_standards = {
            "tropical": {
                "temperature_range": (-10, 50),
                "humidity_range": (40, 100),
                "expected_completeness": 0.85,
                "seasonal_variation": "low"
            },
            "arid": {
                "temperature_range": (-20, 55),
                "humidity_range": (5, 80),
                "expected_completeness": 0.80,
                "seasonal_variation": "moderate"
            },
            "temperate": {
                "temperature_range": (-40, 45),
                "humidity_range": (20, 100),
                "expected_completeness": 0.90,
                "seasonal_variation": "high"
            },
            "continental": {
                "temperature_range": (-50, 45),
                "humidity_range": (10, 100),
                "expected_completeness": 0.85,
                "seasonal_variation": "very_high"
            },
            "polar": {
                "temperature_range": (-70, 20),
                "humidity_range": (30, 100),
                "expected_completeness": 0.70,
                "seasonal_variation": "extreme"
            }
        }
        
        # Data source reliability by region
        self.source_reliability = {
            "north_america": {"air_quality": 0.95, "meteorological": 0.98, "climate_projections": 0.92},
            "europe": {"air_quality": 0.93, "meteorological": 0.96, "climate_projections": 0.94},
            "asia": {"air_quality": 0.85, "meteorological": 0.90, "climate_projections": 0.88},
            "africa": {"air_quality": 0.75, "meteorological": 0.82, "climate_projections": 0.80},
            "south_america": {"air_quality": 0.80, "meteorological": 0.85, "climate_projections": 0.83},
            "oceania": {"air_quality": 0.88, "meteorological": 0.92, "climate_projections": 0.85},
            "antarctica": {"air_quality": 0.60, "meteorological": 0.70, "climate_projections": 0.65}
        }
        
        self.validation_metrics = {}
        logger.info("🌍 Global Data Validator initialized")
    
    def validate_regional_data_quality(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any],
        data_sources: List[str]
    ) -> Dict[str, Any]:
        """
        Comprehensive regional data quality validation.
        
        Args:
            data: Climate data DataFrame
            location_info: Location metadata (lat, lon, climate_zone, etc.)
            data_sources: List of data sources used
            
        Returns:
            Comprehensive validation report
        """
        validation_start = datetime.now()
        
        try:
            # Determine regional context
            climate_zone = self._determine_climate_zone(location_info)
            geographic_region = self._determine_geographic_region(location_info)
            
            # Initialize validation report
            validation_report = {
                "overall_quality_score": 0.0,
                "regional_context": {
                    "climate_zone": climate_zone,
                    "geographic_region": geographic_region,
                    "expected_standards": self.regional_standards.get(climate_zone, {})
                },
                "validation_components": {},
                "recommendations": [],
                "critical_issues": [],
                "metadata": {
                    "validation_timestamp": validation_start.isoformat(),
                    "data_shape": data.shape,
                    "data_sources": data_sources
                }
            }
            
            # Component 1: Range Validation
            range_validation = self._validate_value_ranges(data, climate_zone)
            validation_report["validation_components"]["range_validation"] = range_validation
            
            # Component 2: Completeness Assessment
            completeness_assessment = self._assess_data_completeness(data, climate_zone)
            validation_report["validation_components"]["completeness"] = completeness_assessment
            
            # Component 3: Temporal Consistency
            temporal_consistency = self._check_temporal_consistency(data)
            validation_report["validation_components"]["temporal_consistency"] = temporal_consistency
            
            # Component 4: Source Reliability
            source_reliability = self._assess_source_reliability(data_sources, geographic_region)
            validation_report["validation_components"]["source_reliability"] = source_reliability
            
            # Component 5: Cross-Variable Consistency
            cross_variable_consistency = self._check_cross_variable_consistency(data)
            validation_report["validation_components"]["cross_variable_consistency"] = cross_variable_consistency
            
            # Component 6: Regional Anomaly Detection
            regional_anomalies = self._detect_regional_anomalies(data, location_info)
            validation_report["validation_components"]["regional_anomalies"] = regional_anomalies
            
            # Calculate overall quality score
            component_scores = [
                range_validation["score"],
                completeness_assessment["score"], 
                temporal_consistency["score"],
                source_reliability["score"],
                cross_variable_consistency["score"],
                regional_anomalies["score"]
            ]
            
            validation_report["overall_quality_score"] = np.mean(component_scores)
            
            # Generate recommendations and identify issues
            validation_report["recommendations"] = self._generate_recommendations(validation_report)
            validation_report["critical_issues"] = self._identify_critical_issues(validation_report)
            
            # Update metrics
            processing_time = (datetime.now() - validation_start).total_seconds()
            validation_report["metadata"]["processing_time"] = processing_time
            
            self._update_validation_metrics(validation_report)
            
            logger.info(f"✅ Regional validation complete: {validation_report['overall_quality_score']:.1f}/100")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"❌ Regional validation failed: {e}")
            return self._create_fallback_validation_report(data, str(e))
    
    def assess_data_completeness_by_region(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess data completeness with regional expectations.
        
        Args:
            data: Climate data DataFrame
            location_info: Location metadata
            
        Returns:
            Regional completeness assessment
        """
        try:
            climate_zone = self._determine_climate_zone(location_info)
            expected_completeness = self.regional_standards.get(climate_zone, {}).get("expected_completeness", 0.80)
            
            # Calculate actual completeness
            total_values = data.size
            missing_values = data.isnull().sum().sum()
            actual_completeness = 1 - (missing_values / total_values)
            
            # Feature-level completeness
            feature_completeness = {}
            for column in data.columns:
                feature_missing = data[column].isnull().sum()
                feature_total = len(data[column])
                feature_completeness[column] = 1 - (feature_missing / feature_total)
            
            # Regional adjustment
            completeness_score = min(100, (actual_completeness / expected_completeness) * 100)
            
            return {
                "actual_completeness": actual_completeness,
                "expected_completeness": expected_completeness,
                "completeness_score": completeness_score,
                "meets_regional_standard": actual_completeness >= expected_completeness,
                "feature_completeness": feature_completeness,
                "missing_data_summary": {
                    "total_missing": int(missing_values),
                    "missing_percentage": (missing_values / total_values) * 100,
                    "most_incomplete_features": sorted(
                        [(col, 1-comp) for col, comp in feature_completeness.items()],
                        key=lambda x: x[1], reverse=True
                    )[:5]
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Completeness assessment failed: {e}")
            return {"actual_completeness": 0.0, "error": str(e)}
    
    def identify_regional_data_gaps(
        self, 
        data: pd.DataFrame, 
        location_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Identify and categorize data gaps based on regional characteristics.
        
        Args:
            data: Climate data DataFrame  
            location_info: Location metadata
            
        Returns:
            Regional data gap analysis
        """
        try:
            climate_zone = self._determine_climate_zone(location_info)
            geographic_region = self._determine_geographic_region(location_info)
            
            # Expected features by climate zone
            expected_features = self._get_expected_features_by_climate_zone(climate_zone)
            
            # Identify missing features
            present_features = set(data.columns)
            missing_features = expected_features - present_features
            
            # Identify sparse features (< 50% complete)
            sparse_features = []
            for column in data.columns:
                completeness = 1 - (data[column].isnull().sum() / len(data[column]))
                if completeness < 0.5:
                    sparse_features.append((column, completeness))
            
            # Identify temporal gaps
            temporal_gaps = self._identify_temporal_gaps(data)
            
            # Regional context for gaps
            gap_severity = self._assess_gap_severity(missing_features, sparse_features, geographic_region)
            
            return {
                "regional_context": {
                    "climate_zone": climate_zone,
                    "geographic_region": geographic_region
                },
                "missing_features": {
                    "count": len(missing_features),
                    "features": list(missing_features),
                    "expected_features": list(expected_features),
                    "coverage_percentage": (len(present_features & expected_features) / len(expected_features)) * 100
                },
                "sparse_features": {
                    "count": len(sparse_features),
                    "features": sparse_features
                },
                "temporal_gaps": temporal_gaps,
                "gap_severity": gap_severity,
                "impact_assessment": self._assess_gap_impact(missing_features, sparse_features, climate_zone),
                "recommendations": self._get_gap_mitigation_recommendations(missing_features, sparse_features)
            }
            
        except Exception as e:
            logger.error(f"❌ Gap identification failed: {e}")
            return {"error": str(e)}
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation system summary."""
        return {
            "validation_metrics": self.validation_metrics,
            "regional_standards": self.regional_standards,
            "source_reliability": self.source_reliability,
            "capabilities": {
                "regional_validation": True,
                "completeness_assessment": True,
                "gap_identification": True,
                "temporal_consistency": True,
                "cross_variable_validation": True,
                "anomaly_detection": True
            }
        }
    
    # ========================================
    # PRIVATE HELPER METHODS
    # ========================================
    
    def _determine_climate_zone(self, location_info: Dict[str, Any]) -> str:
        """Determine climate zone from location info."""
        # Check if climate zone is provided
        if "climate_zone" in location_info:
            return location_info["climate_zone"]
        
        # Simple latitude-based classification
        lat = abs(location_info.get("latitude", 0))
        
        if lat >= 66.5:
            return "polar"
        elif lat >= 45:
            return "continental"
        elif lat >= 23.5:
            return "temperate"
        else:
            return "tropical"
    
    def _determine_geographic_region(self, location_info: Dict[str, Any]) -> str:
        """Determine geographic region from coordinates."""
        lat = location_info.get("latitude", 0)
        lon = location_info.get("longitude", 0)
        
        # Simple continent classification based on coordinates
        if -180 <= lon <= -30:
            if lat >= 15:
                return "north_america"
            else:
                return "south_america"
        elif -30 < lon <= 60:
            if lat >= 35:
                return "europe"
            else:
                return "africa"
        elif 60 < lon <= 180:
            if lat >= -10:
                return "asia"
            else:
                return "oceania"
        else:
            return "unknown"
    
    def _validate_value_ranges(self, data: pd.DataFrame, climate_zone: str) -> Dict[str, Any]:
        """Validate data values are within expected ranges for climate zone."""
        try:
            standards = self.regional_standards.get(climate_zone, {})
            violations = []
            score = 100.0
            
            # Temperature validation
            if "temperature_2m" in data.columns:
                temp_range = standards.get("temperature_range", (-50, 60))
                temp_violations = ((data["temperature_2m"] < temp_range[0]) | 
                                 (data["temperature_2m"] > temp_range[1])).sum()
                if temp_violations > 0:
                    violations.append(f"Temperature violations: {temp_violations}")
                    score -= min(20, temp_violations / len(data) * 100)
            
            # Humidity validation
            if "relative_humidity" in data.columns:
                humidity_range = standards.get("humidity_range", (0, 100))
                humidity_violations = ((data["relative_humidity"] < humidity_range[0]) | 
                                     (data["relative_humidity"] > humidity_range[1])).sum()
                if humidity_violations > 0:
                    violations.append(f"Humidity violations: {humidity_violations}")
                    score -= min(15, humidity_violations / len(data) * 100)
            
            return {
                "score": max(0, score),
                "violations": violations,
                "standards_applied": standards
            }
            
        except Exception as e:
            return {"score": 50.0, "error": str(e)}
    
    def _assess_data_completeness(self, data: pd.DataFrame, climate_zone: str) -> Dict[str, Any]:
        """Assess data completeness against regional expectations."""
        try:
            expected_completeness = self.regional_standards.get(climate_zone, {}).get("expected_completeness", 0.80)
            
            total_values = data.size
            missing_values = data.isnull().sum().sum()
            actual_completeness = 1 - (missing_values / total_values)
            
            score = min(100, (actual_completeness / expected_completeness) * 100)
            
            return {
                "score": score,
                "actual_completeness": actual_completeness,
                "expected_completeness": expected_completeness,
                "missing_count": int(missing_values)
            }
            
        except Exception as e:
            return {"score": 50.0, "error": str(e)}
    
    def _check_temporal_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check temporal consistency and patterns."""
        try:
            score = 100.0
            issues = []
            
            # Check for duplicate timestamps
            if data.index.duplicated().any():
                duplicate_count = data.index.duplicated().sum()
                issues.append(f"Duplicate timestamps: {duplicate_count}")
                score -= min(20, duplicate_count / len(data) * 100)
            
            # Check for temporal gaps
            if isinstance(data.index, pd.DatetimeIndex):
                expected_freq = pd.infer_freq(data.index[:10])  # Infer from first 10 points
                if expected_freq:
                    complete_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq=expected_freq)
                    missing_timestamps = len(complete_index) - len(data.index)
                    if missing_timestamps > 0:
                        issues.append(f"Missing timestamps: {missing_timestamps}")
                        score -= min(30, missing_timestamps / len(complete_index) * 100)
            
            return {
                "score": max(0, score),
                "issues": issues
            }
            
        except Exception as e:
            return {"score": 70.0, "error": str(e)}
    
    def _assess_source_reliability(self, data_sources: List[str], geographic_region: str) -> Dict[str, Any]:
        """Assess reliability of data sources for the geographic region."""
        try:
            region_reliability = self.source_reliability.get(geographic_region, {})
            
            if not region_reliability:
                return {"score": 70.0, "warning": f"No reliability data for region: {geographic_region}"}
            
            source_scores = []
            for source in data_sources:
                # Map source names to reliability categories
                if "air_quality" in source.lower() or "meteo" in source.lower():
                    if "air_quality" in source.lower():
                        reliability = region_reliability.get("air_quality", 0.80)
                    else:
                        reliability = region_reliability.get("meteorological", 0.85)
                elif "climate" in source.lower() or "projection" in source.lower():
                    reliability = region_reliability.get("climate_projections", 0.80)
                else:
                    reliability = 0.75  # Default for unknown sources
                
                source_scores.append(reliability)
            
            overall_reliability = np.mean(source_scores) if source_scores else 0.75
            score = overall_reliability * 100
            
            return {
                "score": score,
                "overall_reliability": overall_reliability,
                "source_reliabilities": dict(zip(data_sources, source_scores)),
                "geographic_region": geographic_region
            }
            
        except Exception as e:
            return {"score": 70.0, "error": str(e)}
    
    def _check_cross_variable_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check consistency between related variables."""
        try:
            score = 100.0
            issues = []
            
            # Temperature and humidity relationship
            if "temperature_2m" in data.columns and "relative_humidity" in data.columns:
                # High temperatures should generally correlate with lower humidity (in many climates)
                correlation = data["temperature_2m"].corr(data["relative_humidity"])
                if abs(correlation) < 0.1:  # Very weak correlation might indicate issues
                    issues.append("Weak temperature-humidity relationship")
                    score -= 10
            
            # PM2.5 and PM10 relationship
            if "pm2_5" in data.columns and "pm10" in data.columns:
                # PM2.5 should generally be less than or equal to PM10
                violations = (data["pm2_5"] > data["pm10"]).sum()
                if violations > len(data) * 0.05:  # More than 5% violations
                    issues.append(f"PM2.5 > PM10 violations: {violations}")
                    score -= min(20, violations / len(data) * 100)
            
            return {
                "score": max(0, score),
                "issues": issues
            }
            
        except Exception as e:
            return {"score": 80.0, "error": str(e)}
    
    def _detect_regional_anomalies(self, data: pd.DataFrame, location_info: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies specific to regional expectations."""
        try:
            score = 100.0
            anomalies = []
            
            climate_zone = self._determine_climate_zone(location_info)
            
            # Climate zone specific anomaly detection
            if climate_zone == "tropical":
                # Tropical regions should have relatively stable temperatures
                if "temperature_2m" in data.columns:
                    temp_std = data["temperature_2m"].std()
                    if temp_std > 10:  # High variation for tropical
                        anomalies.append(f"High temperature variation for tropical zone: {temp_std:.1f}°C std")
                        score -= 15
            
            elif climate_zone == "polar":
                # Polar regions should have low temperatures
                if "temperature_2m" in data.columns:
                    avg_temp = data["temperature_2m"].mean()
                    if avg_temp > 5:  # Too warm for polar
                        anomalies.append(f"Unexpectedly high average temperature for polar zone: {avg_temp:.1f}°C")
                        score -= 20
            
            return {
                "score": max(0, score),
                "anomalies": anomalies,
                "climate_zone": climate_zone
            }
            
        except Exception as e:
            return {"score": 80.0, "error": str(e)}
    
    def _generate_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        overall_score = validation_report["overall_quality_score"]
        
        if overall_score < 60:
            recommendations.append("⚠️ Overall data quality is concerning - consider additional validation")
        
        # Component-specific recommendations
        components = validation_report["validation_components"]
        
        if components.get("completeness", {}).get("score", 100) < 70:
            recommendations.append("📊 Improve data completeness through additional data sources")
        
        if components.get("temporal_consistency", {}).get("score", 100) < 80:
            recommendations.append("🕐 Address temporal consistency issues in data collection")
        
        if components.get("source_reliability", {}).get("score", 100) < 75:
            recommendations.append("🔍 Consider more reliable data sources for this region")
        
        return recommendations
    
    def _identify_critical_issues(self, validation_report: Dict[str, Any]) -> List[str]:
        """Identify critical issues requiring immediate attention."""
        critical_issues = []
        
        overall_score = validation_report["overall_quality_score"]
        
        if overall_score < 40:
            critical_issues.append("🚨 Overall data quality is critically low")
        
        # Check each component for critical thresholds
        components = validation_report["validation_components"]
        
        for component_name, component_data in components.items():
            if isinstance(component_data, dict) and component_data.get("score", 100) < 30:
                critical_issues.append(f"🚨 Critical issue in {component_name}")
        
        return critical_issues
    
    def _create_fallback_validation_report(self, data: pd.DataFrame, error_msg: str) -> Dict[str, Any]:
        """Create fallback validation report when validation fails."""
        return {
            "overall_quality_score": 50.0,
            "error": error_msg,
            "fallback": True,
            "basic_stats": {
                "shape": data.shape,
                "missing_values": data.isnull().sum().sum(),
                "columns": list(data.columns)
            }
        }
    
    def _update_validation_metrics(self, validation_report: Dict[str, Any]):
        """Update internal validation metrics."""
        if "overall_scores" not in self.validation_metrics:
            self.validation_metrics["overall_scores"] = []
        
        self.validation_metrics["overall_scores"].append(validation_report["overall_quality_score"])
        self.validation_metrics["last_validation"] = datetime.now().isoformat()
    
    def _get_expected_features_by_climate_zone(self, climate_zone: str) -> set:
        """Get expected features for a climate zone."""
        base_features = {"temperature_2m", "relative_humidity", "precipitation"}
        
        if climate_zone in ["tropical", "arid"]:
            base_features.update({"pm2_5", "pm10", "ozone"})
        elif climate_zone in ["temperate", "continental"]:
            base_features.update({"wind_speed_2m", "pressure"})
        
        return base_features
    
    def _identify_temporal_gaps(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Identify temporal gaps in the data."""
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                return {"error": "Non-temporal index"}
            
            # Calculate time differences
            time_diffs = data.index.to_series().diff()
            median_diff = time_diffs.median()
            
            # Find large gaps (> 2x median difference)
            large_gaps = time_diffs[time_diffs > median_diff * 2]
            
            return {
                "total_gaps": len(large_gaps),
                "largest_gap": str(large_gaps.max()) if len(large_gaps) > 0 else "None",
                "median_frequency": str(median_diff)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _assess_gap_severity(self, missing_features: set, sparse_features: List[Tuple], geographic_region: str) -> str:
        """Assess severity of data gaps."""
        if len(missing_features) > 5 or len(sparse_features) > 3:
            return "High"
        elif len(missing_features) > 2 or len(sparse_features) > 1:
            return "Medium"
        else:
            return "Low"
    
    def _assess_gap_impact(self, missing_features: set, sparse_features: List[Tuple], climate_zone: str) -> Dict[str, Any]:
        """Assess impact of data gaps on analysis."""
        impact = {"analysis_impact": "Low", "prediction_impact": "Low"}
        
        critical_features = {"temperature_2m", "precipitation", "relative_humidity"}
        
        if any(feature in missing_features for feature in critical_features):
            impact["analysis_impact"] = "High"
            impact["prediction_impact"] = "High"
        
        return impact
    
    def _get_gap_mitigation_recommendations(self, missing_features: set, sparse_features: List[Tuple]) -> List[str]:
        """Get recommendations for mitigating data gaps."""
        recommendations = []
        
        if missing_features:
            recommendations.append("🔄 Seek alternative data sources for missing features")
        
        if sparse_features:
            recommendations.append("📈 Implement data interpolation for sparse features")
        
        recommendations.append("🔧 Consider synthetic data generation for critical missing data")
        
        return recommendations


if __name__ == "__main__":
    # Example usage and testing
    print("🌍 Global Data Validator - Day 6")
    print("=" * 50)
    
    # Create test data
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    test_data = pd.DataFrame({
        "temperature_2m": np.random.normal(20, 10, 100),
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
    
    # Test validator
    validator = GlobalDataValidator()
    validation_report = validator.validate_regional_data_quality(
        test_data, 
        test_location, 
        ["air_quality", "meteorological"]
    )
    
    print(f"✅ Validation complete!")
    print(f"📊 Overall Quality Score: {validation_report['overall_quality_score']:.1f}/100")
    print(f"🌍 Climate Zone: {validation_report['regional_context']['climate_zone']}")
    print(f"📋 Recommendations: {len(validation_report['recommendations'])}")