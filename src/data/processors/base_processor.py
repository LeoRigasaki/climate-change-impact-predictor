"""
Base data processor classes for climate data transformation.
Provides common functionality for all data processing operations.
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

from config.settings import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

class BaseDataProcessor(ABC):
    """Abstract base class for all data processors."""
    
    def __init__(self, name: str):
        self.name = name
        self.processing_metadata = {
            "processor": name,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        # Ensure processed data directory exists
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Process raw data into standardized DataFrame format."""
        pass
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """Validate input data structure and content."""
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")
        
        if '_metadata' not in data:
            logger.warning("Input data missing metadata")
        
        return True
    
    def add_processing_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add processing metadata to DataFrame."""
        df.attrs.update(self.processing_metadata)
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> Path:
        """Save processed DataFrame to file with metadata."""
        filepath = PROCESSED_DATA_DIR / f"{filename}.parquet"
        
        # Add processing timestamp
        df_with_meta = df.copy()
        df_with_meta.attrs.update(self.processing_metadata)
        
        # Save as Parquet for efficiency
        df_with_meta.to_parquet(filepath, index=False)
        
        # Save metadata separately
        meta_filepath = PROCESSED_DATA_DIR / f"{filename}_metadata.json"
        with open(meta_filepath, 'w') as f:
            json.dump(df_with_meta.attrs, f, indent=2, default=str)
        
        logger.info(f"Processed data saved to {filepath}")
        return filepath
    
    def get_processing_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for processed data."""
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
            "date_range": {
                "start": df.index.min() if hasattr(df.index, 'min') else None,
                "end": df.index.max() if hasattr(df.index, 'max') else None
            } if pd.api.types.is_datetime64_any_dtype(df.index) else None
        }

class DataQualityChecker:
    """Comprehensive data quality assessment and monitoring."""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def assess_quality(self, df: pd.DataFrame, source: str) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        quality_report = {
            "source": source,
            "assessment_time": datetime.now().isoformat(),
            "shape": df.shape,
            "completeness": self._assess_completeness(df),
            "consistency": self._assess_consistency(df),
            "validity": self._assess_validity(df),
            "duplicates": self._assess_duplicates(df),
            "outliers": self._assess_outliers(df),
            "overall_score": 0.0
        }
        
        # Calculate overall quality score
        quality_report["overall_score"] = self._calculate_quality_score(quality_report)
        
        logger.info(f"Data quality assessment complete for {source}")
        logger.info(f"Overall quality score: {quality_report['overall_score']:.2f}/100")
        
        return quality_report
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data completeness (missing values)."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        
        return {
            "missing_percentage": (missing_cells / total_cells) * 100,
            "missing_by_column": (df.isnull().sum() / len(df) * 100).to_dict(),
            "complete_rows": len(df.dropna()),
            "complete_rows_percentage": (len(df.dropna()) / len(df)) * 100
        }
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data consistency (format, type consistency)."""
        consistency_issues = []
        
        # Check for mixed data types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                except:
                    unique_types = set(type(x).__name__ for x in df[col].dropna())
                    if len(unique_types) > 1:
                        consistency_issues.append(f"Mixed types in {col}: {unique_types}")
        
        return {
            "issues": consistency_issues,
            "consistent_columns": len(df.columns) - len(consistency_issues)
        }
    
    def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data validity (reasonable ranges, formats)."""
        validity_issues = []
        
        # Check for common validity issues
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].min() == df[col].max():
                validity_issues.append(f"No variance in {col}")
            
            # Check for extreme values (more than 3 standard deviations)
            if len(df[col].dropna()) > 0:
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    extreme_count = len(df[abs(df[col] - mean_val) > 3 * std_val])
                    if extreme_count > 0:
                        validity_issues.append(f"{extreme_count} extreme values in {col}")
        
        return {
            "issues": validity_issues,
            "valid_columns": len(df.select_dtypes(include=[np.number]).columns) - len(validity_issues)
        }
    
    def _assess_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess duplicate records."""
        duplicate_rows = df.duplicated().sum()
        
        return {
            "duplicate_rows": duplicate_rows,
            "duplicate_percentage": (duplicate_rows / len(df)) * 100,
            "unique_rows": len(df) - duplicate_rows
        }
    
    def _assess_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess outliers using IQR method."""
        outlier_counts = {}
        
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
        
        return {
            "outliers_by_column": outlier_counts,
            "total_outliers": sum(outlier_counts.values())
        }
    
    def _calculate_quality_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)."""
        completeness_score = max(0, 100 - report["completeness"]["missing_percentage"])
        consistency_score = (report["consistency"]["consistent_columns"] / 
                           max(1, len(report["consistency"]["issues"]) + 
                               report["consistency"]["consistent_columns"])) * 100
        validity_score = (report["validity"]["valid_columns"] / 
                         max(1, len(report["validity"]["issues"]) + 
                             report["validity"]["valid_columns"])) * 100
        duplicate_score = max(0, 100 - report["duplicates"]["duplicate_percentage"])
        
        # Weighted average
        overall_score = (
            completeness_score * 0.3 +
            consistency_score * 0.25 +
            validity_score * 0.25 +
            duplicate_score * 0.2
        )
        
        return round(overall_score, 2)