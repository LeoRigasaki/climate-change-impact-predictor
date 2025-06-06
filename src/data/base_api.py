"""
Base API client for all climate data sources.
Provides common functionality for API requests, error handling, and rate limiting.
"""

import time
import logging
import requests
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class BaseAPIClient(ABC):
    """Abstract base class for all climate API clients."""
    
    def __init__(self, base_url: str, timeout: int = 30, rate_limit: Optional[int] = None):
        self.base_url = base_url
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.session = requests.Session()
        
        # Set common headers
        self.session.headers.update({
            'User-Agent': 'Climate-Change-Impact-Predictor/1.0',
            'Accept': 'application/json'
        })
    
    def _rate_limit_wait(self):
        """Implement rate limiting if specified."""
        if self.rate_limit:
            min_interval = 86400 / self.rate_limit  # seconds between requests
            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with error handling and retries."""
        self._rate_limit_wait()
        
        try:
            logger.info(f"Making request to: {url}")
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            
            # Log response info
            logger.info(f"Request successful: {response.status_code}")
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
    
    @abstractmethod
    def fetch_data(self, **kwargs) -> Dict[str, Any]:
        """Fetch data from the API. Must be implemented by subclasses."""
        pass
    
    def validate_date_range(self, start_date: str, end_date: str) -> tuple:
        """Validate and parse date range."""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            if start > end:
                raise ValueError("Start date must be before end date")
            
            return start, end
            
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            raise
    
    def validate_coordinates(self, latitude: float, longitude: float):
        """Validate geographic coordinates."""
        if not (-90 <= latitude <= 90):
            raise ValueError(f"Invalid latitude: {latitude}. Must be between -90 and 90")
        
        if not (-180 <= longitude <= 180):
            raise ValueError(f"Invalid longitude: {longitude}. Must be between -180 and 180")