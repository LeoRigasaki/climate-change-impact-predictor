from datetime import datetime, timedelta
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
LOGS_DIR = PROJECT_ROOT / "logs"

# API Configuration
API_CONFIGS = {
    "open_meteo": {
        "base_url": "https://air-quality-api.open-meteo.com/v1/air-quality",
        "rate_limit": 10000,  # requests per day
        "timeout": 30
    },
    "world_bank": {
        "base_url": "https://cckpapi.worldbank.org/cckp/v1",
        "rate_limit": None,  # No explicit limit
        "timeout": 60
    },
    "nasa_power": {
        "base_url": "https://power.larc.nasa.gov/api/temporal/daily/point",
        "rate_limit": None,  # No explicit limit
        "timeout": 60
    }
}

# Default locations for testing
DEFAULT_LOCATIONS = {
    "berlin": {"latitude": 52.52, "longitude": 13.41},
    "houston": {"latitude": 29.76, "longitude": -95.39},
    "london": {"latitude": 51.51, "longitude": -0.13},
    "tokyo": {"latitude": 35.68, "longitude": 139.69}
}

# Data parameters
AIR_QUALITY_PARAMS = [
    "pm10", "pm2_5", "carbon_monoxide", "carbon_dioxide", 
    "nitrogen_dioxide", "sulphur_dioxide", "ozone", "dust", 
    "uv_index", "ammonia", "methane"
]

NASA_POWER_PARAMS = ["T2M", "PRECTOTCORR", "WS2M", "RH2M"]

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": LOGS_DIR / "climate_predictor.log",
            "mode": "a",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}

def get_full_historical_range():
    """Get the full historical date range (2015 to yesterday)."""
    start_date = "2015-01-01"
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    return start_date, end_date

def get_recent_range(days=365):
    """Get recent date range (default: last year)."""
    end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return start_date, end_date