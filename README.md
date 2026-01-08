# Climate Change Impact Predictor

A comprehensive portfolio project building a global climate change impact prediction system with advanced machine learning and real-time data pipeline integration.

## Project Overview

This project demonstrates end-to-end data science and machine learning engineering skills by:
- **Advanced ML Architecture**: LSTM time series forecasting + Multi-output deep learning
- **Global Scale**: Trained on 150+ world capitals with real-time predictions for any location
- **Production Ready**: Professional code architecture with comprehensive error handling
- **Multi-Model Ensemble**: 3 specialized ML models working together

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/climate-change-impact-predictor.git
cd climate-change-impact-predictor

# Set up virtual environment
python -m venv climate_env
source climate_env/bin/activate  # On Windows: climate_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Web Interface

```bash
# Activate the virtual environment
source climate_env/bin/activate

# Start the ML API server (required for predictions)
PYTHONPATH=$(pwd) python app/api_server.py &

# Start the Streamlit web interface
PYTHONPATH=$(pwd) streamlit run app/location_picker.py
```

The web interface will be available at: **http://localhost:8501**
The ML API will be running at: **http://localhost:8000**

### Running CLI Predictions

```bash
# Single city prediction
PYTHONPATH=$(pwd) python -m tools.day10_advanced_demo Lahore

# Interactive demo
PYTHONPATH=$(pwd) python -m tools.day10_advanced_demo

# Multiple cities
PYTHONPATH=$(pwd) python -m tools.day10_advanced_demo "New York"
PYTHONPATH=$(pwd) python -m tools.day10_advanced_demo "Tokyo"
```

### Python API Usage

```python
from src.models.enhanced_climate_predictor import EnhancedGlobalClimatePredictor

# Initialize enhanced predictor
predictor = EnhancedGlobalClimatePredictor()

# Basic prediction
result = predictor.predict_climate("London")

# LSTM long-term forecasting
forecast = predictor.predict_long_term("Tokyo", days=30)

# Multi-output comprehensive prediction
comprehensive = predictor.predict_comprehensive("Berlin")

# Ultimate prediction (all models combined)
advanced = predictor.predict_climate_advanced("Paris")
```

## Data Sources & Coverage

| API Source | Data Type | Coverage | Purpose |
|------------|-----------|----------|---------|
| **Open-Meteo** | Air Quality & Weather | Global hourly | Real-time pollution monitoring & forecasts |
| **NASA POWER** | Meteorological | Global daily | Historical weather foundation |
| **World Bank CCKP** | Climate Projections | Country-level | IPCC CMIP6 future scenarios |

**Dataset Scale**: 150+ world capitals, 60+ climate features, multi-source validation

## Machine Learning Models

### 1. Base Climate Model
- **Architecture**: Feed-forward neural network
- **Training**: 144 world capitals
- **Capability**: Basic climate prediction for any city

### 2. LSTM Time Series Forecaster
- **Architecture**: Bidirectional LSTM with attention mechanisms
- **Training**: Enhanced dataset with time series sequences
- **Capability**: 7-30 day weather forecasting

### 3. Multi-Output Deep Learning
- **Architecture**: Multi-head dense networks with shared feature extraction
- **Training**: Comprehensive multi-target learning
- **Capability**: Simultaneous prediction of 5+ climate variables
- **Outputs**: Temperature, precipitation, UV index, air quality, wind

## System Architecture

```
Climate Change Impact Predictor
├── Global Location Service (any coordinates)
├── Multi-Source Data Pipeline (3 professional APIs)
├── Advanced ML Ensemble (3 specialized models)
├── Enhanced Prediction Engine (production-ready)
├── REST API Server (FastAPI)
└── Interactive Web Dashboard (Streamlit)
```

## Project Structure

```
├── app/                       # Web application
│   ├── api_server.py          # FastAPI ML prediction server
│   └── location_picker.py     # Streamlit web interface
├── data/                      # Multi-source climate datasets
│   ├── raw/                   # Raw API responses
│   ├── processed/             # Processed and integrated data
│   ├── capitals/              # World capitals training data
│   └── day10/                 # Advanced ML training datasets
├── models/                    # Trained ML models and preprocessors
│   ├── global_climate_model.keras
│   ├── advanced_lstm_forecaster.keras
│   └── production_multi_output_predictor.keras
├── notebooks/                 # Training and analysis notebooks
├── src/                       # Core system implementation
│   ├── api/                   # Multi-API integration layer
│   ├── core/                  # Data management and processing
│   ├── features/              # Universal feature engineering
│   ├── models/                # ML model implementations
│   └── validation/            # Quality assurance and testing
└── tools/                     # Professional tooling and demos
```

## Development Status

### Phase 1: Foundation & Data Pipeline (Complete)
- Multi-API integration (Open-Meteo, NASA POWER, World Bank CCKP)
- Advanced data processing pipeline with 60+ climate features
- Universal location service supporting any global coordinates
- Adaptive data collection with smart source selection

### Phase 2: Advanced Machine Learning (Complete)
- Global neural network trained on 144 world capitals
- LSTM Time Series Forecasting for 7-30 day predictions
- Multi-Output Deep Learning for simultaneous predictions

### Phase 3: Web Application (Complete)
- REST API server for model serving
- Interactive Streamlit dashboard
- Real-time climate predictions

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check API health status |
| `/predict/basic` | POST | Basic climate prediction |
| `/predict/forecast` | POST | LSTM weather forecast |
| `/location/search` | GET | Search for locations |

## Requirements

- Python 3.10+
- TensorFlow 2.x
- Streamlit
- FastAPI
- See `requirements.txt` for full dependencies

## License

MIT License - See LICENSE file for details.

---

**This project showcases advanced machine learning engineering skills with real-world climate data and production-ready architecture.**
