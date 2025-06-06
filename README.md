# 🌍 Climate Change Impact Predictor

A comprehensive 21-day portfolio project building a global climate change impact prediction system with real-time data pipeline integration.

## 🎯 Project Overview

This project demonstrates end-to-end data science skills by:
- Integrating multiple professional climate APIs
- Building machine learning models for impact prediction
- Creating interactive visualizations and dashboards
- Implementing production-ready code architecture

## 📊 Data Sources

| API Source | Data Type | Purpose |
|------------|-----------|---------|
| Open-Meteo | Air Quality | Real-time pollution monitoring |
| World Bank CCKP | Climate Projections | Future scenario modeling |
| NASA POWER | Meteorological | Historical weather foundation |

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/climate-change-impact-predictor.git
cd climate-change-impact-predictor

# Set up virtual environment
python -m venv climate_env
source climate_env/bin/activate  # On Windows: climate_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app/main.py
```
## 🎯 Current Status: Day 1 COMPLETE ✅

### **API Integration Status:**
- ✅ **Open-Meteo Air Quality API**: Fully integrated and tested
  - 192 hourly records per location
  - Air quality indicators (PM2.5, CO₂, O₃, AQI)
  - Rate limiting implemented
  
- ✅ **NASA POWER Meteorological API**: Fully integrated and tested  
  - Daily meteorological data
  - Temperature, precipitation, wind, humidity
  - Professional error handling

- ✅ **World Bank CCKP API**: Fully integrated and tested
  - IPCC CMIP6 climate projections
  - Temperature extremes to 2100
  - Multiple emission scenarios

### **Locations Tested:**
- ✅ Berlin, Germany
- ✅ Houston, Texas  
- ✅ London, UK
- ✅ Tokyo, Japan

### **Next Phase:**
Ready for Day 2: Advanced data processing and feature engineering