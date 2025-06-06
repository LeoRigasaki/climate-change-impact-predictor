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