# 🏧 ATM Cash Forecasting System

An advanced AI-powered ATM cash demand forecasting system for 10,000+ ATMs across India. Built for operations teams to optimize cash loading and minimize stockout risks.

## 🎯 Features

### Core Capabilities
- **Multivariate Forecasting**: Advanced ML models with 10+ engineered features
- **Multiple Models**: Benchmarking between LightGBM, KATS, and N-BEATS
- **Rigorous Backtesting**: Time-series cross-validation with stockout-cost optimization
- **SHAP Explainability**: Natural language explanations for every prediction
- **Interactive Dashboard**: Streamlit-based UI for operations teams

### Key Components

1. **Data Pipeline** (`src/data_pipeline.py`)
   - Daily CSV ingestion
   - Missing value imputation (Linear/Spline)
   - Feature engineering (lags, rolling stats, holidays, paydays)
   - Indian holiday calendar integration

2. **Models** (`src/models/`)
   - **LightGBM**: Global gradient boosting model
   - Support for per-ATM or global training modes
   - Uncertainty quantification

3. **Backtesting** (`src/backtesting.py`)
   - Time-series cross-validation
   - Stockout-cost metric
   - Comprehensive evaluation (RMSE, MAE, MAPE, R²)

4. **Explainability** (`src/explainability.py`)
   - SHAP value calculation
   - Natural language generation (NLG)
   - Waterfall charts for prediction drivers
   - Business impact messaging

5. **Dashboard** (`app.py`)
   - **View 1: Network Health** - India map with risk indicators
   - **View 2: ATM Detail** - Detailed forecast with SHAP explanations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/bishtNitin/atm-cash-forecasting.git
cd atm-cash-forecasting

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# Train models with sample data
python src/train.py

# Or with custom config
python src/train.py --config config/config.yaml
```

### Running the Dashboard

```bash
# Launch Streamlit dashboard
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Dashboard Views

### View 1: Network Health Overview
- **Map Visualization**: Interactive map of India showing all ATMs
- **Risk Indicators**: Color-coded by stockout risk (Red/Orange/Green)
- **Summary Metrics**: Total ATMs, high/medium/low risk counts
- **High-Risk Table**: Sortable list of ATMs requiring attention

### View 2: ATM Detail Analysis
- **Forecast Summary**: Recommended cash load amount
- **Business Impact**: Stockout probability and risk assessment
- **Drivers of Demand**: SHAP waterfall chart showing key factors
- **Natural Language Explanation**: Plain English breakdown of forecast
- **Historical Trend**: Time-series plot with forecast point
- **Feature Importance**: Top factors driving predictions

## 🧮 Feature Engineering

The system automatically creates:

**Time Features:**
- Day of week, month, year
- Week of year
- Weekend indicator
- Start/end of month

**Holiday Features:**
- Indian public holidays
- Festival seasons (Diwali, Holi)
- Payday indicators (1st, 15th, month-end)

**Lag Features:**
- 1-day, 7-day, 30-day lags
- Rolling mean (7-day, 30-day)
- Rolling standard deviation
- Exponential moving average

**Multivariate Features:**
- ATM group covariances
- Regional patterns

## 📈 Model Performance Metrics

The system evaluates models using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **MAPE** (Mean Absolute Percentage Error)
- **R²** (Coefficient of Determination)
- **Stockout Cost** (Custom business metric)
- **Bias** (Systematic over/under-prediction)

## 🔍 Explainability

### SHAP Analysis
Every prediction includes:
1. **SHAP Values**: Quantified impact of each feature
2. **Waterfall Chart**: Visual breakdown of prediction components
3. **Natural Language**: Plain English explanation

### Example Explanation
```
Forecast for ATM_9921 on 2024-01-01 is High (₹12.00 Lakhs) primarily because:
- it is a Payday (Start/Mid/End of month) is increasing demand by ₹5.00 Lakhs
- it is the Start of Month is increasing demand by ₹3.00 Lakhs
- it is NOT a Weekend is decreasing demand by ₹2.00 Lakhs
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Model hyperparameters
- Backtesting settings
- Explainability options
- Dashboard settings

## 📁 Project Structure

```
atm-cash-forecasting/
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Python dependencies
├── config/
│   └── config.yaml            # Configuration file
├── src/
│   ├── data_pipeline.py       # Data processing
│   ├── train.py               # Training script
│   ├── backtesting.py         # Backtesting framework
│   ├── explainability.py      # SHAP & NLG
│   └── models/
│       └── lightgbm_model.py  # LightGBM implementation
├── data/
│   ├── raw/                   # Raw CSV files
│   └── processed/             # Processed data
└── models/                    # Saved model files
```

## 🔬 Technical Details

### Technology Stack
- **ML Framework**: LightGBM, Darts, KATS
- **Explainability**: SHAP
- **Visualization**: Plotly, Matplotlib, Folium
- **Dashboard**: Streamlit
- **Data Processing**: Pandas, NumPy, SciPy

### Data Requirements
CSV files should contain:
- `atm_id`: ATM identifier
- `date`: Transaction date (YYYY-MM-DD)
- `cash_withdrawn`: Cash amount
- `latitude`, `longitude`: ATM location (optional)

## 🎓 Use Cases

1. **Cash Loading Optimization**: Determine optimal cash amounts for each ATM
2. **Risk Management**: Identify high-risk ATMs before stockouts occur
3. **Resource Planning**: Allocate CIT (Cash-in-Transit) vehicles efficiently
4. **Capacity Planning**: Understand demand patterns across the network

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for ATM Operations Teams**
