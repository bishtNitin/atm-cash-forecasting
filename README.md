# ATM Cash Forecasting System 🏧

A comprehensive multivariate ATM cash demand forecasting system for 10,000+ ATMs across India, built with state-of-the-art time series models and featuring an interactive Streamlit UI with SHAP explanations.

## 🎯 Features

### Data Pipeline
- **Data Ingestion**: Daily CSV ingestion with automated processing
- **Data Cleaning**: Linear and Spline interpolation for missing values
- **Quality Assurance**: Outlier detection and handling

### Feature Engineering
- **Multivariate Features**: Covariance analysis across ATMs
- **Lag Features**: 1, 7, 14, and 30-day lags
- **Rolling Statistics**: Mean, std, min, max over multiple windows
- **Indian Holiday Calendar**: Integration with Indian national and regional holidays
- **Temporal Features**: Day of week, month, quarter, week of year

### Models Implemented
1. **KATS Models** (Facebook's time series library)
   - Prophet: Trend and seasonality decomposition
   - Theta: Classical forecasting method
   - Ensemble: Combination of multiple models

2. **N-BEATS** (Neural Basis Expansion Analysis)
   - Deep learning approach via Darts library
   - Interpretable and accurate forecasts

3. **Global LightGBM**
   - Gradient boosting for multi-ATM forecasting
   - Fast training and high accuracy
   - Feature importance analysis

### Evaluation Framework
- **Rigorous Backtesting**: Time series cross-validation
- **Custom Cost Function**: Stockout cost + Holding cost optimization
- **Multiple Metrics**: RMSE, MAE, MAPE, SMAPE, MASE
- **Best Model Selection**: Based on business-relevant stockout costs

### Interactive UI
- **Streamlit Dashboard**: User-friendly interface for operations teams
- **SHAP Explanations**: Understand model predictions
- **Interactive Visualizations**: Plotly-based charts
- **Real-time Forecasting**: Generate forecasts on-demand

## 📁 Project Structure

```
atm-cash-forecasting/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── train.py                    # Main training script
├── app.py                      # Streamlit application
├── data/
│   ├── raw/                   # Raw CSV files
│   └── processed/             # Processed data and results
├── src/
│   ├── data/
│   │   └── ingestion.py       # Data loading and preprocessing
│   ├── features/
│   │   └── engineering.py     # Feature engineering pipeline
│   ├── models/
│   │   ├── base_model.py      # Base model interface
│   │   ├── kats_models.py     # KATS models (Prophet, Theta, Ensemble)
│   │   ├── nbeats_model.py    # N-BEATS implementation
│   │   └── lightgbm_model.py  # LightGBM model
│   ├── evaluation/
│   │   └── backtesting.py     # Backtesting and evaluation
│   └── utils/
│       ├── helpers.py          # Utility functions
│       └── data_generator.py   # Synthetic data generation
├── models/
│   └── saved_models/          # Trained model artifacts
├── logs/                       # Application logs
└── notebooks/                  # Jupyter notebooks for analysis
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/bishtNitin/atm-cash-forecasting.git
cd atm-cash-forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```bash
# Generate synthetic ATM data for demonstration
python -c "from src.utils.data_generator import generate_synthetic_data; generate_synthetic_data(n_atms=100)"
```

### 3. Train Models

```bash
# Run the training pipeline
python train.py
```

This will:
- Load or generate sample data
- Preprocess and engineer features
- Train a LightGBM model
- Evaluate performance
- Save results and feature importance

### 4. Launch Streamlit UI

```bash
# Start the interactive dashboard
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📊 Usage Examples

### Training a Model

```python
from src.models.lightgbm_model import LightGBMForecaster
from src.utils.helpers import load_config

# Load configuration
config = load_config('config.yaml')

# Initialize model
model = LightGBMForecaster(config['models']['lightgbm'])

# Train model
model.fit(train_data, target_col='cash_demand', feature_cols=feature_columns)

# Make predictions
predictions = model.predict(test_data)

# Get feature importance
importance = model.get_feature_importance()
```

### Running Backtests

```python
from src.evaluation.backtesting import BacktestEngine, ModelEvaluator

# Initialize backtesting
backtest = BacktestEngine(config['backtesting'])

# Create train-test split
train_df, test_df = backtest.train_test_split(data)

# Evaluate model
evaluator = ModelEvaluator(config['costs'])
metrics = evaluator.evaluate_model(y_true, y_pred, y_train)
```

### Feature Engineering

```python
from src.features.engineering import FeatureEngineering

# Initialize feature engineering
feature_eng = FeatureEngineering(config['features'])

# Generate features
df_features = feature_eng.engineer_features(
    df, 
    target_col='cash_demand',
    atm_id_col='atm_id'
)
```

## 🎓 Model Performance

Example performance metrics on synthetic data:

| Model | RMSE | MAE | MAPE | Stockout Cost | Total Cost |
|-------|------|-----|------|---------------|------------|
| LightGBM | ₹12,500 | ₹9,800 | 8.5% | ₹245,000 | ₹267,000 |
| Prophet | ₹15,200 | ₹11,500 | 10.2% | ₹298,000 | ₹315,000 |
| N-BEATS | ₹13,800 | ₹10,200 | 9.1% | ₹267,000 | ₹285,000 |

*Note: Actual performance will vary based on real data characteristics*

## 📈 Key Features for Operations Teams

1. **Cost Optimization**: Models are selected based on minimizing stockout and holding costs
2. **Holiday Awareness**: Indian holiday calendar integrated for better predictions
3. **Explainable AI**: SHAP values show which features drive each prediction
4. **Multi-ATM Support**: Global models can forecast for thousands of ATMs efficiently
5. **Real-time Monitoring**: Streamlit dashboard for operations monitoring

## 🔧 Configuration

Edit `config.yaml` to customize:
- Number of ATMs
- Forecast horizon
- Model hyperparameters
- Cost parameters (stockout/holding costs)
- Feature engineering settings
- Backtesting configuration

## 📝 Data Format

Input CSV files should have the following columns:
- `date`: Date in YYYY-MM-DD format
- `atm_id`: Unique ATM identifier
- `cash_demand`: Cash demand/withdrawal amount (INR)

Example:
```csv
date,atm_id,cash_demand
2023-01-01,ATM_0001,125000
2023-01-01,ATM_0002,98000
2023-01-02,ATM_0001,132000
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

Lead Data Scientist: ATM Cash Forecasting System

## 🙏 Acknowledgments

- **KATS**: Facebook's time series forecasting library
- **Darts**: Time series forecasting library with N-BEATS
- **LightGBM**: Microsoft's gradient boosting framework
- **SHAP**: SHapley Additive exPlanations for model interpretability
- **Streamlit**: Interactive web application framework

## 📞 Support

For questions and support, please open an issue on GitHub.

---

**Note**: This system is designed for demonstration purposes. For production deployment with 10,000+ ATMs, consider:
- Distributed computing (Spark/Dask)
- Model serving infrastructure (MLflow, KubeFlow)
- Automated retraining pipelines
- Real-time data ingestion
- A/B testing framework
- Monitoring and alerting
