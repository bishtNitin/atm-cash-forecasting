# Quick Reference Guide

## Installation

```bash
# Clone repository
git clone https://github.com/bishtNitin/atm-cash-forecasting.git
cd atm-cash-forecasting

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train the Model

```bash
python src/train.py
```

Expected output:
- Generates sample data (or loads existing)
- Trains LightGBM model
- Runs backtesting
- Calculates SHAP values
- Saves model to `models/lightgbm_model.pkl`

### 2. Launch Dashboard

```bash
streamlit run app.py
```

Access at: http://localhost:8501

## Dashboard Navigation

### View 1: Network Health
- **What**: Overview of all ATMs on India map
- **Use**: Identify high-risk ATMs quickly
- **Key Metrics**: Total ATMs, High/Medium/Low risk counts

### View 2: ATM Detail Analysis
- **What**: Detailed forecast for specific ATM
- **Use**: Understand forecast drivers and plan cash loading
- **Key Features**:
  - Recommended load amount
  - Stockout risk percentage
  - SHAP waterfall explanation
  - Historical trend
  - Feature importance

## API Usage

### Data Pipeline

```python
from src.data_pipeline import ATMDataPipeline

config = {...}
pipeline = ATMDataPipeline(config)

# Load data
df = pipeline.ingest_csv('data/raw/atm_transactions.csv')

# Clean data
df_clean = pipeline.clean_data(df, method='linear')

# Engineer features
df_features = pipeline.engineer_features(df_clean)
```

### Model Training

```python
from src.models.lightgbm_model import MultiATMLightGBM

config = {'n_estimators': 500, 'learning_rate': 0.05}
model = MultiATMLightGBM(config, mode='global')
model.train(df_features, target_col='cash_withdrawn')

# Make predictions
predictions = model.predict(df_test)
```

### Explainability

```python
from src.explainability import ATMExplainer

explainer = ATMExplainer(model.model, feature_names)
explainer.calculate_shap_values(X_test)

# Get explanation
explanation = explainer.explain_prediction(
    atm_id='ATM_0001',
    date='2024-01-01',
    shap_values=shap_values[0],
    features=X_test.iloc[0],
    prediction=predictions[0]
)
```

## Configuration

Edit `config/config.yaml`:

```yaml
models:
  lightgbm:
    n_estimators: 500
    learning_rate: 0.05
    max_depth: 7

backtesting:
  n_splits: 5
  test_size: 30

explainability:
  shap_sample_size: 100
  top_features: 10
```

## Testing

```bash
# Run all tests
python -m unittest discover tests

# Run specific test
python -m unittest tests.test_forecasting.TestDataPipeline
```

## Data Format

CSV files should contain:
- `atm_id`: ATM identifier (e.g., ATM_0001)
- `date`: Date in YYYY-MM-DD format
- `cash_withdrawn`: Cash amount in rupees
- `latitude`, `longitude`: Location (optional)

Example:
```csv
atm_id,date,cash_withdrawn,n_transactions,latitude,longitude
ATM_0001,2023-01-01,1250000.50,45,28.6139,77.2090
```

## Troubleshooting

### Issue: ImportError
**Solution**: Ensure all dependencies installed: `pip install -r requirements.txt`

### Issue: No data found
**Solution**: Run training script first: `python src/train.py`

### Issue: SHAP calculation slow
**Solution**: Reduce `shap_sample_size` in config.yaml

### Issue: Dashboard not loading
**Solution**: Check Streamlit version: `streamlit version`

## Performance Tips

1. **Large Datasets**: Use `mode='global'` instead of `mode='per_atm'`
2. **Faster Training**: Reduce `n_estimators` in config
3. **Memory Usage**: Reduce `shap_sample_size` for SHAP calculations
4. **Dashboard Speed**: Cache is enabled - first load may be slow

## Feature Engineering

The system automatically creates:

**Time Features:**
- Day of week, month, year
- Weekend indicator
- Start/end of month

**Business Features:**
- Payday indicator (1st, 15th, month-end)
- Indian holidays
- Festival seasons (Diwali, Holi)

**Statistical Features:**
- Lag features (1, 7, 30 days)
- Rolling mean and std (7, 30 days)
- Exponential moving average

## Model Selection

The system uses stockout-cost as primary metric:
- **Underestimation penalty**: 2.0x (stockout risk)
- **Overestimation penalty**: 0.1x (excess cash)

Lower stockout-cost = better model

## Support

- GitHub Issues: Report bugs or request features
- Documentation: See README.md
- Examples: See notebooks/example_usage.ipynb
