"""
Training script for ATM Cash Forecasting models
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_pipeline import ATMDataPipeline
from models.lightgbm_model import LightGBMForecaster, MultiATMLightGBM
from backtesting import ATMBacktester
from explainability import ATMExplainer


def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(config):
    """Prepare data for training"""
    print("=" * 80)
    print("DATA PREPARATION")
    print("=" * 80)
    
    pipeline = ATMDataPipeline(config)
    
    # Check if data exists, otherwise generate
    data_path = os.path.join(config['data']['raw_data_path'], 'atm_transactions.csv')
    
    if not os.path.exists(data_path):
        print("\n📊 Generating sample data...")
        pipeline.generate_sample_data(
            config['data']['raw_data_path'],
            n_atms=100,
            n_days=365
        )
    else:
        print(f"\n📊 Loading data from {data_path}")
    
    # Load and process data
    df = pipeline.ingest_csv(data_path)
    print(f"✓ Loaded {len(df)} records for {df['atm_id'].nunique()} ATMs")
    
    # Clean data
    print("\n🧹 Cleaning data and imputing missing values...")
    df_clean = pipeline.clean_data(df, method='linear')
    print("✓ Data cleaned")
    
    # Engineer features
    print("\n⚙️  Engineering features...")
    df_features = pipeline.engineer_features(df_clean, target_col='cash_withdrawn')
    print(f"✓ Created {len(df_features.columns)} features")
    
    # Remove rows with NaN
    feature_cols = [col for col in df_features.columns 
                   if col not in ['date', 'cash_withdrawn', 'atm_id', 'latitude', 'longitude']]
    df_features = df_features.dropna(subset=feature_cols + ['cash_withdrawn'])
    
    print(f"✓ Final dataset: {len(df_features)} records")
    
    return df_features, feature_cols


def train_models(df_features, feature_cols, config):
    """Train and evaluate models"""
    print("\n" + "=" * 80)
    print("MODEL TRAINING")
    print("=" * 80)
    
    # Initialize model
    lgb_config = config.get('models', {}).get('lightgbm', {})
    
    print("\n🤖 Training Global LightGBM Model...")
    model_wrapper = MultiATMLightGBM(lgb_config, mode='global')
    model_wrapper.train(df_features, target_col='cash_withdrawn')
    print("✓ Model trained successfully")
    
    # Save model
    model_path = os.path.join(config['models']['output_path'], 'lightgbm_model.pkl')
    os.makedirs(config['models']['output_path'], exist_ok=True)
    model_wrapper.global_model.save_model(model_path)
    print(f"✓ Model saved to {model_path}")
    
    return model_wrapper


def backtest_models(df_features, model_wrapper, config):
    """Perform backtesting"""
    print("\n" + "=" * 80)
    print("BACKTESTING")
    print("=" * 80)
    
    backtester = ATMBacktester(config)
    
    # Define trainer and predictor functions
    def train_fn(train_df):
        feature_cols = [col for col in train_df.columns 
                       if col not in ['date', 'cash_withdrawn', 'atm_id', 'latitude', 'longitude']]
        X = train_df[feature_cols]
        y = train_df['cash_withdrawn']
        
        lgb_config = config.get('models', {}).get('lightgbm', {})
        model = LightGBMForecaster(lgb_config)
        model.train(X, y)
        return model
    
    def predict_fn(model, test_df):
        feature_cols = [col for col in test_df.columns 
                       if col not in ['date', 'cash_withdrawn', 'atm_id', 'latitude', 'longitude']]
        X = test_df[feature_cols]
        return model.predict(X)
    
    print("\n📈 Running time-series cross-validation...")
    backtest_config = config.get('backtesting', {})
    results = backtester.backtest_model(
        train_fn,
        predict_fn,
        df_features,
        target_col='cash_withdrawn',
        n_splits=backtest_config.get('n_splits', 5),
        test_size=backtest_config.get('test_size', 30)
    )
    
    print("\n✓ Backtesting completed")
    print("\n📊 Average Metrics:")
    for metric, value in results['average_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n📊 Overall Metrics:")
    for metric, value in results['overall_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    return results


def create_explainer(model_wrapper, df_features, feature_cols, config):
    """Create SHAP explainer"""
    print("\n" + "=" * 80)
    print("EXPLAINABILITY ANALYSIS")
    print("=" * 80)
    
    print("\n🔍 Calculating SHAP values...")
    
    # Sample data for SHAP
    shap_config = config.get('explainability', {})
    sample_size = shap_config.get('shap_sample_size', 100)
    X_sample = df_features[feature_cols].sample(n=min(sample_size, len(df_features)), random_state=42)
    
    # Create explainer
    explainer = ATMExplainer(
        model_wrapper.global_model.model,
        feature_cols,
        shap_config
    )
    
    explainer.calculate_shap_values(X_sample)
    print("✓ SHAP values calculated")
    
    # Get top features
    print("\n📊 Top 10 Most Important Features:")
    top_features = explainer.get_top_features(n=10)
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return explainer


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train ATM Cash Forecasting Models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("ATM CASH FORECASTING - TRAINING PIPELINE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Prepare data
    df_features, feature_cols = prepare_data(config)
    
    # Train models
    model_wrapper = train_models(df_features, feature_cols, config)
    
    # Backtest
    backtest_results = backtest_models(df_features, model_wrapper, config)
    
    # Create explainer
    explainer = create_explainer(model_wrapper, df_features, feature_cols, config)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 Next Steps:")
    print("  1. Run the Streamlit dashboard: streamlit run app.py")
    print("  2. Review model performance in the dashboard")
    print("  3. Analyze SHAP explanations for key predictions")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
