"""
Main training script for ATM Cash Forecasting System
Demonstrates full pipeline: data loading -> preprocessing -> feature engineering -> model training -> evaluation
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.helpers import load_config, setup_logging
from src.utils.data_generator import generate_synthetic_data
from src.data.ingestion import DataIngestion, DataPreprocessor
from src.features.engineering import FeatureEngineering
from src.models.lightgbm_model import LightGBMForecaster
from src.evaluation.backtesting import BacktestEngine, ModelEvaluator

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    
    logger.info("=" * 80)
    logger.info("ATM Cash Forecasting System - Training Pipeline")
    logger.info("=" * 80)
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config('config.yaml')
    
    # Step 1: Generate/Load Data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Data Generation/Loading")
    logger.info("=" * 80)
    
    data_path = Path('data/raw/synthetic_atm_data.csv')
    
    if not data_path.exists():
        logger.info("Generating synthetic data...")
        df = generate_synthetic_data(
            n_atms=100,  # Use 100 ATMs for demo
            start_date='2022-01-01',
            end_date='2023-12-31',
            output_path=str(data_path)
        )
    else:
        logger.info("Loading existing data...")
        ingestion = DataIngestion('data/raw')
        df = ingestion.load_csv('synthetic_atm_data.csv')
    
    logger.info(f"Loaded {len(df)} records for {df['atm_id'].nunique()} ATMs")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Step 2: Data Preprocessing
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Data Preprocessing")
    logger.info("=" * 80)
    
    preprocessor = DataPreprocessor(imputation_method='linear')
    df = preprocessor.preprocess(df, handle_outliers=True)
    
    logger.info("Data preprocessing completed")
    
    # Step 3: Feature Engineering
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 80)
    
    feature_config = {
        'lag_periods': config['features']['lag_periods'],
        'rolling_windows': config['features']['rolling_windows'],
        'use_holidays': config['features']['use_holidays'],
        'use_covariance': False,  # Disable for demo due to computational cost
        'country': config['data']['country'],
        'covariance_window': 30
    }
    
    feature_eng = FeatureEngineering(feature_config)
    df_features = feature_eng.engineer_features(df, target_col='cash_demand', atm_id_col='atm_id')
    
    logger.info(f"Created {len(df_features.columns)} features")
    
    # Step 4: Train-Test Split
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Train-Test Split")
    logger.info("=" * 80)
    
    backtest_engine = BacktestEngine(config['backtesting'])
    
    # Sort by date
    df_features = df_features.sort_values(['atm_id', 'date']).reset_index(drop=True)
    
    # Select one ATM for demo
    demo_atm = df_features['atm_id'].unique()[0]
    df_atm = df_features[df_features['atm_id'] == demo_atm].copy()
    
    train_df, test_df = backtest_engine.train_test_split(df_atm)
    
    logger.info(f"Training on ATM: {demo_atm}")
    logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Step 5: Model Training
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Model Training - LightGBM")
    logger.info("=" * 80)
    
    # Train LightGBM model
    lgbm_model = LightGBMForecaster(config['models']['lightgbm'])
    
    # Identify feature columns (exclude target, date, atm_id)
    exclude_cols = ['cash_demand', 'date', 'atm_id']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    lgbm_model.fit(train_df, target_col='cash_demand', feature_cols=feature_cols)
    
    # Step 6: Model Evaluation
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Model Evaluation")
    logger.info("=" * 80)
    
    # Make predictions on test set
    y_test = test_df['cash_demand'].values
    y_pred = lgbm_model.predict(test_df)
    y_train = train_df['cash_demand'].values
    
    # Evaluate
    evaluator = ModelEvaluator(config['costs'])
    metrics = evaluator.evaluate_model(y_test, y_pred, y_train)
    
    logger.info("\nModel Performance Metrics:")
    logger.info("-" * 50)
    for metric, value in metrics.items():
        logger.info(f"{metric:20s}: {value:,.2f}")
    
    # Step 7: Feature Importance
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Feature Importance Analysis")
    logger.info("=" * 80)
    
    feature_importance = lgbm_model.get_feature_importance()
    logger.info("\nTop 10 Most Important Features:")
    logger.info("-" * 50)
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"{row['feature']:30s}: {row['importance']:,.0f}")
    
    # Save feature importance
    importance_path = 'data/processed/feature_importance.csv'
    feature_importance.to_csv(importance_path, index=False)
    logger.info(f"\nFeature importance saved to {importance_path}")
    
    # Step 8: Save Results
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Saving Results")
    logger.info("=" * 80)
    
    # Save predictions
    results_df = test_df[['date', 'atm_id', 'cash_demand']].copy()
    results_df['predicted_demand'] = y_pred
    results_df['prediction_error'] = results_df['cash_demand'] - results_df['predicted_demand']
    
    results_path = 'data/processed/predictions.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Predictions saved to {results_path}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nSummary:")
    logger.info(f"  - Model: LightGBM")
    logger.info(f"  - ATM: {demo_atm}")
    logger.info(f"  - Test RMSE: ₹{metrics['rmse']:,.0f}")
    logger.info(f"  - Test MAE: ₹{metrics['mae']:,.0f}")
    logger.info(f"  - Test MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  - Total Cost: ₹{metrics['total_cost']:,.0f}")
    logger.info(f"  - Stockout Cost: ₹{metrics['stockout_cost']:,.0f}")
    logger.info(f"  - Holding Cost: ₹{metrics['holding_cost']:,.0f}")
    logger.info("\nNext Steps:")
    logger.info("  1. Run 'streamlit run app.py' to launch the UI")
    logger.info("  2. Compare with other models (Prophet, N-BEATS, etc.)")
    logger.info("  3. Scale to all 10,000 ATMs")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
