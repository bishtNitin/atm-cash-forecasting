"""
Unit tests for ATM Cash Forecasting System
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_pipeline import ATMDataPipeline
from models.lightgbm_model import LightGBMForecaster
from backtesting import ATMBacktester


class TestDataPipeline(unittest.TestCase):
    """Test cases for data pipeline"""
    
    def setUp(self):
        self.config = {
            'data': {
                'raw_data_path': 'data/raw',
                'processed_data_path': 'data/processed'
            }
        }
        self.pipeline = ATMDataPipeline(self.config)
    
    def test_generate_sample_data(self):
        """Test sample data generation"""
        file_path = self.pipeline.generate_sample_data('/tmp', n_atms=5, n_days=30)
        self.assertTrue(os.path.exists(file_path))
        
        df = pd.read_csv(file_path)
        self.assertEqual(len(df), 5 * 30)
        self.assertEqual(df['atm_id'].nunique(), 5)
        
        # Cleanup
        os.remove(file_path)
    
    def test_clean_data(self):
        """Test data cleaning"""
        # Create sample data with missing values
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'cash_withdrawn': [100, np.nan, 200, 300, np.nan, 400, 500, 600, 700, 800]
        })
        
        df_clean = self.pipeline.clean_data(df, method='linear')
        
        # Check no missing values
        self.assertEqual(df_clean['cash_withdrawn'].isna().sum(), 0)
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=60),
            'cash_withdrawn': np.random.uniform(100000, 200000, 60)
        })
        
        df_features = self.pipeline.engineer_features(df, target_col='cash_withdrawn')
        
        # Check key features are created
        expected_features = ['day_of_week', 'day_of_month', 'is_weekend', 
                           'is_holiday', 'is_payday', 'lag_1', 'lag_7']
        
        for feature in expected_features:
            self.assertIn(feature, df_features.columns)


class TestLightGBMModel(unittest.TestCase):
    """Test cases for LightGBM model"""
    
    def setUp(self):
        self.config = {
            'n_estimators': 10,
            'learning_rate': 0.1,
            'max_depth': 3
        }
        self.model = LightGBMForecaster(self.config)
    
    def test_train_and_predict(self):
        """Test model training and prediction"""
        # Create sample data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
        y = pd.Series(np.random.randn(100) * 100000 + 500000)
        
        # Train model
        metrics = self.model.train(X, y)
        
        # Check metrics exist
        self.assertIn('train_rmse', metrics)
        self.assertIn('train_mae', metrics)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Check predictions shape
        self.assertEqual(len(predictions), len(X))
        
        # Check predictions are reasonable
        self.assertTrue(all(predictions > 0))


class TestBacktester(unittest.TestCase):
    """Test cases for backtesting"""
    
    def setUp(self):
        self.config = {
            'backtesting': {
                'n_splits': 3,
                'test_size': 10
            }
        }
        self.backtester = ATMBacktester(self.config)
    
    def test_calculate_metrics(self):
        """Test metric calculation"""
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([110, 190, 310, 390, 510])
        
        metrics = self.backtester.calculate_metrics(y_true, y_pred)
        
        # Check metrics exist
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('stockout_cost', metrics)
        
        # Check metrics are reasonable
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
    
    def test_stockout_cost(self):
        """Test stockout cost calculation"""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([90, 210, 280])  # Under, over, under
        
        cost = self.backtester.calculate_stockout_cost(y_true, y_pred)
        
        # Cost should be positive
        self.assertGreater(cost, 0)


if __name__ == '__main__':
    unittest.main()
