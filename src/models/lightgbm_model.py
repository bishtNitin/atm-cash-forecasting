"""
Global LightGBM Model for ATM Cash Forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from .base_model import BaseForecaster

logger = logging.getLogger(__name__)


class LightGBMForecaster(BaseForecaster):
    """Global LightGBM model for multi-ATM forecasting"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LightGBM forecaster
        
        Args:
            config: Model configuration
        """
        super().__init__('LightGBM', config)
        self.feature_names = None
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'cash_demand',
            feature_cols: Optional[List[str]] = None):
        """
        Fit LightGBM model
        
        Args:
            train_data: Training DataFrame with features and target
            target_col: Target column name
            feature_cols: List of feature columns (if None, use all except target and date)
        """
        logger.info(f"Fitting {self.model_name} model")
        
        # Identify feature columns
        if feature_cols is None:
            exclude_cols = [target_col, 'date', 'atm_id']
            feature_cols = [col for col in train_data.columns if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        # Prepare features and target
        X = train_data[feature_cols].copy()
        y = train_data[target_col].copy()
        
        # Remove rows with NaN (from lag features)
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create LightGBM datasets
        train_data_lgb = lgb.Dataset(X_train, label=y_train)
        val_data_lgb = lgb.Dataset(X_val, label=y_val, reference=train_data_lgb)
        
        # Model parameters
        params = {
            'objective': self.config.get('objective', 'regression'),
            'metric': self.config.get('metric', 'rmse'),
            'boosting_type': self.config.get('boosting_type', 'gbdt'),
            'num_leaves': self.config.get('num_leaves', 31),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'max_depth': self.config.get('max_depth', 7),
            'min_child_samples': self.config.get('min_child_samples', 20),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'verbose': -1
        }
        
        # Train model
        self.model = lgb.train(
            params,
            train_data_lgb,
            num_boost_round=self.config.get('n_estimators', 200),
            valid_sets=[train_data_lgb, val_data_lgb],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        self.is_fitted = True
        logger.info(f"{self.model_name} model fitted successfully")
        
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using LightGBM
        
        Args:
            test_data: Test DataFrame with features
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating forecasts with {self.model_name}")
        
        # Prepare features
        X = test_data[self.feature_names].copy()
        
        # Make predictions
        predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
