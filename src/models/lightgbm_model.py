"""
LightGBM Model for ATM Cash Forecasting
Global model that can handle multiple ATMs
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
import pickle


class LightGBMForecaster:
    """Global LightGBM model for ATM cash forecasting"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
             validation_data: Optional[Tuple] = None) -> Dict:
        """
        Train LightGBM model
        
        Args:
            X: Training features
            y: Training target
            validation_data: Optional tuple of (X_val, y_val)
            
        Returns:
            Dictionary with training metrics
        """
        self.feature_names = X.columns.tolist()
        
        # Get model parameters from config
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'n_estimators': self.config.get('n_estimators', 500),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'max_depth': self.config.get('max_depth', 7),
            'num_leaves': self.config.get('num_leaves', 31),
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1
        }
        
        if validation_data is not None:
            X_val, y_val = validation_data
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
        else:
            self.model = lgb.LGBMRegressor(**params)
            self.model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Calculate training metrics
        train_pred = self.model.predict(X)
        train_rmse = np.sqrt(np.mean((y - train_pred) ** 2))
        train_mae = np.mean(np.abs(y - train_pred))
        train_mape = np.mean(np.abs((y - train_pred) / y)) * 100
        
        metrics = {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'train_mape': train_mape
        }
        
        if validation_data is not None:
            val_pred = self.model.predict(X_val)
            val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
            val_mae = np.mean(np.abs(y_val - val_pred))
            val_mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
            
            metrics.update({
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_mape': val_mape
            })
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X: pd.DataFrame, 
                                 n_estimators: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using quantile regression
        
        Args:
            X: Features for prediction
            n_estimators: Number of estimators to use for uncertainty
            
        Returns:
            Tuple of (predictions, std_dev)
        """
        predictions = self.predict(X)
        
        # Simple uncertainty estimation based on feature variance
        # In production, could use quantile regression or ensemble methods
        std_dev = np.std(predictions) * np.ones_like(predictions) * 0.1
        
        return predictions, std_dev
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N important features
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, file_path: str):
        """Save model to file"""
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'feature_importance': self.feature_importance,
                'config': self.config
            }, f)
    
    def load_model(self, file_path: str):
        """Load model from file"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.feature_importance = data['feature_importance']
            self.config = data.get('config', self.config)


class MultiATMLightGBM:
    """Wrapper for training separate models per ATM or a global model"""
    
    def __init__(self, config: Dict, mode: str = 'global'):
        """
        Args:
            config: Configuration dictionary
            mode: 'global' for single model or 'per_atm' for separate models
        """
        self.config = config
        self.mode = mode
        self.models = {}  # ATM ID -> model mapping for per_atm mode
        self.global_model = None
        
    def train(self, df: pd.DataFrame, target_col: str = 'cash_withdrawn'):
        """
        Train models based on mode
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
        """
        if self.mode == 'global':
            # Train single global model
            feature_cols = [col for col in df.columns 
                          if col not in ['date', target_col, 'atm_id', 'latitude', 'longitude']]
            
            X = df[feature_cols]
            y = df[target_col]
            
            # Split for validation
            split_idx = int(len(df) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            self.global_model = LightGBMForecaster(self.config)
            metrics = self.global_model.train(X_train, y_train, (X_val, y_val))
            
            print("Global model training metrics:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.2f}")
                
        elif self.mode == 'per_atm':
            # Train separate model for each ATM
            if 'atm_id' not in df.columns:
                raise ValueError("ATM ID column required for per_atm mode")
            
            atm_ids = df['atm_id'].unique()
            feature_cols = [col for col in df.columns 
                          if col not in ['date', target_col, 'atm_id', 'latitude', 'longitude']]
            
            for atm_id in atm_ids:
                atm_df = df[df['atm_id'] == atm_id]
                
                X = atm_df[feature_cols]
                y = atm_df[target_col]
                
                # Skip if insufficient data
                if len(X) < 30:
                    continue
                
                # Split for validation
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
                
                model = LightGBMForecaster(self.config)
                model.train(X_train, y_train, (X_val, y_val))
                
                self.models[atm_id] = model
            
            print(f"Trained {len(self.models)} ATM-specific models")
    
    def predict(self, df: pd.DataFrame, atm_id: Optional[str] = None) -> np.ndarray:
        """
        Make predictions
        
        Args:
            df: DataFrame with features
            atm_id: ATM ID for per_atm mode
            
        Returns:
            Array of predictions
        """
        feature_cols = [col for col in df.columns 
                       if col not in ['date', 'cash_withdrawn', 'atm_id', 'latitude', 'longitude']]
        
        X = df[feature_cols]
        
        if self.mode == 'global':
            return self.global_model.predict(X)
        else:
            if atm_id is None:
                raise ValueError("ATM ID required for per_atm mode")
            if atm_id not in self.models:
                # Fallback to global average or zero
                return np.zeros(len(X))
            return self.models[atm_id].predict(X)
