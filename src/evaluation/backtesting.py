"""
Backtesting Framework for ATM Cash Forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Backtesting engine for time series forecasting models"""
    
    def __init__(self, config: Dict):
        """
        Initialize BacktestEngine
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.test_size = config.get('test_size', 0.2)
        self.validation_size = config.get('validation_size', 0.1)
        self.cv_splits = config.get('cv_splits', 5)
        self.forecast_horizon = config.get('forecast_horizon', 7)
        
    def train_test_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (train_df, test_df)
        """
        n = len(df)
        split_idx = int(n * (1 - self.test_size))
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        return train_df, test_df
    
    def time_series_cv_split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create time series cross-validation splits
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of (train, validation) splits
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        splits = []
        
        for train_idx, val_idx in tscv.split(df):
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()
            splits.append((train_fold, val_fold))
        
        logger.info(f"Created {len(splits)} time series CV splits")
        return splits
    
    def rolling_forecast_origin(self, df: pd.DataFrame, 
                                initial_train_size: int,
                                step_size: int = 1) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create rolling forecast origin for backtesting
        
        Args:
            df: Input DataFrame
            initial_train_size: Initial training set size
            step_size: Number of periods to roll forward
            
        Returns:
            List of (train, test) splits
        """
        splits = []
        n = len(df)
        
        current_train_end = initial_train_size
        
        while current_train_end + self.forecast_horizon <= n:
            train_df = df.iloc[:current_train_end].copy()
            test_df = df.iloc[current_train_end:current_train_end + self.forecast_horizon].copy()
            splits.append((train_df, test_df))
            current_train_end += step_size
        
        logger.info(f"Created {len(splits)} rolling forecast origin splits")
        return splits


class MetricsCalculator:
    """Calculate various forecast accuracy metrics"""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        return np.mean(numerator[mask] / denominator[mask]) * 100
    
    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """Mean Absolute Scaled Error"""
        mae_forecast = np.mean(np.abs(y_true - y_pred))
        mae_naive = np.mean(np.abs(np.diff(y_train)))
        
        if mae_naive == 0:
            return np.inf
        return mae_forecast / mae_naive
    
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data for MASE calculation
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'rmse': MetricsCalculator.rmse(y_true, y_pred),
            'mae': MetricsCalculator.mae(y_true, y_pred),
            'mape': MetricsCalculator.mape(y_true, y_pred),
            'smape': MetricsCalculator.smape(y_true, y_pred)
        }
        
        if y_train is not None:
            metrics['mase'] = MetricsCalculator.mase(y_true, y_pred, y_train)
        
        return metrics


class StockoutCostCalculator:
    """Calculate stockout and holding costs for cash forecasting"""
    
    def __init__(self, stockout_cost: float = 100, holding_cost: float = 1):
        """
        Initialize StockoutCostCalculator
        
        Args:
            stockout_cost: Cost per stockout transaction (INR)
            holding_cost: Cost per day per 1000 INR held (INR)
        """
        self.stockout_cost = stockout_cost
        self.holding_cost = holding_cost
        
    def calculate_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate total cost based on forecast
        
        Args:
            y_true: True cash demand
            y_pred: Predicted cash demand (used for replenishment)
            
        Returns:
            Dictionary with cost components
        """
        # Stockout occurs when demand > prediction
        stockouts = np.maximum(0, y_true - y_pred)
        stockout_cost_total = np.sum(stockouts) * self.stockout_cost / 1000  # per 1000 INR
        
        # Holding cost occurs when prediction > demand
        excess_cash = np.maximum(0, y_pred - y_true)
        holding_cost_total = np.sum(excess_cash) * self.holding_cost / 1000  # per 1000 INR
        
        total_cost = stockout_cost_total + holding_cost_total
        
        return {
            'stockout_cost': stockout_cost_total,
            'holding_cost': holding_cost_total,
            'total_cost': total_cost,
            'avg_stockout': np.mean(stockouts),
            'avg_excess': np.mean(excess_cash)
        }


class ModelEvaluator:
    """Evaluate and compare forecasting models"""
    
    def __init__(self, config: Dict):
        """
        Initialize ModelEvaluator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.metrics_calc = MetricsCalculator()
        self.cost_calc = StockoutCostCalculator(
            stockout_cost=config.get('stockout_cost_per_transaction', 100),
            holding_cost=config.get('holding_cost_per_day', 1)
        )
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      y_train: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate a single model
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Accuracy metrics
        metrics = self.metrics_calc.calculate_all_metrics(y_true, y_pred, y_train)
        
        # Cost metrics
        cost_metrics = self.cost_calc.calculate_cost(y_true, y_pred)
        
        # Combine all metrics
        all_metrics = {**metrics, **cost_metrics}
        
        return all_metrics
    
    def compare_models(self, results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            results: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame with comparison results
        """
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('total_cost')
        
        logger.info("Model comparison completed")
        return comparison_df
    
    def select_best_model(self, results: Dict[str, Dict[str, float]],
                         primary_metric: str = 'total_cost') -> str:
        """
        Select best model based on primary metric
        
        Args:
            results: Dictionary mapping model names to their metrics
            primary_metric: Metric to use for selection (lower is better)
            
        Returns:
            Name of best model
        """
        best_model = min(results.keys(), key=lambda k: results[k][primary_metric])
        
        logger.info(f"Best model: {best_model} with {primary_metric}={results[best_model][primary_metric]:.2f}")
        return best_model
