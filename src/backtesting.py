"""
Backtesting Framework for ATM Cash Forecasting
Implements time-series cross-validation and model evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Callable
from datetime import timedelta


class ATMBacktester:
    """Backtesting framework for time series forecasting models"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = []
        
    def time_series_split(self, df: pd.DataFrame, 
                         n_splits: int = 5, 
                         test_size: int = 30) -> List[Tuple]:
        """
        Create time series splits for backtesting
        
        Args:
            df: DataFrame with date column
            n_splits: Number of splits
            test_size: Size of test set in days
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        df = df.sort_values('date').reset_index(drop=True)
        dates = df['date'].unique()
        
        splits = []
        total_days = len(dates)
        
        for i in range(n_splits):
            # Calculate split point
            test_end_idx = total_days - i * test_size
            test_start_idx = test_end_idx - test_size
            
            if test_start_idx < test_size:  # Not enough training data
                break
            
            test_dates = dates[test_start_idx:test_end_idx]
            train_dates = dates[:test_start_idx]
            
            train_idx = df[df['date'].isin(train_dates)].index
            test_idx = df[df['date'].isin(test_dates)].index
            
            splits.append((train_idx, test_idx))
        
        return splits[::-1]  # Reverse to go chronologically
    
    def calculate_stockout_cost(self, y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               stockout_penalty: float = 2.0,
                               overload_penalty: float = 0.1) -> float:
        """
        Calculate stockout-cost metric
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            stockout_penalty: Penalty multiplier for underestimation
            overload_penalty: Penalty multiplier for overestimation
            
        Returns:
            Stockout cost
        """
        errors = y_true - y_pred
        
        # Underestimation (stockout) - higher penalty
        stockout_cost = np.sum(np.maximum(errors, 0) * stockout_penalty)
        
        # Overestimation (excess cash) - lower penalty
        overload_cost = np.sum(np.maximum(-errors, 0) * overload_penalty)
        
        total_cost = stockout_cost + overload_cost
        
        return total_cost / len(y_true)
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Ensure no zero values for MAPE calculation
        y_true_safe = np.where(y_true == 0, 1, y_true)
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        
        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Stockout cost
        stockout_cost = self.calculate_stockout_cost(y_true, y_pred)
        
        # Bias (mean error)
        bias = np.mean(y_pred - y_true)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'stockout_cost': stockout_cost,
            'bias': bias
        }
    
    def backtest_model(self, 
                      model_trainer: Callable,
                      model_predictor: Callable,
                      df: pd.DataFrame,
                      target_col: str = 'cash_withdrawn',
                      n_splits: int = 5,
                      test_size: int = 30) -> Dict:
        """
        Perform backtesting for a model
        
        Args:
            model_trainer: Function to train model (takes train_df, returns model)
            model_predictor: Function to make predictions (takes model, test_df, returns predictions)
            df: Full DataFrame with features and target
            target_col: Name of target column
            n_splits: Number of CV splits
            test_size: Test set size in days
            
        Returns:
            Dictionary with backtesting results
        """
        splits = self.time_series_split(df, n_splits=n_splits, test_size=test_size)
        
        all_metrics = []
        all_predictions = []
        all_actuals = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            # Train model
            model = model_trainer(train_df)
            
            # Make predictions
            predictions = model_predictor(model, test_df)
            actuals = test_df[target_col].values
            
            # Calculate metrics
            fold_metrics = self.calculate_metrics(actuals, predictions)
            fold_metrics['fold'] = fold_idx
            all_metrics.append(fold_metrics)
            
            all_predictions.extend(predictions)
            all_actuals.extend(actuals)
        
        # Aggregate metrics across folds
        metrics_df = pd.DataFrame(all_metrics)
        
        avg_metrics = {
            'mean_rmse': metrics_df['rmse'].mean(),
            'std_rmse': metrics_df['rmse'].std(),
            'mean_mae': metrics_df['mae'].mean(),
            'std_mae': metrics_df['mae'].std(),
            'mean_mape': metrics_df['mape'].mean(),
            'std_mape': metrics_df['mape'].std(),
            'mean_r2': metrics_df['r2'].mean(),
            'std_r2': metrics_df['r2'].std(),
            'mean_stockout_cost': metrics_df['stockout_cost'].mean(),
            'std_stockout_cost': metrics_df['stockout_cost'].std(),
            'mean_bias': metrics_df['bias'].mean()
        }
        
        # Overall metrics on all predictions
        overall_metrics = self.calculate_metrics(
            np.array(all_actuals), 
            np.array(all_predictions)
        )
        
        return {
            'fold_metrics': metrics_df,
            'average_metrics': avg_metrics,
            'overall_metrics': overall_metrics,
            'predictions': all_predictions,
            'actuals': all_actuals
        }
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models based on backtesting results
        
        Args:
            results_dict: Dictionary mapping model names to backtest results
            
        Returns:
            Comparison DataFrame
        """
        comparison = []
        
        for model_name, results in results_dict.items():
            avg_metrics = results['average_metrics']
            comparison.append({
                'model': model_name,
                'rmse': avg_metrics['mean_rmse'],
                'mae': avg_metrics['mean_mae'],
                'mape': avg_metrics['mean_mape'],
                'r2': avg_metrics['mean_r2'],
                'stockout_cost': avg_metrics['mean_stockout_cost'],
                'bias': avg_metrics['mean_bias']
            })
        
        comparison_df = pd.DataFrame(comparison)
        comparison_df = comparison_df.sort_values('stockout_cost')
        
        return comparison_df
    
    def select_best_model(self, results_dict: Dict[str, Dict], 
                         metric: str = 'stockout_cost') -> str:
        """
        Select best model based on a metric
        
        Args:
            results_dict: Dictionary mapping model names to backtest results
            metric: Metric to use for selection (lower is better)
            
        Returns:
            Name of best model
        """
        best_model = None
        best_score = float('inf')
        
        for model_name, results in results_dict.items():
            score = results['average_metrics'][f'mean_{metric}']
            
            if score < best_score:
                best_score = score
                best_model = model_name
        
        return best_model


def calculate_stockout_probability(prediction: float, 
                                   actual_std: float,
                                   threshold: float) -> float:
    """
    Calculate probability of stockout given prediction uncertainty
    
    Args:
        prediction: Predicted demand
        actual_std: Standard deviation of prediction error
        threshold: Cash threshold
        
    Returns:
        Probability of stockout
    """
    from scipy import stats
    
    # Calculate z-score
    z = (threshold - prediction) / actual_std if actual_std > 0 else 0
    
    # Probability that actual demand exceeds threshold
    stockout_prob = 1 - stats.norm.cdf(z)
    
    return stockout_prob


if __name__ == "__main__":
    print("Backtesting module loaded successfully")
