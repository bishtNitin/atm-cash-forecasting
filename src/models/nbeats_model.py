"""
N-BEATS Model using Darts for ATM Cash Forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler

from .base_model import BaseForecaster

logger = logging.getLogger(__name__)


class NBEATSForecaster(BaseForecaster):
    """N-BEATS model implementation using Darts"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize N-BEATS forecaster
        
        Args:
            config: Model configuration
        """
        super().__init__('N-BEATS', config)
        self.scaler = Scaler()
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'cash_demand'):
        """
        Fit N-BEATS model
        
        Args:
            train_data: Training DataFrame with 'date' and target columns
            target_col: Target column name
        """
        logger.info(f"Fitting {self.model_name} model")
        
        # Prepare data for Darts
        ts_data = train_data[['date', target_col]].copy()
        ts_data = ts_data.set_index('date')
        series = TimeSeries.from_dataframe(ts_data, value_cols=target_col)
        
        # Scale the data
        series_scaled = self.scaler.fit_transform(series)
        
        # Create N-BEATS model
        self.model = NBEATSModel(
            input_chunk_length=self.config.get('input_chunk_length', 30),
            output_chunk_length=self.config.get('output_chunk_length', 7),
            num_stacks=self.config.get('num_stacks', 30),
            num_blocks=self.config.get('num_blocks', 1),
            num_layers=self.config.get('num_layers', 4),
            layer_widths=self.config.get('layer_widths', 256),
            expansion_coefficient_dim=self.config.get('expansion_coefficient_dim', 5),
            n_epochs=self.config.get('n_epochs', 100),
            batch_size=self.config.get('batch_size', 32),
            random_state=42
        )
        
        # Fit model
        self.model.fit(series_scaled, verbose=False)
        self.is_fitted = True
        self.train_series = series_scaled
        
        logger.info(f"{self.model_name} model fitted successfully")
        
    def predict(self, horizon: int, exog_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions using N-BEATS
        
        Args:
            horizon: Forecast horizon (number of periods)
            exog_data: Not used for N-BEATS
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {horizon}-step forecast with {self.model_name}")
        
        # Make prediction
        forecast_scaled = self.model.predict(n=horizon, series=self.train_series)
        
        # Inverse transform
        forecast = self.scaler.inverse_transform(forecast_scaled)
        predictions = forecast.values().flatten()
        
        return predictions
