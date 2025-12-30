"""
KATS Models (Prophet, Theta, Ensemble) for ATM Cash Forecasting
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from kats.consts import TimeSeriesData
from kats.models.prophet import ProphetModel, ProphetParams
from kats.models.theta import ThetaModel, ThetaParams
from kats.models.ensemble.ensemble import EnsembleParams, BaseModelParams
from kats.models.ensemble.kats_ensemble import KatsEnsemble

from .base_model import BaseForecaster

logger = logging.getLogger(__name__)


class ProphetForecaster(BaseForecaster):
    """Prophet model implementation using KATS"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Prophet forecaster
        
        Args:
            config: Model configuration
        """
        super().__init__('Prophet', config)
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'cash_demand'):
        """
        Fit Prophet model
        
        Args:
            train_data: Training DataFrame with 'date' and target columns
            target_col: Target column name
        """
        logger.info(f"Fitting {self.model_name} model")
        
        # Prepare data for KATS
        ts_data = train_data[['date', target_col]].copy()
        ts_data.columns = ['time', 'value']
        ts_data = TimeSeriesData(ts_data)
        
        # Create Prophet parameters
        params = ProphetParams(
            changepoint_prior_scale=self.config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=self.config.get('seasonality_prior_scale', 10.0),
            seasonality_mode=self.config.get('seasonality_mode', 'multiplicative')
        )
        
        # Fit model
        self.model = ProphetModel(ts_data, params)
        self.model.fit()
        self.is_fitted = True
        
        logger.info(f"{self.model_name} model fitted successfully")
        
    def predict(self, horizon: int, exog_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions using Prophet
        
        Args:
            horizon: Forecast horizon (number of periods)
            exog_data: Not used for Prophet
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {horizon}-step forecast with {self.model_name}")
        
        forecast = self.model.predict(steps=horizon)
        predictions = forecast['fcst'].values
        
        return predictions


class ThetaForecaster(BaseForecaster):
    """Theta model implementation using KATS"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Theta forecaster
        
        Args:
            config: Model configuration
        """
        super().__init__('Theta', config)
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'cash_demand'):
        """
        Fit Theta model
        
        Args:
            train_data: Training DataFrame with 'date' and target columns
            target_col: Target column name
        """
        logger.info(f"Fitting {self.model_name} model")
        
        # Prepare data for KATS
        ts_data = train_data[['date', target_col]].copy()
        ts_data.columns = ['time', 'value']
        ts_data = TimeSeriesData(ts_data)
        
        # Create Theta parameters
        params = ThetaParams(m=self.config.get('m', 12))
        
        # Fit model
        self.model = ThetaModel(ts_data, params)
        self.model.fit()
        self.is_fitted = True
        
        logger.info(f"{self.model_name} model fitted successfully")
        
    def predict(self, horizon: int, exog_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions using Theta
        
        Args:
            horizon: Forecast horizon (number of periods)
            exog_data: Not used for Theta
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {horizon}-step forecast with {self.model_name}")
        
        forecast = self.model.predict(steps=horizon)
        predictions = forecast['fcst'].values
        
        return predictions


class EnsembleForecaster(BaseForecaster):
    """Ensemble of KATS models"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Ensemble forecaster
        
        Args:
            config: Model configuration
        """
        super().__init__('KATSEnsemble', config)
        
    def fit(self, train_data: pd.DataFrame, target_col: str = 'cash_demand'):
        """
        Fit Ensemble model
        
        Args:
            train_data: Training DataFrame with 'date' and target columns
            target_col: Target column name
        """
        logger.info(f"Fitting {self.model_name} model")
        
        # Prepare data for KATS
        ts_data = train_data[['date', target_col]].copy()
        ts_data.columns = ['time', 'value']
        ts_data = TimeSeriesData(ts_data)
        
        # Create ensemble with Prophet and Theta
        model_params = EnsembleParams([
            BaseModelParams("prophet", ProphetParams()),
            BaseModelParams("theta", ThetaParams(m=12))
        ])
        
        # Fit model
        self.model = KatsEnsemble(ts_data, model_params)
        self.model.fit()
        self.is_fitted = True
        
        logger.info(f"{self.model_name} model fitted successfully")
        
    def predict(self, horizon: int, exog_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions using Ensemble
        
        Args:
            horizon: Forecast horizon (number of periods)
            exog_data: Not used for Ensemble
            
        Returns:
            Predictions array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {horizon}-step forecast with {self.model_name}")
        
        forecast = self.model.predict(steps=horizon)
        predictions = forecast['fcst'].values
        
        return predictions
