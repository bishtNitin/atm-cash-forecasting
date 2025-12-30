"""
Base model interface for ATM Cash Forecasting
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models"""
    
    def __init__(self, model_name: str, config: Optional[Dict] = None):
        """
        Initialize base forecaster
        
        Args:
            model_name: Name of the model
            config: Model configuration
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, target_col: str = 'cash_demand'):
        """
        Fit the model
        
        Args:
            train_data: Training DataFrame
            target_col: Target column name
        """
        pass
    
    @abstractmethod
    def predict(self, horizon: int, exog_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Make predictions
        
        Args:
            horizon: Forecast horizon
            exog_data: Exogenous variables
            
        Returns:
            Predictions array
        """
        pass
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'config': self.config,
            'is_fitted': self.is_fitted
        }
