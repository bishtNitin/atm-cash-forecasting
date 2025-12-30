"""
Utility functions for ATM Cash Forecasting
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np


def setup_logging(log_file: str = 'logs/atm_forecasting.log', level: int = logging.INFO):
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = 'config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_predictions(predictions: pd.DataFrame, output_path: str):
    """
    Save predictions to CSV
    
    Args:
        predictions: DataFrame with predictions
        output_path: Output file path
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")


def load_predictions(input_path: str) -> pd.DataFrame:
    """
    Load predictions from CSV
    
    Args:
        input_path: Input file path
        
    Returns:
        DataFrame with predictions
    """
    predictions = pd.read_csv(input_path)
    if 'date' in predictions.columns:
        predictions['date'] = pd.to_datetime(predictions['date'])
    return predictions
