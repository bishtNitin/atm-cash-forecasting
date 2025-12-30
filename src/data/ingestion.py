"""
Data Ingestion Module for ATM Cash Forecasting
Handles CSV ingestion, cleaning, and imputation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
import logging
from scipy.interpolate import interp1d
from datetime import datetime

logger = logging.getLogger(__name__)


class DataIngestion:
    """Handles ingestion of daily CSV files for ATM cash data"""
    
    def __init__(self, data_path: str):
        """
        Initialize DataIngestion
        
        Args:
            data_path: Path to directory containing CSV files
        """
        self.data_path = Path(data_path)
        
    def load_csv(self, filename: str) -> pd.DataFrame:
        """
        Load a single CSV file
        
        Args:
            filename: Name of CSV file
            
        Returns:
            DataFrame with loaded data
        """
        filepath = self.data_path / filename
        logger.info(f"Loading CSV: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
            df = df.drop('Date', axis=1)
            
        return df
    
    def load_multiple_csvs(self, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load multiple CSV files and concatenate
        
        Args:
            pattern: Glob pattern for matching CSV files
            
        Returns:
            Concatenated DataFrame
        """
        csv_files = list(self.data_path.glob(pattern))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        dfs = []
        for filepath in csv_files:
            df = pd.read_csv(filepath)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'])
                df = df.drop('Date', axis=1)
            dfs.append(df)
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_df = combined_df.sort_values('date').reset_index(drop=True)
            return combined_df
        else:
            raise ValueError("No CSV files found")


class DataCleaner:
    """Handles data cleaning and imputation"""
    
    def __init__(self, method: str = 'linear'):
        """
        Initialize DataCleaner
        
        Args:
            method: Imputation method ('linear', 'spline', 'forward', 'backward')
        """
        self.method = method
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values, outliers, etc.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Log missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        if missing_pct.any():
            logger.info(f"Missing values percentage:\n{missing_pct[missing_pct > 0]}")
        
        return df
    
    def impute_missing(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Impute missing values using specified method
        
        Args:
            df: Input DataFrame
            columns: Columns to impute (if None, impute all numeric columns)
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [c for c in columns if c != 'date']
        
        for col in columns:
            if df[col].isnull().any():
                logger.info(f"Imputing missing values in {col} using {self.method}")
                
                if self.method == 'linear':
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                elif self.method == 'spline':
                    df[col] = df[col].interpolate(method='spline', order=3, limit_direction='both')
                elif self.method == 'forward':
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                elif self.method == 'backward':
                    df[col] = df[col].fillna(method='bfill').fillna(method='ffill')
                else:
                    # Default to linear
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers using IQR or z-score method
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            method: 'iqr' or 'zscore'
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            columns = [c for c in columns if c != 'date']
        
        for col in columns:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df


class DataPreprocessor:
    """Main data preprocessing pipeline"""
    
    def __init__(self, imputation_method: str = 'linear'):
        """
        Initialize DataPreprocessor
        
        Args:
            imputation_method: Method for imputation
        """
        self.cleaner = DataCleaner(method=imputation_method)
        
    def preprocess(self, df: pd.DataFrame, handle_outliers: bool = True) -> pd.DataFrame:
        """
        Run full preprocessing pipeline
        
        Args:
            df: Input DataFrame
            handle_outliers: Whether to handle outliers
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing")
        
        # Clean data
        df = self.cleaner.clean_data(df)
        
        # Impute missing values
        df = self.cleaner.impute_missing(df)
        
        # Handle outliers if requested
        if handle_outliers:
            df = self.cleaner.handle_outliers(df, method='iqr', threshold=3.0)
        
        logger.info("Data preprocessing completed")
        return df
