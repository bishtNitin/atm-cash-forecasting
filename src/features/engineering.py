"""
Feature Engineering Module for ATM Cash Forecasting
Handles multivariate covariance, lags, and holiday features
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging
import holidays
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LagFeatureGenerator:
    """Generate lag features for time series"""
    
    def __init__(self, lag_periods: List[int] = [1, 7, 14, 30]):
        """
        Initialize LagFeatureGenerator
        
        Args:
            lag_periods: List of lag periods to create
        """
        self.lag_periods = lag_periods
        
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           atm_id_col: str = 'atm_id') -> pd.DataFrame:
        """
        Create lag features for target column
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            atm_id_col: ATM identifier column
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in self.lag_periods:
            col_name = f'{target_col}_lag_{lag}'
            df[col_name] = df.groupby(atm_id_col)[target_col].shift(lag)
            logger.info(f"Created lag feature: {col_name}")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str,
                               windows: List[int] = [7, 14, 30],
                               atm_id_col: str = 'atm_id') -> pd.DataFrame:
        """
        Create rolling window features (mean, std, min, max)
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            windows: List of window sizes
            atm_id_col: ATM identifier column
            
        Returns:
            DataFrame with rolling features
        """
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            col_name = f'{target_col}_roll_mean_{window}'
            df[col_name] = df.groupby(atm_id_col)[target_col].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            # Rolling std
            col_name = f'{target_col}_roll_std_{window}'
            df[col_name] = df.groupby(atm_id_col)[target_col].rolling(
                window=window, min_periods=1
            ).std().reset_index(0, drop=True)
            
            # Rolling min
            col_name = f'{target_col}_roll_min_{window}'
            df[col_name] = df.groupby(atm_id_col)[target_col].rolling(
                window=window, min_periods=1
            ).min().reset_index(0, drop=True)
            
            # Rolling max
            col_name = f'{target_col}_roll_max_{window}'
            df[col_name] = df.groupby(atm_id_col)[target_col].rolling(
                window=window, min_periods=1
            ).max().reset_index(0, drop=True)
            
            logger.info(f"Created rolling features for window: {window}")
        
        return df


class HolidayFeatureGenerator:
    """Generate holiday-related features for Indian holidays"""
    
    def __init__(self, country: str = 'IN'):
        """
        Initialize HolidayFeatureGenerator
        
        Args:
            country: Country code for holidays
        """
        self.country = country
        
    def create_holiday_features(self, df: pd.DataFrame, 
                               date_col: str = 'date') -> pd.DataFrame:
        """
        Create holiday features
        
        Args:
            df: Input DataFrame
            date_col: Date column name
            
        Returns:
            DataFrame with holiday features
        """
        df = df.copy()
        
        # Get years range
        years = pd.to_datetime(df[date_col]).dt.year.unique()
        
        # Create holiday calendar
        holiday_dict = {}
        for year in years:
            holiday_dict.update(holidays.country_holidays(self.country, years=int(year)))
        
        # Is holiday
        df['is_holiday'] = df[date_col].apply(
            lambda x: 1 if pd.to_datetime(x).date() in holiday_dict else 0
        )
        
        # Days to/from holiday
        df['days_to_holiday'] = 0
        df['days_from_holiday'] = 0
        
        for idx, row in df.iterrows():
            date = pd.to_datetime(row[date_col]).date()
            
            # Find next holiday
            future_holidays = [h for h in holiday_dict.keys() if h > date]
            if future_holidays:
                next_holiday = min(future_holidays)
                df.at[idx, 'days_to_holiday'] = (next_holiday - date).days
            
            # Find previous holiday
            past_holidays = [h for h in holiday_dict.keys() if h < date]
            if past_holidays:
                prev_holiday = max(past_holidays)
                df.at[idx, 'days_from_holiday'] = (date - prev_holiday).days
        
        # Day of week
        df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek
        
        # Is weekend
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Month
        df['month'] = pd.to_datetime(df[date_col]).dt.month
        
        # Quarter
        df['quarter'] = pd.to_datetime(df[date_col]).dt.quarter
        
        # Day of month
        df['day_of_month'] = pd.to_datetime(df[date_col]).dt.day
        
        # Week of year
        df['week_of_year'] = pd.to_datetime(df[date_col]).dt.isocalendar().week
        
        logger.info("Created holiday and temporal features")
        return df


class CovarianceFeatureGenerator:
    """Generate multivariate covariance features"""
    
    def __init__(self, window: int = 30):
        """
        Initialize CovarianceFeatureGenerator
        
        Args:
            window: Rolling window for covariance calculation
        """
        self.window = window
        
    def create_covariance_features(self, df: pd.DataFrame, 
                                   target_col: str,
                                   atm_id_col: str = 'atm_id',
                                   n_top_atms: int = 5) -> pd.DataFrame:
        """
        Create covariance features between ATMs
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            atm_id_col: ATM identifier column
            n_top_atms: Number of top correlated ATMs to use
            
        Returns:
            DataFrame with covariance features
        """
        df = df.copy()
        
        # Pivot data to have ATMs as columns
        pivot_df = df.pivot(index='date', columns=atm_id_col, values=target_col)
        
        # Calculate rolling correlation for each ATM
        for atm in pivot_df.columns:
            # Calculate correlation with all other ATMs
            correlations = pivot_df.corrwith(pivot_df[atm], axis=0)
            correlations = correlations.drop(atm).sort_values(ascending=False)
            
            # Get top correlated ATMs
            top_corr_atms = correlations.head(n_top_atms).index.tolist()
            
            # Create average of top correlated ATMs as feature
            if top_corr_atms:
                avg_col_name = f'{target_col}_corr_avg_{atm}'
                pivot_df[avg_col_name] = pivot_df[top_corr_atms].mean(axis=1)
        
        # Melt back to original format
        result_df = pivot_df.reset_index().melt(
            id_vars=['date'], 
            var_name=atm_id_col,
            value_name=target_col
        )
        
        # Merge with original dataframe
        df = df.merge(result_df, on=['date', atm_id_col], how='left', suffixes=('', '_new'))
        
        logger.info("Created covariance features")
        return df


class FeatureEngineering:
    """Main feature engineering pipeline"""
    
    def __init__(self, config: Dict):
        """
        Initialize FeatureEngineering
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.lag_generator = LagFeatureGenerator(
            lag_periods=config.get('lag_periods', [1, 7, 14, 30])
        )
        self.holiday_generator = HolidayFeatureGenerator(
            country=config.get('country', 'IN')
        )
        self.covariance_generator = CovarianceFeatureGenerator(
            window=config.get('covariance_window', 30)
        )
        
    def engineer_features(self, df: pd.DataFrame, 
                         target_col: str = 'cash_demand',
                         atm_id_col: str = 'atm_id') -> pd.DataFrame:
        """
        Run full feature engineering pipeline
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            atm_id_col: ATM identifier column
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering")
        
        # Create lag features
        df = self.lag_generator.create_lag_features(df, target_col, atm_id_col)
        
        # Create rolling features
        rolling_windows = self.config.get('rolling_windows', [7, 14, 30])
        df = self.lag_generator.create_rolling_features(
            df, target_col, windows=rolling_windows, atm_id_col=atm_id_col
        )
        
        # Create holiday features if enabled
        if self.config.get('use_holidays', True):
            df = self.holiday_generator.create_holiday_features(df)
        
        # Create covariance features if enabled
        if self.config.get('use_covariance', True):
            try:
                df = self.covariance_generator.create_covariance_features(
                    df, target_col, atm_id_col
                )
            except Exception as e:
                logger.warning(f"Could not create covariance features: {e}")
        
        logger.info("Feature engineering completed")
        return df
