"""
Data Pipeline for ATM Cash Forecasting
Handles data ingestion, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from scipy import interpolate
from datetime import datetime, timedelta
import holidays
from typing import List, Dict, Optional
import os


class ATMDataPipeline:
    """Pipeline for processing ATM transaction data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indian_holidays = holidays.India(years=range(2020, 2026))
        
    def ingest_csv(self, file_path: str) -> pd.DataFrame:
        """
        Ingest daily CSV file with ATM transaction data
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with transaction data
        """
        df = pd.read_csv(file_path)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def clean_data(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """
        Clean data and impute missing values
        
        Args:
            df: Input DataFrame
            method: Imputation method ('linear' or 'spline')
            
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Sort by date and ATM ID
        if 'atm_id' in df.columns:
            df = df.sort_values(['atm_id', 'date'])
        else:
            df = df.sort_values('date')
        
        # Identify numeric columns for imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'linear':
            # Linear interpolation
            for col in numeric_cols:
                if df[col].isna().any():
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        elif method == 'spline':
            # Spline interpolation
            for col in numeric_cols:
                if df[col].isna().any():
                    # Only use spline if we have enough non-null values
                    if df[col].notna().sum() >= 4:
                        df[col] = df[col].interpolate(method='spline', order=3, limit_direction='both')
                    else:
                        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        
        # Fill any remaining missing values with forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = 'cash_withdrawn') -> pd.DataFrame:
        """
        Create features for model training
        
        Args:
            df: Input DataFrame with date and target column
            target_col: Name of target column
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Holiday features
        df['is_holiday'] = df['date'].apply(lambda x: 1 if x in self.indian_holidays else 0)
        
        # Payday features (typically 1st, 15th, and last day of month)
        df['is_payday'] = df['day_of_month'].apply(
            lambda x: 1 if x in [1, 15] or x >= 28 else 0
        )
        
        # Start/End of month
        df['is_month_start'] = (df['day_of_month'] <= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 26).astype(int)
        
        # Festival seasons (major Indian festivals)
        df['is_diwali_season'] = ((df['month'] == 10) | (df['month'] == 11)).astype(int)
        df['is_holi_season'] = (df['month'] == 3).astype(int)
        
        # Lag features (if target column exists)
        if target_col in df.columns:
            if 'atm_id' in df.columns:
                # Create lags per ATM
                df['lag_1'] = df.groupby('atm_id')[target_col].shift(1)
                df['lag_7'] = df.groupby('atm_id')[target_col].shift(7)
                df['lag_30'] = df.groupby('atm_id')[target_col].shift(30)
                
                # Rolling statistics
                df['rolling_mean_7'] = df.groupby('atm_id')[target_col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )
                df['rolling_std_7'] = df.groupby('atm_id')[target_col].transform(
                    lambda x: x.rolling(window=7, min_periods=1).std()
                )
                df['rolling_mean_30'] = df.groupby('atm_id')[target_col].transform(
                    lambda x: x.rolling(window=30, min_periods=1).mean()
                )
                
                # Exponential moving average
                df['ema_7'] = df.groupby('atm_id')[target_col].transform(
                    lambda x: x.ewm(span=7, adjust=False).mean()
                )
            else:
                # Single time series
                df['lag_1'] = df[target_col].shift(1)
                df['lag_7'] = df[target_col].shift(7)
                df['lag_30'] = df[target_col].shift(30)
                df['rolling_mean_7'] = df[target_col].rolling(window=7, min_periods=1).mean()
                df['rolling_std_7'] = df[target_col].rolling(window=7, min_periods=1).std()
                df['rolling_mean_30'] = df[target_col].rolling(window=30, min_periods=1).mean()
                df['ema_7'] = df[target_col].ewm(span=7, adjust=False).mean()
        
        return df
    
    def calculate_covariance_features(self, df: pd.DataFrame, 
                                     atm_groups: Optional[List[List[str]]] = None) -> pd.DataFrame:
        """
        Calculate multivariate covariance features between ATMs
        
        Args:
            df: DataFrame with ATM data
            atm_groups: List of ATM groups for covariance calculation
            
        Returns:
            DataFrame with covariance features
        """
        df = df.copy()
        
        if atm_groups and 'atm_id' in df.columns:
            for group_idx, group in enumerate(atm_groups):
                # Calculate mean cash withdrawn for the group
                group_data = df[df['atm_id'].isin(group)]
                group_mean = group_data.groupby('date')['cash_withdrawn'].mean()
                
                # Merge back to main dataframe
                df[f'group_{group_idx}_mean'] = df['date'].map(group_mean)
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, 
                             target_col: str = 'cash_withdrawn') -> tuple:
        """
        Prepare data for model training
        
        Args:
            df: Processed DataFrame with features
            target_col: Name of target column
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        # Exclude non-feature columns
        exclude_cols = ['date', target_col, 'atm_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove rows with NaN in features or target
        df_clean = df.dropna(subset=feature_cols + [target_col])
        
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        
        return X, y, feature_cols
    
    def generate_sample_data(self, output_path: str, 
                            n_atms: int = 100, 
                            n_days: int = 365) -> str:
        """
        Generate sample ATM transaction data for testing
        
        Args:
            output_path: Directory to save the CSV file
            n_atms: Number of ATMs to simulate
            n_days: Number of days of data
            
        Returns:
            Path to generated CSV file
        """
        np.random.seed(42)
        
        # Generate date range
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_days)]
        
        # Generate data for each ATM
        data = []
        for atm_idx in range(n_atms):
            atm_id = f"ATM_{atm_idx + 1:04d}"
            
            # Base cash withdrawal amount (varies by ATM)
            base_amount = np.random.uniform(500000, 1500000)
            
            # Add location (random coordinates in India)
            latitude = np.random.uniform(8.0, 35.0)
            longitude = np.random.uniform(68.0, 97.0)
            
            for date in dates:
                # Seasonal pattern
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)
                
                # Weekly pattern (more on weekdays)
                weekly_factor = 1.2 if date.weekday() < 5 else 0.8
                
                # Payday boost
                payday_factor = 1.5 if date.day in [1, 15] or date.day >= 28 else 1.0
                
                # Holiday impact
                holiday_factor = 1.3 if date in holidays.India(years=date.year) else 1.0
                
                # Random noise
                noise = np.random.normal(1.0, 0.1)
                
                # Calculate cash withdrawn
                cash_withdrawn = base_amount * seasonal_factor * weekly_factor * payday_factor * holiday_factor * noise
                cash_withdrawn = max(0, cash_withdrawn)  # Ensure non-negative
                
                # Number of transactions
                n_transactions = int(np.random.poisson(50))
                
                data.append({
                    'atm_id': atm_id,
                    'date': date.strftime('%Y-%m-%d'),
                    'cash_withdrawn': round(cash_withdrawn, 2),
                    'n_transactions': n_transactions,
                    'latitude': round(latitude, 4),
                    'longitude': round(longitude, 4)
                })
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        
        os.makedirs(output_path, exist_ok=True)
        file_path = os.path.join(output_path, 'atm_transactions.csv')
        df.to_csv(file_path, index=False)
        
        print(f"Generated sample data: {len(df)} rows, {n_atms} ATMs, {n_days} days")
        print(f"Saved to: {file_path}")
        
        return file_path


if __name__ == "__main__":
    # Example usage
    config = {
        'data': {
            'raw_data_path': 'data/raw',
            'processed_data_path': 'data/processed'
        }
    }
    
    pipeline = ATMDataPipeline(config)
    
    # Generate sample data
    data_path = pipeline.generate_sample_data('data/raw', n_atms=100, n_days=365)
    
    # Load and process
    df = pipeline.ingest_csv(data_path)
    df_clean = pipeline.clean_data(df, method='linear')
    df_features = pipeline.engineer_features(df_clean, target_col='cash_withdrawn')
    
    print(f"\nProcessed data shape: {df_features.shape}")
    print(f"Features: {df_features.columns.tolist()}")
