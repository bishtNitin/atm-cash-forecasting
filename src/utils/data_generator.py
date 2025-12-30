"""
Generate synthetic ATM cash demand data for demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_data(
    n_atms: int = 100,
    start_date: str = '2022-01-01',
    end_date: str = '2023-12-31',
    output_path: str = 'data/raw/synthetic_atm_data.csv'
) -> pd.DataFrame:
    """
    Generate synthetic ATM cash demand data
    
    Args:
        n_atms: Number of ATMs
        start_date: Start date
        end_date: End date
        output_path: Path to save CSV
        
    Returns:
        DataFrame with synthetic data
    """
    logger.info(f"Generating synthetic data for {n_atms} ATMs")
    
    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    
    for atm_id in range(1, n_atms + 1):
        # Base demand varies by ATM
        base_demand = np.random.uniform(50000, 200000)
        
        # Daily variation
        for date in dates:
            # Trend component (slight growth over time)
            trend = base_demand * (1 + 0.0001 * (date - dates[0]).days)
            
            # Seasonal component (weekly pattern)
            day_of_week = date.dayofweek
            weekly_pattern = 1.0
            if day_of_week in [0, 4, 5]:  # Monday, Friday, Saturday - higher demand
                weekly_pattern = 1.3
            elif day_of_week in [2, 6]:  # Wednesday, Sunday - lower demand
                weekly_pattern = 0.8
            
            # Monthly pattern (higher at month end)
            day_of_month = date.day
            monthly_pattern = 1.0
            if day_of_month >= 25 or day_of_month <= 5:
                monthly_pattern = 1.4
            
            # Random noise
            noise = np.random.normal(1.0, 0.15)
            
            # Calculate demand
            demand = trend * weekly_pattern * monthly_pattern * noise
            demand = max(0, demand)  # Ensure non-negative
            
            # Add occasional missing values (2% missing)
            if np.random.random() < 0.02:
                demand = np.nan
            
            data.append({
                'date': date,
                'atm_id': f'ATM_{atm_id:04d}',
                'cash_demand': demand
            })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic data saved to {output_path}")
    
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Generate data for 100 ATMs (smaller subset for demo)
    df = generate_synthetic_data(
        n_atms=100,
        start_date='2022-01-01',
        end_date='2023-12-31',
        output_path='data/raw/synthetic_atm_data.csv'
    )
    
    print(f"Generated {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of ATMs: {df['atm_id'].nunique()}")
    print(f"\nSample data:\n{df.head(10)}")
