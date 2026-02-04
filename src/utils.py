"""
FraudGuard - Utility Functions
Helper functions for the entire project
"""

import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime


def load_data(filepath: str, nrows: int = None) -> pd.DataFrame:
    """
     Loads CSV data
    
    Args:
        filepath: Path to the CSV file
        nrows: Optional – number of rows to load (for quick testing)
    
    Returns:
        DataFrame
    """
    print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(filepath, nrows=nrows)
    
    print(f"✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    return df


def temporal_train_test_split(
    df: pd.DataFrame, 
    time_col: str = 'trans_date_trans_time',
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
   Temporal train/test split – important for time series data
    First 70% = training set, last 30% = test set
    
    Args:
        df: DataFrame
        time_col: Name of the time column
        train_ratio: Proportion of training data (default 0.7)
    
    Returns:
        train_df, test_df
    """
    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train: {len(train_df):,} rows ({train_ratio:.0%})")
    print(f"Test:  {len(test_df):,} rows ({1-train_ratio:.0%})")
    
    return train_df, test_df


def calculate_haversine_distance(
    lat1: float, 
    lon1: float, 
    lat2: float, 
    lon2: float
) -> float:
    """
    Calculates the distance between two geographic coordinates in kilometers
    (Haversine formula)
    
    Args:
        lat1, lon1: Coordinates of point 1
        lat2, lon2: Coordinates of point 2
    
    Returns:
        Distance in kilometers
    """
    # Radius of the Earth in kilometers
    R = 6371.0
    
    # Convert degrees to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine Formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    distance = R * c
    
    return distance


def print_fraud_stats(df: pd.DataFrame, label_col: str = 'is_fraud'):
    """
    Prints fraud statistics
    
    Args:
        df: DataFrame
        label_col: Name of the label column
    """
    total = len(df)
    fraud_count = df[label_col].sum()
    fraud_rate = fraud_count / total
    
    print("="*50)
    print("FRAUD STATISTICS")
    print("="*50)
    print(f"Total Transactions: {total:,}")
    print(f"Fraud Cases:        {fraud_count:,}")
    print(f"Fraud Rate:         {fraud_rate:.2%}")
    print(f"Legitimate Cases:   {total - fraud_count:,}")
    print("="*50)


def format_currency(amount: float) -> str:
    """Formats a numeric amount as currency"""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Formats a numeric value as a percentage"""
    return f"{value:.2%}"


if __name__ == "__main__":
    # Test
    print("FraudGuard Utils - Ready!")
    
    # Test Haversine
    # Portland, OR to Miami, FL
    distance = calculate_haversine_distance(45.5152, -122.6784, 25.7617, -80.1918)
    print(f"Portland → Miami: {distance:.0f} km")
