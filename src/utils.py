"""
FraudGuard - Utility Functions
Helper functions for the entire project
"""

import pandas as pd
from typing import Tuple


def load_data(filepath: str, nrows: int = None) -> pd.DataFrame:
    """
    Loads CSV data
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
    Train/test split
    First 70% = training set, last 30% = test set
    """
    # Sort by time
    df = df.sort_values(time_col).reset_index(drop=True)
    
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train: {len(train_df):,} rows ({train_ratio:.0%})")
    print(f"Test:  {len(test_df):,} rows ({1-train_ratio:.0%})")
    
    return train_df, test_df


def print_fraud_stats(df: pd.DataFrame, label_col: str = 'is_fraud'):
    """
    Prints fraud statistics
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


if __name__ == "__main__":
    print("FraudGuard Utility Functions")
    print("=" * 50)
    print("\nFunctions:")
    print("  - load_data(filepath, nrows)")
    print("  - temporal_train_test_split(df, time_col, train_ratio)")
    print("  - print_fraud_stats(df, label_col)")
