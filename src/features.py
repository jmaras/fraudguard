"""
FraudGuard – Feature Engineering (SIMPLIFIED). Creates essential features for the ML model
"""

import pandas as pd
import numpy as np
from typing import List


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates features for ML training
    """
    print("Engineering features for ML...")
    
    df = df.copy()
    
    # 1. Time features
    print("  1. Time features...")
    df = _create_time_features(df)
    
    # === 2. Aggregated features ===
    print("  2. Aggregated features...")
    df = _create_aggregated_features(df)
    
    # === 3. Categorical features ===
    print("  3. Categorical features...")
    df = _create_categorical_features(df)
    
    print(f"✓ Feature engineering complete: {len(df.columns)} total columns")
    
    return df


def _create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-based features"""
    
    # Parse datetime
    if 'trans_datetime' not in df.columns:
        df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
    
    # Hour (0-23)
    if 'hour' not in df.columns:
        df['hour'] = df['trans_datetime'].dt.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek
    
    # Is weekend
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Time of day categories
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    
    return df


def _create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregated features (per user/card)
    """
    
    # Sort by time
    df = df.sort_values(['cc_num', 'trans_datetime']).reset_index(drop=True)
    
    # === Transaction Count ===
    df['txn_count_total'] = df.groupby('cc_num').cumcount() + 1
    
    # === Amount Features ===
    
    # Average of all previous transactions
    df['avg_amount_expanding'] = df.groupby('cc_num')['amt'].transform(
        lambda x: x.expanding(min_periods=1).mean()
    )
    
    # Standard deviation
    df['std_amount_expanding'] = df.groupby('cc_num')['amt'].transform(
        lambda x: x.expanding(min_periods=1).std()
    ).fillna(0)
    
    # === Deviation Features ===
    
    # Amount vs. User Average
    df['amount_vs_avg_ratio'] = df['amt'] / (df['avg_amount_expanding'] + 1)
    
    # Z-score (how many standard deviations away?)
    df['amount_zscore'] = (
        (df['amt'] - df['avg_amount_expanding']) / 
        (df['std_amount_expanding'] + 1)
    )
    
    return df


def _create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Categorical feature encoding"""
    
    # === Gender ===
    df['gender_M'] = (df['gender'] == 'M').astype(int)
    
    # === State frequency encoding ===
    state_freq = df['state'].value_counts() / len(df)
    df['state_frequency'] = df['state'].map(state_freq)
    
    # === Category frequency encoding ===
    category_freq = df['category'].value_counts() / len(df)
    df['category_frequency'] = df['category'].map(category_freq)
    
    # === City population (Log Transform) ===
    df['city_pop_log'] = np.log1p(df['city_pop'])
    
    # === Age ===
    if 'dob' in df.columns:
        df['dob_datetime'] = pd.to_datetime(df['dob'])
        df['age'] = (df['trans_datetime'] - df['dob_datetime']).dt.days / 365.25
        df = df.drop('dob_datetime', axis=1)
    
    return df


def select_ml_features(df: pd.DataFrame, include_rules: bool = False) -> List[str]:
    """
    Selects features for ML training (simplified)
    """
    
    features = []
    
    # === Base features ===
    base_features = [
        'amt',
        'hour',
        'day_of_week',
        'is_weekend',
        'is_night',
        'is_morning'
    ]
    features.extend([f for f in base_features if f in df.columns])
    
    # === Aggregated features ===
    agg_features = [
        'txn_count_total',
        'avg_amount_expanding',
        'std_amount_expanding',
        'amount_vs_avg_ratio',
        'amount_zscore'
    ]
    features.extend([f for f in agg_features if f in df.columns])
    
    # === Categorical features ===
    cat_features = [
        'gender_M',
        'state_frequency',
        'category_frequency',
        'city_pop_log',
        'age'
    ]
    features.extend([f for f in cat_features if f in df.columns])
    
    # === Rule features ===
    if include_rules:
        rule_features = [
            'rule_high_frequency',
            'rule_night_transaction',
            'rule_high_amount',
            'rule_round_amount',
            'rule_risky_category',
            'rules_triggered'
        ]
        features.extend([f for f in rule_features if f in df.columns])
    
    # Remove duplicates
    features = list(dict.fromkeys(features))
    
    print(f"\nSelected {len(features)} features for ML:")
    if include_rules:
        print("  Mode: HYBRID (with rule features)")
    else:
        print("  Mode: ML-ONLY (without rule features)")
    
    return features


def prepare_for_ml(df: pd.DataFrame, feature_cols: List[str], label_col: str = 'is_fraud'):
    """
    Prepares DataFrame for ML    
    Returns:
        X, y (Features, Labels)
    """
    # Only existing features
    available_features = [col for col in feature_cols if col in df.columns]
    
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"Warning: {len(missing)} features not found")
    
    # Features
    X = df[available_features].copy()
    
    # Fill NaN with 0
    X = X.fillna(0)
    
    # Label
    y = df[label_col].copy()
    
    print(f"\nPrepared for ML:")
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Fraud rate: {y.mean():.2%}")
    
    return X, y


if __name__ == "__main__":
    print("FraudGuard Feature Engineering Module (Simplified)")
    print("=" * 50)
    print("\nFunctions:")
    print("  - engineer_features(df)")
    print("  - select_ml_features(df, include_rules=True/False)")
    print("  - prepare_for_ml(df, feature_cols)")
