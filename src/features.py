"""
FraudGuard - Feature Engineering
Erstellt Features für ML-Modell
"""

import pandas as pd
import numpy as np
from typing import List
from src.utils import calculate_haversine_distance


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt alle Features für ML-Training
    
    Args:
        df: DataFrame mit Transaktionen (mit Regeln!)
        
    Returns:
        DataFrame mit zusätzlichen Features
    """
    print("Engineering features for ML...")
    
    df = df.copy()
    
    # === 1. Zeit-Features ===
    print("  1. Time features...")
    df = _create_time_features(df)
    
    # === 2. Geo-Features ===
    print("  2. Geographic features...")
    df = _create_geo_features(df)
    
    # === 3. Aggregierte Features ===
    print("  3. Aggregated features...")
    df = _create_aggregated_features(df)
    
    # === 4. Deviation Features ===
    print("  4. Deviation features...")
    df = _create_deviation_features(df)
    
    # === 5. Kategorische Features ===
    print("  5. Categorical features...")
    df = _create_categorical_features(df)
    
    print(f"✓ Feature engineering complete: {len(df.columns)} total columns")
    
    return df


def _create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Zeit-basierte Features"""
    
    # Parse datetime falls noch nicht vorhanden
    if 'trans_datetime' not in df.columns:
        df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
    
    # Hour (0-23)
    if 'hour' not in df.columns:
        df['hour'] = df['trans_datetime'].dt.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek
    
    # Is weekend (Sat/Sun)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Time of day categories
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
    
    return df


def _create_geo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Geografische Features"""
    
    # Distance zwischen Customer und Merchant
    # (nur wenn noch nicht vorhanden von Rules)
    if 'distance_to_prev' not in df.columns:
        print("    Calculating customer-merchant distances...")
        df['cust_merch_distance'] = df.apply(
            lambda row: calculate_haversine_distance(
                row['lat'], row['long'],
                row['merch_lat'], row['merch_long']
            ),
            axis=1
        )
    
    # Distance Categories
    if 'cust_merch_distance' in df.columns:
        df['is_local'] = (df['cust_merch_distance'] < 10).astype(int)  # <10km
        df['is_regional'] = ((df['cust_merch_distance'] >= 10) & 
                             (df['cust_merch_distance'] < 100)).astype(int)
        df['is_distant'] = (df['cust_merch_distance'] >= 100).astype(int)
    
    return df


def _create_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregierte Features (pro User/Karte)
    Wichtig: Temporal Leakage vermeiden!
    """
    
    # Sortiere nach Zeit
    df = df.sort_values(['cc_num', 'trans_datetime']).reset_index(drop=True)
    
    # === Transaction Count Features ===
    
    # Count bis zu diesem Zeitpunkt (expanding window)
    df['txn_count_total'] = df.groupby('cc_num').cumcount() + 1
    
    # Count letzte 24h (falls noch nicht von Rules)
    if 'txn_count_1h' not in df.columns:
        # Vereinfachte Version: Rolling basierend auf Index
        # In Production: Zeitbasiertes Rolling
        df['txn_count_24h'] = df.groupby('cc_num')['amt'].transform(
            lambda x: x.rolling(window=24, min_periods=1).count()
        )
    
    # === Amount Features ===
    
    # Durchschnitt aller bisherigen Transaktionen (expanding)
    df['avg_amount_expanding'] = df.groupby('cc_num')['amt'].transform(
        lambda x: x.expanding(min_periods=1).mean()
    )
    
    # Std Dev (expanding)
    df['std_amount_expanding'] = df.groupby('cc_num')['amt'].transform(
        lambda x: x.expanding(min_periods=1).std()
    ).fillna(0)
    
    # Min/Max bisherige Transaktionen
    df['min_amount_so_far'] = df.groupby('cc_num')['amt'].transform(
        lambda x: x.expanding(min_periods=1).min()
    )
    df['max_amount_so_far'] = df.groupby('cc_num')['amt'].transform(
        lambda x: x.expanding(min_periods=1).max()
    )
    
    # === Category Features ===
    
    # Häufigste Kategorie des Users (bis jetzt)
    # Vereinfacht: Mode der bisherigen Transaktionen
    # (In Production: Rolling mode)
    
    return df


def _create_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features die Abweichung von User-Normal messen"""
    
    # Amount vs. User Average
    if 'avg_amount_expanding' in df.columns:
        df['amount_vs_avg_ratio'] = df['amt'] / (df['avg_amount_expanding'] + 1)
        
        # Z-Score (wie viele StdDevs entfernt?)
        df['amount_zscore'] = (
            (df['amt'] - df['avg_amount_expanding']) / 
            (df['std_amount_expanding'] + 1)
        )
    
    # Amount vs. Min/Max
    if 'min_amount_so_far' in df.columns:
        df['is_new_max'] = (df['amt'] > df['max_amount_so_far']).astype(int)
        df['is_new_min'] = (df['amt'] < df['min_amount_so_far']).astype(int)
    
    return df


def _create_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kategorische Features encoding"""
    
    # === Gender ===
    df['gender_M'] = (df['gender'] == 'M').astype(int)
    df['gender_F'] = (df['gender'] == 'F').astype(int)
    
    # === State Frequency Encoding ===
    # Wie häufig ist dieser State? (häufiger = weniger verdächtig)
    state_freq = df['state'].value_counts() / len(df)
    df['state_frequency'] = df['state'].map(state_freq)
    
    # === Category Frequency Encoding ===
    category_freq = df['category'].value_counts() / len(df)
    df['category_frequency'] = df['category'].map(category_freq)
    
    # === Job (vereinfacht) ===
    # Nur die häufigsten Jobs als Features
    top_jobs = df['job'].value_counts().head(10).index
    for job in top_jobs:
        df[f'job_{job.replace(" ", "_").lower()}'] = (df['job'] == job).astype(int)
    
    # === City Population (Größe der Stadt) ===
    # Log-Transform für bessere Verteilung
    df['city_pop_log'] = np.log1p(df['city_pop'])
    
    # === Age ===
    # Berechne Alter aus DOB
    if 'dob' in df.columns:
        df['dob_datetime'] = pd.to_datetime(df['dob'])
        df['age'] = (df['trans_datetime'] - df['dob_datetime']).dt.days / 365.25
        df = df.drop('dob_datetime', axis=1)
    
    return df


def select_ml_features(df: pd.DataFrame, include_rules: bool = True) -> List[str]:
    """
    Wählt Features für ML-Training aus
    
    Args:
        df: DataFrame mit allen Features
        include_rules: Ob Regel-Features inkludiert werden sollen
                      True = Hybrid Model
                      False = ML-Only Model
    
    Returns:
        Liste von Feature-Namen
    """
    
    features = []
    
    # === Base Features ===
    base_features = [
        'amt',
        'hour',
        'day_of_week',
        'is_weekend',
        'is_night',
        'is_morning',
        'is_afternoon',
        'is_evening'
    ]
    features.extend([f for f in base_features if f in df.columns])
    
    # === Geo Features ===
    geo_features = [
        'cust_merch_distance',
        'is_local',
        'is_regional',
        'is_distant',
        'distance_to_prev',  # Von Rules
        'velocity_kmh'       # Von Rules
    ]
    features.extend([f for f in geo_features if f in df.columns])
    
    # === Aggregated Features ===
    agg_features = [
        'txn_count_total',
        'txn_count_24h',
        'txn_count_1h',  # Von Rules
        'avg_amount_expanding',
        'std_amount_expanding',
        'min_amount_so_far',
        'max_amount_so_far'
    ]
    features.extend([f for f in agg_features if f in df.columns])
    
    # === Deviation Features ===
    dev_features = [
        'amount_vs_avg_ratio',
        'amount_zscore',
        'is_new_max',
        'is_new_min'
    ]
    features.extend([f for f in dev_features if f in df.columns])
    
    # === Categorical Features ===
    cat_features = [
        'gender_M',
        'gender_F',
        'state_frequency',
        'category_frequency',
        'city_pop_log',
        'age'
    ]
    features.extend([f for f in cat_features if f in df.columns])
    
    # === Rule Features (optional) ===
    if include_rules:
        rule_features = [
            'rule_high_frequency',
            'rule_geographic_impossible',
            'rule_night_transaction',
            'rule_high_amount',
            'rule_out_of_state',
            'rule_round_amount',
            'rule_risky_category',
            'rules_triggered'
        ]
        features.extend([f for f in rule_features if f in df.columns])
    
    # Remove duplicates (falls welche)
    features = list(dict.fromkeys(features))
    
    print(f"\nSelected {len(features)} features for ML:")
    if include_rules:
        print("  Mode: HYBRID (with rule features)")
    else:
        print("  Mode: ML-ONLY (without rule features)")
    
    return features


def prepare_for_ml(df: pd.DataFrame, feature_cols: List[str], label_col: str = 'is_fraud'):
    """
    Bereitet DataFrame für ML vor
    
    Args:
        df: DataFrame
        feature_cols: Liste von Feature-Namen
        label_col: Name der Label-Spalte
    
    Returns:
        X, y (Features, Labels)
    """
    # Nur vorhandene Features
    available_features = [col for col in feature_cols if col in df.columns]
    
    missing = set(feature_cols) - set(available_features)
    if missing:
        print(f"Warning: {len(missing)} features not found: {missing}")
    
    # Features
    X = df[available_features].copy()
    
    # Fill NaN mit 0 (oder andere Strategie)
    X = X.fillna(0)
    
    # Label
    y = df[label_col].copy()
    
    print(f"\nPrepared for ML:")
    print(f"  Features: {X.shape}")
    print(f"  Labels: {y.shape}")
    print(f"  Fraud rate: {y.mean():.2%}")
    
    return X, y


if __name__ == "__main__":
    print("FraudGuard Feature Engineering Module")
    print("=" * 50)
    print("\nFunctions:")
    print("  - engineer_features(df)")
    print("  - select_ml_features(df, include_rules=True/False)")
    print("  - prepare_for_ml(df, feature_cols)")
