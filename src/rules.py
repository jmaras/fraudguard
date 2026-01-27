"""
FraudGuard - Rule Engine
7 Business-Regeln zur Betrugserkennung
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src.utils import calculate_haversine_distance


class FraudRuleEngine:
    """
    Regelbasiertes Fraud Detection System
    
    7 Regeln:
    1. High Frequency: >5 Transaktionen pro Stunde
    2. Geographic Impossible: Velocity >500 km/h
    3. Night Transaction: 2-5 Uhr morgens
    4. High Amount: >3x User-Durchschnitt
    5. Out-of-State: Anderer State als üblich
    6. Round Amount: Verdächtig runde Beträge
    7. Risky Category: Kategorien mit hoher Fraud-Rate
    """
    
    def __init__(self):
        # Kategorien mit historisch hoher Fraud-Rate
        # (werden später aus EDA angepasst)
        self.risky_categories = [
            'shopping_net',
            'misc_net',
            'grocery_pos',
            'shopping_pos'
        ]
        
        # Verdächtig runde Beträge
        self.suspicious_amounts = [
            50, 100, 200, 250, 500, 750, 1000, 
            1500, 2000, 2500, 5000
        ]
    
    
    def apply_all_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wendet alle 7 Regeln auf DataFrame an
        
        Args:
            df: DataFrame mit Transaktionen
            
        Returns:
            DataFrame mit zusätzlichen Regel-Spalten
        """
        print("Applying fraud detection rules...")
        
        # Kopie erstellen
        df = df.copy()
        
        # Parse Zeit (falls noch nicht vorhanden)
        if 'trans_datetime' not in df.columns:
            df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour'] = df['trans_datetime'].dt.hour
        
        # Regel 1: High Frequency
        df = self._rule_high_frequency(df)
        
        # Regel 2: Geographic Impossible
        df = self._rule_geographic_impossible(df)
        
        # Regel 3: Night Transaction
        df = self._rule_night_transaction(df)
        
        # Regel 4: High Amount
        df = self._rule_high_amount(df)
        
        # Regel 5: Out-of-State
        df = self._rule_out_of_state(df)
        
        # Regel 6: Round Amount
        df = self._rule_round_amount(df)
        
        # Regel 7: Risky Category
        df = self._rule_risky_category(df)
        
        # Zähle getriggerte Regeln
        rule_cols = [col for col in df.columns if col.startswith('rule_')]
        df['rules_triggered'] = df[rule_cols].sum(axis=1)
        
        # Finale Entscheidung: ≥2 Regeln → Fraud
        df['rule_based_prediction'] = (df['rules_triggered'] >= 2).astype(int)
        
        print(f"✓ Applied {len(rule_cols)} rules")
        print(f"  Transactions flagged (≥2 rules): {df['rule_based_prediction'].sum():,}")
        
        return df
    
    
    def _rule_high_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 1: High Frequency
        >5 Transaktionen pro Stunde von derselben Karte
        """
        # Runde Zeit auf Stunde
        df['hour_bucket'] = df['trans_datetime'].dt.floor('H')
        
        # Zähle Transaktionen pro cc_num + Stunde
        txn_counts = df.groupby(['cc_num', 'hour_bucket']).size().reset_index(name='txn_count_1h')
        
        # Merge zurück
        df = df.merge(txn_counts, on=['cc_num', 'hour_bucket'], how='left')
        
        # Regel: >5 Transaktionen
        df['rule_high_frequency'] = (df['txn_count_1h'] > 5).astype(int)
        
        # Cleanup
        df = df.drop('hour_bucket', axis=1)
        
        triggered = df['rule_high_frequency'].sum()
        print(f"  Rule 1 (High Frequency): {triggered:,} triggered")
        
        return df
    
    
    def _rule_geographic_impossible(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 2: Geographic Impossible
        Velocity >500 km/h zwischen Transaktionen (schneller als Flugzeug!)
        """
        # Sortiere nach Karte + Zeit
        df = df.sort_values(['cc_num', 'trans_datetime']).reset_index(drop=True)
        
        # Berechne Distance zur vorherigen Transaktion (pro Karte)
        df['prev_lat'] = df.groupby('cc_num')['lat'].shift(1)
        df['prev_long'] = df.groupby('cc_num')['long'].shift(1)
        df['prev_time'] = df.groupby('cc_num')['trans_datetime'].shift(1)
        
        # Nur wenn vorherige Transaktion existiert
        mask = df['prev_lat'].notna()
        
        # Berechne Distance (Haversine)
        df.loc[mask, 'distance_to_prev'] = df.loc[mask].apply(
            lambda row: calculate_haversine_distance(
                row['prev_lat'], row['prev_long'],
                row['lat'], row['long']
            ),
            axis=1
        )
        
        # Berechne Zeit-Differenz in Stunden
        df.loc[mask, 'time_diff_hours'] = (
            (df.loc[mask, 'trans_datetime'] - df.loc[mask, 'prev_time']).dt.total_seconds() / 3600
        )
        
        # Berechne Velocity (km/h)
        # Nur wenn time_diff > 0 (nicht exakt gleiche Zeit)
        velocity_mask = mask & (df['time_diff_hours'] > 0)
        df.loc[velocity_mask, 'velocity_kmh'] = (
            df.loc[velocity_mask, 'distance_to_prev'] / df.loc[velocity_mask, 'time_diff_hours']
        )
        
        # Regel: Velocity >500 km/h (unmöglich!)
        df['rule_geographic_impossible'] = (df['velocity_kmh'] > 500).fillna(0).astype(int)
        
        # Cleanup
        df = df.drop(['prev_lat', 'prev_long', 'prev_time'], axis=1)
        
        triggered = df['rule_geographic_impossible'].sum()
        print(f"  Rule 2 (Geographic Impossible): {triggered:,} triggered")
        
        return df
    
    
    def _rule_night_transaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 3: Night Transaction
        Transaktionen zwischen 2-5 Uhr morgens sind verdächtig
        """
        df['rule_night_transaction'] = (
            (df['hour'] >= 2) & (df['hour'] < 5)
        ).astype(int)
        
        triggered = df['rule_night_transaction'].sum()
        print(f"  Rule 3 (Night Transaction): {triggered:,} triggered")
        
        return df
    
    
    def _rule_high_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 4: High Amount
        Betrag >3x Durchschnitt des Users
        """
        # Berechne User-Durchschnitt (pro cc_num)
        user_avg = df.groupby('cc_num')['amt'].transform('mean')
        
        # Regel: >3x Durchschnitt
        df['rule_high_amount'] = (df['amt'] > 3 * user_avg).astype(int)
        
        triggered = df['rule_high_amount'].sum()
        print(f"  Rule 4 (High Amount): {triggered:,} triggered")
        
        return df
    
    
    def _rule_out_of_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 5: Out-of-State
        Transaktion in anderem State als üblich für diesen User
        """
        # Häufigster State pro User
        user_home_state = df.groupby('cc_num')['state'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else None
        ).to_dict()
        
        # Map zurück
        df['home_state'] = df['cc_num'].map(user_home_state)
        
        # Regel: Aktueller State != Home State
        df['rule_out_of_state'] = (df['state'] != df['home_state']).astype(int)
        
        # Cleanup
        df = df.drop('home_state', axis=1)
        
        triggered = df['rule_out_of_state'].sum()
        print(f"  Rule 5 (Out-of-State): {triggered:,} triggered")
        
        return df
    
    
    def _rule_round_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 6: Round Amount
        Verdächtig runde Beträge (z.B. genau 100, 500, 1000)
        """
        df['rule_round_amount'] = df['amt'].isin(self.suspicious_amounts).astype(int)
        
        triggered = df['rule_round_amount'].sum()
        print(f"  Rule 6 (Round Amount): {triggered:,} triggered")
        
        return df
    
    
    def _rule_risky_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 7: Risky Category
        Kategorien mit historisch hoher Fraud-Rate
        """
        df['rule_risky_category'] = df['category'].isin(self.risky_categories).astype(int)
        
        triggered = df['rule_risky_category'].sum()
        print(f"  Rule 7 (Risky Category): {triggered:,} triggered")
        
        return df
    
    
    def get_rule_explanations(self) -> Dict[str, str]:
        """
        Gibt Erklärungen für jede Regel zurück
        
        Returns:
            Dictionary mit Regel-Namen und Erklärungen
        """
        return {
            'rule_high_frequency': 'More than 5 transactions per hour',
            'rule_geographic_impossible': 'Velocity >500 km/h (physically impossible)',
            'rule_night_transaction': 'Transaction between 2-5 AM',
            'rule_high_amount': 'Amount >3x user average',
            'rule_out_of_state': 'Transaction in different state than usual',
            'rule_round_amount': f'Suspicious round amount ({self.suspicious_amounts})',
            'rule_risky_category': f'High-risk category ({self.risky_categories})'
        }


def evaluate_rules(df: pd.DataFrame, label_col: str = 'is_fraud') -> pd.DataFrame:
    """
    Evaluiert Performance der Regeln
    
    Args:
        df: DataFrame mit Regeln und Labels
        label_col: Name der Label-Spalte
    
    Returns:
        DataFrame mit Rule Performance Metrics
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    results = []
    
    rule_cols = [col for col in df.columns if col.startswith('rule_') and col != 'rules_triggered']
    
    # Individual Rules
    for rule in rule_cols:
        triggered = df[df[rule] == 1]
        
        if len(triggered) > 0:
            precision = precision_score(triggered[label_col], [1] * len(triggered), zero_division=0)
            recall = triggered[label_col].sum() / df[label_col].sum()
        else:
            precision = 0
            recall = 0
        
        results.append({
            'Rule': rule.replace('rule_', '').replace('_', ' ').title(),
            'Triggered': len(triggered),
            'Fraud_Found': triggered[label_col].sum() if len(triggered) > 0 else 0,
            'Precision': precision,
            'Recall': recall
        })
    
    # Combined (≥2 rules)
    y_true = df[label_col]
    y_pred = df['rule_based_prediction']
    
    results.append({
        'Rule': 'COMBINED (≥2 rules)',
        'Triggered': y_pred.sum(),
        'Fraud_Found': df[y_pred == 1][label_col].sum(),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0)
    })
    
    results_df = pd.DataFrame(results)
    results_df['F1_Score'] = 2 * (results_df['Precision'] * results_df['Recall']) / (
        results_df['Precision'] + results_df['Recall']
    )
    results_df['F1_Score'] = results_df['F1_Score'].fillna(0)
    
    return results_df


if __name__ == "__main__":
    print("FraudGuard Rule Engine")
    print("=" * 50)
    
    engine = FraudRuleEngine()
    explanations = engine.get_rule_explanations()
    
    print("\nConfigured Rules:")
    for i, (rule, explanation) in enumerate(explanations.items(), 1):
        print(f"{i}. {rule}: {explanation}")
