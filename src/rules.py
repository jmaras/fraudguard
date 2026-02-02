"""
FraudGuard - Rule Engine (SIMPLIFIED)
5 Business-Regeln zur Betrugserkennung
"""

import pandas as pd
import numpy as np
from typing import Dict


class FraudRuleEngine:
    """
    Regelbasiertes Fraud Detection System
    
    5 Regeln:
    1. High Frequency: >5 Transaktionen pro Stunde
    2. Night Transaction: 2-5 Uhr morgens
    3. High Amount: >3x User-Durchschnitt
    4. Round Amount: Verdächtig runde Beträge
    5. Risky Category: Kategorien mit hoher Fraud-Rate
    """
    
    def __init__(self):
        # Kategorien mit historisch hoher Fraud-Rate
        self.risky_categories = [
            'shopping_net',
            'misc_net',
            'grocery_pos',
            'shopping_pos',
            'gas_transport'
        ]
        
        # Verdächtig runde Beträge
        self.suspicious_amounts = [
            50, 100, 200, 250, 500, 750, 1000, 
            1500, 2000, 2500, 5000
        ]
    
    
    def apply_all_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Wendet alle 5 Regeln auf DataFrame an
        
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
        
        # Regel 2: Night Transaction
        df = self._rule_night_transaction(df)
        
        # Regel 3: High Amount
        df = self._rule_high_amount(df)
        
        # Regel 4: Round Amount
        df = self._rule_round_amount(df)
        
        # Regel 5: Risky Category
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
    
    
    def _rule_night_transaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 2: Night Transaction
        Transaktionen zwischen 2-5 Uhr morgens sind verdächtig
        """
        df['rule_night_transaction'] = (
            (df['hour'] >= 2) & (df['hour'] < 5)
        ).astype(int)
        
        triggered = df['rule_night_transaction'].sum()
        print(f"  Rule 2 (Night Transaction): {triggered:,} triggered")
        
        return df
    
    
    def _rule_high_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 3: High Amount
        Betrag >3x Durchschnitt des Users
        """
        # Berechne User-Durchschnitt (pro cc_num)
        user_avg = df.groupby('cc_num')['amt'].transform('mean')
        
        # Regel: >3x Durchschnitt
        df['rule_high_amount'] = (df['amt'] > 3 * user_avg).astype(int)
        
        triggered = df['rule_high_amount'].sum()
        print(f"  Rule 3 (High Amount): {triggered:,} triggered")
        
        return df
    
    
    def _rule_round_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 4: Round Amount
        Verdächtig runde Beträge (z.B. genau 100, 500, 1000)
        """
        df['rule_round_amount'] = df['amt'].isin(self.suspicious_amounts).astype(int)
        
        triggered = df['rule_round_amount'].sum()
        print(f"  Rule 4 (Round Amount): {triggered:,} triggered")
        
        return df
    
    
    def _rule_risky_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Regel 5: Risky Category
        Kategorien mit historisch hoher Fraud-Rate
        """
        df['rule_risky_category'] = df['category'].isin(self.risky_categories).astype(int)
        
        triggered = df['rule_risky_category'].sum()
        print(f"  Rule 5 (Risky Category): {triggered:,} triggered")
        
        return df
    
    
    def get_rule_explanations(self) -> Dict[str, str]:
        """
        Gibt Erklärungen für jede Regel zurück
        
        Returns:
            Dictionary mit Regel-Namen und Erklärungen
        """
        return {
            'rule_high_frequency': 'More than 5 transactions per hour',
            'rule_night_transaction': 'Transaction between 2-5 AM',
            'rule_high_amount': 'Amount >3x user average',
            'rule_round_amount': f'Suspicious round amount ({self.suspicious_amounts})',
            'rule_risky_category': f'High-risk category ({self.risky_categories})'
        }


if __name__ == "__main__":
    print("FraudGuard Rule Engine (Simplified)")
    print("=" * 50)
    
    engine = FraudRuleEngine()
    explanations = engine.get_rule_explanations()
    
    print("\nConfigured Rules:")
    for i, (rule, explanation) in enumerate(explanations.items(), 1):
        print(f"{i}. {rule}: {explanation}")
