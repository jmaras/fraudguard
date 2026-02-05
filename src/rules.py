"""
FraudGuard - Rule Engine
5 business rules for fraud detection
"""

import pandas as pd
from typing import Dict


class FraudRuleEngine:
    """
    Rule-based fraud detection system

    5 rules:
    Rule 1: High Frequency
    Rule 2: Night Transaction
    Rule 3: High Amount
    Rule 4: Round Amount
    Rule 5: Risky Category
    """
    
    def __init__(self):
        # Categories with historically high fraud rates
        self.risky_categories = [
            'shopping_net',
            'misc_net',
            'grocery_pos',
            'shopping_pos',
            'gas_transport'
        ]
        
        # Suspiciously round transaction amounts
        self.suspicious_amounts = [
            50, 100, 200, 250, 500, 750, 1000, 
            1500, 2000, 2500, 5000
        ]
    
    
    def apply_all_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all 5 fraud detection rules to the DataFrame
        """
        print("Applying fraud detection rules...")
        
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()
        
        # Parse transaction time
        if 'trans_datetime' not in df.columns:
            df['trans_datetime'] = pd.to_datetime(df['trans_date_trans_time'])
            df['hour'] = df['trans_datetime'].dt.hour
        
        # Rule 1: High Frequency
        df = self._rule_high_frequency(df)
        
        # Rule 2: Night Transaction
        df = self._rule_night_transaction(df)
        
        # Rule 3: High Amount
        df = self._rule_high_amount(df)
        
        # Rule 4: Round Amount
        df = self._rule_round_amount(df)
        
        # Rule 5: Risky Category
        df = self._rule_risky_category(df)
        
        # Count how many rules were triggered per transaction
        rule_cols = [col for col in df.columns if col.startswith('rule_')]
        df['rules_triggered'] = df[rule_cols].sum(axis=1)
        
        # Final decision: fraud if at least 2 rules are triggered
        df['rule_based_prediction'] = (df['rules_triggered'] >= 2).astype(int)
        
        print(f"✓ Applied {len(rule_cols)} rules\n")
        print(f"  Transactions flagged (≥2 rules): {df['rule_based_prediction'].sum():,}\n")
        
        return df
    
    
    def _rule_high_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 1: High Frequency
        More than 5 transactions per hour from the same card
        """
        # Round transaction time down to the hour
        df['hour_bucket'] = df['trans_datetime'].dt.floor('H')
        
        # Count transactions per card and hour
        txn_counts = df.groupby(['cc_num', 'hour_bucket']).size().reset_index(name='txn_count_1h')
        
        # Merge counts back into the main DataFrame
        df = df.merge(txn_counts, on=['cc_num', 'hour_bucket'], how='left')
        
        # Rule condition: more than 5 transactions
        df['rule_high_frequency'] = (df['txn_count_1h'] > 5).astype(int)
        
        # Remove temporary column
        df = df.drop('hour_bucket', axis=1)
        
        triggered = df['rule_high_frequency'].sum()
        print(f"  Rule 1 (High Frequency): {triggered:,} triggered")
        
        return df
    
    
    def _rule_night_transaction(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 2: Night Transaction
        Transactions between 2–5 AM are considered suspicious
        """
        df['rule_night_transaction'] = (
            (df['hour'] >= 2) & (df['hour'] < 5)
        ).astype(int)
        
        triggered = df['rule_night_transaction'].sum()
        print(f"  Rule 2 (Night Transaction): {triggered:,} triggered")
        
        return df
    
    
    def _rule_high_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 3: High Amount
        Transaction amount greater than 3x the user's average amount
        """
        # Calculate user average transaction amount (per card)
        user_avg = df.groupby('cc_num')['amt'].transform('mean')
        
        # Rule condition: amount greater than 3x average
        df['rule_high_amount'] = (df['amt'] > 3 * user_avg).astype(int)
        
        triggered = df['rule_high_amount'].sum()
        print(f"  Rule 3 (High Amount): {triggered:,} triggered")
        
        return df
    
    
    def _rule_round_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 4: Round Amount
        Suspiciously round transaction amounts (e.g. exactly 100, 500, 1000)
        """
        df['rule_round_amount'] = df['amt'].isin(self.suspicious_amounts).astype(int)
        
        triggered = df['rule_round_amount'].sum()
        print(f"  Rule 4 (Round Amount): {triggered:,} triggered")
        
        return df
    
    
    def _rule_risky_category(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule 5: Risky Category
        Merchant categories with historically high fraud rates
        """
        df['rule_risky_category'] = df['category'].isin(self.risky_categories).astype(int)
        
        triggered = df['rule_risky_category'].sum()
        print(f"  Rule 5 (Risky Category): {triggered:,} triggered")
        
        return df
    
    def get_rule_explanations(self) -> Dict[str, str]:
        """
        Returns human-readable explanations for each rule
        """
        return {
            'rule_high_frequency': 'More than 5 transactions per hour',
            'rule_night_transaction': 'Transaction between 2-5 AM',
            'rule_high_amount': 'Amount >3x user average',
            'rule_round_amount': f'Suspicious round amount ({self.suspicious_amounts})',
            'rule_risky_category': f'High-risk category ({self.risky_categories})'
        }


if __name__ == "__main__":
    print("FraudGuard Rule Engine")
    print("=" * 50)
    print("\nFunctions:")
    print("  - FraudRuleEngine.apply_all_rules(df)")
    print("\nConfigured Rules:")
    print("  1. High Frequency (>5 txn/hour)")
    print("  2. Night Transaction (2-5 AM)")
    print("  3. High Amount (>3x user avg)")
    print("  4. Round Amount (suspicious amounts)")
    print("  5. Risky Category (high-risk merchants)")
