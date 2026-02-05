"""
FraudGuard - Streamlit Dashboard
Rule-Based vs. ML-Based Fraud Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Page Config
st.set_page_config(
    page_title="FraudGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# DATA LOADING

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/processed/predictions_comparison.csv')
        
        if 'trans_datetime' in df.columns:
            df['trans_datetime'] = pd.to_datetime(df['trans_datetime'])
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Please run the notebook first.")
        return None


# HEADER

def render_header():
    """Render dashboard header"""
    st.markdown('<div class="main-header">FraudGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fraud Detection System - Rule-Based vs. ML Comparison</div>', 
                unsafe_allow_html=True)
    st.markdown("---")


# TAB 1: OVERVIEW

def render_overview(df):
    """Overview Tab"""
    st.header("📊 Overview")
    
    # KPI Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{len(df):,}"
        )
    
    with col2:
        fraud_count = df['is_fraud'].sum()
        fraud_rate = df['is_fraud'].mean() * 100
        st.metric(
            label="Fraud Cases",
            value=f"{fraud_count:,}",
            delta=f"{fraud_rate:.2f}%"
        )
    
    with col3:
        if 'ml_prediction' in df.columns:
            ml_detected = df['ml_prediction'].sum()
            st.metric(
                label="ML Detected",
                value=f"{ml_detected:,}"
            )
    
    st.markdown("---")
    
    # Fraud Distribution
    st.subheader("Fraud Distribution")
    
    fraud_dist = df['is_fraud'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=['Legitimate', 'Fraud'],
        values=fraud_dist.values,
        hole=0.4,
        marker=dict(colors=['#2ecc71', '#e74c3c'])
    )])
    fig.update_layout(height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


# TAB 2: COMPARISON

def render_comparison(df):
    """Comparison Tab"""
    st.header("Rule-Based vs. ML-Only Comparison")
    
    y_true = df['is_fraud']
    
    # Rule-Based
    y_pred_rules = df['rule_based_prediction']
    precision_rules = precision_score(y_true, y_pred_rules, zero_division=0)
    recall_rules = recall_score(y_true, y_pred_rules, zero_division=0)
    f1_rules = f1_score(y_true, y_pred_rules, zero_division=0)
    
    # ML
    y_pred_ml = df['ml_prediction']
    precision_ml = precision_score(y_true, y_pred_ml, zero_division=0)
    recall_ml = recall_score(y_true, y_pred_ml, zero_division=0)
    f1_ml = f1_score(y_true, y_pred_ml, zero_division=0)
    
    # Performance Metrics
    st.subheader("Performance Metrics")
    
    comparison = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Rule-Based': [f"{precision_rules:.1%}", f"{recall_rules:.1%}", f"{f1_rules:.1%}"],
        'ML-Only': [f"{precision_ml:.1%}", f"{recall_ml:.1%}", f"{f1_ml:.1%}"]
    })
    
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    
    improvement = ((f1_ml - f1_rules) / f1_rules * 100) if f1_rules > 0 else 0
    st.success(f"**✅ ML improves F1-Score by {improvement:.0f}% over Rule-Based!**")
    
    st.markdown("---")
    
    # Bar Chart
    st.subheader("Metrics Comparison")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Rule-Based',
        x=['Precision', 'Recall', 'F1-Score'],
        y=[precision_rules, recall_rules, f1_rules],
        marker_color='#e74c3c'
    ))
    fig.add_trace(go.Bar(
        name='ML-Only',
        x=['Precision', 'Recall', 'F1-Score'],
        y=[precision_ml, recall_ml, f1_ml],
        marker_color='#3498db'
    ))
    fig.update_layout(
        barmode='group',
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion Matrices
    st.subheader("Confusion Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Rule-Based")
        cm_rules = confusion_matrix(y_true, y_pred_rules)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_rules,
            x=['Predicted Legitimate', 'Predicted Fraud'],
            y=['Actual Legitimate', 'Actual Fraud'],
            colorscale='Reds',
            text=cm_rules,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"TN: {cm_rules[0,0]:,} | FP: {cm_rules[0,1]:,} | FN: {cm_rules[1,0]:,} | TP: {cm_rules[1,1]:,}")
    
    with col2:
        st.markdown("#### ML-Only")
        cm_ml = confusion_matrix(y_true, y_pred_ml)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm_ml,
            x=['Predicted Legitimate', 'Predicted Fraud'],
            y=['Actual Legitimate', 'Actual Fraud'],
            colorscale='Blues',
            text=cm_ml,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption(f"TN: {cm_ml[0,0]:,} | FP: {cm_ml[0,1]:,} | FN: {cm_ml[1,0]:,} | TP: {cm_ml[1,1]:,}")


# TAB 3: TRANSACTION LOOKUP

def render_lookup(df):
    """Transaction Lookup Tab"""
    st.header("🔍 Transaction Lookup")
    st.write("Analyze how each approach handles individual transactions")
    
    if 'random_txn_id' not in st.session_state:
        st.session_state.random_txn_id = None
    
    st.info(f"**Available Indices:** 0 to {len(df)-1:,} (total {len(df):,} transactions)")
    
    example_indices = df.sample(5).index.tolist()
    st.caption(f"Try these examples: {', '.join([str(x) for x in example_indices])}")
    
    # Search Input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        default_value = str(st.session_state.random_txn_id) if st.session_state.random_txn_id else ""
        
        txn_idx = st.text_input(
            "Enter Transaction Index",
            value=default_value,
            placeholder="e.g., 1234"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random", use_container_width=True):
            random_idx = df.sample(1).index[0]
            st.session_state.random_txn_id = random_idx
            st.rerun()
    
    if txn_idx:
        try:
            txn_idx_int = int(txn_idx)
            
            if txn_idx_int < 0 or txn_idx_int >= len(df):
                st.error(f"Index out of range (0 to {len(df)-1})")
            else:
                txn = df.iloc[txn_idx_int]
                
                # Ground Truth
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    st.markdown("### Ground Truth")
                    if txn['is_fraud'] == 1:
                        st.error("❌ **ACTUAL FRAUD**")
                    else:
                        st.success("✅ **LEGITIMATE**")
                
                with col2:
                    st.markdown("### Transaction Info")
                    st.write(f"**Index:** {txn_idx_int}")
                    st.write(f"**Amount:** ${txn['amt']:.2f}")
                    st.write(f"**Category:** {txn['category']}")
                
                with col3:
                    st.markdown("### Customer")
                    if 'first' in txn.index and 'last' in txn.index:
                        st.write(f"**Name:** {txn['first']} {txn['last']}")
                    if 'city' in txn.index and 'state' in txn.index:
                        st.write(f"**Location:** {txn['city']}, {txn['state']}")
                
                # Comparison
                st.markdown("---")
                st.subheader("🔬 Side-by-Side Comparison")
                
                col1, col2 = st.columns(2)
                
                # Rule-Based
                with col1:
                    st.markdown("#### 🔧 Rule-Based")
                    
                    decision = "FRAUD" if txn['rule_based_prediction'] == 1 else "LEGITIMATE"
                    correct = txn['rule_based_prediction'] == txn['is_fraud']
                    
                    if correct:
                        st.success(f"✅ Predicted: **{decision}**")
                    else:
                        st.error(f"❌ Predicted: **{decision}**")
                    
                    if 'rules_triggered' in txn.index:
                        st.write(f"**Rules Triggered:** {int(txn['rules_triggered'])}/5")
                
                # ML-Based
                with col2:
                    st.markdown("#### 🤖 ML-Only")
                    
                    decision = "FRAUD" if txn['ml_prediction'] == 1 else "LEGITIMATE"
                    correct = txn['ml_prediction'] == txn['is_fraud']
                    
                    if correct:
                        st.success(f"✅ Predicted: **{decision}**")
                    else:
                        st.error(f"❌ Predicted: **{decision}**")
                    
                    if 'ml_probability' in txn.index:
                        prob = txn['ml_probability']
                        st.write(f"**Probability:** {prob:.1%}")
                        st.progress(float(prob))
        
        except ValueError:
            st.error("Please enter a valid numeric index")


# TAB 4: METHODOLOGY

def render_methodology(df):
    """Methodology Tab"""
    st.header("Methodology")
    
    # Overview
    st.subheader("Project Overview")
    st.write("""
    FraudGuard compares two approaches to credit card fraud detection:
    1. **Rule-Based System**: Business rules based on domain knowledge
    2. **ML-Based System**: XGBoost classifier with engineered features
    """)
    
    st.markdown("---")
    
    # Dataset
    st.subheader("Dataset")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Source:** Synthetic Fraud Detection (Kaggle)")
        st.write(f"**Transactions:** {len(df):,}")
        st.write(f"**Fraud Rate:** {df['is_fraud'].mean():.2%}")
    
    with col2:
        st.write(f"**Split:** 70/30 Train/Test")
        st.write(f"**Features:** Time, Amount, Location, Demographics")
        st.write(f"**Target:** is_fraud (binary)")
    
    st.markdown("---")
    
    # Rule-Based
    st.subheader("Rule-Based Approach")
    st.write("""
    **5 Business Rules:**
    - High Frequency (>5 transactions/hour)
    - Night Transaction (2-5 AM)
    - High Amount (>3x user average)
    - Round Amount (suspicious amounts)
    - Risky Category (high-risk merchants)
    
    **Decision:** ≥2 rules triggered → Flag as Fraud
    """)
    
    with st.expander("Code Example"):
        st.code("""
# Apply rules
engine = FraudRuleEngine()
df = engine.apply_all_rules(df)

# Prediction
df['fraud'] = (df['rules_triggered'] >= 2)
        """, language='python')
    
    st.markdown("---")
    
    # ML Approach
    st.subheader("ML-Based Approach")
    st.write("""
    **Model:** XGBoost Classifier
    
    **Features:** Time, Geo, Aggregations, Deviations, Demographics
    
    **Training:** SMOTE for class imbalance, StandardScaler
    """)
    
    with st.expander("Code Example"):
        st.code("""
# Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
        """, language='python')
    
    st.markdown("---")
    
    # Comparison
    st.subheader("Key Differences")
    
    comparison_table = pd.DataFrame({
        'Aspect': ['Transparency', 'Adaptability', 'Performance'],
        'Rule-Based': ['High ✅', 'Low ❌', 'Moderate'],
        'ML-Based': ['Lower', 'High ✅', 'High ✅']
    })
    
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)


# MAIN APP

def main():
    """Main application"""
    
    df = load_data()
    
    if df is None:
        st.stop()
    
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛡️ FraudGuard")
        st.markdown("**Fraud Detection Analysis**")
        st.markdown("---")
        st.markdown(f"📊 **Dataset:** {len(df):,} transactions")
        st.markdown(f"🚨 **Fraud:** {df['is_fraud'].sum():,} cases")
        st.markdown(f"📈 **Rate:** {df['is_fraud'].mean():.2%}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔬 Comparison",
        "🔍 Transaction Lookup",
        "📚 Methodology"
    ])
    
    with tab1:
        render_overview(df)
    
    with tab2:
        render_comparison(df)
    
    with tab3:
        render_lookup(df)
    
    with tab4:
        render_methodology(df)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "FraudGuard Dashboard | Built with Streamlit & XGBoost"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
