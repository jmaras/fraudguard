"""
FraudGuard - Streamlit Dashboard
Vergleich: Rule-Based vs. ML-Based Fraud Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load predictions and models"""
    try:
        # Load predictions
        df = pd.read_csv('data/processed/predictions_comparison.csv')
        
        # Parse datetime
        if 'trans_datetime' in df.columns:
            df['trans_datetime'] = pd.to_datetime(df['trans_datetime'])
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        return df
    except FileNotFoundError:
        st.error("❌ Data file not found! Please run Notebook 03 first.")
        return None


@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model = joblib.load('models/xgboost_ml_only.pkl')
        scaler = joblib.load('models/scaler_ml_only.pkl')
        features = joblib.load('models/ml_features.pkl')
        return model, scaler, features
    except FileNotFoundError:
        st.warning("⚠️ Model files not found. Some features may be limited.")
        return None, None, None


# ============================================================================
# HEADER
# ============================================================================

def render_header():
    """Render dashboard header"""
    st.markdown('<div class="main-header">🛡️ FraudGuard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fraud Detection System - Rule-Based vs. ML Comparison</div>', 
                unsafe_allow_html=True)
    st.markdown("---")


# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

def render_overview(df):
    """Overview Tab"""
    st.header("📊 Overview")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Transactions",
            value=f"{len(df):,}",
            delta="Test Set"
        )
    
    with col2:
        fraud_count = df['is_fraud'].sum()
        fraud_rate = df['is_fraud'].mean() * 100
        st.metric(
            label="Fraud Cases",
            value=f"{fraud_count:,}",
            delta=f"{fraud_rate:.2f}% of total"
        )
    
    with col3:
        if 'ml_prediction' in df.columns:
            ml_detected = df['ml_prediction'].sum()
            st.metric(
                label="ML Detected",
                value=f"{ml_detected:,}",
                delta="Flagged by ML"
            )
    
    with col4:
        if 'rule_based_prediction' in df.columns:
            rule_detected = df['rule_based_prediction'].sum()
            st.metric(
                label="Rules Detected",
                value=f"{rule_detected:,}",
                delta="Flagged by Rules"
            )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud Distribution
        fraud_dist = df['is_fraud'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Legitimate', 'Fraud'],
            values=fraud_dist.values,
            hole=0.4,
            marker=dict(colors=['#2ecc71', '#e74c3c'])
        )])
        fig.update_layout(
            title="Transaction Distribution",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ML Risk Level Distribution
        if 'ml_risk_level' in df.columns:
            risk_dist = df['ml_risk_level'].value_counts()
            
            fig = go.Figure(data=[go.Bar(
                x=risk_dist.index,
                y=risk_dist.values,
                marker=dict(color=['#2ecc71', '#f39c12', '#e74c3c'])
            )])
            fig.update_layout(
                title="ML Risk Level Distribution",
                xaxis_title="Risk Level",
                yaxis_title="Count",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Amount Distribution
    st.subheader("💰 Transaction Amount Distribution")
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[df['is_fraud'] == 0]['amt'],
        name='Legitimate',
        marker_color='#2ecc71',
        opacity=0.7,
        nbinsx=50
    ))
    fig.add_trace(go.Histogram(
        x=df[df['is_fraud'] == 1]['amt'],
        name='Fraud',
        marker_color='#e74c3c',
        opacity=0.7,
        nbinsx=50
    ))
    fig.update_layout(
        barmode='overlay',
        xaxis_title="Amount ($)",
        yaxis_title="Frequency",
        height=400,
        xaxis=dict(range=[0, 500])  # Focus on <$500
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# TAB 2: COMPARISON
# ============================================================================

def render_comparison(df):
    """Comparison Tab - KERN DES DASHBOARDS!"""
    st.header("🔬 Rule-Based vs. ML-Only Comparison")
    
    # Calculate Metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    y_true = df['is_fraud']
    
    # Rule-Based
    y_pred_rules = df['rule_based_prediction']
    precision_rules = precision_score(y_true, y_pred_rules, zero_division=0)
    recall_rules = recall_score(y_true, y_pred_rules, zero_division=0)
    f1_rules = f1_score(y_true, y_pred_rules, zero_division=0)
    
    # ML
    y_pred_ml = df['ml_prediction']
    y_proba_ml = df['ml_probability']
    precision_ml = precision_score(y_true, y_pred_ml, zero_division=0)
    recall_ml = recall_score(y_true, y_pred_ml, zero_division=0)
    f1_ml = f1_score(y_true, y_pred_ml, zero_division=0)
    roc_auc_ml = roc_auc_score(y_true, y_proba_ml)
    
    # Performance Metrics Cards
    st.subheader("📈 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Rule-Based")
        st.metric("Precision", f"{precision_rules:.1%}")
        st.metric("Recall", f"{recall_rules:.1%}")
        st.metric("F1-Score", f"{f1_rules:.1%}")
    
    with col2:
        st.markdown("### ML-Only")
        st.metric("Precision", f"{precision_ml:.1%}")
        st.metric("Recall", f"{recall_ml:.1%}")
        st.metric("F1-Score", f"{f1_ml:.1%}")
        st.metric("ROC-AUC", f"{roc_auc_ml:.3f}")
    
    with col3:
        st.markdown("### 🏆 Improvement")
        improvement = ((f1_ml - f1_rules) / f1_rules * 100) if f1_rules > 0 else 0
        st.metric("F1-Score", f"+{improvement:.1f}%", delta="ML vs Rules")
        st.success(f"**ML improves F1-Score by {improvement:.0f}%!**")
    
    st.markdown("---")
    
    # Comparison Chart
    st.subheader("📊 Metrics Comparison")
    
    comparison_data = pd.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Rule-Based': [precision_rules, recall_rules, f1_rules],
        'ML-Only': [precision_ml, recall_ml, f1_ml]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Rule-Based',
        x=comparison_data['Metric'],
        y=comparison_data['Rule-Based'],
        marker_color='#e74c3c'
    ))
    fig.add_trace(go.Bar(
        name='ML-Only',
        x=comparison_data['Metric'],
        y=comparison_data['ML-Only'],
        marker_color='#3498db'
    ))
    fig.update_layout(
        barmode='group',
        yaxis_title="Score",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion Matrices
    st.subheader("🎯 Confusion Matrices")
    
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
        
        st.write(f"**TN:** {cm_rules[0,0]:,} | **FP:** {cm_rules[0,1]:,}")
        st.write(f"**FN:** {cm_rules[1,0]:,} | **TP:** {cm_rules[1,1]:,}")
    
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
        
        st.write(f"**TN:** {cm_ml[0,0]:,} | **FP:** {cm_ml[0,1]:,}")
        st.write(f"**FN:** {cm_ml[1,0]:,} | **TP:** {cm_ml[1,1]:,}")
    
    # ROC Curve
    st.subheader("📉 ROC Curve")
    
    fpr_ml, tpr_ml, _ = roc_curve(y_true, y_proba_ml)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr_ml,
        y=tpr_ml,
        mode='lines',
        name=f'ML-Only (AUC={roc_auc_ml:.3f})',
        line=dict(color='#3498db', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(color='gray', dash='dash')
    ))
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Findings
    st.markdown("---")
    st.subheader("🔑 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ ML Strengths:**")
        st.write("- Much higher precision (fewer false alarms)")
        st.write("- Better recall (catches more fraud)")
        st.write("- Balanced performance")
        st.write("- Probability scores for ranking")
    
    with col2:
        st.markdown("**⚠️ Rule-Based Limitations:**")
        st.write("- High false positive rate")
        st.write("- Rigid thresholds")
        st.write("- Misses complex patterns")
        st.write("- No probability scores")


# ============================================================================
# TAB 3: TRANSACTION LOOKUP
# ============================================================================

def render_lookup(df):
    """Transaction Lookup Tab"""
    st.header("🔍 Transaction Lookup")
    st.write("Analyze how each approach handles individual transactions")
    
    # Initialize session state for random transaction
    if 'random_txn_id' not in st.session_state:
        st.session_state.random_txn_id = None
    
    # Show available indices (use index instead of trans_num)
    st.info(f"💡 **Available Indices:** 0 to {len(df)-1:,} (total {len(df):,} transactions)")
    
    # Example indices
    example_indices = df.sample(5).index.tolist()
    st.caption(f"Try these examples: {', '.join([str(x) for x in example_indices])}")
    
    # Search Input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Use session state value if random button was clicked
        default_value = str(st.session_state.random_txn_id) if st.session_state.random_txn_id else ""
        
        txn_idx = st.text_input(
            "Enter Transaction Index (0-based)",
            value=default_value,
            placeholder="e.g., 1234",
            help="Enter an index number (0 to {}) to see detailed analysis".format(len(df)-1)
        )
    
    with col2:
        # Add vertical spacing to align button with input field
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎲 Random Transaction", use_container_width=True):
            random_idx = df.sample(1).index[0]
            st.session_state.random_txn_id = random_idx
            st.rerun()
    
    if txn_idx:
        try:
            txn_idx_int = int(txn_idx)
            
            if txn_idx_int < 0 or txn_idx_int >= len(df):
                st.error(f"❌ Index {txn_idx_int} out of range. Please use 0 to {len(df)-1}")
            else:
                txn = df.iloc[txn_idx_int]
                
                # Ground Truth
                st.markdown("---")
                col1, col2, col3 = st.columns([2, 2, 3])
                
                with col1:
                    st.markdown("### Ground Truth")
                    if txn['is_fraud'] == 1:
                        st.error("🚨 **ACTUAL FRAUD**")
                    else:
                        st.success("✅ **LEGITIMATE**")
                
                with col2:
                    st.markdown("### Transaction Info")
                    st.write(f"**Index:** {txn_idx_int}")
                    if 'trans_num' in txn.index:
                        st.write(f"**ID:** {str(txn['trans_num'])[:16]}...")
                    st.write(f"**Amount:** ${txn['amt']:.2f}")
                    st.write(f"**Category:** {txn['category']}")
                    if 'trans_datetime' in txn.index:
                        st.write(f"**Time:** {txn['trans_datetime']}")
                
                with col3:
                    st.markdown("### Customer Info")
                    if 'first' in txn.index and 'last' in txn.index:
                        st.write(f"**Name:** {txn['first']} {txn['last']}")
                    if 'city' in txn.index and 'state' in txn.index:
                        st.write(f"**Location:** {txn['city']}, {txn['state']}")
                
                # Comparison
                st.markdown("---")
                st.subheader("🔬 How Each Approach Handled This Transaction")
                
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
                    
                    # Rules Triggered
                    if 'rules_triggered' in txn.index:
                        st.write(f"**Rules Triggered:** {int(txn['rules_triggered'])}/7")
                        
                        # Show which rules
                        rule_cols = [c for c in df.columns if c.startswith('rule_') and c != 'rules_triggered']
                        triggered_rules = [c.replace('rule_', '').replace('_', ' ').title() 
                                         for c in rule_cols if txn[c] == 1]
                        
                        if triggered_rules:
                            st.write("**Triggered:**")
                            for rule in triggered_rules:
                                st.write(f"  ⚠️ {rule}")
                
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
                        st.write(f"**Fraud Probability:** {prob:.1%}")
                        
                        # Probability bar
                        st.progress(float(prob))
                        
                        if 'ml_risk_level' in txn.index:
                            risk = txn['ml_risk_level']
                            if risk == 'High':
                                st.error(f"**Risk Level:** {risk}")
                            elif risk == 'Medium':
                                st.warning(f"**Risk Level:** {risk}")
                            else:
                                st.success(f"**Risk Level:** {risk}")
                
                # Analysis
                st.markdown("---")
                st.subheader("💡 Analysis")
                
                both_correct = (txn['rule_based_prediction'] == txn['is_fraud'] and 
                               txn['ml_prediction'] == txn['is_fraud'])
                
                if both_correct:
                    st.success("✅ Both approaches got it right!")
                elif txn['ml_prediction'] == txn['is_fraud']:
                    st.info("✅ ML was correct, Rules failed - shows ML's superior pattern recognition")
                elif txn['rule_based_prediction'] == txn['is_fraud']:
                    st.warning("✅ Rules were correct, ML failed - rare edge case")
                else:
                    st.error("❌ Both approaches failed - challenging transaction")
        
        except ValueError:
            st.error("Please enter a valid numeric index")


# ============================================================================
# TAB 4: METHODOLOGY
# ============================================================================

def render_methodology(df):
    """Methodology Tab"""
    st.header("📚 Methodology")
    
    # Overview
    st.subheader("🎯 Project Overview")
    st.write("""
    FraudGuard compares two approaches to credit card fraud detection:
    
    1. **Rule-Based System**: 7 business rules based on domain knowledge
    2. **ML-Based System**: XGBoost classifier with 26+ engineered features
    """)
    
    # Dataset
    st.markdown("---")
    st.subheader("📊 Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Source:** Synthetic Fraud Detection (Kaggle)")
        st.write(f"**Total Transactions:** {len(df):,}")
        st.write(f"**Fraud Rate:** {df['is_fraud'].mean():.2%}")
    
    with col2:
        st.write("**Split:** 70/30 Train/Test (Temporal)")
        st.write("**Features:** Transaction amount, time, location, customer info")
        st.write("**Target:** is_fraud (binary)")
    
    # Rule-Based Approach
    st.markdown("---")
    st.subheader("🔧 Rule-Based Approach")
    
    st.write("""
    **7 Business Rules:**
    
    1. **High Frequency:** >5 transactions per hour
    2. **Geographic Impossible:** Velocity >500 km/h between transactions
    3. **Night Transaction:** Between 2-5 AM
    4. **High Amount:** >3x user's average
    5. **Out-of-State:** State changed from previous transaction
    6. **Round Amount:** Suspicious round amounts (100, 500, 1000...)
    7. **Risky Category:** High-risk merchant categories
    
    **Decision Logic:** If ≥2 rules trigger → Flag as Fraud
    """)
    
    with st.expander("📝 See Rule Code Example"):
        st.code("""
# Example: Night Transaction Rule
def rule_night_transaction(df):
    hour = df['trans_datetime'].dt.hour
    return ((hour >= 2) & (hour < 5)).astype(int)
        """, language='python')
    
    # ML Approach
    st.markdown("---")
    st.subheader("🤖 ML-Based Approach")
    
    st.write("""
    **Model:** XGBoost Classifier
    
    **Features (26+):**
    - **Time:** hour, day_of_week, is_weekend, is_night
    - **Geo:** customer-merchant distance, velocity
    - **Aggregated:** transaction count, average amount, std deviation
    - **Deviation:** amount vs. user average, z-scores
    - **Categorical:** gender, state frequency, age, job
    
    **Training:**
    - SMOTE for class imbalance (0.99% fraud → 30% after SMOTE)
    - StandardScaler for feature scaling
    - 300 trees, max_depth=5, learning_rate=0.1
    """)
    
    with st.expander("📝 See Model Training Code"):
        st.code("""
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Handle imbalance
smote = SMOTE(sampling_strategy=0.3)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train model
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1
)
model.fit(X_train, y_train)
        """, language='python')
    
    # Key Differences
    st.markdown("---")
    st.subheader("⚖️ Key Differences")
    
    comparison = pd.DataFrame({
        'Aspect': ['Transparency', 'Adaptability', 'Performance', 'Maintenance', 'Explainability'],
        'Rule-Based': ['High ✅', 'Low ❌', 'Moderate', 'Manual updates', 'Very clear'],
        'ML-Based': ['Low ❌', 'High ✅', 'High ✅', 'Automatic', 'Feature importance']
    })
    
    st.dataframe(comparison, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main application"""
    
    # Load data
    df = load_data()
    
    if df is None:
        st.stop()
    
    # Header
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/security-shield-green.png", width=150)
        st.markdown("### 🛡️ FraudGuard")
        st.markdown("**Fraud Detection Analysis**")
        st.markdown("---")
        st.markdown(f"📊 **Dataset Size:** {len(df):,}")
        st.markdown(f"🚨 **Fraud Cases:** {df['is_fraud'].sum():,}")
        st.markdown(f"📈 **Fraud Rate:** {df['is_fraud'].mean():.2%}")
        st.markdown("---")
        st.markdown("**Navigation:**")
        st.markdown("Use tabs above to explore")
    
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