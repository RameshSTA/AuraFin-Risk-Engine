
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import joblib
import json
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="AuraFin | AI Risk Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# 2. INTERNAL MODEL LOADER (The "Brain")
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Loads model, encoders, and config directly into memory.
    Cached so it only runs once, making the app fast.
    """
    # Verify paths (Streamlit Cloud runs from root)
    base_path = os.getcwd()
    model_path = os.path.join(base_path, "models", "xgb_champion.pkl")
    encoder_path = os.path.join(base_path, "models", "label_encoders.pkl")
    config_path = os.path.join(base_path, "models", "threshold_config.json")

    # Graceful error handling if files are missing
    if not os.path.exists(model_path):
        st.error(f"❌ Critical Error: Model file not found at {model_path}")
        return None, None, 0.50

    try:
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        with open(config_path, "r") as f:
            config = json.load(f)
            threshold = config.get("optimal_threshold", 0.50)
        return model, encoders, threshold
    except Exception as e:
        st.error(f"Failed to load AI Engine: {e}")
        return None, None, 0.50

# Load everything immediately
model, encoders, OPTIMAL_THRESHOLD = load_artifacts()

# ---------------------------------------------------------
# 3. PREDICTION LOGIC (Moved from API to Dashboard)
# ---------------------------------------------------------
def make_prediction(input_data):
    """
    Runs the full prediction pipeline locally.
    Replaces the API call.
    """
    if model is None:
        return None

    # Create DataFrame
    df = pd.DataFrame([input_data])

    # --- Feature Engineering ---
    # 1. LTI (Loan-to-Income)
    income_safe = df['income'].replace(0, 1.0)
    df['LTI'] = df['loan_amount'] / income_safe

    # 2. LTV Safety Check
    if 'LTV' not in df.columns or df['LTV'].iloc[0] == 0:
        prop_safe = df['property_value'].fillna(0).replace(0, 1.0)
        df['LTV'] = (df['loan_amount'] / prop_safe) * 100

    # 3. Disposable Income
    df['est_monthly_payment'] = df['loan_amount'] * 0.005
    df['disposable_income'] = df['income'] - df['est_monthly_payment']

    # 4. Risk Interaction Flags
    df['risky_ltv'] = (df['LTV'] > 80).astype(int)
    df['risky_credit'] = (df['Credit_Score'] < 650).astype(int)
    df['double_risk'] = df['risky_ltv'] * df['risky_credit']

    # --- Label Encoding ---
    for col, le in encoders.items():
        if col in df.columns:
            val = str(df.loc[0, col])
            if val in le.classes_:
                df.loc[0, col] = le.transform([val])[0]
            else:
                # Handle unknown categories safely
                df.loc[0, col] = le.transform([le.classes_[0]])[0]

    # --- Column Alignment ---
    try:
        # Try to get feature names from XGBoost model
        expected_cols = model.get_booster().feature_names
    except:
        # Fallback if attribute missing
        expected_cols = df.columns 

    # Ensure all columns exist
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0
            
    # Reorder and Convert to Numeric
    final_df = df[expected_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # --- Prediction ---
    probs = model.predict_proba(final_df)
    risk_probability = float(probs[0][1])
    
    decision = "REJECT" if risk_probability >= OPTIMAL_THRESHOLD else "APPROVE"
    exposure = input_data['loan_amount'] * 0.60 * risk_probability
    
    return {
        "decision": decision,
        "risk_score": risk_probability,
        "exposure": exposure,
        "threshold": OPTIMAL_THRESHOLD
    }

# ---------------------------------------------------------
# 4. CSS STYLING (Google/Stripe Professional Theme)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* GLOBAL SETTINGS */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
        background-color: #F8FAFC; /* Slate 50 */
    }
    
    /* Card Styling */
    div.stExpander, div.css-1r6slb0 {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        padding: 20px;
    }
    
    /* Metrics Styling */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    label[data-testid="stMetricLabel"] {
        color: #64748B;
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A;
        font-size: 28px;
        font-weight: 700;
    }
    
    /* Custom Badge Styling */
    .verdict-box {
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid;
    }
    .verdict-approve {
        background-color: #ECFDF5; /* Emerald 50 */
        border-color: #10B981;
        color: #047857;
    }
    .verdict-reject {
        background-color: #FEF2F2; /* Rose 50 */
        border-color: #F43F5E;
        color: #BE123C;
    }
    .verdict-title {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
        opacity: 0.8;
    }
    .verdict-value {
        font-size: 36px;
        font-weight: 800;
        margin: 0;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] * {
        color: #1E293B !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 2px solid #E2E8F0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 14px;
        font-weight: 600;
        color: #64748B;
        padding-bottom: 12px;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB;
        border-bottom-color: #2563EB;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 5. SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=50)
    st.markdown("### AuraFin Control")
    st.caption("Production v9.0")
    st.markdown("---")
    
    st.markdown("**💰 Financial Profile**")
    income = st.number_input("Monthly Income ($)", value=9500, step=500)
    loan_amt = st.number_input("Loan Amount ($)", value=320000, step=1000)
    prop_val = st.number_input("Property Value ($)", value=450000, step=1000)
    
    ltv = (loan_amt / prop_val) * 100 if prop_val > 0 else 0
    st.markdown(f"**LTV Ratio:** `{ltv:.1f}%`")
    if ltv > 80:
        st.warning("⚠️ High Risk Leverage")
    
    st.markdown("---")
    st.markdown("**👤 Applicant Details**")
    credit_score = st.slider("FICO Score", 300, 850, 720)
    region = st.selectbox("Region", ["North", "South", "Central", "North-East"])
    loan_purpose = st.selectbox("Purpose", ["p1", "p2", "p3", "p4"])
    
    # Defaults
    payload_extras = {"Gender": "Male", "loan_type": "type1", "credit_type": "CIB", "age": "35-44", "dtir1": 40.0}
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("RUN RISK ASSESSMENT", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 6. MAIN HEADER & TABS
# ---------------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    st.title("Credit Risk Intelligence System")
    st.markdown("Cost-Sensitive Decision Engine • Optimized for Net Profitability")
with c2:
    st.markdown("")
    st.markdown('<div style="text-align: right; color: #10B981; font-weight: 600;">● SYSTEM ONLINE</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: right; color: #64748B; font-size: 14px;">Threshold: {OPTIMAL_THRESHOLD}</div>', unsafe_allow_html=True)

st.markdown("---")

tab_live, tab_impact, tab_research = st.tabs([
    "⚡ LIVE DECISION", 
    "💰 BUSINESS IMPACT", 
    "🔬 RESEARCH & METHODOLOGY"
])

# =========================================================
# TAB 1: LIVE DECISION
# =========================================================
with tab_live:
    if run_btn:
        # Build Input Data
        input_data = {
            "loan_amount": loan_amt, "income": income, "property_value": prop_val,
            "Credit_Score": credit_score, "LTV": ltv, "Region": region,
            "loan_purpose": loan_purpose, **payload_extras
        }
        
        # --- EXECUTE MODEL INTERNALLY ---
        # No more HTTP Request to localhost!
        result = make_prediction(input_data)
        
        if result:
            decision = result['decision']
            prob = result['risk_score']
            exposure = result['exposure']
            threshold = result['threshold']
            
            # 1. VERDICT CARD
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                style = "verdict-approve" if decision == "APPROVE" else "verdict-reject"
                icon = "✅" if decision == "APPROVE" else "🚫"
                
                st.markdown(f"""
                <div class="verdict-box {style}">
                    <div class="verdict-title">AI RECOMMENDATION</div>
                    <div class="verdict-value">{decision}</div>
                    <div style="margin-top: 10px; font-weight: 500;">{icon} Confidence: {100-(prob*100):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_right:
                # 2. GAUGE CHART
                st.markdown("##### 📊 Risk Probability Spectrum")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': ""},
                    delta = {'reference': 59, 'increasing': {'color': "#EF4444"}, 'decreasing': {'color': "#10B981"}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                        'bar': {'color': "#1E293B"}, # Dark Pointer
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#E2E8F0",
                        'steps': [
                            {'range': [0, 59], 'color': "#D1FAE5"},  # Green Zone
                            {'range': [59, 100], 'color': "#FEE2E2"}], # Red Zone
                        'threshold': {
                            'line': {'color': "#E11D48", 'width': 4},
                            'thickness': 0.75,
                            'value': 59}}))
                
                fig.update_layout(height=180, margin=dict(l=30,r=30,t=20,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"})
                st.plotly_chart(fig, use_container_width=True)

            # 3. METRICS ROW
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Calculated Risk", f"{prob:.1%}")
            m2.metric("Financial Exposure", f"${exposure:,.0f}", help="Capital at Risk")
            m3.metric("Safety Threshold", f"{threshold:.2f}")

            # 4. EXPLAINABILITY
            st.markdown("---")
            with st.expander("🔍 **View Logic Explanation**", expanded=True):
                st.markdown(f"""
                **Decision Logic:**
                The model calculated a probability of **{prob:.1%}**. 
                This is **{'LOWER' if decision=='APPROVE' else 'HIGHER'}** than the optimized threshold of **{threshold}**, triggering the **{decision}** verdict.
                
                **Primary Risk Drivers:**
                1.  **LTV Ratio:** {ltv:.1f}% ({'High Risk' if ltv > 80 else 'Safe Equity'})
                2.  **Credit History:** {credit_score} ({'Subprime' if credit_score < 650 else 'Prime'})
                3.  **LTI Ratio:** {loan_amt/income:.1f}x Annual Income
                """)
    else:
        st.info("👈 Enter applicant details in the sidebar to generate a live risk assessment.")

# =========================================================
# TAB 2: FINANCIAL IMPACT
# =========================================================
with tab_impact:
    st.header("Financial Impact Analysis")
    
    col_i1, col_i2 = st.columns([1, 1])
    
    with col_i1:
        st.subheader("The $21.8 Million Advantage")
        st.markdown("""
        **1. The "Vanilla" Baseline (Red Bar)**
        Standard machine learning models optimize for **Accuracy**. On a dataset with 94% good loans, a standard model achieves 94% accuracy by simply predicting "No Default" for everyone.
        * **Result:** It captures **0%** of actual defaults.
        * **Financial Consequence:** The bank absorbs **$35.3 Million** in losses per batch.
        
        **2. The AuraFin Solution (Green Bar)**
        We optimized for **Net Profit**. By implementing Cost-Sensitive Learning (15x penalty) and Threshold Optimization (0.59), we catch 71% of defaults while preserving safe customers.
        * **Result:** Minimizes loss to **$13.5 Million**.
        """)
        st.success("🚀 **NET SAVINGS: $21.8 Million per batch**")
        
    with col_i2:
        if os.path.exists("images/11_true_business_impact.png"):
            st.image("images/11_true_business_impact.png", caption="Portfolio Loss Simulation (Lower is Better)", use_container_width=True)
        else:
            st.warning("Chart missing: images/11_true_business_impact.png")

# =========================================================
# TAB 3: METHODOLOGY
# =========================================================
with tab_research:
    st.header("Research & Methodology")
    
    # 1. PROBLEM
    st.markdown("### 1. Problem Statement")
    st.markdown("Standard ML models optimize for accuracy. In credit risk, this is fatal because the cost of a False Negative (Default: $60k) is 10x the cost of a False Positive (Rejection: $6k).")
    
    st.markdown("---")
    
    # 2. FORENSICS
    st.markdown("### 2. Data Forensics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Leakage Detection:** We identified that `rate_of_interest` was missing for 99% of rejected loans. This confirms it is a post-approval variable. We removed it to prevent Data Leakage.")
    with c2:
        if os.path.exists("images/01_leakage_evidence.png"):
            st.image("images/01_leakage_evidence.png", caption="Leakage Evidence", use_container_width=True)
    
    st.markdown("---")
    
    # 3. ENGINEERING
    st.markdown("### 3. Feature Engineering")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**'Skin in the Game' Theory:** We validated that Loan-to-Value (LTV) is the strongest predictor of default. High LTV (>80%) correlates with strategic default.")
    with c4:
        if os.path.exists("images/05_ltv_separation.png"):
            st.image("images/05_ltv_separation.png", caption="LTV Separation", use_container_width=True)

    st.markdown("---")
    
    # 4. OPTIMIZATION
    st.markdown("### 4. Financial Optimization")
    st.markdown("**The Cost Function:** We solved for the optimal decision threshold using the P&L equation:")
    st.latex(r"J(t) = (FN \times \$60,000) + (FP \times \$6,000)")
    
    c5, c6 = st.columns(2)
    with c5:
        if os.path.exists("images/09_financial_optimization.png"):
            st.image("images/09_financial_optimization.png", caption="Cost Minimization Curve", use_container_width=True)
    with c6:
        st.markdown("**Why 0.59?** Our weighted model was aggressive (paranoid). The optimization loop found that raising the threshold to 0.59 recovered **$300k** in profit without increasing risk.")