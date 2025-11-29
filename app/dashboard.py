# ---------------------------------------------------------
# app/dashboard.py - Final UI (Fixed Logic)
# ---------------------------------------------------------
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
# 2. INTERNAL LOGIC (The "Brain") - Hidden from UI
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_path = os.getcwd()
    model_path = os.path.join(base_path, "models", "xgb_champion.pkl")
    encoder_path = os.path.join(base_path, "models", "label_encoders.pkl")
    config_path = os.path.join(base_path, "models", "threshold_config.json")

    if not os.path.exists(model_path):
        st.error("Model not found. Please check repository structure.")
        return None, None, 0.50

    try:
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        with open(config_path, "r") as f:
            config = json.load(f)
            threshold = config.get("optimal_threshold", 0.50)
        return model, encoders, threshold
    except Exception as e:
        st.error(f"Error loading AI engine: {e}")
        return None, None, 0.50

model, encoders, OPTIMAL_THRESHOLD = load_artifacts()

def make_prediction(input_data):
    if model is None: return None
    df = pd.DataFrame([input_data])
    
    # Feature Engineering
    income_val = df['income'].replace(0, 1.0)
    df['LTI'] = df['loan_amount'] / income_val
    if 'LTV' not in df.columns or df['LTV'].iloc[0] == 0:
        prop_val = df['property_value'].replace(0, 1.0)
        df['LTV'] = (df['loan_amount'] / prop_val) * 100
    df['est_monthly_payment'] = df['loan_amount'] * 0.005
    df['disposable_income'] = df['income'] - df['est_monthly_payment']
    df['risky_ltv'] = (df['LTV'] > 80).astype(int)
    df['risky_credit'] = (df['Credit_Score'] < 650).astype(int)
    df['double_risk'] = df['risky_ltv'] * df['risky_credit']

    # Encoding
    for col, le in encoders.items():
        if col in df.columns:
            val = str(df.loc[0, col])
            if val in le.classes_:
                df.loc[0, col] = le.transform([val])[0]
            else:
                df.loc[0, col] = le.transform([le.classes_[0]])[0]

    # Alignment
    try: expected_cols = model.get_booster().feature_names
    except: expected_cols = df.columns
    for c in expected_cols:
        if c not in df.columns: df[c] = 0
            
    final_df = df[expected_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
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
# 3. YOUR PREFERRED CSS (Professional, Clean, & Robust)
# ---------------------------------------------------------
st.markdown("""
   /* Global Settings */
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
       border: 2px solid #E2E8F0;
       padding: 20px;
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
       height: 200px; /* Fixed height for alignment */
       display: flex;
       flex-direction: column;
       justify-content: center;
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
  
   /* Metric Box Styling (Similar to Verdict) */
   .metric-box {
       padding: 16px;
       border-radius: 12px;
       text-align: center;
       border: 1px solid #E2E8F0;
       background-color: #FFFFFF;
       box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
   }
   .metric-title {
       font-size: 12px;
       font-weight: 600;
       text-transform: uppercase;
       letter-spacing: 1px;
       margin-bottom: 4px;
       color: #64748B;
   }
   .metric-value {
       font-size: 24px;
       font-weight: 700;
       color: #0F172A;
   }
  
   /* Tab Styling */
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
  
   /* Gauge Card */
   .gauge-card {
       border: 1px solid #E2E8F0;
       border-radius: 12px;
       background-color: #FFFFFF;
       box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
       padding: 10px;
       height: 200px; /* Fixed height for alignment */
   }

    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=50)
    st.markdown("### AuraFin Control")
    st.caption("Production v1.0 | XGBoost")
    st.markdown("---")
    
    st.markdown("** Financial Profile**")
    income = st.number_input("Monthly Income ($)", value=9500, step=500)
    loan_amt = st.number_input("Loan Amount ($)", value=320000, step=1000)
    prop_val = st.number_input("Property Value ($)", value=450000, step=1000)
    
    ltv = (loan_amt / prop_val) * 100 if prop_val > 0 else 0
    st.markdown(f"**LTV Ratio:** `{ltv:.1f}%`")
    if ltv > 80:
        st.warning("High Leverage Risk")
    
    st.markdown("---")
    st.markdown("** Applicant Details**")
    credit_score = st.slider("FICO Score", 300, 850, 720)
    region = st.selectbox("Region", ["North", "South", "Central", "North-East"])
    loan_purpose = st.selectbox("Purpose", ["p1", "p2", "p3", "p4"])
    
    payload_extras = {"Gender": "Male", "loan_type": "type1", "credit_type": "CIB", "age": "35-44", "dtir1": 40.0}
    
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("RUN RISK ASSESSMENT", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 5. MAIN HEADER
# ---------------------------------------------------------
c1, c2 = st.columns([3, 1])
with c1:
    st.title("AuraFin: Hybrid XGBoost/PyTorch System")
    st.markdown("Cost-Sensitive Decision Engine • Optimized for Net Profitability")
with c2:
    st.markdown("")
    st.markdown('<div style="text-align: right; color: #10B981; font-weight: 600;">● SYSTEM ONLINE</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="text-align: right; color: #64748B; font-size: 14px;">Threshold: {OPTIMAL_THRESHOLD}</div>', unsafe_allow_html=True)

st.markdown("---")

tab_live, tab_impact, tab_research = st.tabs([
    " LIVE DECISION", 
    " BUSINESS IMPACT", 
    " METHODOLOGY"
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
        
        # EXECUTE LOGIC (No Requests)
        result = make_prediction(input_data)
        
        if result:
            decision = result['decision']
            prob = result['risk_score']
            exposure = result['exposure']
            threshold = result['threshold']
            
            # --- 1. TOP ROW: VERDICT CARD & GAUGE CHART ---
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                style_class = "verdict-approve" if decision == "APPROVE" else "verdict-reject"
                icon = "✅" if decision == "APPROVE" else "🚫"
                
                st.markdown(f"""
                <div class="verdict-box {style_class}">
                    <div class="verdict-title">AI RECOMMENDATION</div>
                    <h1 class="verdict-value">{decision}</h1>
                    <div style="margin-top: 10px; font-weight: 500;">{icon} Confidence: {100-(prob*100):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_right:
                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "<b>Risk Probability Score</b>", 'font': {'size': 16, 'color': '#64748B'}},
                    delta = {'reference': 59, 'increasing': {'color': "#EF4444"}, 'decreasing': {'color': "#10B981"}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748B"},
                        'bar': {'color': "#1E293B"}, # Dark Pointer
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#E2E8F0",
                        'steps': [
                            {'range': [0, 59], 'color': "#D1FAE5"},  # Safe Green
                            {'range': [59, 100], 'color': "#FEE2E2"}], # Danger Red
                        'threshold': {
                            'line': {'color': "#EF4444", 'width': 4},
                            'thickness': 0.75,
                            'value': 59}}))
                
                fig.update_layout(height=200, margin=dict(l=20,r=20,t=30,b=20), paper_bgcolor="rgba(0,0,0,0)", font={'family': "Inter"})
                st.plotly_chart(fig, use_container_width=True)

            # --- 3. METRICS ROW ---
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">CALCULATED RISK</div>
                    <div class="metric-value">{prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">FINANCIAL EXPOSURE</div>
                    <div class="metric-value">${exposure:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with m3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">SAFETY THRESHOLD</div>
                    <div class="metric-value">{threshold:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            # --- 4. EXPLAINABILITY ---
            st.markdown("---")
            st.subheader(" Logic Explanation")
            
            exp_c1, exp_c2 = st.columns([1, 1])
            with exp_c1:
                st.info(f"""
                **The Verdict:**
                The applicant's risk score is **{prob:.1%}**.
                
                This is **{'LOWER' if decision=='APPROVE' else 'HIGHER'}** than the optimized threshold of **{threshold}**, triggering the **{decision}** outcome.
                """)
            with exp_c2:
                st.write("**Primary Risk Drivers:**")
                st.markdown(f"""
                - **LTV Ratio:** `{ltv:.1f}%` 
                  *Status:* {'🔴 High Leverage' if ltv > 80 else '🟢 Safe Equity'}
                
                - **Credit History:** `{credit_score}` 
                  *Status:* {'🔴 Subprime' if credit_score < 650 else '🟢 Prime'}
                
                - **LTI Ratio:** `{loan_amt/income:.1f}x` 
                  *Status:* Debt-to-Income load
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
        **1. The "Vanilla" Baseline (Red Bar):**
        Standard models optimize for *Accuracy*. On a dataset with 94% good loans, a standard model achieves 94% accuracy by simply predicting "No Default" for everyone.
        * **Result:** It captures **0%** of actual defaults.
        * **Financial Consequence:** The bank absorbs **$35.3 Million** in losses.
        
        **2. The AuraFin Solution (Green Bar):**
        We optimized for **Net Profit**. By implementing Cost-Sensitive Learning (15x penalty) and Threshold Optimization (0.59), we catch 71% of defaults while preserving safe customers.
        * **Result:** Minimizes loss to **$13.5 Million**.
        """)
        st.success(" **NET SAVINGS: $21.8 Million per batch**")
        
    with col_i2:
        if os.path.exists("images/11_true_business_impact.png"):
            st.image("images/11_true_business_impact.png", caption="Portfolio Loss Simulation", use_container_width=True)
        else:
            st.warning("Chart missing: images/11_true_business_impact.png")

# =========================================================
# TAB 3: METHODOLOGY
# =========================================================
with tab_research:
    st.markdown("## Engineering Methodology")
    st.markdown("A deep dive into the end-to-end data science lifecycle.")
    
    # 1. PROBLEM
    st.markdown("### 1. Problem Statement")
    st.markdown("Standard ML models optimize for accuracy. In credit risk, this is fatal because the cost of a False Negative (Default: \$60k) is 10x the cost of a False Positive (Rejection: \$6k).")
    
    st.markdown("---")
    
    # 2. FORENSICS
    st.markdown("### 2. Data Forensics & Cleaning")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Leakage Detection:** We identified that `rate_of_interest` was missing for 99% of rejected loans. This confirms it is a post-approval variable. We removed it to prevent Data Leakage.")
    with c2:
        if os.path.exists("images/01_leakage_evidence.png"):
            st.image("images/01_leakage_evidence.png", caption="Leakage Evidence", use_container_width=True)
    
    st.markdown("---")
    
    # 3. ENGINEERING
    st.markdown("### 3. Strategic Feature Engineering")
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