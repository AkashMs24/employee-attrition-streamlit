# app.py
"""
Enhanced AttritionIQ - Explainable ML Decision Support System
Production-ready Streamlit application with research-level features
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.temporal_analysis import TemporalAttritionAnalysis, EarlyWarningSystem
from src.causal_interventions import InterventionRecommender
from src.risk_clustering import RiskSegmentation
from src.fairness_audit import FairnessAudit, generate_fairness_report
from src.monitoring import ModelPerformanceMonitor, DataQualityMonitor, AlertingSystem
from src.explainability import SHAPExplainer, FeatureImportanceExplainer, CounterfactualExplainer

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="AttritionIQ - Research Edition",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING
# ============================================================================

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ff0000;
    }
    .risk-medium {
        background-color: #ffffcc;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #ffaa00;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #00aa00;
    }
    .header-main {
        color: #1f77b4;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model():
    """Load pre-trained model"""
    model_path = "rf_model_4_features.pkl"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found")
        st.stop()
    return joblib.load(model_path)

@st.cache_resource
def load_scaler():
    """Load feature scaler"""
    scaler_path = "scaler.pkl"
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

model = load_model()
scaler = load_scaler()

# ============================================================================
# INITIALIZATION
# ============================================================================

temporal_analyzer = TemporalAttritionAnalysis(model)
intervention_recommender = InterventionRecommender(model)
risk_segmenter = RiskSegmentation(n_clusters=4)
fairness_auditor = FairnessAudit(model)
performance_monitor = ModelPerformanceMonitor()
data_quality_monitor = DataQualityMonitor()
explainability_explainer = SHAPExplainer(model)
feature_importance_explainer = FeatureImportanceExplainer(model)
counterfactual_explainer = CounterfactualExplainer(model)

FEATURE_NAMES = ['Age', 'Monthly Income', 'Years at Company', 'OverTime']

# ============================================================================
# HELPER: ROBUST FEATURE CLEANING FOR UPLOADED CSVs
# ============================================================================

def encode_binary_labels(series):
    """
    Convert a column that may contain Yes/No, True/False, 1/0, or mixed
    case/whitespace variants into clean numeric 0/1 (float, NaN for unparseable).
    """
    label_map = {
        'yes': 1, 'no': 0,
        'true': 1, 'false': 0,
        '1': 1, '0': 0,
        1: 1, 0: 0, True: 1, False: 0
    }
    cleaned = series.apply(lambda v: v.strip().lower() if isinstance(v, str) else v)
    mapped = cleaned.map(label_map)
    # Fall back to numeric coercion for anything not caught by the map (e.g. floats like 1.0)
    still_missing = mapped.isna() & series.notna()
    if still_missing.any():
        mapped.loc[still_missing] = pd.to_numeric(series.loc[still_missing], errors='coerce')
    return mapped


def prepare_hr_features(df, feature_cols=('Age', 'MonthlyIncome', 'YearsAtCompany', 'OverTime')):
    """
    Validate and clean the 4 model features from an uploaded CSV.

    - Checks the required columns exist at all (returns a clear error message if not).
    - Coerces Age / MonthlyIncome / YearsAtCompany to numeric (strips commas/currency symbols).
    - Maps OverTime (Yes/No, True/False, 1/0, various casing) to 0/1.
    - Drops rows that still can't be made numeric after cleaning, rather than crashing.

    Returns:
        clean_df: DataFrame with only valid, fully-numeric rows for feature_cols
        X: numpy float64 array of shape (n_valid_rows, 4), safe to pass to model.predict_proba
        dropped_count: number of rows dropped due to bad/missing values
        missing_cols: list of required columns not present in df at all (if any)
    """
    feature_cols = list(feature_cols)
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        return None, None, 0, missing_cols

    df = df.copy()

    # --- OverTime: handle Yes/No, True/False, 1/0, and stray whitespace/case ---
    df['OverTime'] = encode_binary_labels(df['OverTime'])

    # --- Numeric columns: strip commas/currency symbols/whitespace, coerce ---
    for col in ['Age', 'MonthlyIncome', 'YearsAtCompany']:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.replace('$', '', regex=False)
            .str.strip()
        )
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Find rows that failed conversion in ANY required column ---
    bad_mask = df[feature_cols].isna().any(axis=1)
    dropped_count = int(bad_mask.sum())

    clean_df = df.loc[~bad_mask].reset_index(drop=True)
    X = clean_df[feature_cols].astype(float).values

    return clean_df, X, dropped_count, []


def get_audit_labels(clean_df):
    """
    Build a clean 0/1 integer numpy array for the target column ('Attrition')
    to use as y_true in fairness auditing. Handles Yes/No, True/False, 1/0.
    Rows where Attrition can't be parsed are treated as 0 (non-attrition),
    matching the previous default behavior, but a warning count is returned.
    """
    if 'Attrition' not in clean_df.columns:
        return np.zeros(len(clean_df), dtype=int), 0

    encoded = encode_binary_labels(clean_df['Attrition'])
    unparseable = int(encoded.isna().sum())
    encoded = encoded.fillna(0).astype(int)
    return encoded.values, unparseable

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.markdown("# 🎯 AttritionIQ Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Feature:",
    [
        "🏠 Dashboard",
        "👤 Individual Prediction",
        "📈 Risk Trajectory",
        "💡 Interventions",
        "👥 Employee Segmentation",
        "⚖️ Fairness Audit",
        "📊 Model Monitoring",
        "🔍 Explainability Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
### 📖 About AttritionIQ Research Edition
- **Individual Predictions** with confidence scores
- **Risk Trajectory Analysis** - predict how risk changes over time
- **Causal Interventions** - specific actions with ROI
- **Employee Clustering** - segment by risk profile
- **Fairness Audits** - ensure no demographic bias
- **Production Monitoring** - detect model drift
- **SHAP Explanations** - detailed feature contributions
""")

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================

if page == "🏠 Dashboard":
    st.markdown('<div class="header-main">📊 AttritionIQ Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to **AttritionIQ** - an explainable ML system for predicting and preventing employee attrition.
    
    ### 🌟 Key Features
    - **Individual Risk Scoring** - Get probability of attrition (0-100%)
    - **Risk Trajectory** - Predict how risk changes over months/quarters
    - **Smart Interventions** - Recommended actions with expected impact
    - **Employee Clusters** - Identify high-risk groups with common issues
    - **Fairness Checks** - Ensure fair treatment across demographics
    - **Production Monitoring** - Track model performance over time
    """)
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", "Random Forest")
    with col2:
        st.metric("Features Used", "4 Core")
    with col3:
        st.metric("Risk Scale", "0-100%")
    with col4:
        st.metric("Status", "🟢 Active")
    
    st.markdown("---")
    
    # Quick prediction demo
    st.subheader("⚡ Quick Demo - Try a Prediction")
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        demo_age = st.slider("Demo Age", 18, 60, 35)
        demo_tenure = st.slider("Demo Years", 0, 40, 5)
    
    with demo_col2:
        demo_income = st.slider("Demo Income", 1000, 200000, 50000)
        demo_ot = st.selectbox("Demo Overtime", ["No", "Yes"])
    
    if st.button("🔍 Get Quick Prediction"):
        demo_ot_enc = 1 if demo_ot == "Yes" else 0
        demo_risk = model.predict_proba([[demo_age, demo_income, demo_tenure, demo_ot_enc]])[0][1]
        
        st.success(f"**Attrition Risk: {demo_risk*100:.1f}%**")
        
        if demo_risk > 0.65:
            st.error("🔴 HIGH RISK - Immediate intervention needed")
        elif demo_risk > 0.35:
            st.warning("🟡 MEDIUM RISK - Monitor closely")
        else:
            st.success("🟢 LOW RISK - Employee engagement is strong")

# ============================================================================
# PAGE 2: INDIVIDUAL PREDICTION
# ============================================================================

elif page == "👤 Individual Prediction":
    st.markdown('<div class="header-main">👤 Individual Employee Risk Assessment</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", 18, 65, 30, step=1)
        tenure = st.number_input("Years at Company", 0, 50, 5, step=1)
    
    with col2:
        income = st.number_input("Monthly Income (₹)", 1000, 500000, 50000, step=1000)
        overtime = st.selectbox("Overtime", ["No", "Yes"])
    
    overtime_enc = 1 if overtime == "Yes" else 0
    
    # What-if slider
    st.markdown("### 📊 What-If Salary Simulation")
    salary_change = st.slider(
        "Simulate salary change (%)",
        min_value=-20,
        max_value=50,
        value=0,
        step=5,
    )
    
    adjusted_income = int(income * (1 + salary_change / 100))
    st.info(f"Adjusted Monthly Income: ₹{adjusted_income:,}")
    
    # Make prediction
    if st.button("🔍 Analyze Employee Risk", use_container_width=True):
        
        # Get prediction
        input_data = np.array([[age, adjusted_income, tenure, overtime_enc]])
        proba = model.predict_proba(input_data)[0][1]
        risk_pct = int(proba * 100)
        
        # Determine confidence
        if proba < 0.25 or proba > 0.75:
            confidence = "High"
        elif proba < 0.4 or proba > 0.6:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Risk categorization
        if proba >= 0.60:
            risk_color = "#ff4444"
            verdict = "🔴 HIGH RISK"
        elif proba >= 0.35:
            risk_color = "#ffb800"
            verdict = "🟡 MEDIUM RISK"
        else:
            risk_color = "#00e676"
            verdict = "🟢 LOW RISK"
        
        # Display risk score
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Score", f"{risk_pct}%")
        with col2:
            st.metric("Confidence", confidence)
        with col3:
            st.metric("Verdict", verdict)
        
        st.progress(risk_pct / 100)
        
        # Feature importance
        st.subheader("📊 Factor Analysis")
        
        fi_df = feature_importance_explainer.get_model_feature_importance(FEATURE_NAMES)
        if fi_df is not None:
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#c8ff00' if v == fi_df['importance'].max() else '#2a2a2a' 
                     for v in fi_df['importance']]
            ax.barh(fi_df['feature'], fi_df['importance'], color=colors)
            ax.set_xlabel('Importance Score')
            ax.set_title('Key Factors in Attrition Prediction')
            st.pyplot(fig)
        
        # Interventions
        st.subheader("💡 Recommended Interventions")
        
        interventions = intervention_recommender.recommend_interventions(
            {'age': age, 'income': adjusted_income, 'years': tenure, 'ot_encoded': overtime_enc},
            proba
        )
        
        if interventions:
            for i, intervention in enumerate(interventions[:3], 1):
                with st.expander(f"Option {i}: {intervention['action']}", expanded=i==1):
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Current Risk", f"{intervention['current_risk']:.1%}")
                    col_b.metric("After Action", f"{intervention['new_risk']:.1%}")
                    col_c.metric("Improvement", f"{intervention['improvement']:.1f}%")
                    
                    st.write(f"**Details:** {intervention.get('details', '')}")
                    st.write(f"**Cost:** {intervention.get('cost', 'N/A')}")
                    st.write(f"**Timeline:** {intervention.get('timeline', 'N/A')}")
        
        # Download report
        report_data = pd.DataFrame({
            'Metric': ['Age', 'Monthly Income', 'Years at Company', 'Overtime', 'Attrition Risk', 'Verdict'],
            'Value': [age, f"₹{adjusted_income:,}", tenure, overtime, f"{risk_pct}%", verdict]
        })
        
        csv = report_data.to_csv(index=False)
        st.download_button(
            "📥 Download Report",
            csv,
            "prediction_report.csv",
            "text/csv"
        )

# ============================================================================
# PAGE 3: RISK TRAJECTORY
# ============================================================================

elif page == "📈 Risk Trajectory":
    st.markdown('<div class="header-main">📈 Risk Trajectory Analysis</div>', unsafe_allow_html=True)
    
    st.info("Track how employee attrition risk changes over time to catch early warning signs")
    
    # Sample trajectory data
    st.subheader("📊 Example: Employee Risk Over Time")
    
    # Create sample data
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    col1, col2 = st.columns(2)
    
    with col1:
        emp_age = st.number_input("Employee Age (for trajectory)", 25, 60, 32, key="traj_age")
        emp_income = st.number_input("Base Income (for trajectory)", 10000, 200000, 50000, key="traj_income")
    
    with col2:
        emp_tenure = st.number_input("Current Tenure (for trajectory)", 1, 40, 5, key="traj_tenure")
        emp_ot = st.selectbox("Overtime Status (for trajectory)", ["No", "Yes"], key="traj_ot")
    
    emp_ot_enc = 1 if emp_ot == "Yes" else 0
    
    # Simulate risk trajectory
    sample_history = [
        {'age': emp_age - 0.25, 'income': emp_income, 'years': emp_tenure - 0.25, 'overtime': emp_ot_enc, 'period': 'Q1'},
        {'age': emp_age - 0.15, 'income': emp_income, 'years': emp_tenure - 0.15, 'overtime': emp_ot_enc, 'period': 'Q2'},
        {'age': emp_age - 0.05, 'income': emp_income, 'years': emp_tenure - 0.05, 'overtime': emp_ot_enc, 'period': 'Q3'},
        {'age': emp_age, 'income': emp_income, 'years': emp_tenure, 'overtime': emp_ot_enc, 'period': 'Q4'},
    ]
    
    if st.button("📈 Analyze Trajectory"):
        trajectory = temporal_analyzer.calculate_risk_trajectory(sample_history)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Risk", f"{trajectory['current_risk']:.1%}")
        with col2:
            st.metric("Risk Velocity", f"{trajectory['risk_velocity']:+.2%}/quarter")
        with col3:
            st.metric("Trajectory", trajectory['risk_trajectory'])
        
        # Plot trajectory
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(trajectory['risk_history'])), trajectory['risk_history'], 
               marker='o', linewidth=2, markersize=8, label='Historical Risk')
        
        # Projection
        future_periods = 3
        future_risks = [
            trajectory['current_risk'] + (trajectory['risk_velocity'] * (i+1))
            for i in range(future_periods)
        ]
        future_x = [len(trajectory['risk_history']) - 1 + i for i in range(1, future_periods + 1)]
        
        ax.plot(future_x, future_risks, marker='s', linestyle='--', 
               linewidth=2, markersize=6, label='Projection', alpha=0.7)
        
        ax.axhline(y=0.35, color='orange', linestyle=':', alpha=0.5, label='Medium Risk Threshold')
        ax.axhline(y=0.60, color='red', linestyle=':', alpha=0.5, label='High Risk Threshold')
        
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Attrition Risk')
        ax.set_title('Employee Risk Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        st.pyplot(fig)
        
        # Alert level
        st.markdown(f"### Alert Level: {trajectory['alert_level']}")
        
        if trajectory['recommendation']:
            st.markdown("### 📋 Recommendations")
            for rec in trajectory['recommendation']:
                st.write(f"• {rec}")

# ============================================================================
# PAGE 4: INTERVENTIONS
# ============================================================================

elif page == "💡 Interventions":
    st.markdown('<div class="header-main">💡 Intervention Recommendation Engine</div>', unsafe_allow_html=True)
    
    st.info("Get specific HR actions with predicted ROI for each at-risk employee")
    
    col1, col2 = st.columns(2)
    
    with col1:
        int_age = st.number_input("Age (Intervention)", 18, 65, 32, key="int_age")
        int_tenure = st.number_input("Tenure (Intervention)", 0, 50, 5, key="int_tenure")
    
    with col2:
        int_income = st.number_input("Income (Intervention)", 1000, 500000, 50000, key="int_income")
        int_ot = st.selectbox("Overtime (Intervention)", ["No", "Yes"], key="int_ot")
    
    int_ot_enc = 1 if int_ot == "Yes" else 0
    
    if st.button("💡 Get Interventions"):
        current_risk = model.predict_proba([[int_age, int_income, int_tenure, int_ot_enc]])[0][1]
        
        interventions = intervention_recommender.recommend_interventions(
            {'age': int_age, 'income': int_income, 'years': int_tenure, 'ot_encoded': int_ot_enc},
            current_risk
        )
        
        st.subheader(f"Current Risk: {current_risk:.1%}")
        
        for i, intervention in enumerate(interventions[:5], 1):
            with st.expander(
                f"💼 {i}. {intervention['action']} - {intervention['improvement']:.1f}% improvement",
                expanded=i==1
            ):
                col_x, col_y, col_z = st.columns(3)
                
                col_x.metric(
                    "Current Risk",
                    f"{intervention['current_risk']:.1%}"
                )
                col_y.metric(
                    "After Intervention",
                    f"{intervention['new_risk']:.1%}"
                )
                col_z.metric(
                    "Risk Reduction",
                    f"{intervention['improvement']:.1f}%"
                )
                
                st.markdown(f"""
                **Intervention:** {intervention.get('details', 'N/A')}
                
                **Expected Cost:** {intervention.get('cost', 'N/A')}
                
                **Timeline:** {intervention.get('timeline', 'N/A')}
                
                **Effort Level:** {intervention.get('effort', 'N/A')}
                
                **Success Probability:** {intervention.get('probability', 'N/A')}
                """)

# ============================================================================
# PAGE 5: EMPLOYEE SEGMENTATION
# ============================================================================

elif page == "👥 Employee Segmentation":
    st.markdown('<div class="header-main">👥 Employee Risk Segmentation</div>', unsafe_allow_html=True)
    
    st.info("Segment employees into clusters with similar risk profiles for targeted strategies")
    
    # Upload CSV for batch analysis
    uploaded_file = st.file_uploader("Upload employee CSV for segmentation", type=['csv'])
    
    if uploaded_file is not None:
        employees_df = pd.read_csv(uploaded_file)
        
        st.success(f"Loaded {len(employees_df)} employees")
        
        if st.button("🔬 Segment Employees"):
            clean_df, X_check, dropped_count, missing_cols = prepare_hr_features(employees_df)

            if missing_cols:
                st.error(
                    f"❌ This file is missing required column(s): **{', '.join(missing_cols)}**\n\n"
                    f"This model expects **employee HR data** with these columns: "
                    f"`Age, MonthlyIncome, YearsAtCompany, OverTime`.\n\n"
                    f"Your file has these columns instead: `{', '.join(employees_df.columns.tolist())}`\n\n"
                    f"💡 It looks like you may have uploaded a different dataset — please upload "
                    f"an employee HR dataset (like the IBM HR Attrition dataset) that contains "
                    f"`Age`, `MonthlyIncome`, `YearsAtCompany`, and `OverTime` columns."
                )
                with st.expander("📄 Preview of uploaded file"):
                    st.dataframe(employees_df.head(), use_container_width=True)
                st.stop()

            if dropped_count > 0:
                st.warning(
                    f"⚠️ Dropped {dropped_count} row(s) with missing/non-numeric "
                    f"values in Age, MonthlyIncome, YearsAtCompany, or OverTime. "
                    f"Proceeding with {len(clean_df)} valid rows."
                )

            if len(clean_df) == 0:
                st.error("❌ No valid rows remain after cleaning. Please check your CSV.")
                st.stop()

            with st.spinner("Segmenting employees..."):
                try:
                    segmented_df, clusters = risk_segmenter.segment_employees(clean_df, model)
                except Exception as e:
                    st.error(f"❌ Segmentation failed: {e}")
                    st.stop()
            
            # Display cluster summary
            st.subheader("📊 Cluster Summary")
            
            summary_df = risk_segmenter.get_cluster_summary(clusters)
            st.dataframe(summary_df, use_container_width=True)
            
            # Visualize clusters
            try:
                fig = risk_segmenter.visualize_clusters(segmented_df, clusters)
                st.pyplot(fig)
            except Exception:
                st.info("Visualization not available")
            
            # Detailed cluster information
            st.subheader("📋 Cluster Recommendations")
            
            for cluster_id, cluster_info in clusters.items():
                with st.expander(f"{cluster_info['name']} ({cluster_info['size']} employees)", 
                               expanded=cluster_id==0):
                    st.write(f"**Description:** {cluster_info['description']}")
                    st.write(f"**Primary Issue:** {cluster_info['primary_issue']}")
                    st.write(f"**Recommendation:** {cluster_info['recommendation']}")
                    
                    if cluster_info['urgent_actions']:
                        st.write("**Urgent Actions:**")
                        for action in cluster_info['urgent_actions']:
                            st.write(f"  • {action}")
                    
                    col_a, col_b = st.columns(2)
                    col_a.metric("Avg Risk", f"{cluster_info['avg_risk']:.1%}")
                    col_b.metric("Avg Income", f"₹{cluster_info['avg_income']:,.0f}")

# ============================================================================
# PAGE 6: FAIRNESS AUDIT
# ============================================================================

elif page == "⚖️ Fairness Audit":
    st.markdown('<div class="header-main">⚖️ Fairness & Bias Audit</div>', unsafe_allow_html=True)
    
    st.info("Ensure the model treats all demographic groups fairly")
    
    # Upload data for fairness audit
    uploaded_file = st.file_uploader("Upload data for fairness audit", type=['csv'], key="fairness_upload")
    
    if uploaded_file is not None:
        audit_df = pd.read_csv(uploaded_file)
        
        # Select demographic column
        demographic_col = st.selectbox("Select demographic column to audit", audit_df.columns)
        
        if st.button("🔍 Run Fairness Audit"):
            st.markdown("### ⏳ Running audit...")

            clean_df, X_audit, dropped_count, missing_cols = prepare_hr_features(audit_df)

            if missing_cols:
                st.error(
                    f"❌ This file is missing required column(s): **{', '.join(missing_cols)}**\n\n"
                    f"This model expects **employee HR data** with these columns: "
                    f"`Age, MonthlyIncome, YearsAtCompany, OverTime`.\n\n"
                    f"Your file has these columns instead: `{', '.join(audit_df.columns.tolist())}`\n\n"
                    f"💡 It looks like you may have uploaded a different dataset — please upload "
                    f"an employee HR dataset (like the IBM HR Attrition dataset) that contains "
                    f"`Age`, `MonthlyIncome`, `YearsAtCompany`, and `OverTime` columns."
                )
                with st.expander("📄 Preview of uploaded file"):
                    st.dataframe(audit_df.head(), use_container_width=True)
                st.stop()

            if dropped_count > 0:
                st.warning(
                    f"⚠️ Dropped {dropped_count} row(s) with missing/non-numeric "
                    f"values in Age, MonthlyIncome, YearsAtCompany, or OverTime. "
                    f"Proceeding with {len(clean_df)} valid rows."
                )

            if len(clean_df) == 0:
                st.error("❌ No valid rows remain after cleaning. Please check your CSV.")
                st.stop()

            try:
                # Build clean 0/1 target labels (handles Yes/No, True/False, etc.)
                y_audit, unparseable_labels = get_audit_labels(clean_df)

                if unparseable_labels > 0:
                    st.warning(
                        f"⚠️ {unparseable_labels} row(s) had an unrecognized 'Attrition' "
                        f"value and were treated as 'No' (0) for this audit."
                    )

                # Get audit results
                audit_results = fairness_auditor.audit_by_demographic(
                    X_audit,
                    y_audit,
                    clean_df,
                    demographic_col
                )
            except Exception as e:
                st.error(f"❌ Audit failed: {e}")
                st.stop()
            
            st.success("✓ Audit complete")
            
            # Display results by group
            st.subheader(f"Fairness Metrics by {demographic_col}")
            
            results_table = []
            for group, metrics in audit_results.items():
                results_table.append({
                    'Group': group,
                    'Sample Size': metrics['sample_size'],
                    'FPR': f"{metrics['false_positive_rate']:.2%}",
                    'FNR': f"{metrics['false_negative_rate']:.2%}",
                    'Precision': f"{metrics['precision']:.2%}"
                })
            
            st.dataframe(pd.DataFrame(results_table), use_container_width=True)
            
            # Bias detection
            bias_findings = fairness_auditor.detect_bias({demographic_col: audit_results})
            
            if bias_findings:
                st.warning("⚠️ Bias Detected")
                for finding in bias_findings:
                    st.error(f"**{finding['metric']}** - {finding['description']}")
            else:
                st.success("✓ No significant bias detected")

# ============================================================================
# PAGE 7: MODEL MONITORING
# ============================================================================

elif page == "📊 Model Monitoring":
    st.markdown('<div class="header-main">📊 Production Monitoring</div>', unsafe_allow_html=True)
    
    st.info("Monitor model performance and detect data/model drift")
    
    col1, col2 = st.columns(2)
    
    with col1:
        monitoring_metric = st.selectbox(
            "Select metric to monitor",
            ["Calibration", "Prediction Distribution", "Data Quality"]
        )
    
    with col2:
        window_size = st.number_input("Window size (predictions)", 50, 1000, 100)
    
    if st.button("📊 Generate Monitoring Report"):
        
        if monitoring_metric == "Calibration":
            st.subheader("📊 Calibration Report")
            st.info("Checks if predicted probabilities match actual outcomes")
            
            st.markdown("""
            **What is Calibration?**
            - When we say "70% risk", do approximately 70% actually leave?
            - Good calibration = trustworthy predictions
            
            **Expected vs Observed Rate:**
            - Expected: Average of predicted probabilities
            - Observed: Actual attrition rate
            
            **Calibration Error:**
            - < 10%: GOOD (model is well-calibrated)
            - 10-20%: WARNING (some drift)
            - > 20%: CRITICAL (retrain recommended)
            """)
            
            st.warning("ℹ️ Add historical prediction data to enable calibration monitoring")
        
        elif monitoring_metric == "Prediction Distribution":
            st.subheader("📊 Prediction Distribution Shift")
            st.info("Detects if model prediction patterns are changing")
            
            st.markdown("""
            **What is Distribution Shift?**
            - Are prediction probability distributions changing?
            - Indicates employee population or behavior changing
            
            **Actions if Shift Detected:**
            - Investigate what changed in the organization
            - Review recent hiring, layoffs, reorganizations
            - Consider model retraining
            """)
        
        elif monitoring_metric == "Data Quality":
            st.subheader("📊 Data Quality Checks")
            st.info("Monitor incoming data quality")
            
            st.markdown("""
            **Key Quality Checks:**
            - Missing values
            - Out-of-range values
            - Duplicate records
            - Anomalous values (outliers)
            
            **Quality Thresholds:**
            - Score > 90%: PASS
            - Score 70-90%: WARNING
            - Score < 70%: FAIL
            """)

# ============================================================================
# PAGE 8: EXPLAINABILITY ANALYSIS
# ============================================================================

elif page == "🔍 Explainability Analysis":
    st.markdown('<div class="header-main">🔍 Deep Explainability Analysis</div>', unsafe_allow_html=True)
    
    st.info("Understand predictions at multiple levels using SHAP, feature importance, and counterfactuals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exp_age = st.number_input("Age (Explainability)", 18, 65, 35, key="exp_age")
        exp_tenure = st.number_input("Tenure (Explainability)", 0, 50, 5, key="exp_tenure")
    
    with col2:
        exp_income = st.number_input("Income (Explainability)", 1000, 500000, 50000, key="exp_income")
        exp_ot = st.selectbox("Overtime (Explainability)", ["No", "Yes"], key="exp_ot")
    
    exp_ot_enc = 1 if exp_ot == "Yes" else 0
    
    if st.button("🔍 Analyze Explainability"):
        
        features_array = np.array([exp_age, exp_income, exp_tenure, exp_ot_enc])
        current_risk = model.predict_proba([features_array])[0][1]
        
        # Feature Importance
        st.subheader("📊 Feature Importance Analysis")
        
        fi_df = feature_importance_explainer.get_model_feature_importance(FEATURE_NAMES)
        if fi_df is not None:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(fi_df['feature'], fi_df['importance'], color='#4CAF50')
            ax.set_xlabel('Importance Score')
            st.pyplot(fig)
            
            st.write("**Interpretation:** Shows which features have the most influence on predictions globally")
        
        # Counterfactual Analysis
        st.subheader("🎯 Counterfactual Analysis")
        st.write("What changes would reduce attrition risk to low levels?")
        
        counterfactual = counterfactual_explainer.find_counterfactual(
            features_array,
            FEATURE_NAMES,
            target_probability=0.30
        )
        
        if 'counterfactuals' in counterfactual and counterfactual['counterfactuals']:
            for cf in counterfactual['counterfactuals'][:3]:
                with st.expander(cf['explanation']):
                    col_i, col_j = st.columns(2)
                    col_i.write(f"**Current:** {cf['current_value']:.0f}")
                    col_j.write(f"**Suggested:** {cf['suggested_value']:.0f}")
                    st.metric("Expected New Risk", f"{cf['new_risk']:.1%}")
        else:
            st.info("✓ Already at low risk level or limited improvement possible")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>AttritionIQ Research Edition | Built with Streamlit | Powered by scikit-learn, SHAP & XGBoost</p>
    <p>For production deployment, implement monitoring pipelines and retraining automation</p>
</div>
""", unsafe_allow_html=True)
