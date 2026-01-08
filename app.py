import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("rf_model_4_features.pkl")

import pandas as pd
import matplotlib.pyplot as plt


# Page config
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📊 Employee Attrition Prediction")
st.caption("Predict attrition risk using machine learning")

st.markdown("---")

# Input section
st.subheader("🧑‍💼 Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 60, 30)
    years_at_company = st.number_input("Years at Company", 0, 40, 5)

with col2:
    monthly_income = st.number_input("Monthly Income", 1000, 200000, 30000)
    overtime = st.selectbox("OverTime", ["No", "Yes"])

# Encode OverTime
overtime = 1 if overtime == "Yes" else 0

st.markdown("---")

# Prediction
st.markdown("### 🔄 What-If Analysis")

salary_change = st.slider(
    "Simulate salary change (%)",
    min_value=-20,
    max_value=50,
    value=0,
    step=5
)

adjusted_income = int(monthly_income * (1 + salary_change / 100))
st.caption(f"Adjusted Monthly Income: ₹{adjusted_income}")

if st.button("🔍 Predict Attrition Risk"):
    input_data = np.array([[age, adjusted_income, years_at_company, overtime]])

    # Probability of attrition
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("📈 Prediction Result")

    risk_percent = int(proba * 100)
    st.metric(label="Attrition Risk", value=f"{risk_percent} %")
    st.progress(risk_percent)

    # Confidence
    if proba < 0.25 or proba > 0.75:
        confidence = "High"
    elif proba < 0.4 or proba > 0.6:
        confidence = "Medium"
    else:
        confidence = "Low"

    st.caption(f"🔍 Model Confidence: **{confidence}**")

    # Risk interpretation
    if proba >= 0.6:
        st.error("🔴 High Risk: Immediate HR attention recommended")
    elif proba >= 0.35:
        st.warning("🟡 Medium Risk: Monitor and engage employee")
    else:
        st.success("🟢 Low Risk: Employee likely to stay")
st.markdown("---")
st.subheader("📊 Feature Importance")

feature_names = ["Age", "Monthly Income", "Years at Company", "OverTime"]
importances = model.feature_importances_

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=True)

fig, ax = plt.subplots()
ax.barh(fi_df["Feature"], fi_df["Importance"])
ax.set_xlabel("Importance Score")
ax.set_title("Feature Importance (Random Forest)")

st.pyplot(fig)

    # ---------- DOWNLOAD REPORT (INSIDE BLOCK) ----------
if st.button("🔍 Predict Attrition Risk"):
    input_data = np.array([[age, adjusted_income, years_at_company, overtime]])

    # Predict probability
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("📈 Prediction Result")

    risk_percent = int(proba * 100)
    st.metric(label="Attrition Risk", value=f"{risk_percent} %")
    st.progress(risk_percent)

    # Confidence logic
    if proba < 0.25 or proba > 0.75:
        confidence = "High"
    elif proba < 0.4 or proba > 0.6:
        confidence = "Medium"
    else:
        confidence = "Low"

    st.caption(f"🔍 Model Confidence: **{confidence}**")

    # Risk interpretation
    if proba >= 0.6:
        st.error("🔴 High Risk: Immediate HR attention recommended")
    elif proba >= 0.35:
        st.warning("🟡 Medium Risk: Monitor and engage employee")
    else:
        st.success("🟢 Low Risk: Employee likely to stay")

    # -------- DOWNLOAD REPORT (INSIDE BLOCK) --------
    report_df = pd.DataFrame({
        "Age": [age],
        "Monthly Income": [adjusted_income],
        "Years at Company": [years_at_company],
        "OverTime": ["Yes" if overtime == 1 else "No"],
        "Attrition Risk Probability": [round(proba, 2)]
    })

    csv = report_df.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv,
        file_name="attrition_prediction_report.csv",
        mime="text/csv"
    )

    st.caption(
        "⚠️ Predictions are probabilistic and should be used as decision support, not final judgment."
    )

with st.expander("🧠 How is this prediction made?"):
    st.write("""
    - The model is trained on historical HR data.
    - It uses a Random Forest classifier.
    - The output is a probability score, not a hard rule.
    - Higher probability indicates higher attrition risk.
    """)
st.markdown("---")
st.subheader("📁 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload employee CSV", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    batch_df["OverTime"] = batch_df["OverTime"].map({"Yes": 1, "No": 0})

    batch_predictions = model.predict_proba(batch_df)[:, 1]
    batch_df["Attrition Risk"] = batch_predictions.round(2)

    st.dataframe(batch_df)
with st.expander("ℹ️ Feature Explanation"):
    st.markdown("""
    - **Age**: Career stage indicator  
    - **Monthly Income**: Compensation satisfaction proxy  
    - **Years at Company**: Employee loyalty & stability  
    - **OverTime**: Workload & burnout signal  
    """)

