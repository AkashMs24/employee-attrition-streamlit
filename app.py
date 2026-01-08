import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("rf_model_4_features.pkl")

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
if st.button("🔍 Predict Attrition Risk"):
    input_data = np.array([[age, monthly_income, years_at_company, overtime]])

    # Probability of attrition
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("📈 Prediction Result")

    # Progress bar
    st.progress(int(proba * 100))

    st.write(f"**Attrition Risk Probability:** `{proba:.2f}`")

    # Risk interpretation
    if proba >= 0.6:
        st.error("🔴 High Risk: Employee is very likely to leave")
    elif proba >= 0.35:
        st.warning("🟡 Medium Risk: Employee may leave")
    else:
        st.success("🟢 Low Risk: Employee likely to stay")

    st.markdown("---")
    st.caption(
        "⚠️ This prediction is based on historical patterns and should be used as decision support, not as a final judgment."
    )
