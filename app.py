import streamlit as st
import numpy as np
import joblib

# Load trained model (4-feature model)
model = joblib.load("rf_model_4_features.pkl")

st.title("Employee Attrition Prediction")
st.write("Enter employee details below:")

# User inputs (must match training features order)
age = st.number_input("Age", min_value=18, max_value=60, value=30)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=30000)
years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
overtime = st.selectbox("OverTime", ["No", "Yes"])

# Encode OverTime (same as training)
overtime = 1 if overtime == "Yes" else 0

# Predict
if st.button("Predict Attrition"):
    input_data = np.array([[age, monthly_income, years_at_company, overtime]])

    # get probability of attrition (class = 1)
    proba = model.predict_proba(input_data)[0][1]

    st.write(f"Attrition risk probability: {proba:.2f}")

    if proba >= 0.35:
        st.error("⚠️ Employee is at risk of leaving")
    else:
        st.success("✅ Employee is likely to stay")

