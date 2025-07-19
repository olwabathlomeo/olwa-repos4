import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

# Load model
with open("best_rf_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load test data for SHAP summary plot
x_test = pd.read_csv("x_test.csv")

# SHAP explainer
explainer = shap.TreeExplainer(model)

# Streamlit app layout
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts **Loan Approval** and explains the result using **SHAP**.")

# Input form
st.header("Enter Applicant Details")

no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (₹)", step=1000)
loan_amount = st.number_input("Loan Amount (₹)", step=1000)
loan_term = st.number_input("Loan Term (in months)", step=1)
cibil_score = st.slider("CIBIL Score", 300, 900)
residential_assets_value = st.number_input("Residential Asset Value", step=1000)
commercial_assets_value = st.number_input("Commercial Asset Value", step=1000)
luxury_assets_value = st.number_input("Luxury Asset Value", step=1000)
bank_asset_value = st.number_input("Bank Asset Value", step=1000)

# Predict
if st.button("Predict Loan Status"):
    input_data = pd.DataFrame([[
        no_of_dependents, education, self_employed, income_annum, loan_amount,
        loan_term, cibil_score, residential_assets_value, commercial_assets_value,
        luxury_assets_value, bank_asset_value
    ]], columns=[
        'no_of_dependents', 'education', 'self_employed', 'income_annum',
        'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
        'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
    ])

    # Encode categorical
    input_data['education'] = input_data['education'].map({"Graduate": 1, "Not Graduate": 0})
    input_data['self_employed'] = input_data['self_employed'].map({"Yes": 1, "No": 0})

    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
        st.subheader(f"Prediction: {result}")

        # SHAP force plot for this applicant
        shap_values = explainer.shap_values(input_data)
        st.subheader("🔍 SHAP Explanation for this Applicant")
        shap.initjs()
        st.components.v1.html(shap.plots.force(shap_values[1][0], matplotlib=False), height=300)

        # SHAP summary plot for model explanation
        st.subheader("📊 Overall Feature Importance (SHAP Summary Plot)")
        shap_values_all = explainer.shap_values(x_test)
        plt.figure()
        shap.summary_plot(shap_values_all[1], x_test, show=False)
        st.pyplot(bbox_inches='tight')

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction or explanation: {e}")
