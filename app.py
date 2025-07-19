import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Load model
with open("best_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load x_test to get correct column structure for SHAP input
x_test = pd.read_csv("x_test.csv")
expected_columns = x_test.columns.tolist()

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** based on applicant details and explains the decision using SHAP.")

st.header("📋 Enter Applicant Details")

# Input form
no_of_dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
income_annum = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, value=700)
residential_assets_value = st.number_input("Residential Asset Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Asset Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Asset Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

if st.button("Predict"):
    try:
        # Input as DataFrame
        input_data = pd.DataFrame([{
            'no_of_dependents': no_of_dependents,
            'education': education,
            'self_employed': self_employed,
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value
        }])

        # One-hot encode and reindex to match model input
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

        # Prediction
        prediction = model.predict(input_encoded)[0]
        prediction_proba = model.predict_proba(input_encoded)[0]

        # Display result
        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        st.subheader("🔍 SHAP Explanation for this Applicant")

        # SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_encoded)

        # Show force plot only if binary classification
        if len(shap_values) == 2:
            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], input_encoded), height=300)
        else:
            st_shap(shap.force_plot(explainer.expected_value, shap_values, input_encoded), height=300)

        # Optional: Bar chart
        st.markdown("#### Feature Impact (Bar)")
        plt.title("SHAP Feature Impact")
        shap.summary_plot(shap_values[1], input_encoded, plot_type="bar", show=False)
        st.pyplot(bbox_inches='tight')
        plt.clf()

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction or explanation: {e}")
