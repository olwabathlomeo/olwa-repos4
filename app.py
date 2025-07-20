import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle

# Load model
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# App configuration
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** based on applicant information.")

st.header("📋 Applicant Information")

# User input form
with st.form(key='input_form'):
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income (KES)", min_value=0)
    loan_amount = st.number_input("Loan Amount Requested (KES)", min_value=0)
    loan_term = st.number_input("Loan Term (in months)", min_value=0)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
    residential_assets_value = st.number_input("Residential Assets Value (KES)", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value (KES)", min_value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value (KES)", min_value=0)
    bank_asset_value = st.number_input("Bank Asset Value (KES)", min_value=0)

    submit = st.form_submit_button("Predict")

if submit:
    # Create DataFrame from inputs
    input_dict = {
        'no_of_dependents': [no_of_dependents],
        'education': [1 if education == "Graduate" else 0],
        'self_employed': [1 if self_employed == "Yes" else 0],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    }

    input_df = pd.DataFrame(input_dict)

    # ✅ Predict
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # ✅ Display prediction
    st.subheader("🔍 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Loan Approved with a probability of {prediction_proba:.2f}")
    else:
        st.error(f"❌ Loan Rejected with a probability of {1 - prediction_proba:.2f}")

    # ✅ SHAP explanation
    st.subheader("📊 SHAP Explanation")

    # Initialize TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)

    # SHAP waterfall plot for this individual
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.write("Feature contributions to this prediction:")
    shap.plots.waterfall(shap_values[0], max_display=11)
