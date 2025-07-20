import streamlit as st
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt

from streamlit_shap import st_shap

# 🌟 Load the trained XGBoost model
with open('best_xgb_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 🌟 Define app UI
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("Predict **Loan Approval** and understand feature impact using **SHAP**.")

# 📥 User input form
with st.form("loan_form"):
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    loan_amount = st.number_input("Loan Amount", min_value=10000, step=1000)
    loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=360, step=6)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

    submitted = st.form_submit_button("Predict")

# 🔍 Map categorical values
def map_inputs():
    return {
        "no_of_dependents": no_of_dependents,
        "education": 1 if education == "Graduate" else 0,
        "self_employed": 1 if self_employed == "Yes" else 0,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "bank_asset_value": bank_asset_value
    }

# 📊 Predict and explain
if submitted:
    input_data = map_inputs()
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # 🧠 Display result
    st.subheader("Prediction Result")
    st.markdown(f"🔹 **Loan Status**: {'Approved ✅' if prediction == 1 else 'Rejected ❌'}")
    st.markdown(f"🔹 **Approval Probability**: `{prediction_proba:.2f}`")

    # ⚡ SHAP Explanation
    st.subheader("Feature Impact (SHAP)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    shap.initjs()
    fig = shap.force_plot(explainer.expected_value, shap_values, input_df, matplotlib=True, figsize=(12, 3))
    st.pyplot(fig)
