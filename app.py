import shap
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load x_test for SHAP
x_test = pd.read_csv('x_test.csv')

# Streamlit App
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts Loan Approval and explains the result using SHAP.")

# Input UI
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income (₹)", min_value=0)
loan_amount = st.number_input("Loan Amount (₹)", min_value=0)
loan_term = st.number_input("Loan Term (in months)", min_value=0)
cibil_score = st.slider("CIBIL Score", min_value=300, max_value=900, step=1)
res_asset = st.number_input("Residential Asset Value", min_value=0)
comm_asset = st.number_input("Commercial Asset Value", min_value=0)
lux_asset = st.number_input("Luxury Asset Value", min_value=0)
bank_asset = st.number_input("Bank Asset Value", min_value=0)

# Prepare input
input_data = pd.DataFrame({
    'no_of_dependents': [no_of_dependents],
    'education': [1 if education == "Graduate" else 0],
    'self_employed': [1 if self_employed == "Yes" else 0],
    'income_annum': [income_annum],
    'loan_amount': [loan_amount],
    'loan_term': [loan_term],
    'cibil_score': [cibil_score],
    'residential_assets_value': [res_asset],
    'commercial_assets_value': [comm_asset],
    'luxury_assets_value': [lux_asset],
    'bank_asset_value': [bank_asset]
})

# Prediction
if st.button("Predict Loan Approval"):
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]

    st.markdown(f"**Prediction:** {'✅ Loan Approved' if prediction == 1 else '❌ Loan Rejected'}")
    st.markdown(f"**Approval Probability:** `{prediction_proba:.2f}`")

    # SHAP Explanation
    st.subheader("SHAP Explanation for this Applicant")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_test)
        st.markdown("🔍 Feature impact:")

        # Combine x_test and input for display context
        x_display = pd.concat([input_data, x_test.iloc[:10]], axis=0)

        # SHAP force plot fallback using summary bar plot
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[1], x_display, plot_type="bar", show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ SHAP explanation failed: {e}")
