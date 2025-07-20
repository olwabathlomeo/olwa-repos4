import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle

# Load your trained model and explainer
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

explainer = shap.Explainer(model)

# Streamlit form for user input
st.title("🏦 Loan Approval Predictor with SHAP Explanation")

st.subheader("🔢 Enter Applicant Details:")
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education Level", options=["Not Graduate", "Graduate"])
self_employed = st.selectbox("Self Employed", options=["No", "Yes"])
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (Months)", min_value=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900)
res_asset = st.number_input("Residential Asset Value", min_value=0)
comm_asset = st.number_input("Commercial Asset Value", min_value=0)
lux_asset = st.number_input("Luxury Asset Value", min_value=0)
bank_asset = st.number_input("Bank Asset Value", min_value=0)

# Convert inputs to model format
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

applicant_data = pd.DataFrame([{
    'no_of_dependents': no_of_dependents,
    'education': education_val,
    'self_employed': self_employed_val,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': res_asset,
    'commercial_assets_value': comm_asset,
    'luxury_assets_value': lux_asset,
    'bank_asset_value': bank_asset
}])

# Predict and explain
if st.button("Predict and Explain"):
    prediction = model.predict(applicant_data)[0]
    prediction_label = "✅ Approved" if prediction == 1 else "❌ Rejected"
    st.subheader(f"Prediction: {prediction_label}")

    shap_values = explainer(applicant_data)

    st.subheader("🔍 SHAP Explanation:")

    # SHAP waterfall plot for class 1 (Approved)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap.Explanation(
        values=shap_values.values[0, :, 1],  # class 1
        base_values=shap_values.base_values[0, 1],
        data=applicant_data.values[0],
        feature_names=applicant_data.columns.tolist()
    ), show=False)
    st.pyplot(fig)
