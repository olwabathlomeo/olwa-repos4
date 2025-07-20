import streamlit as st
import pandas as pd
import numpy as np
import shap
import pickle

# 🌟 Load model
with open('best_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# 📌 Define input features
st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦")
st.title("🏦 Loan Approval Predictor")
st.markdown("This app predicts whether a loan will be **Approved** or **Rejected** and explains why.")

# 🌟 Collect user inputs
income_annum = st.number_input("Annual Income (in KES)", min_value=0)
loan_amount = st.number_input("Loan Amount (in KES)", min_value=0)
cibil_score = st.slider("CIBIL Score", 300, 900, 650)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
asset_value = st.number_input("Total Asset Value (in KES)", min_value=0)

# 🌟 Preprocess input
input_dict = {
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'cibil_score': cibil_score,
    'education': 1 if education == "Graduate" else 0,
    'self_employed': 1 if self_employed == "Yes" else 0,
    'asset_value': asset_value
}
input_df = pd.DataFrame([input_dict])

# 🔮 Prediction
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result:")
st.write("✅ **Loan Approved**" if prediction == 1 else "❌ **Loan Rejected**")
st.write(f"Confidence: **{prediction_proba * 100:.2f}%**")

# 📊 SHAP Explanation
st.subheader("Explanation (Why this prediction?)")
explainer = shap.Explainer(model, input_df)
shap_values = explainer(input_df)

# Show waterfall plot
st.set_option('deprecation.showPyplotGlobalUse', False)
import matplotlib.pyplot as plt
fig = shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)
