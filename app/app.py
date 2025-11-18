#importing the libraries
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from model_loader import load_model

st.set_page_config(page_title="Lung Cancer Prediction", layout="wide")

model, scaler = load_model()

st.title("Lung Cancer Risk Prediction App")
st.markdown("This tool predicts lung cancer risk using your symptoms and lifestyle indicators.")

# ------------------------------
# Input Form
# ------------------------------
st.header("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 45)
    smoking = st.selectbox("Smoking", ["No", "Yes"])
    yellow_fingers = st.selectbox("Yellow Fingers", ["No", "Yes"])
    anxiety = st.selectbox("Anxiety", ["No", "Yes"])
    peer_pressure = st.selectbox("Peer Pressure", ["No", "Yes"])
    chronic_disease = st.selectbox("Chronic Disease", ["No", "Yes"])

with col2:
    fatigue = st.selectbox("Fatigue", ["No", "Yes"])
    allergy = st.selectbox("Allergy", ["No", "Yes"])
    wheezing = st.selectbox("Wheezing", ["No", "Yes"])
    alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
    coughing = st.selectbox("Coughing", ["No", "Yes"])
    shortness_breath = st.selectbox("Shortness of Breath", ["No", "Yes"])
    swallowing = st.selectbox("Swallowing Difficulty", ["No", "Yes"])
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"])

# Convert Yes/No to 1/0
mapping = {"Yes": 1, "No": 0}

data = {
    "AGE": age,
    "SMOKING": mapping[smoking],
    "YELLOW_FINGERS": mapping[yellow_fingers],
    "ANXIETY": mapping[anxiety],
    "PEER_PRESSURE": mapping[peer_pressure],
    "CHRONIC_DISEASE": mapping[chronic_disease],
    "FATIGUE": mapping[fatigue],
    "ALLERGY": mapping[allergy],
    "WHEEZING": mapping[wheezing],
    "ALCOHOL_CONSUMING": mapping[alcohol],
    "COUGHING": mapping[coughing],
    "SHORTNESS_OF_BREATH": mapping[shortness_breath],
    "SWALLOWING_DIFFICULTY": mapping[swallowing],
    "CHEST_PAIN": mapping[chest_pain],
}

input_df = pd.DataFrame([data])

if st.button("Predict"):
    scaled_input = scaler.transform(input_df)

    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"High Likelihood of Lung Cancer (Probability: {prob:.2f})")
    else:
        st.success(f"Low Likelihood of Lung Cancer (Probability: {prob:.2f})")

    # ------------------------------
    # SHAP Explainability
    # ------------------------------
    st.subheader("Feature Contribution (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if shap_values.shape[1] == scaled_input.shape[1] + 1:
        shap_values = shap_values[:, :-1]

    fig, ax = plt.subplots()
    shap.bar_plot(shap_values[0], feature_names=input_df.columns, max_display=10)
    st.pyplot(fig)
