import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lung Cancer Risk Predictor", layout="wide", page_icon="🫁")

@st.cache_resource
def load_artifacts():
    with open("artifacts/models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

FEATURES = ["GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
            "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY",
            "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
            "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN"]

st.title("🫁 Lung Cancer Risk Prediction")
st.markdown("**Explainable AI-powered clinical decision support tool**")
st.markdown("*Based on research replication of Pavithran et al. (2025) and Alsinglawi et al. (2022)*")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 20, 90, 55)
    smoking = st.selectbox("Smoking", ["Yes", "No"])
    alcohol = st.selectbox("Alcohol Consuming", ["Yes", "No"])

with col2:
    st.subheader("Symptoms")
    coughing = st.selectbox("Coughing", ["Yes", "No"])
    shortness = st.selectbox("Shortness of Breath", ["Yes", "No"])
    wheezing = st.selectbox("Wheezing", ["Yes", "No"])
    chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])
    fatigue = st.selectbox("Fatigue", ["Yes", "No"])
    swallowing = st.selectbox("Swallowing Difficulty", ["Yes", "No"])

with col3:
    st.subheader("Risk Factors")
    yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
    anxiety = st.selectbox("Anxiety", ["Yes", "No"])
    peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
    chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
    allergy = st.selectbox("Allergy", ["Yes", "No"])

def encode(val): return 2 if val == "Yes" else 1
def encode_gender(val): return 1 if val == "Male" else 0

if st.button("🔍 Predict Risk", type="primary"):
    input_data = np.array([[
        encode_gender(gender), age, encode(smoking), encode(yellow_fingers),
        encode(anxiety), encode(peer_pressure), encode(chronic_disease),
        encode(fatigue), encode(allergy), encode(wheezing),
        encode(alcohol), encode(coughing), encode(shortness),
        encode(swallowing), encode(chest_pain)
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.divider()
    col_res1, col_res2 = st.columns(2)

    with col_res1:
        if prediction == 1:
            st.error(f"⚠️ **HIGH RISK** — Predicted probability: {probability:.1%}")
        else:
            st.success(f"✅ **LOW RISK** — Predicted probability: {probability:.1%}")

        st.progress(float(probability))
        st.caption("⚠️ This tool is for research purposes only. Always consult a medical professional.")

    with col_res2:
        st.subheader("Feature Explanation (LIME)")
        # LIME explanation for this instance
        try:
            import lime.lime_tabular
            with open("artifacts/models/X_train.pkl", "rb") as f:
                X_train = pickle.load(f)
            
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=FEATURES,
                class_names=["No Cancer", "Cancer"],
                mode="classification"
            )
            exp = explainer.explain_instance(input_scaled[0], model.predict_proba, num_features=8)
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
        except Exception as e:
            st.info("LIME explanation unavailable in this environment.")

st.divider()
st.subheader("📊 Model Performance Summary")
perf_data = {
    "Augmentation": ["K-Means SMOTE", "SMOTE", "ADASYN", "Random Oversampling"],
    "Classifier": ["MLP", "XGBoost", "Random Forest", "SVM"],
    "Accuracy (%)": ["93.55", "91.94", "91.94", "88.71"],
    "AUC-ROC (%)": ["96.76", "95.83", "94.56", "96.06"]
}
st.dataframe(pd.DataFrame(perf_data), use_container_width=True)