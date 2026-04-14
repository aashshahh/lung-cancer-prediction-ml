import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "../artifacts/final_model.pkl"
SCALER_PATH = "../artifacts/final_scaler.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def preprocess_and_predict(input_dict):
    """
    input_dict = {
        'AGE': 45,
        'GENDER': 'Male',
        'SMOKING': 1,
        ...
    }
    """

    df = pd.DataFrame([input_dict])

    #convert categorical like GENDER if needed
    if "GENDER" in df.columns:
        df["GENDER"] = df["GENDER"].map({"Male": 1, "Female": 0})
    
    scaled = scaler.transform(df)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return prediction, probability