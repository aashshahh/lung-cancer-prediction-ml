from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc
import joblib
import os

ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")

def load_model():
    model = joblib.load(os.path.join(ARTIFACT_PATH, "tuned_rf.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_PATH, "scaler.pkl"))
    return model, scaler
