from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load models
vectorizer = joblib.load("tfidf_vectorizer.pkl")
log_model = joblib.load("logistic_model.pkl")
rf_model = joblib.load("random_forest.pkl")
svm_model = joblib.load("svm_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
lgb_model = joblib.load("lgb_model.pkl")
tokenizer = joblib.load("tokenizer.pkl")
lstm_model = load_model("lstm_model.h5")

# Initialize FastAPI
app = FastAPI()

# Text Input Model
class TextRequest(BaseModel):
    text: str

# Function to clean text (Assuming you have a function)
def clean_text(text):
    return text.lower().strip()

# Prediction Function
def predict_sensitive_info(text, model, vectorizer=None, tokenizer=None, lstm=False):
    text_clean = clean_text(text)
    
    if lstm:
        text_seq = tokenizer.texts_to_sequences([text_clean])
        text_padded = pad_sequences(text_seq, maxlen=100, padding="post")
        prob = model.predict(text_padded)[0][0]
    else:
        text_tfidf = vectorizer.transform([text_clean])
        prob = model.predict_proba(text_tfidf)[0][1]

    prob_percentage = round(prob * 100, 2)
    prediction = "Sensitive Info Detected" if prob >= 0.5 else "Safe Message"
    
    return {"prediction": prediction, "probability": f"{prob_percentage}%", "raw_prob": round(prob, 2)}

# API Endpoint
@app.post("/predict/")
def classify_text(request: TextRequest):
    text = request.text

    results = {
        "logistic_regression": predict_sensitive_info(text, log_model, vectorizer),
        "random_forest": predict_sensitive_info(text, rf_model, vectorizer),
        "svm": predict_sensitive_info(text, svm_model, vectorizer),
        "xgboost": predict_sensitive_info(text, xgb_model, vectorizer),
        "lightgbm": predict_sensitive_info(text, lgb_model, vectorizer),
        "lstm": predict_sensitive_info(text, lstm_model, tokenizer=tokenizer, lstm=True)
    }

    return {"input_text": text, "results": results}
