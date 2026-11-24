from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="SMS Spam Detection API",
    description="Machine Learning service for classifying SMS messages as ham or spam.",
    version="0.1.0",
)

# Load the saved pipeline
model = joblib.load("spam_model.pkl")


class SMSRequest(BaseModel):
    message: str


class SMSResponse(BaseModel):
    message: str
    is_spam: bool
    prediction: str
    confidence_percentage: float


@app.get("/")
def read_root():
    return {
        "service": "SMS Spam Detection API",
        "status": "online",
        "version": "0.1.0",
        "docs_url": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=SMSResponse)
def predict_sms(sms: SMSRequest):
    text_to_predict = [sms.message]

    prediction = model.predict(text_to_predict)[0]
    probabilities = model.predict_proba(text_to_predict)[0]

    is_spam = bool(prediction == "spam")
    confidence = max(probabilities) * 100

    return {
        "message": sms.message,
        "is_spam": is_spam,
        "prediction": prediction,
        "confidence_percentage": round(confidence, 2),
    }
