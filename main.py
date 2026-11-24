from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib

app = FastAPI(
    title="SMS Spam Detection API",
    description="Machine Learning service for classifying SMS messages as ham or spam.",
    version="0.1.0",
)

# Enable CORS for frontend web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the saved pipeline
model = joblib.load("spam_model.pkl")


class SMSRequest(BaseModel):
    message: str = Field(..., min_length=1, description="SMS text content to classify")


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
    cleaned_message = sms.message.strip()
    if not cleaned_message:
        raise HTTPException(status_code=400, detail="SMS message cannot be empty or whitespace only.")

    text_to_predict = [cleaned_message]

    prediction = model.predict(text_to_predict)[0]
    probabilities = model.predict_proba(text_to_predict)[0]

    is_spam = bool(prediction == "spam")
    confidence = max(probabilities) * 100

    return {
        "message": cleaned_message,
        "is_spam": is_spam,
        "prediction": prediction,
        "confidence_percentage": round(confidence, 2),
    }
