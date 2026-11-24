from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the saved pipeline
model = joblib.load("spam_model.pkl")


class SMSRequest(BaseModel):
    message: str


@app.post("/predict")
def predict_sms(sms: SMSRequest):
    text_to_predict = [sms.message]

    prediction = model.predict(text_to_predict)[0]
    probabilities = model.predict_proba(text_to_predict)[0]
    spam_probability = probabilities[1]

    is_spam = bool(prediction == "spam")
    confidence = max(probabilities) * 100

    return {
        "message": sms.message,
        "is_spam": is_spam,
        "prediction": prediction,
        "confidence_percentage": round(confidence, 2),
    }
