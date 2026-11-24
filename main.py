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
    probability = model.predict_proba(text_to_predict)[0][
        1
    ]  # Probability of being spam

    is_spam = bool(prediction == "spam")
    confidence = max(probability) * 100
