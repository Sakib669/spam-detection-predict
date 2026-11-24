from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Load the saved pipeline
model = joblib.load("spam_model.pkl")


class SMSRequest(BaseModel):
    message: str
