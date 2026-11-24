# SMS Spam Detection & Prediction Service

A production-ready NLP Machine Learning pipeline and FastAPI REST API service for real-time classification of SMS messages into `ham` (legitimate) or `spam`.

## Features
- **Scikit-Learn NLP Pipeline:** TF-IDF Vectorizer with English stop-word filtering combined with a Logistic Regression classifier.
- **FastAPI Inference Microservice:** High-performance, asynchronous REST API with Swagger UI documentation.
- **Data Validation:** Strict Pydantic schemas validating payloads and rejecting invalid/empty inputs.
- **CORS Enabled:** Cross-Origin Resource Sharing enabled for seamless web and mobile application integration.
- **Health Checks & Observability:** Root service metadata and `/health` probe endpoints.
- **Comprehensive Test Suite:** Automated test suite verifying classification logic, edge cases, and schema validations.

---

## Dataset

Trained on the standard SMS Spam Collection dataset (`SMSSpamCollection.csv`):
- Total Messages: **5,574**
- **Ham (legitimate):** 4,825 instances (~86.56%)
- **Spam (unsolicited):** 749 instances (~13.44%)

---

## Setup & Installation

The project uses Python >= 3.14 and `uv` package manager:

```bash
# Sync dependencies with virtual environment
uv sync
```

Or run via standard Python venv:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # or use pyproject.toml
```

---

## Running the API

Start the FastAPI application with Uvicorn:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Interactive ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Endpoints

### 1. Health Check
`GET /health`
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 2. Predict SMS
`POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"message": "Congratulations! You have won a $1,000 gift card. Call now to claim!"}'
```

**Response:**
```json
{
  "message": "Congratulations! You have won a $1,000 gift card. Call now to claim!",
  "is_spam": true,
  "prediction": "spam",
  "confidence_percentage": 94.25
}
```

---

## Running Automated Tests

Run the test runner:
```bash
python run_tests.py
```
