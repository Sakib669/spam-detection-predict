from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["service"] == "SMS Spam Detection API"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict_ham_message():
    response = client.post("/predict", json={"message": "Hey are we still meeting for lunch at 1pm?"})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "ham"
    assert data["is_spam"] is False
    assert 0 <= data["confidence_percentage"] <= 100


def test_predict_spam_message():
    response = client.post(
        "/predict",
        json={"message": "URGENT! You have won a £1000 prize. Call 09061701461 now to claim your reward!"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "spam"
    assert data["is_spam"] is True
    assert 0 <= data["confidence_percentage"] <= 100


def test_predict_empty_message_rejected():
    response = client.post("/predict", json={"message": "   "})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_predict_missing_field_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422
