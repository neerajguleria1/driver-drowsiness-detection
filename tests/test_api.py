from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.app import app

client = TestClient(app)

API_KEY = "dev_secure_key_123"
HEADERS = {"x-api-key": API_KEY}

TEST_DATA = {
    "Speed": 60.0,
    "Alertness": 0.8,
    "Seatbelt": 1,
    "HR": 75.0,
    "Fatigue": 3,
    "speed_change": 5.0,
    "prev_alertness": 0.85
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_single_prediction():
    response = client.post(
        "/v1/analyze",
        json=TEST_DATA,
        headers=HEADERS
    )
    assert response.status_code == 200
    data = response.json()
    assert "ml_prediction" in data
    assert data["ml_prediction"] in ["Alert", "Drowsy"]
    assert 0 <= data["ml_confidence"] <= 1

def test_batch_prediction():
    batch = [TEST_DATA for _ in range(5)]
    response = client.post(
        "/v1/analyze/batch",
        json=batch,
        headers=HEADERS
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 5

def test_metrics():
    response = client.get("/v1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "drowsy_predictions" in data

def test_diagnostics():
    response = client.get("/v1/diagnostics")
    assert response.status_code == 200
    data = response.json()
    assert data["model_loaded"] == True

def test_api_key_required():
    response = client.post("/v1/analyze", json=TEST_DATA)
    assert response.status_code == 403 or response.status_code == 401

def test_model_performance():
    response = client.get("/v1/model/performance", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "prediction_distribution" in data
    assert "avg_confidence" in data
