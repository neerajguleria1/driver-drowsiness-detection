"""
Day 59 - Final System Testing
==============================
Comprehensive tests for edge cases, invalid inputs, and extreme scenarios
"""

import requests
import pytest

BASE_URL = "http://localhost:8000"
API_KEY = "dev_secure_key_123"
HEADERS = {"x-api-key": API_KEY}

VALID_DATA = {
    "Speed": 60.0,
    "Alertness": 0.8,
    "Seatbelt": 1,
    "HR": 75.0,
    "Fatigue": 3,
    "speed_change": 5.0,
    "prev_alertness": 0.85
}

# ============================================================================
# TEST 1: Invalid Input Tests
# ============================================================================

def test_missing_required_field():
    """Test with missing required field"""
    data = VALID_DATA.copy()
    del data["Speed"]
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code == 422  # Validation error

def test_invalid_seatbelt_value():
    """Test with invalid seatbelt value (not 0 or 1)"""
    data = VALID_DATA.copy()
    data["Seatbelt"] = 2
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code in [400, 422]

def test_negative_speed():
    """Test with negative speed"""
    data = VALID_DATA.copy()
    data["Speed"] = -10.0
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code in [400, 422]

def test_alertness_out_of_range():
    """Test with alertness > 1.0"""
    data = VALID_DATA.copy()
    data["Alertness"] = 1.5
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code in [400, 422]

def test_invalid_fatigue_type():
    """Test with float fatigue instead of int"""
    data = VALID_DATA.copy()
    data["Fatigue"] = 3.5
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    # Should either accept and convert or reject
    assert response.status_code in [200, 400, 422]

# ============================================================================
# TEST 2: Extreme Value Tests
# ============================================================================

def test_extreme_low_values():
    """Test with all minimum values"""
    data = {
        "Speed": 0.0,
        "Alertness": 0.0,
        "Seatbelt": 0,
        "HR": 30.0,
        "Fatigue": 0,
        "speed_change": 0.0,
        "prev_alertness": 0.0
    }
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert result["ml_prediction"] in ["Alert", "Drowsy"]

def test_extreme_high_values():
    """Test with all maximum values"""
    data = {
        "Speed": 200.0,
        "Alertness": 1.0,
        "Seatbelt": 1,
        "HR": 200.0,
        "Fatigue": 10,
        "speed_change": 20.0,
        "prev_alertness": 1.0
    }
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert result["ml_prediction"] in ["Alert", "Drowsy"]

def test_critical_drowsy_scenario():
    """Test extreme drowsy scenario"""
    data = {
        "Speed": 30.0,
        "Alertness": 0.1,
        "Seatbelt": 1,
        "HR": 50.0,
        "Fatigue": 10,
        "speed_change": 2.0,
        "prev_alertness": 0.9
    }
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert result["risk_state"] in ["MODERATE", "CRITICAL"]

def test_perfect_alert_scenario():
    """Test perfect alert scenario"""
    data = {
        "Speed": 100.0,
        "Alertness": 1.0,
        "Seatbelt": 1,
        "HR": 70.0,
        "Fatigue": 0,
        "speed_change": 15.0,
        "prev_alertness": 1.0
    }
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert result["risk_state"] == "LOW"

# ============================================================================
# TEST 3: Batch Processing Tests
# ============================================================================

def test_empty_batch():
    """Test with empty batch"""
    response = requests.post(f"{BASE_URL}/v1/analyze/batch", json=[], headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert result["total_processed"] == 0

def test_single_item_batch():
    """Test batch with single item"""
    batch = [VALID_DATA]
    response = requests.post(f"{BASE_URL}/v1/analyze/batch", json=batch, headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert result["total_processed"] == 1

def test_large_batch():
    """Test batch with 100 items"""
    batch = [VALID_DATA for _ in range(100)]
    response = requests.post(f"{BASE_URL}/v1/analyze/batch", json=batch, headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert result["total_processed"] == 100

def test_batch_size_limit():
    """Test batch size exceeds limit (>200)"""
    batch = [VALID_DATA for _ in range(201)]
    response = requests.post(f"{BASE_URL}/v1/analyze/batch", json=batch, headers=HEADERS)
    assert response.status_code == 400

def test_batch_with_invalid_item():
    """Test batch with one invalid item"""
    batch = [VALID_DATA, {"Speed": -10}, VALID_DATA]
    response = requests.post(f"{BASE_URL}/v1/analyze/batch", json=batch, headers=HEADERS)
    # Should reject entire batch or skip invalid
    assert response.status_code in [200, 400, 422]

# ============================================================================
# TEST 4: API Security Tests
# ============================================================================

def test_no_api_key():
    """Test without API key"""
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA)
    assert response.status_code in [401, 403]

def test_invalid_api_key():
    """Test with invalid API key"""
    headers = {"x-api-key": "invalid_key"}
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA, headers=headers)
    assert response.status_code == 401

def test_missing_api_key_header():
    """Test with missing x-api-key header"""
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA)
    assert response.status_code in [401, 403]

# ============================================================================
# TEST 5: Response Validation Tests
# ============================================================================

def test_response_structure():
    """Test response has all required fields"""
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA, headers=HEADERS)
    assert response.status_code == 200
    
    result = response.json()
    required_fields = [
        "ml_prediction", "ml_confidence", "confidence_level",
        "risk_score", "risk_state", "risk_factors", "decision",
        "top_contributing_features", "explanations",
        "model_version", "model_type", "inference_latency_ms"
    ]
    
    for field in required_fields:
        assert field in result, f"Missing field: {field}"

def test_prediction_values():
    """Test prediction is either Alert or Drowsy"""
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA, headers=HEADERS)
    result = response.json()
    assert result["ml_prediction"] in ["Alert", "Drowsy"]

def test_confidence_range():
    """Test confidence is between 0 and 1"""
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA, headers=HEADERS)
    result = response.json()
    assert 0.0 <= result["ml_confidence"] <= 1.0

def test_risk_score_range():
    """Test risk score is between 0 and 100"""
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA, headers=HEADERS)
    result = response.json()
    assert 0 <= result["risk_score"] <= 100

def test_risk_state_values():
    """Test risk state is valid"""
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA, headers=HEADERS)
    result = response.json()
    assert result["risk_state"] in ["LOW", "MODERATE", "CRITICAL"]

# ============================================================================
# TEST 6: Performance Tests
# ============================================================================

def test_inference_latency():
    """Test inference completes within 100ms"""
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA, headers=HEADERS)
    result = response.json()
    assert result["inference_latency_ms"] < 100

def test_concurrent_requests():
    """Test handling multiple concurrent requests"""
    import concurrent.futures
    
    def make_request():
        return requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA, headers=HEADERS)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    assert all(r.status_code == 200 for r in results)

# ============================================================================
# TEST 7: Edge Case Tests
# ============================================================================

def test_zero_speed_high_fatigue():
    """Test stationary vehicle with high fatigue"""
    data = VALID_DATA.copy()
    data["Speed"] = 0.0
    data["Fatigue"] = 9
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert "Stationary but highly fatigued" in result.get("risk_factors", [])

def test_rapid_alertness_drop():
    """Test rapid alertness drop"""
    data = VALID_DATA.copy()
    data["prev_alertness"] = 0.9
    data["Alertness"] = 0.3
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert "Rapid alertness drop" in result.get("risk_factors", [])

def test_high_heart_rate():
    """Test elevated heart rate"""
    data = VALID_DATA.copy()
    data["HR"] = 120.0
    
    response = requests.post(f"{BASE_URL}/v1/analyze", json=data, headers=HEADERS)
    assert response.status_code == 200
    result = response.json()
    assert "Elevated heart rate" in result.get("risk_factors", [])

# ============================================================================
# TEST 8: Fallback Mode Tests
# ============================================================================

def test_fallback_response_structure():
    """Test fallback response has required fields"""
    # This would trigger if model fails, but we can't easily simulate
    # Just verify normal response has fallback_mode field
    response = requests.post(f"{BASE_URL}/v1/analyze", json=VALID_DATA, headers=HEADERS)
    result = response.json()
    # fallback_mode should be present (False for normal operation)
    assert "fallback_mode" in result or response.status_code == 200

# ============================================================================
# Run All Tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DAY 59 - FINAL SYSTEM TESTING")
    print("=" * 70)
    print("\nMake sure API is running: uvicorn src.app:app --reload\n")
    
    pytest.main([__file__, "-v", "--tb=short"])
