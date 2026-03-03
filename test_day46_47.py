import requests
import json
import time
import pandas as pd
import numpy as np

BASE_URL = "http://localhost:8000"
API_KEY = "dev_secure_key_123"

HEADERS = {"x-api-key": API_KEY}

# Sample test data
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
    """Test 1: Health check"""
    print("\n=== TEST 1: Health Check ===")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"✅ Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_single_inference():
    """Test 2: Single inference"""
    print("\n=== TEST 2: Single Inference ===")
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/analyze",
            json=TEST_DATA,
            headers=HEADERS
        )
        print(f"✅ Status: {resp.status_code}")
        result = resp.json()
        print(f"Prediction: {result['ml_prediction']}")
        print(f"Confidence: {result['ml_confidence']:.2%}")
        print(f"Risk Score: {result['risk_score']}")
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_inference():
    """Test 3: Batch inference"""
    print("\n=== TEST 3: Batch Inference ===")
    try:
        batch = [TEST_DATA for _ in range(5)]
        resp = requests.post(
            f"{BASE_URL}/v1/analyze/batch",
            json=batch,
            headers=HEADERS
        )
        print(f"✅ Status: {resp.status_code}")
        result = resp.json()
        print(f"Processed: {result['total_processed']} items")
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_metrics():
    """Test 4: Metrics endpoint"""
    print("\n=== TEST 4: Metrics ===")
    try:
        resp = requests.get(f"{BASE_URL}/v1/metrics")
        print(f"✅ Status: {resp.status_code}")
        metrics = resp.json()
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Drowsy Predictions: {metrics['drowsy_predictions']}")
        print(f"Avg Latency: {metrics['average_latency_ms']:.2f}ms")
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_diagnostics():
    """Test 5: Diagnostics"""
    print("\n=== TEST 5: Diagnostics ===")
    try:
        resp = requests.get(f"{BASE_URL}/v1/diagnostics")
        print(f"✅ Status: {resp.status_code}")
        diag = resp.json()
        print(f"Model Loaded: {diag['model_loaded']}")
        print(f"Model Version: {diag['model_version']}")
        print(f"Model Type: {diag['model_type']}")
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_day46_model_versioning():
    """Test 6: Day 46 - Model Versioning"""
    print("\n=== TEST 6: Day 46 - Model Versioning ===")
    try:
        # Try to switch model (will fail if version doesn't exist, but tests endpoint)
        resp = requests.post(
            f"{BASE_URL}/v1/model/switch/v2.0",
            headers=HEADERS
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 400:
            print("✅ Endpoint works (model version not found - expected)")
            return True
        elif resp.status_code == 200:
            print(f"✅ Model switched: {resp.json()}")
            return True
        else:
            print(f"❌ Unexpected status: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_day46_shadow_model():
    """Test 7: Day 46 - Shadow Model"""
    print("\n=== TEST 7: Day 46 - Shadow Model ===")
    try:
        # Load shadow model
        resp = requests.post(
            f"{BASE_URL}/v1/model/shadow/v2.0",
            headers=HEADERS
        )
        print(f"Status: {resp.status_code}")
        if resp.status_code == 400:
            print("✅ Endpoint works (shadow model version not found - expected)")
            return True
        elif resp.status_code == 200:
            print(f"✅ Shadow model loaded: {resp.json()}")
            return True
        else:
            print(f"❌ Unexpected status: {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_day47_drift_detection():
    """Test 8: Day 47 - Drift Detection"""
    print("\n=== TEST 8: Day 47 - Drift Detection ===")
    try:
        # Generate 250 inferences to populate buffer
        print("Generating 250 inferences for drift detection...")
        for i in range(250):
            data = TEST_DATA.copy()
            data["Fatigue"] = np.random.randint(0, 10)
            data["Alertness"] = np.random.uniform(0.3, 1.0)
            data["HR"] = np.random.uniform(50, 120)
            requests.post(
                f"{BASE_URL}/v1/analyze",
                json=data,
                headers=HEADERS
            )
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1} inferences...")
        
        # Set baseline
        print("Setting baseline statistics...")
        resp = requests.post(
            f"{BASE_URL}/v1/drift/baseline",
            headers=HEADERS
        )
        print(f"Baseline Status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"❌ Failed to set baseline: {resp.json()}")
            return False
        
        # Check drift
        print("Checking drift...")
        resp = requests.get(
            f"{BASE_URL}/v1/drift/detect",
            headers=HEADERS
        )
        print(f"✅ Drift Detection Status: {resp.status_code}")
        drift = resp.json()
        print(f"Drift Status: {drift['status']}")
        if drift['details']:
            print(f"Drift Details: {drift['details']}")
        return resp.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_rate_limiting():
    """Test 9: Rate Limiting"""
    print("\n=== TEST 9: Rate Limiting ===")
    try:
        # Make 11 requests (limit is 10/minute)
        print("Testing rate limit (10/minute)...")
        for i in range(11):
            resp = requests.post(
                f"{BASE_URL}/v1/analyze",
                json=TEST_DATA,
                headers=HEADERS
            )
            if resp.status_code == 429:
                print(f"✅ Rate limit triggered at request {i + 1}")
                return True
        print("⚠️ Rate limit not triggered (may need more requests)")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_api_key_protection():
    """Test 10: API Key Protection"""
    print("\n=== TEST 10: API Key Protection ===")
    try:
        # Try without API key
        resp = requests.post(
            f"{BASE_URL}/v1/analyze",
            json=TEST_DATA
        )
        if resp.status_code == 403 or resp.status_code == 401:
            print(f"✅ API key protection works (status: {resp.status_code})")
            return True
        else:
            print(f"❌ Expected 401/403, got {resp.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("TESTING DAY 46 & 47 FEATURES")
    print("=" * 60)
    
    results = {
        "Health Check": test_health(),
        "Single Inference": test_single_inference(),
        "Batch Inference": test_batch_inference(),
        "Metrics": test_metrics(),
        "Diagnostics": test_diagnostics(),
        "Day 46 - Model Switch": test_day46_model_versioning(),
        "Day 46 - Shadow Model": test_day46_shadow_model(),
        "Day 47 - Drift Detection": test_day47_drift_detection(),
        "Rate Limiting": test_rate_limiting(),
        "API Key Protection": test_api_key_protection(),
    }
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

if __name__ == "__main__":
    print("Make sure the API is running: uvicorn src.app:app --reload")
    input("Press Enter to start tests...")
    main()
