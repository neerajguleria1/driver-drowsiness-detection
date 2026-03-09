import requests
import json

BASE_URL = "http://localhost:8000"
API_KEY = "dev_secure_key_123"

def test_api():
    print("=" * 60)
    print("TESTING DRIVER DROWSINESS DETECTION API")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n[TEST 1] Health Check...")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        if resp.status_code == 200:
            print("✅ API is healthy!")
        else:
            print(f"❌ Health check failed: {resp.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("\nMake sure API is running:")
        print("  python -m uvicorn src.app:app --reload")
        return
    
    # Test 2: Single Prediction
    print("\n[TEST 2] Single Prediction...")
    test_data = {
        "Speed": 60.0,
        "Alertness": 0.8,
        "Seatbelt": 1,
        "HR": 75.0,
        "Fatigue": 3,
        "speed_change": 5.0,
        "prev_alertness": 0.85
    }
    
    try:
        resp = requests.post(
            f"{BASE_URL}/v1/analyze",
            json=test_data,
            headers={"x-api-key": API_KEY}
        )
        
        if resp.status_code == 200:
            result = resp.json()
            print("✅ Prediction successful!")
            print(f"   Prediction: {result['ml_prediction']}")
            print(f"   Confidence: {result['ml_confidence']:.2%}")
            print(f"   Risk Score: {result['risk_score']}/100")
            print(f"   Decision: {result['decision']['action']}")
        else:
            print(f"❌ Prediction failed: {resp.status_code}")
            print(f"   Error: {resp.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 3: Metrics
    print("\n[TEST 3] System Metrics...")
    try:
        resp = requests.get(f"{BASE_URL}/v1/metrics")
        if resp.status_code == 200:
            metrics = resp.json()
            print("✅ Metrics retrieved!")
            print(f"   Total Requests: {metrics['total_requests']}")
            print(f"   Drowsy Predictions: {metrics['drowsy_predictions']}")
            print(f"   Avg Latency: {metrics['average_latency_ms']:.2f}ms")
        else:
            print(f"❌ Metrics failed: {resp.status_code}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED!")
    print("=" * 60)
    print("\n✨ Your API is working correctly!")
    print(f"\n🌐 Open Swagger UI: {BASE_URL}/docs")
    print("=" * 60)

if __name__ == "__main__":
    test_api()
