"""
Driver Drowsiness Detection System - Live Demo
===============================================
This demo showcases the production ML system with real-time predictions.
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_KEY = "dev_secure_key_123"
HEADERS = {"x-api-key": API_KEY}

def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_section(text):
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print(f"{'─' * 70}")

def demo_scenario_1_alert_driver():
    """Scenario 1: Alert Driver - Normal Driving"""
    print_section("SCENARIO 1: Alert Driver (Normal Conditions)")
    
    driver_data = {
        "Speed": 80.0,
        "Alertness": 0.9,
        "Seatbelt": 1,
        "HR": 72.0,
        "Fatigue": 2,
        "speed_change": 10.0,
        "prev_alertness": 0.92
    }
    
    print("\n📊 Driver Status:")
    print(f"   Speed: {driver_data['Speed']} km/h")
    print(f"   Alertness: {driver_data['Alertness']*100:.0f}%")
    print(f"   Heart Rate: {driver_data['HR']} bpm")
    print(f"   Fatigue Level: {driver_data['Fatigue']}/10")
    print(f"   Seatbelt: {'✓ On' if driver_data['Seatbelt'] else '✗ Off'}")
    
    print("\n🔄 Analyzing...")
    response = requests.post(
        f"{BASE_URL}/v1/analyze",
        json=driver_data,
        headers=HEADERS
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ ANALYSIS COMPLETE")
        print(f"\n🎯 Prediction: {result['ml_prediction']}")
        print(f"📈 Confidence: {result['ml_confidence']*100:.1f}% ({result['confidence_level']})")
        print(f"⚠️  Risk Score: {result['risk_score']}/100 ({result['risk_state']})")
        print(f"\n💡 Decision: {result['decision']['action']}")
        print(f"   Severity: {result['decision']['severity']}")
        print(f"   Message: {result['decision']['message']}")
        
        if result['risk_factors']:
            print(f"\n⚠️  Risk Factors:")
            for factor in result['risk_factors']:
                print(f"   • {factor}")
        
        print(f"\n🔍 Top Contributing Features:")
        for feat in result['top_contributing_features'][:3]:
            print(f"   • {feat['feature']}: {feat['feature_value']:.2f} (importance: {feat['global_importance']:.3f})")
        
        print(f"\n⏱️  Inference Time: {result['inference_latency_ms']:.2f}ms")
    else:
        print(f"❌ Error: {response.status_code}")

def demo_scenario_2_drowsy_driver():
    """Scenario 2: Drowsy Driver - High Risk"""
    print_section("SCENARIO 2: Drowsy Driver (High Risk)")
    
    driver_data = {
        "Speed": 45.0,
        "Alertness": 0.3,
        "Seatbelt": 1,
        "HR": 58.0,
        "Fatigue": 8,
        "speed_change": 3.0,
        "prev_alertness": 0.7
    }
    
    print("\n📊 Driver Status:")
    print(f"   Speed: {driver_data['Speed']} km/h ⚠️  (Low)")
    print(f"   Alertness: {driver_data['Alertness']*100:.0f}% ⚠️  (Very Low)")
    print(f"   Heart Rate: {driver_data['HR']} bpm")
    print(f"   Fatigue Level: {driver_data['Fatigue']}/10 🔴 (High)")
    print(f"   Seatbelt: {'✓ On' if driver_data['Seatbelt'] else '✗ Off'}")
    
    print("\n🔄 Analyzing...")
    response = requests.post(
        f"{BASE_URL}/v1/analyze",
        json=driver_data,
        headers=HEADERS
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ ANALYSIS COMPLETE")
        print(f"\n🎯 Prediction: {result['ml_prediction']} 🔴")
        print(f"📈 Confidence: {result['ml_confidence']*100:.1f}% ({result['confidence_level']})")
        print(f"⚠️  Risk Score: {result['risk_score']}/100 ({result['risk_state']}) 🔴")
        print(f"\n💡 Decision: {result['decision']['action']} 🚨")
        print(f"   Severity: {result['decision']['severity']}")
        print(f"   Message: {result['decision']['message']}")
        
        if result['risk_factors']:
            print(f"\n⚠️  Risk Factors Detected:")
            for factor in result['risk_factors']:
                print(f"   🔴 {factor}")
        
        print(f"\n📝 Explanations:")
        for exp in result['explanations'][:4]:
            print(f"   • {exp}")
        
        print(f"\n⏱️  Inference Time: {result['inference_latency_ms']:.2f}ms")
    else:
        print(f"❌ Error: {response.status_code}")

def demo_scenario_3_moderate_risk():
    """Scenario 3: Moderate Risk - Early Warning"""
    print_section("SCENARIO 3: Moderate Risk (Early Warning)")
    
    driver_data = {
        "Speed": 65.0,
        "Alertness": 0.55,
        "Seatbelt": 1,
        "HR": 85.0,
        "Fatigue": 5,
        "speed_change": 7.0,
        "prev_alertness": 0.75
    }
    
    print("\n📊 Driver Status:")
    print(f"   Speed: {driver_data['Speed']} km/h")
    print(f"   Alertness: {driver_data['Alertness']*100:.0f}% ⚠️  (Declining)")
    print(f"   Heart Rate: {driver_data['HR']} bpm")
    print(f"   Fatigue Level: {driver_data['Fatigue']}/10 ⚠️  (Moderate)")
    print(f"   Seatbelt: {'✓ On' if driver_data['Seatbelt'] else '✗ Off'}")
    
    print("\n🔄 Analyzing...")
    response = requests.post(
        f"{BASE_URL}/v1/analyze",
        json=driver_data,
        headers=HEADERS
    )
    
    if response.status_code == 200:
        result = response.json()
        print("\n✅ ANALYSIS COMPLETE")
        print(f"\n🎯 Prediction: {result['ml_prediction']}")
        print(f"📈 Confidence: {result['ml_confidence']*100:.1f}% ({result['confidence_level']})")
        print(f"⚠️  Risk Score: {result['risk_score']}/100 ({result['risk_state']}) ⚠️")
        print(f"\n💡 Decision: {result['decision']['action']}")
        print(f"   Severity: {result['decision']['severity']}")
        print(f"   Message: {result['decision']['message']}")
        
        print(f"\n⏱️  Inference Time: {result['inference_latency_ms']:.2f}ms")
    else:
        print(f"❌ Error: {response.status_code}")

def demo_system_metrics():
    """Show system performance metrics"""
    print_section("SYSTEM PERFORMANCE METRICS")
    
    response = requests.get(f"{BASE_URL}/v1/metrics")
    if response.status_code == 200:
        metrics = response.json()
        print(f"\n📊 Total Requests: {metrics['total_requests']}")
        print(f"😴 Drowsy Predictions: {metrics['drowsy_predictions']}")
        print(f"⚡ Average Latency: {metrics['average_latency_ms']:.2f}ms")
    
    response = requests.get(f"{BASE_URL}/v1/model/performance", headers=HEADERS)
    if response.status_code == 200:
        perf = response.json()
        print(f"\n🎯 Prediction Distribution:")
        print(f"   Alert: {perf['prediction_distribution']['Alert']}")
        print(f"   Drowsy: {perf['prediction_distribution']['Drowsy']}")
        print(f"📈 Average Confidence: {perf['avg_confidence']:.3f}")

def main():
    print_header("🚗 DRIVER DROWSINESS DETECTION SYSTEM - LIVE DEMO")
    print("\n🎯 Production ML System with Real-Time Risk Assessment")
    print("📅 Demo Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # Check API health
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("\n❌ API is not running. Start it with: uvicorn src.app:app --reload")
            return
        
        print("\n✅ API Status: Healthy")
        
        # Run demo scenarios
        time.sleep(1)
        demo_scenario_1_alert_driver()
        
        time.sleep(2)
        demo_scenario_2_drowsy_driver()
        
        time.sleep(2)
        demo_scenario_3_moderate_risk()
        
        time.sleep(1)
        demo_system_metrics()
        
        print_header("✅ DEMO COMPLETE")
        print("\n🎓 Key Features Demonstrated:")
        print("   ✓ Real-time ML predictions")
        print("   ✓ Multi-factor risk assessment")
        print("   ✓ Explainable AI (feature importance)")
        print("   ✓ Actionable decision recommendations")
        print("   ✓ Sub-100ms inference latency")
        print("   ✓ Production-grade monitoring")
        
        print("\n📚 For more info:")
        print("   • API Docs: http://localhost:8000/docs")
        print("   • Architecture: docs/ARCHITECTURE.md")
        print("   • GitHub: [Your Repository URL]")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API. Make sure it's running:")
        print("   uvicorn src.app:app --reload")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
