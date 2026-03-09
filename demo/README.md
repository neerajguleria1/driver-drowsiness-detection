# 🚗 Driver Drowsiness Detection System - Demo

## Quick Start

### 1. Start the API
```bash
uvicorn src.app:app --reload
```

### 2. Run the Demo
```bash
python demo/demo_run.py
```

## What the Demo Shows

### Scenario 1: Alert Driver ✅
- Normal driving conditions
- High alertness (90%)
- Low fatigue (2/10)
- **Result**: Alert prediction, LOW risk

### Scenario 2: Drowsy Driver 🔴
- Low speed (45 km/h)
- Very low alertness (30%)
- High fatigue (8/10)
- **Result**: Drowsy prediction, CRITICAL risk

### Scenario 3: Moderate Risk ⚠️
- Declining alertness
- Moderate fatigue (5/10)
- **Result**: Early warning, MODERATE risk

## Demo Output

The demo displays:
- 🎯 **ML Prediction** (Alert/Drowsy)
- 📈 **Confidence Score** (0-100%)
- ⚠️ **Risk Score** (0-100)
- 💡 **Decision Recommendation**
- 🔍 **Feature Importance**
- ⏱️ **Inference Latency**
- 📊 **System Metrics**

## For Recruiters

This demo showcases:
- ✅ Production ML system
- ✅ Real-time predictions (<100ms)
- ✅ Explainable AI
- ✅ Risk assessment engine
- ✅ Decision intelligence
- ✅ Full monitoring

## Technical Stack

- **ML**: scikit-learn, Random Forest
- **API**: FastAPI, uvicorn
- **Features**: 20+ engineered features
- **Security**: API key auth, rate limiting
- **Monitoring**: Drift detection, audit logging

---

**Built by**: Neeraj Guleria  
**Role**: Upcoming SWE @ Amazon  
**Focus**: Production ML Systems
