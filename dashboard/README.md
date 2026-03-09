# 🚗 Driver Drowsiness Detection - Interactive Dashboard

## Quick Start

### 1. Install Dependencies
```bash
pip install streamlit plotly
```

### 2. Start the API (Required)
```bash
# Terminal 1
uvicorn src.app:app --reload
```

### 3. Run Dashboard
```bash
# Terminal 2
streamlit run dashboard/app.py
```

### 4. Open Browser
Dashboard will open at: http://localhost:8501

## Features

### 📊 Interactive Inputs
- Speed slider (0-200 km/h)
- Alertness level (0-100%)
- Seatbelt status
- Heart rate (30-200 bpm)
- Fatigue level (0-10)
- Speed variability
- Previous alertness

### 📈 Visual Outputs
- **Prediction**: Alert/Drowsy with color coding
- **Confidence Score**: ML model confidence
- **Risk Score Gauge**: 0-100 visual indicator
- **Risk State**: LOW/MODERATE/CRITICAL
- **Recommended Action**: What driver should do
- **Risk Factors**: List of detected issues
- **Feature Importance**: Top contributing factors
- **Explanations**: Human-readable reasoning

### 🎨 Color Coding
- 🟢 Green: Safe (Risk < 40)
- 🟡 Yellow: Moderate (Risk 40-69)
- 🔴 Red: Critical (Risk ≥ 70)

## Example Scenarios

### Scenario 1: Alert Driver ✅
```
Speed: 80 km/h
Alertness: 90%
Fatigue: 2/10
HR: 72 bpm
→ Result: Alert, LOW risk
```

### Scenario 2: Moderate Risk ⚠️
```
Speed: 65 km/h
Alertness: 55%
Fatigue: 5/10
HR: 85 bpm
→ Result: Moderate risk warning
```

### Scenario 3: High Risk 🔴
```
Speed: 45 km/h
Alertness: 30%
Fatigue: 8/10
HR: 58 bpm
→ Result: Drowsy, CRITICAL risk
```

## Technology Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Backend**: FastAPI (must be running)
- **ML**: scikit-learn

## For Recruiters

This dashboard demonstrates:
- ✅ Full-stack ML deployment
- ✅ Interactive UI/UX
- ✅ Real-time predictions
- ✅ Data visualization
- ✅ Production-ready interface

---

**Built by**: Neeraj Guleria  
**Role**: Upcoming SWE @ Amazon
