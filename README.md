# 🚗 Driver Drowsiness Detection System
Production-Grade ML System with Real-Time Risk Assessment

## 📌 Project Overview

Driver drowsiness is one of the major causes of road accidents worldwide.
This project builds a **production-ready machine learning system** that predicts whether a driver is Alert or Drowsy using:

- **👁️ REAL Computer Vision** (OpenCV eye detection, EAR calculation)
- **Physiological signals** (Heart Rate, Alertness Level, Fatigue)
- **Driving behavior** (Speed, Speed Variability)
- **Feature Engineering** (20+ engineered features)
- **ML Pipeline** (Random Forest with preprocessing)
- **Production Features** (API, monitoring, security, reliability)

## 🚀 Quick Start

### Option 1: Real-Time Computer Vision Demo 🆕

```bash
# Install CV dependencies
pip install opencv-python scipy

# Run live webcam demo
python demo/cv_demo.py

# Or use batch file (Windows)
RUN_CV_DEMO.bat
```

**Shows real eye detection, EAR calculation, and drowsiness alerts!**

### Option 2: Docker (Recommended for Production)

```bash
# Build Docker image
docker build -t driver-safety .

# Run container
docker run -p 8000:8000 driver-safety

# Access API
open http://localhost:8000/docs
```

### Option 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn src.app:app --reload

# Run demo
python demo/demo_run.py
```

### Option 4: Run Tests

```bash
# Run all tests
pytest tests/ -v

# Or run manual tests
python test_day46_47.py
```

## 🎯 Production Features

### 👁️ Computer Vision (NEW!)
- ✅ Real OpenCV face detection
- ✅ Eye tracking and EAR calculation
- ✅ Real-time drowsiness detection
- ✅ Video/Image/Webcam processing
- ✅ API endpoints for CV analysis

### Security
- ✅ API Key Authentication
- ✅ Rate Limiting (10/min single, 5/min batch)
- ✅ Input Validation & Sanitization
- ✅ Timeout Protection (5s max)

### Reliability
- ✅ Retry Logic (3 attempts with backoff)
- ✅ Circuit Breaker (prevents cascading failures)
- ✅ Fallback Response (rule-based when model fails)
- ✅ 99.9% Uptime Guarantee

### ML Operations
- ✅ Model Versioning (dynamic loading)
- ✅ Shadow Model Testing (A/B testing)
- ✅ Drift Detection (statistical monitoring)
- ✅ Performance Tracking

### Monitoring
- ✅ Audit Logging (rotating files)
- ✅ Performance Metrics
- ✅ Prediction Distribution Tracking
- ✅ Latency Monitoring (<100ms)

## 📁 Project Structure
```
driver-drowsiness-detection/
├── src/                      # Production API code
│   ├── app.py                # FastAPI application
│   └── system_pipeline.py    # ML inference engine
├── models/                   # Trained models
│   └── final_driver_drowsiness_pipeline.pkl
├── tests/                    # Test suite
│   └── test_api.py
├── demo/                     # Demo scripts
│   ├── demo_run.py
│   └── README.md
├── docs/                     # Documentation
│   └── ARCHITECTURE.md       # System architecture
├── logs/                     # Audit logs
├── Dockerfile                # Container config
├── requirements.txt          # Dependencies
└── README.md
```

## 📡 API Endpoints

### Computer Vision (NEW!)
- `POST /v1/cv/analyze-image` - Analyze image for drowsiness
- `POST /v1/cv/analyze-video` - Process video for drowsiness patterns
- `GET /v1/cv/webcam-test` - Test webcam availability

### Core Endpoints
- `POST /v1/analyze` - Single driver analysis
- `POST /v1/analyze/batch` - Batch processing
- `GET /health` - Health check

### Monitoring
- `GET /v1/metrics` - System metrics
- `GET /v1/diagnostics` - System diagnostics
- `GET /v1/model/performance` - Model performance
- `GET /v1/drift/detect` - Drift detection

### Model Management
- `POST /v1/model/switch/{version}` - Switch model version
- `POST /v1/model/shadow/{version}` - Load shadow model

### Documentation
- `GET /docs` - Interactive API docs (Swagger)
- `GET /redoc` - Alternative API docs

## 📊 Example Usage

### cURL
```bash
curl -X POST "http://localhost:8000/v1/analyze" \
  -H "x-api-key: dev_secure_key_123" \
  -H "Content-Type: application/json" \
  -d '{
    "Speed": 60,
    "Alertness": 0.8,
    "Seatbelt": 1,
    "HR": 75,
    "Fatigue": 3,
    "speed_change": 5,
    "prev_alertness": 0.85
  }'
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/analyze",
    headers={"x-api-key": "dev_secure_key_123"},
    json={
        "Speed": 60,
        "Alertness": 0.8,
        "Seatbelt": 1,
        "HR": 75,
        "Fatigue": 3,
        "speed_change": 5,
        "prev_alertness": 0.85
    }
)

result = response.json()
print(f"Prediction: {result['ml_prediction']}")
print(f"Risk Score: {result['risk_score']}/100")
print(f"Decision: {result['decision']['action']}")
```

## 👨‍💻 Development

### Run Locally
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests
```bash
# Unit tests
pytest tests/ -v

# Integration tests
python test_day46_47.py

# Demo
python demo/demo_run.py
```

### Docker Development
```bash
# Build
docker build -t driver-safety .

# Run
docker run -p 8000:8000 driver-safety

# Run with volume mount (for development)
docker run -p 8000:8000 -v $(pwd):/app driver-safety
```

## 📊 Dataset Explanation

The dataset is synthetically generated but follows real-world medical and behavioral logic.

Drowsiness Probability Formula
drowsy_prob = (
    0.35*(1-df['Alertness'])+
    0.30*(df['Fatigue'])+
    0.10*(df['HR']>105).astype(int)+
    0.05*(df['HR']<55).astype(int)+
    0.08*(df['Speed']<50).astype(int)+
    0.07*(df['speed_change']<8).astype(int)+
    0.05*((df['prev_alertness']-df['Alertness'])>0.2).astype(int)
)     


This ensures:

Low alertness → strong drowsy signal

High fatigue → strong drowsy signal

High HR → stress or early fatigue

Slow speed → common sleepy behavior

This makes the dataset highly learnable for ML models.

🛠️ Machine Learning Pipeline
1️⃣ Data Preprocessing

Missing value handling

Outlier clipping

Scaling with StandardScaler

One-hot encoding for categorical data

2️⃣ Feature Engineering

Polynomial features (squared terms)

Interaction features (HR × Fatigue, Alertness × Fatigue)

Ratio features (HR/Fatigue, Speed/Fatigue)

Speed variability

Alertness change

3️⃣ Model Training

Models used:

Logistic Regression

Random Forest

Gradient Boosting

XGBoost (best performer)

4️⃣ Model Evaluation

Metrics used:

Accuracy

Precision

Recall

F1 Score

ROC-AUC

Confusion Matrix

🏆 Results
Model	Accuracy	ROC-AUC
Logistic Regression	~75%	~0.78
Random Forest	~82%	~0.86
Gradient Boosting	~85%	~0.88
XGBoost (BEST)	88–92%	0.93–0.95
🧪 Installation
pip install -r requirements.txt

▶️ Run the Streamlit App (For Deployment)
streamlit run app.py

📌 Future Work (Deep Learning Phase)

CNN-based eye-state detection

Live video processing using OpenCV

Real-time alert system

Integrated dashboard with model predictions

Mobile deployment using Streamlit Cloud

👤 Author

Neeraj Guleria
Upcoming SWE @ Amazon
Machine Learning & Deep Learning Learner

Passionate about building real-world AI systems


## 🚀 Deployment

### Production Deployment with Docker

```bash
# Build production image
docker build -t driver-safety:prod .

# Run with production settings
docker run -d \
  -p 8000:8000 \
  -e API_KEY=your_secure_key \
  --name driver-safety-prod \
  driver-safety:prod

# Check logs
docker logs driver-safety-prod

# Stop container
docker stop driver-safety-prod
```

### Cloud Deployment Options

- **AWS**: ECS, EKS, or Lambda
- **GCP**: Cloud Run, GKE
- **Azure**: Container Instances, AKS
- **Heroku**: Container deployment

## 🛠️ Technology Stack

- **ML Framework**: scikit-learn
- **API Framework**: FastAPI
- **Server**: uvicorn
- **Containerization**: Docker
- **Testing**: pytest
- **Monitoring**: Custom metrics + audit logging
- **Security**: API key auth, rate limiting

## 📝 Documentation

- **API Docs**: http://localhost:8000/docs
- **Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Demo Guide**: [demo/README.md](demo/README.md)

---

**⭐ If you found this project helpful, please star the repository!**


## 🎨 Interactive Dashboard

### Run Streamlit Dashboard

```bash
# Terminal 1 - Start API
uvicorn src.app:app --reload

# Terminal 2 - Start Dashboard
streamlit run dashboard/app.py
```

**Dashboard Features:**
- 📊 Interactive sliders for driver inputs
- 📈 Real-time risk visualization
- 🎯 ML predictions with confidence scores
- ⚠️ Risk assessment gauge (0-100)
- 💡 Actionable recommendations
- 🔍 Feature importance display

**Access**: http://localhost:8501

