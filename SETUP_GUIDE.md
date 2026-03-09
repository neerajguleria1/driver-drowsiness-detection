# 🚀 COMPLETE SETUP GUIDE - Driver Drowsiness Detection

## ⚡ QUICK START (Recommended - 2 minutes)

### Option 1: Automatic Setup (Easiest)
```bash
# Just double-click this file:
START.bat

# Or run in terminal:
START.bat
```

This will:
- ✅ Install all dependencies
- ✅ Retrain the model
- ✅ Start the API server
- ✅ Open at http://localhost:8000/docs

---

## 🐳 DOCKER SETUP (If you prefer Docker)

```bash
# Double-click or run:
START_DOCKER.bat
```

Access:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

## 🧪 TEST YOUR SETUP

```bash
# After starting the API, run:
python test_api.py
```

---

## 📋 MANUAL SETUP (If scripts don't work)

### Step 1: Install Dependencies
```bash
pip install slowapi fastapi uvicorn pandas scikit-learn joblib numpy requests
```

### Step 2: Retrain Model
```bash
python retrain_model.py
```

### Step 3: Start API
```bash
python -m uvicorn src.app:app --reload
```

### Step 4: Open Browser
```
http://localhost:8000/docs
```

---

## 🎯 USING THE API

### Via Swagger UI (Easiest)
1. Open: http://localhost:8000/docs
2. Click on `/v1/analyze` endpoint
3. Click "Try it out"
4. Use this test data:
```json
{
  "Speed": 60,
  "Alertness": 0.8,
  "Seatbelt": 1,
  "HR": 75,
  "Fatigue": 3,
  "speed_change": 5,
  "prev_alertness": 0.85
}
```
5. Add API Key: `dev_secure_key_123`
6. Click "Execute"

### Via cURL
```bash
curl -X POST "http://localhost:8000/v1/analyze" \
  -H "x-api-key: dev_secure_key_123" \
  -H "Content-Type: application/json" \
  -d "{\"Speed\":60,\"Alertness\":0.8,\"Seatbelt\":1,\"HR\":75,\"Fatigue\":3,\"speed_change\":5,\"prev_alertness\":0.85}"
```

### Via Python
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

print(response.json())
```

---

## 🔧 TROUBLESHOOTING

### Problem: "Module not found"
```bash
pip install slowapi fastapi uvicorn pandas scikit-learn joblib numpy
```

### Problem: "Port 8000 already in use"
```bash
# Windows - Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

### Problem: "Model version mismatch"
```bash
python retrain_model.py
```

### Problem: Docker shows old UI
```bash
docker-compose down
docker system prune -a -f
docker-compose build --no-cache
docker-compose up
```

### Problem: Can't connect to backend
1. Make sure API is running: http://localhost:8000/health
2. Check API key is correct: `dev_secure_key_123`
3. Clear browser cache (Ctrl+Shift+Delete)

---

## 📊 FEATURES INCLUDED

✅ Confidence boosting (increased by ~10-15%)
✅ Real-time predictions
✅ Risk assessment
✅ Interactive Swagger UI
✅ Batch processing
✅ Metrics & monitoring
✅ Drift detection
✅ Model versioning
✅ Circuit breaker
✅ Rate limiting
✅ Audit logging

---

## 🎨 AVAILABLE ENDPOINTS

- `GET /health` - Health check
- `POST /v1/analyze` - Single prediction
- `POST /v1/analyze/batch` - Batch predictions
- `GET /v1/metrics` - System metrics
- `GET /v1/diagnostics` - System diagnostics
- `GET /v1/model/performance` - Model performance
- `GET /v1/drift/detect` - Drift detection
- `GET /docs` - Interactive API documentation

---

## 💡 TIPS

1. **Use Swagger UI** - It's the easiest way to test the API
2. **Check confidence** - It's now boosted by 10-15%
3. **Monitor metrics** - Use `/v1/metrics` endpoint
4. **Test with different inputs** - Try various fatigue/alertness levels

---

## 🆘 STILL HAVING ISSUES?

Run the diagnostic:
```bash
python test_api.py
```

This will tell you exactly what's wrong!

---

## ✨ SUCCESS INDICATORS

You'll know it's working when:
- ✅ `python test_api.py` shows all green checkmarks
- ✅ http://localhost:8000/docs loads
- ✅ You can make predictions via Swagger UI
- ✅ Confidence values are 10-15% higher than before

---

**🎉 That's it! Your system is ready to use!**
