# COMPLETE FIX GUIDE - Driver Drowsiness Detection

## Problem Summary
1. Scikit-learn version mismatch (model trained on 1.6.1, you have 1.8.0)
2. Docker caching old UI
3. Backend connection issues

## SOLUTION - Follow These Steps Exactly

### Step 1: Clean Everything
```bash
# Stop all containers
docker-compose down

# Remove old images
docker system prune -a

# Clear browser cache (Ctrl+Shift+Delete)
```

### Step 2: Retrain Model (CRITICAL)
```bash
# Run the retrain script
python retrain_model.py
```

### Step 3: Test Locally First
```bash
# Terminal 1 - Start API
python -m uvicorn src.app:app --reload

# Wait for "Application startup complete"
# Then open: http://localhost:8000/docs
# Test the API with Swagger UI
```

### Step 4: If Local Works, Use Docker
```bash
# Build fresh (no cache)
docker-compose build --no-cache

# Start services
docker-compose up

# Access:
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

## Quick Test Commands

### Test API is Working
```bash
curl -X POST "http://localhost:8000/v1/analyze" \
  -H "x-api-key: dev_secure_key_123" \
  -H "Content-Type: application/json" \
  -d "{\"Speed\":60,\"Alertness\":0.8,\"Seatbelt\":1,\"HR\":75,\"Fatigue\":3,\"speed_change\":5,\"prev_alertness\":0.85}"
```

## If Still Having Issues

### Option 1: Use Only Swagger UI (Simplest)
1. Run: `python -m uvicorn src.app:app --reload`
2. Open: http://localhost:8000/docs
3. Use the interactive API directly

### Option 2: Skip Docker Completely
```bash
# Terminal 1 - API
python -m uvicorn src.app:app --reload

# Terminal 2 - Dashboard (if you want UI)
streamlit run dashboard/app.py
```

## Common Errors & Fixes

### Error: "Module 'slowapi' not found"
```bash
pip install slowapi
```

### Error: "Model version mismatch"
```bash
python retrain_model.py
```

### Error: "Port already in use"
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Then restart
```

### Docker UI shows old version
```bash
# Force rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up
```

## Recommended: Simplest Working Setup

```bash
# 1. Retrain model
python retrain_model.py

# 2. Start API
python -m uvicorn src.app:app --reload

# 3. Use Swagger UI
# Open: http://localhost:8000/docs
```

This avoids Docker complexity and works immediately!
