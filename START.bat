@echo off
echo ========================================
echo DRIVER DROWSINESS DETECTION - AUTO FIX
echo ========================================
echo.

echo [1/5] Installing required packages...
pip install slowapi fastapi uvicorn pandas scikit-learn joblib numpy

echo.
echo [2/5] Retraining model with current scikit-learn version...
python retrain_model.py

echo.
echo [3/5] Cleaning Docker cache...
docker-compose down
docker system prune -f

echo.
echo [4/5] Starting API server...
echo.
echo ========================================
echo API will start at: http://localhost:8000
echo Swagger UI: http://localhost:8000/docs
echo ========================================
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
