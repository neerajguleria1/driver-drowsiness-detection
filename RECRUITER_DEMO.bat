@echo off
cls
echo ========================================
echo   MAANG-LEVEL INTERACTIVE DEMO
echo   For Recruiters / Technical Interviews
echo ========================================
echo.
echo This demo shows:
echo  - Computer Vision feature extraction
echo  - ML model processing
echo  - Real-time predictions
echo  - Production-ready system
echo.
echo Recruiters can:
echo  1. Select pre-built scenarios
echo  2. See INPUT features clearly
echo  3. Watch ML processing
echo  4. See OUTPUT predictions
echo.
echo NO sensors needed - uses realistic scenarios!
echo.
pause

echo.
echo [1/2] Starting API Server...
start cmd /k "title API Server && cd /d %~dp0 && python -m uvicorn src.app:app --reload"

timeout /t 5

echo [2/2] Starting Interactive Demo...
start cmd /k "title Interactive Demo && cd /d %~dp0 && streamlit run dashboard/interactive_demo.py"

timeout /t 3

echo.
echo Opening browser...
start http://localhost:8501

echo.
echo ========================================
echo   DEMO READY!
echo ========================================
echo.
echo Instructions for Recruiter:
echo  1. Select a scenario from dropdown
echo  2. Click "Analyze Driver Condition"
echo  3. Watch the 4-step process
echo.
echo Scenarios available:
echo  - Alert Professional Driver (Safe)
echo  - Tired Long-Haul Driver (Warning)
echo  - Critical Drowsy Driver (Danger)
echo.
echo Each shows:
echo  INPUT: Video analysis + sensor data
echo  PROCESS: ML model inference
echo  OUTPUT: Prediction + risk score
echo.
pause
