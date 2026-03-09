@echo off
echo ========================================
echo DRIVER DROWSINESS DETECTION - COMPLETE SETUP
echo ========================================
echo.

echo Step 1: Starting API Server...
echo.
start cmd /k "cd /d %~dp0 && python -m uvicorn src.app:app --reload"

timeout /t 5

echo Step 2: Starting Dashboard...
echo.
start cmd /k "cd /d %~dp0 && streamlit run dashboard/realtime_monitor.py"

timeout /t 3

echo.
echo ========================================
echo SYSTEM STARTED!
echo ========================================
echo.
echo API: http://localhost:8000/docs
echo Dashboard: http://localhost:8501
echo.
echo Press any key to open dashboard in browser...
pause > nul

start http://localhost:8501

echo.
echo To stop: Close both terminal windows
echo.
pause
