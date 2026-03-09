@echo off
echo ========================================
echo AUTO-PLAY DEMO FOR HR/RECRUITERS
echo ========================================
echo.
echo This will:
echo 1. Start the API
echo 2. Start the auto-play demo
echo 3. Open browser automatically
echo.
echo NO INPUT NEEDED - Just watch!
echo.
pause

echo Starting API...
start cmd /k "cd /d %~dp0 && python -m uvicorn src.app:app --reload"

timeout /t 5

echo Starting Auto-Play Demo...
start cmd /k "cd /d %~dp0 && streamlit run dashboard/auto_demo.py"

timeout /t 3

echo Opening browser...
start http://localhost:8501

echo.
echo ========================================
echo DEMO IS RUNNING!
echo ========================================
echo.
echo The demo will:
echo - Show 3 scenarios automatically
echo - Normal driving (5 updates)
echo - Getting tired (5 updates)  
echo - Critical alert (5 updates)
echo.
echo Total time: ~30 seconds
echo.
echo Just watch the screen!
echo.
pause
