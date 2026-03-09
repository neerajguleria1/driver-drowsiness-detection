@echo off
cls
echo ========================================
echo   VIDEO ANALYSIS - CV Demo
echo ========================================
echo.
echo This will analyze a video file for drowsiness
echo.
echo Usage:
echo   1. Drag and drop a video file onto this batch file
echo   2. Or run: TEST_VIDEO.bat "path\to\video.mp4"
echo.

if "%~1"=="" (
    echo ERROR: No video file provided!
    echo.
    echo Please drag and drop a video file onto this batch file
    echo Or provide the path as an argument
    echo.
    pause
    exit /b 1
)

echo Analyzing video: %~1
echo.
echo Controls:
echo   - Press 'q' to quit
echo   - Press SPACE to pause/resume
echo.
pause

python demo\test_video.py "%~1"

pause
