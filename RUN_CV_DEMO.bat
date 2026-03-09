@echo off
cls
echo ========================================
echo   PROFESSIONAL COMPUTER VISION DEMO
echo   MediaPipe + OpenCV Production-Grade
echo ========================================
echo.
echo This demo shows PROFESSIONAL computer vision:
echo  - MediaPipe 468-point facial landmarks
echo  - Precise eye contour detection  
echo  - Advanced head pose estimation
echo  - Production-grade accuracy
echo  - Real-time performance metrics
echo.
echo Requirements:
echo  - Webcam connected
echo  - pip install opencv-python mediapipe scipy
echo.
echo Choose an option:
echo  1. Professional CV Demo (MediaPipe)
echo  2. Standard CV Demo (OpenCV only)
echo  3. FAANG-Level Simple Demo (Reliable)
echo  4. Test eye detection
echo  5. Install MediaPipe
echo  6. Exit
echo.
set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto professional
if "%choice%"=="2" goto standard
if "%choice%"=="3" goto simple
if "%choice%"=="4" goto test
if "%choice%"=="5" goto install
if "%choice%"=="6" goto end

:professional
echo.
echo Starting Professional CV Demo...
echo Features: 468 landmarks, precise detection, head pose
echo Press 'q' to quit, 's' for stats
python demo\professional_cv_demo.py
goto end

:simple
echo.
echo Starting FAANG-Level Simple Demo...
echo Features: Reliable detection, professional UI, no dependencies
echo Press 'q' to quit
python demo\simple_cv_demo.py
goto end

:standard
echo.
echo Starting Standard CV Demo...
echo Press 'q' to quit
python demo\cv_demo.py
goto end

:test
echo.
echo Starting eye detection test...
python test_eye_detection.py
goto end

:install
echo.
echo Installing MediaPipe for professional features...
echo.
echo Fixing numpy compatibility issue...
pip uninstall numpy opencv-python -y
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install mediapipe>=0.10.0 imutils>=0.5.4 scipy>=1.7.0
echo.
echo Installation complete! Try option 1 now.
pause
goto end

:end
pause
