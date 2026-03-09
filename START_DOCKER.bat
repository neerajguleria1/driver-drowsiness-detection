@echo off
echo ========================================
echo DOCKER DEPLOYMENT - COMPLETE REBUILD
echo ========================================
echo.

echo [1/4] Stopping all containers...
docker-compose down

echo.
echo [2/4] Removing old images and cache...
docker system prune -a -f

echo.
echo [3/4] Rebuilding images (no cache)...
docker-compose build --no-cache

echo.
echo [4/4] Starting services...
docker-compose up

echo.
echo ========================================
echo Services Running:
echo API: http://localhost:8000/docs
echo Dashboard: http://localhost:8501
echo ========================================
