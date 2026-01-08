@echo off
rem Start Frontend server

echo Starting frontend...
cd frontend
python -m http.server 8000

if errorlevel 1 (
    echo Error: Failed to start frontend
    pause
)
    echo.
    pause
    exit /b 1
)
