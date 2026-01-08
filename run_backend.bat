@echo off
rem Start Flask API

echo Starting server...
cd backend
python app.py

if errorlevel 1 (
    echo Error: Failed to start backend
    pause
)
    echo - Model not trained yet (run run_training.bat first)
    echo - Port 5000 already in use
    echo - Missing dependencies (run setup.bat)
    echo.
    pause
    exit /b 1
)
