@echo off
rem Crop Disease Classification Setup Script

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python not found.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Setup complete.
echo Run run_training.bat to start.
pause
