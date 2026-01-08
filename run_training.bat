@echo off
rem Crop Disease Classification Training Script

echo Starting model training...
echo This process may take several hours.
echo Press Ctrl+C to stop.

python train_model.py

if errorlevel 1 (
    echo.
    echo Error: Training failed.
    pause
    exit /b 1
)

echo.
echo Training complete.
echo Model saved to models/crop_disease_model.pth
pause
