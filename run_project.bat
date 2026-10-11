@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo   Bengaluru Road Accident Risk Prediction and Safety Recommendation System
echo   MCA Generative AI & Data Science Academic Project
echo ==============================================================================
echo.

:: 1. Check Python installation
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found in your system PATH!
    echo Please install Python 3.10+ and check 'Add Python to PATH'.
    pause
    exit /b 1
)

:: 2. Create or verify Virtual Environment
if not exist "venv" (
    echo [STEP 1/5] Creating Python virtual environment in 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [STEP 1/5] Virtual environment 'venv' already exists.
)

:: 3. Activate Virtual Environment
echo [STEP 2/5] Activating virtual environment...
call venv\Scripts\activate.bat

:: 4. Install Requirements
echo [STEP 3/5] Installing/verifying dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: 5. Copy .env if not exists
if not exist ".env" (
    echo [STEP 4/5] Creating default .env from .env.example...
    copy .env.example .env >nul
) else (
    echo [STEP 4/5] Environment configuration '.env' verified.
)

:: 6. Check if data and models exist; if not, download and train
if not exist "data\processed\bengaluru_accidents_clean.csv" (
    echo [PIPELINE] Data not preprocessed. Running data download and preprocessing...
    python data\download_data.py
    python data\preprocess.py
)

if not exist "models\accident_risk_model.pkl" (
    echo [PIPELINE] Model artifact not found. Training machine learning models...
    python ml\train_model.py
)

:: 7. Start the Web Application
echo.
echo ==============================================================================
echo   APPLICATION STARTING!
echo   Open your browser at: http://127.0.0.1:5000
echo   Press CTRL+C in this terminal window to stop the server.
echo ==============================================================================
echo.
python app.py

pause
