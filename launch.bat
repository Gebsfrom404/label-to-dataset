@echo off
setlocal

REM Label-to-Dataset Launcher (Windows)
set VENV_DIR=.venv

REM Create venv if it doesn't exist
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment. Ensure Python 3.12+ is installed.
        pause
        exit /b 1
    )
)

REM Activate venv
call %VENV_DIR%\Scripts\activate.bat

REM Install/upgrade PyTorch with CUDA
pip show torch >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
)

REM Install requirements
echo Checking dependencies...
pip install -r requirements.txt --quiet

REM Create model directories
if not exist "models\yolo" mkdir models\yolo
if not exist "models\lama" mkdir models\lama
if not exist "models\caption" mkdir models\caption

REM Run the app
echo Starting Label-to-Dataset...
python main.py

endlocal