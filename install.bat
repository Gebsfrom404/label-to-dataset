@echo off
setlocal

REM Label-to-Dataset Installer (pip)
REM Creates/updates .venv and installs all dependencies.
REM Requires Python 3.13+ installed globally.

set VENV_DIR=.venv
set PYTHON=python

REM --- Check Python version ---
%PYTHON% --version 2>nul | findstr /R "3\.13\. 3\.14\." >nul
if errorlevel 1 (
    echo ERROR: Python 3.13+ is required.
    echo Found:
    %PYTHON% --version 2>nul || echo   Python not found in PATH
    echo.
    echo Install Python 3.13 from https://www.python.org/downloads/
    echo Or use install_uv.bat which can download Python automatically.
    pause
    exit /b 1
)

echo Python version:
%PYTHON% --version

REM --- Create or reuse venv ---
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Existing venv found, updating...
) else (
    echo Creating virtual environment...
    %PYTHON% -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM --- Activate venv ---
call %VENV_DIR%\Scripts\activate.bat

REM --- Upgrade pip ---
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM --- Install/upgrade dependencies ---
echo Installing dependencies...
pip install -r requirements.txt --upgrade
if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    echo If torch fails, check CUDA compatibility at https://pytorch.org/get-started/locally/
    pause
    exit /b 1
)

REM --- Create model directories ---
if not exist "models\yolo" mkdir models\yolo
if not exist "models\lama" mkdir models\lama
if not exist "models\caption" mkdir models\caption

echo.
echo ============================================
echo   Installation complete!
echo   Run launch.bat to start the application.
echo ============================================

pause
endlocal
