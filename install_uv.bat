@echo off
setlocal

REM Label-to-Dataset Installer (uv)
REM Creates/updates .venv with Python 3.13 and installs all dependencies.
REM uv will download Python 3.13 automatically if not installed globally.

set PYTHON_VERSION=3.13

REM --- Check if uv is available ---
uv --version >nul 2>&1
if errorlevel 1 (
    echo uv not found. Installing uv...
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if errorlevel 1 (
        echo ERROR: Failed to install uv.
        echo Install manually: https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    REM Refresh PATH so uv is available
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    uv --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: uv installed but not found in PATH. Please restart your terminal and run again.
        pause
        exit /b 1
    )
)

echo uv version:
uv --version

REM --- Sync: create venv + install dependencies ---
echo Syncing dependencies (Python %PYTHON_VERSION%)...
uv sync --python %PYTHON_VERSION%
if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
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
