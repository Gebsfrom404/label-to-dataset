@echo off
setlocal

REM Label-to-Dataset Launcher (Windows)
REM Run install.bat or install_uv.bat first to set up dependencies.

set VENV_DIR=.venv

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found.
    echo Run install.bat or install_uv.bat first.
    pause
    exit /b 1
)

call %VENV_DIR%\Scripts\activate.bat

echo Starting Label-to-Dataset...
python main.py
if %errorlevel% neq 0 pause

endlocal
