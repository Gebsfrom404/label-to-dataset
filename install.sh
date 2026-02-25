#!/bin/bash

# Label-to-Dataset Installer (pip)
# Creates/updates .venv and installs all dependencies.
# Requires Python 3.13+ installed globally.

set -e

VENV_DIR=".venv"
PYTHON="python3"

# --- Check Python version ---
if ! $PYTHON --version 2>/dev/null | grep -qE "3\.(13|14)\."; then
    echo "ERROR: Python 3.13+ is required."
    echo -n "Found: "
    $PYTHON --version 2>/dev/null || echo "Python not found in PATH"
    echo ""
    echo "Install Python 3.13 from https://www.python.org/downloads/"
    echo "Or use install_uv.sh which can download Python automatically."
    exit 1
fi

echo "Python version: $($PYTHON --version)"

# --- Create or reuse venv ---
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Existing venv found, updating..."
else
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# --- Activate venv ---
source "$VENV_DIR/bin/activate"

# --- Upgrade pip ---
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# --- Install/upgrade dependencies ---
echo "Installing dependencies..."
pip install -r requirements.txt --upgrade
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Dependency installation failed."
    echo "If torch fails, check CUDA compatibility at https://pytorch.org/get-started/locally/"
    exit 1
fi

# --- Create model directories ---
mkdir -p models/yolo models/lama models/caption

echo ""
echo "============================================"
echo "  Installation complete!"
echo "  Run ./launch.sh to start the application."
echo "============================================"
