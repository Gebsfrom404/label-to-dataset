#!/bin/bash

# Label-to-Dataset Launcher (Linux/Mac)
# Run install.sh or install_uv.sh first to set up dependencies.

VENV_DIR=".venv"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Virtual environment not found."
    echo "Run ./install.sh or ./install_uv.sh first."
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Starting Label-to-Dataset..."
python main.py
