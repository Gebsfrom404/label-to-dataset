#!/bin/bash

# Label-to-Dataset Launcher (Linux/Mac)
VENV_DIR=".venv"

# Create venv if it doesn't exist
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment. Ensure Python 3.12+ is installed."
        exit 1
    fi
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install/upgrade PyTorch with CUDA
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# Install requirements
echo "Checking dependencies..."
pip install -r requirements.txt --quiet

# Create model directories
mkdir -p models/yolo models/lama models/caption

# Run the app
echo "Starting Label-to-Dataset..."
python main.py
