#!/bin/bash

# Label-to-Dataset Installer (uv)
# Creates/updates .venv with Python 3.13 and installs all dependencies.
# uv will download Python 3.13 automatically if not installed globally.

set -e

PYTHON_VERSION="3.13"

# --- Check if uv is available, install if not ---
if ! command -v uv &>/dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        echo "ERROR: uv installed but not found in PATH. Restart your terminal and run again."
        exit 1
    fi
fi

echo "uv version: $(uv --version)"

# --- Sync: create venv + install dependencies ---
echo "Syncing dependencies (Python $PYTHON_VERSION)..."
uv sync --python "$PYTHON_VERSION"

# --- Create model directories ---
mkdir -p models/yolo models/lama models/caption

echo ""
echo "============================================"
echo "  Installation complete!"
echo "  Run ./launch.sh to start the application."
echo "============================================"
