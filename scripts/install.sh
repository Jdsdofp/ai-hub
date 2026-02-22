#!/bin/bash
# SmartX Vision Platform v3 — Manual Installation
set -e

echo "==========================================="
echo " SmartX Vision Platform v3 — Installation"
echo "==========================================="

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3.10+ required"
    exit 1
fi

PYVER=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[OK] Python $PYVER"

# Check GPU
if command -v nvidia-smi &>/dev/null; then
    echo "[OK] GPU detected"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true
else
    echo "[WARNING] No GPU detected"
fi

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env if missing
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[OK] Created .env from template — edit credentials before running"
fi

# Create directories
mkdir -p data logs

echo ""
echo "==========================================="
echo " [OK] Installation complete!"
echo "==========================================="
echo ""
echo " 1. Edit .env with your credentials"
echo " 2. Start: source venv/bin/activate && python main.py"
echo " 3. Open: http://localhost:5000"
echo ""
