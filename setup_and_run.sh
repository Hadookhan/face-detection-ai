#!/usr/bin/env bash
set -e  # exit on first error

# Go to the directory where the script lives
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "[INFO] Using project directory: $PROJECT_DIR"

# System packages (Ubuntu)
echo "[INFO] Installing system dependencies (you may be asked for your password)..."
sudo apt update
sudo apt install -y \
    python3 \
    python3-venv \
    python3-pip \
    libgl1 \
    libgtk2.0-dev

# Python virtual environment
if [ ! -d ".venv" ]; then
  echo "[INFO] Creating virtual environment..."
  python3 -m venv .venv
else
  echo "[INFO] Virtual environment already exists, reusing .venv"
fi

echo "[INFO] Activating virtual environment..."
# shellcheck source=/dev/null
source .venv/bin/activate

echo "[INFO] Python version in venv:"
python --version

# Python packages
echo "[INFO] Installing Python packages..."
pip install --upgrade pip
pip install opencv-contrib-python numpy

# Train the recognizer
echo "[INFO] Running training script..."
python src/train_recognizer.py

# Run webcam recognition
echo "[INFO] Starting webcam recognition (press 'q' to quit)..."
python src/webcam_recognition.py
