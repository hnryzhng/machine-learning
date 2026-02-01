#!/usr/bin/env bash
set -e

REQUIRED_PYTHON="3.11"

echo "Checking for Python ${REQUIRED_PYTHON}..."

if command -v python3.11 &> /dev/null; then
  PYTHON_BIN="python3.11"
else
  echo "Python 3.11 not found."
  echo "Install it first (e.g. via Homebrew):"
  echo "brew install python@3.11"
  exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo "✅ Using Python ${PYTHON_VERSION}"

echo "🔧 Creating virtual environment..."
$PYTHON_BIN -m venv .venv

echo "🔌 Activating virtual environment..."
source .venv/bin/activate

echo "⬆️ Upgrading pip..."
python -m pip install --upgrade pip

echo "🧹 Removing any existing torch installs..."
pip uninstall torch torchvision torchaudio -y || true

echo "📦 Installing core packages..."
pip install \
  jupyter \
  ipykernel \
  numpy \
  pandas \
  scipy \
  requests \
  matplotlib \
  Pillow

echo "🔥 Installing PyTorch for Apple Silicon (CPU + MPS)..."
pip install torch torchvision

# Fallback if torch import fails
if ! python -c "import torch" &> /dev/null; then
  echo "⚠️ Default PyTorch install failed — falling back to CPU wheels"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "🧠 Registering Jupyter kernel..."
python -m ipykernel install --user --name m1-py311 --display-name "Python 3.11 (M1 venv)"

echo "✅ Verifying installation..."
python - <<EOF
import sys, torch
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
EOF

echo "🎉 Setup complete!"
echo "👉 Run 'jupyter lab' and select kernel: Python 3.11 (M1 venv)"