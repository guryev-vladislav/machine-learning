#!/bin/bash

set -euo pipefail

echo "YOLO Crack Detection"
echo "Project setup"
echo "===================="
echo ""

cd "$(dirname "$0")" || exit 1

echo "[1/3] Checking dependencies"
if ! python3 -c "import torch" 2>/dev/null; then
    echo "      PyTorch is not available. Installing requirements."
    python3 -m pip install -r requirements.txt
    echo "      Dependencies installed."
else
    echo "      PyTorch is already available."
fi

echo ""
echo "[2/3] Creating project directories"
mkdir -p data/{raw,videos,dataset/{train/{crack,no_crack},val/{crack,no_crack}}} results outputs
echo "      Directories are ready."

echo ""
echo "[3/3] Setup complete"
echo ""
echo "Available commands"
echo "------------------"
echo ""
echo "  Configuration: configs/train_config.yaml"
echo ""
echo "  1. Full pipeline:"
echo "      python main.py"
echo ""
echo "  2. Prepare dataset:"
echo "      python main.py --skip-train"
echo ""
echo "  3. Train model:"
echo "      python main.py --epochs 50"
echo ""
echo "  4. Process video:"
echo "      python -m src.inference.video_processor --video data/videos/input.mp4 --model outputs/<run>/best_model.pt --output outputs/result.mp4"
echo ""
echo "  5. Smoke test:"
echo "      python main.py --config configs/quickstart.yaml --smoke-test --run-name quickstart"
echo ""
echo "Documentation"
echo "-------------"
echo "  README.md"
echo ""
echo "Start with: python main.py"
echo ""

