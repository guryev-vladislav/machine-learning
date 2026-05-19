#!/bin/bash

# YOLO Crack Detection - First Run Script
# Этот скрипт поможет вам начать работу с проектом

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     YOLO CRACK DETECTION - First Run Setup                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Переходим в папку проекта
cd "$(dirname "$0")" || exit 1

echo "📦 Step 1: Checking dependencies..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "   ⚠️  PyTorch not found. Installing requirements..."
    pip install -r requirements.txt
    echo "   ✅ Dependencies installed"
else
    echo "   ✅ Dependencies already installed"
fi

echo ""
echo "📂 Step 2: Creating project directories..."
mkdir -p data/{raw,videos,dataset/{train/{crack,no_crack},val/{crack,no_crack}}} results outputs
echo "   ✅ Directories created"

echo ""
echo "🎯 Step 3: Project setup complete!"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📖 Quick Start Commands:"
echo ""
echo "  1️⃣  Full Pipeline (everything in one):"
echo "      python train.py --all"
echo ""
echo "  2️⃣  Create videos from dataset images:"
echo "      python main.py --create-video all"
echo ""
echo "  3️⃣  Prepare dataset for training:"
echo "      python train.py --prepare-data"
echo ""
echo "  4️⃣  Train YOLO model:"
echo "      python train.py --train --epochs 50"
echo ""
echo "  5️⃣  Process video (detect cracks):"
echo "      python main.py --video data/videos/deepcrack_cracked.mp4 --output results/output.mp4"
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "📚 Documentation:"
echo "   • QUICKSTART_RU.md  - Быстрый старт (русский)"
echo "   • README.md         - Overview"
echo "   • USAGE_GUIDE.md    - Full documentation"
echo "   • ARCHITECTURE.md   - Technical details"
echo ""
echo "✨ Ready to go! Start with: python train.py --all"
echo ""

