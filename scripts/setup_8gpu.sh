#!/bin/bash
# 8-GPU Session Setup Script
# Run this first when you SSH into the 8-GPU machine

set -e

echo "=========================================="
echo "Nangila 8-GPU Test Environment Setup"
echo "=========================================="

# Check GPUs
echo "Checking GPUs..."
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "Found $GPU_COUNT GPUs"

if [ "$GPU_COUNT" -lt 8 ]; then
    echo "WARNING: Expected 8 GPUs, found $GPU_COUNT"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate
pip install huggingface_hub maturin

# Clone Nangila if not present
if [ ! -d "nangila" ]; then
    echo ""
    echo "Cloning Nangila..."
    git clone https://github.com/careltchirara-bot/nangila-db.git nangila
fi

cd nangila

# Build Nangila
echo ""
echo "Building Nangila..."
maturin develop --release -F python

# Download model
echo ""
echo "Downloading Mistral 7B..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('mistralai/Mistral-7B-v0.3', local_dir='./mistral-7b')
print('Download complete!')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run bandwidth benchmark:  ./scripts/run_bandwidth_test.sh"
echo "  2. Run training test:        ./scripts/run_training_test.sh"
echo "  3. Run full validation:      ./scripts/run_full_validation.sh"
