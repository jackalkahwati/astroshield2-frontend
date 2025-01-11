#!/bin/bash

# Create conda environment
conda create -n astro_ml python=3.12 -y

# Source conda for shell script
source ~/miniconda3/etc/profile.d/conda.sh

# Activate environment
conda activate astro_ml

# Install PyTorch for macOS ARM
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
pip3 install onnx numpy

# Run the export script
python ml/models/export_models.py
