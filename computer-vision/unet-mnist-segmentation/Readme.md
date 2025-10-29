# MNIST Segmentation U-Net

Multi-output U-Net for simultaneous digit segmentation and classification on MNIST.

## What It Does

- **Segmentation**: Creates pixel-wise masks around digits
- **Classification**: Predicts digit classes (0-9)
- **Single Model**: One U-Net architecture, two outputs

## Quick Start

```bash
# Install dependencies
pip install tensorflow matplotlib seaborn numpy h5py

# Run complete pipeline (dataset generation + training)
python run_pipeline.py