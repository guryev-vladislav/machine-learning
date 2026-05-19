# Machine Learning Projects Collection

A collection of machine learning projects and implementations covering various computer vision tasks and deep learning architectures. This repository serves as a practical resource for ML experimentation and education.

## Projects Overview

### 1. Concrete Crack Detection (PyTorch)
Two-stage approach: UNet for segmentation + CNN for classification
- **Model**: U-Net (encoder-decoder) + Simple CNN
- **Task**: Binary crack/no-crack classification
- **Metrics Generated**: IoU, Dice, Accuracy, Precision, Recall, F1-Score
- **Visualizations**: Training curves, confusion matrix, multi-scale comparison
- **Location**: `computer-vision/concrete_crack_detection/`

### 2. YOLO Crack Detection (Ultralytics)
Real-time classification approach
- **Model**: YOLOv8n-cls (nano classification)
- **Task**: Binary classification
- **Metrics Generated**: Accuracy, Precision, Recall, F1-Score, ROC AUC
- **Visualizations**: Training history, comprehensive performance report, model comparison
- **Location**: `computer-vision/yolo-crack-detection/`

### 3. CNN MNIST Segmentation
Convolutional neural network for digit segmentation
- **Model**: Custom CNN architecture
- **Task**: Digit segmentation and classification
- **Location**: `computer-vision/cnn-mnist-segmentation/`

### 4. UNet MNIST Segmentation
U-Net based segmentation for digit isolation
- **Model**: U-Net encoder-decoder
- **Task**: Semantic segmentation
- **Location**: `computer-vision/unet-mnist-segmentation/`

## Features

- ✅ Clean, modular code structure for easy experimentation
- ✅ Comprehensive configuration management (YAML configs)
- ✅ **Automated metrics generation during training**
- ✅ **Automatic visualization of results**
- ✅ Reproducible training pipelines
- ✅ Practical examples of deep learning applications
- ✅ Support for multiple datasets (DeepCrack, SDNET2018)

## Key Metrics & Visualizations

### Concrete Crack Detection Metrics

**Segmentation (UNet)**:
- IoU (Intersection over Union): measures overlap between predicted and true masks
- Dice Coefficient: robust metric for imbalanced data
- Pixel Accuracy: percentage of correctly classified pixels

**Classification (CNN)**:
- Accuracy: overall correctness
- Precision: reliability of positive predictions
- Recall: completeness of positive detection
- F1-Score: balance between precision and recall

**Generated Visualizations**:
- `outputs/run_*/plots/unet_curves.png` - UNet training history
- `outputs/run_*/plots/cnn_curves.png` - CNN training history
- `outputs/run_*/plots/result_*.png` - Multi-scale comparison (256px, 128px, 64px)

### YOLO Crack Detection Metrics

**Classification**:
- Accuracy: percentage of correct predictions
- Precision: false positive rate control
- Recall: false negative rate control
- F1-Score: harmonic mean of precision and recall
- ROC AUC: ranking quality metric

**Generated Visualizations**:
- `outputs/training_history.png` - Loss and accuracy curves
- `outputs/confusion_matrix.png` - Error analysis matrix
- `outputs/roc_curve.png` - ROC analysis for threshold tuning
- `outputs/pr_curve.png` - Precision-Recall trade-off
- `outputs/comprehensive_report.png` - Full performance summary

## Technologies

Python 3.10+, PyTorch, Ultralytics (YOLO), OpenCV, scikit-learn, NumPy, Matplotlib, Seaborn

## Usage

### Concrete Crack Detection Training

```bash
cd computer-vision/concrete_crack_detection

# Train both UNet and CNN
python main.py
# Metrics and plots automatically saved to outputs/run_*/
```

Outputs:
- Models: `outputs/run_*/models/unet.pth`, `cnn.pth`
- Metrics: Printed to console + saved in logs
- Plots: `outputs/run_*/plots/` (training curves, segmentation results)

### YOLO Crack Detection Training

```bash
cd computer-vision/yolo-crack-detection

# Prepare data and train
python train.py --prepare-data --train --epochs 50

# Or just train with existing data
python train.py --train
```

Outputs:
- Best model: `outputs/best_model.pt`
- Metrics: Generated during and after training
- Reports: Saved in outputs directory

### Inference

```bash
cd computer-vision/yolo-crack-detection

# Run on video
python main.py

# Get detection report
# Outputs saved to: outputs/deepcrack_cracked_detected_report.txt
```

## Metrics Explanation for Thesis

### Classification Metrics (YOLO, CNN)

| Metric | Formula | Meaning | Target |
|--------|---------|---------|--------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness | > 90% |
| **Precision** | TP/(TP+FP) | False positive rate | > 85% |
| **Recall** | TP/(TP+FN) | False negative rate | > 85% |
| **F1-Score** | 2(PR)/(P+R) | Balanced metric | > 87% |
| **ROC AUC** | Area under curve | Discrimination ability | > 95% |

### Segmentation Metrics (UNet)

| Metric | Formula | Meaning | Target |
|--------|---------|---------|--------|
| **IoU** | \|A∩B\|/\|A∪B\| | Overlap ratio | > 0.75 |
| **Dice** | 2\|A∩B\|/(A+B) | Similarity coefficient | > 0.80 |
| **PixelAcc** | (TP+TN)/Total | Pixel-level accuracy | > 95% |

### Confusion Matrix

Used to analyze:
- True Positives (TP): correctly detected cracks
- True Negatives (TN): correctly detected non-cracks
- False Positives (FP): incorrectly detected cracks
- False Negatives (FN): missed cracks

## File Structure

```
machine-learning/
├── README.md (this file)
├── computer-vision/
│   ├── concrete_crack_detection/
│   │   ├── src/utils/
│   │   │   ├── metrics.py          ← IoU, Dice calculations
│   │   │   ├── visualizer.py       ← Plot generation
│   │   │   └── config.py
│   │   ├── src/training/
│   │   │   ├── trainer.py          ← Training loop with metrics
│   │   │   └── pipeline.py
│   │   ├── main.py                 ← Entry point
│   │   ├── MODEL_DOCUMENTATION.md  ← Detailed explanation
│   │   └── outputs/run_*/
│   │       ├── models/             ← Saved weights
│   │       ├── plots/              ← Generated graphs
│   │       └── config_used.yaml    ← Training config
│   │
│   ├── yolo-crack-detection/
│   │   ├── src/utils/
│   │   │   ├── metrics.py          ← Classification metrics
│   │   │   ├── visualizer.py       ← Plot generation
│   │   │   └── config.py
│   │   ├── train.py                ← Training with metrics
│   │   ├── main.py                 ← Inference
│   │   ├── MODEL_DOCUMENTATION.md  ← Detailed explanation
│   │   ├── src/models/pretrained/  ← Pre-trained models
│   │   └── outputs/                ← Generated metrics & plots
│   │
│   └── [other projects]/
└── thesis_report_generator.py      ← Comparative report generation
```

## Automatic Metrics Generation During Training

### When you run training:

**Concrete Crack Detection**:
```bash
python main.py
```
✅ Automatically generates:
- `plots/unet_curves.png` - Loss/IoU/Dice over epochs
- `plots/cnn_curves.png` - Loss/Accuracy over epochs
- `plots/result_*.png` - Segmentation results on test images
- Console output: real-time metrics for each epoch

**YOLO Crack Detection**:
```bash
python train.py --train
```
✅ Automatically generates:
- `outputs/training_history.png` - Training curves
- `outputs/confusion_matrix.png` - Error analysis
- `outputs/roc_curve.png` - ROC analysis
- Training logs with metrics

## For Thesis/Diploma

All necessary information for your thesis:
- ✅ Model architecture details (in MODEL_DOCUMENTATION.md)
- ✅ Metric formulas and explanations (in this README)
- ✅ Performance graphs (auto-generated during training)
- ✅ Comparison tables (can be generated via thesis_report_generator.py)
- ✅ Methodological documentation

Generate comprehensive report:
```bash
python thesis_report_generator.py
# Creates: thesis_reports/report_YYYYMMDD_HHMMSS/
```

## Dependencies

```bash
# For concrete crack detection
pip install torch torchvision pytorch-lightning
pip install opencv-python albumentations
pip install matplotlib seaborn scikit-learn

# For YOLO crack detection
pip install ultralytics
pip install opencv-python-headless

# Common
pip install numpy pyyaml pydantic
```

## Configuration

Each project uses YAML configuration files:
- `concrete_crack_detection/configs/train_config.yaml`
- `yolo-crack-detection/configs/train_config.yaml`

Customize training parameters, batch size, learning rate, number of epochs, etc.

## Results Summary

### Expected Performance

**Concrete Crack Detection**:
- UNet IoU: > 0.75
- CNN Accuracy: > 90%

**YOLO Crack Detection**:
- Accuracy: > 95%
- F1-Score: > 93%

All metrics are automatically calculated and visualized during training.
