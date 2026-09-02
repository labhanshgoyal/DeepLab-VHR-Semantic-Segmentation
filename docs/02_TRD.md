# 🔧 Technical Requirements Document (TRD)

## DeepLab Semantic Segmentation — Technical Architecture

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Date** | August 16, 2026 |
| **Status** | Draft |

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROJECT STRUCTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Data Layer   │───>│ Training     │───>│  Evaluation &         │  │
│  │              │    │ Pipeline     │    │  Metrics              │  │
│  │ - COCO JSON  │    │              │    │                       │  │
│  │ - DataLoader │    │ - CLI args   │    │ - Confusion Matrix    │  │
│  │ - Augment    │    │ - AMP        │    │ - Per-class IoU       │  │
│  │ - Transform  │    │ - TensorBoard│    │ - Loss Curves         │  │
│  └──────────────┘    │ - Checkpoint │    │ - PR Curves           │  │
│                      └──────────────┘    └───────────────────────┘  │
│                             │                       │               │
│                             ▼                       ▼               │
│                      ┌──────────────┐    ┌───────────────────────┐  │
│                      │  Models      │    │  Inference Pipeline   │  │
│                      │              │    │                       │  │
│                      │ - DeepLabV1  │    │ - predict.py          │  │
│                      │ - DeepLabV2  │    │ - ONNX export         │  │
│                      │ - DeepLabV3  │    │ - Batch prediction    │  │
│                      │ - DeepLabV3+ │    └───────────────────────┘  │
│                      └──────────────┘               │               │
│                                                     ▼               │
│                                          ┌───────────────────────┐  │
│                                          │  Streamlit Demo App   │  │
│                                          │                       │  │
│                                          │ - Upload & Predict    │  │
│                                          │ - Model Comparison    │  │
│                                          │ - Results Dashboard   │  │
│                                          └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Stack

### Core ML

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | >= 3.10 |
| Deep Learning Framework | PyTorch | >= 2.0 |
| Vision Utilities | torchvision | >= 0.15 |
| Dataset Annotations | pycocotools | >= 2.0 |
| Image Processing | scikit-image (skimage) | >= 0.20 |
| Numerical Computing | NumPy | >= 1.24 |
| Experiment Tracking | TensorBoard | >= 2.12 |
| Mixed Precision | torch.cuda.amp (built-in) | — |
| Model Export | ONNX + onnxruntime | >= 1.14 |

### Demo Application

| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | Streamlit | >= 1.28 |
| Plotting | Matplotlib | >= 3.7 |
| Image Handling | Pillow (PIL) | >= 9.0 |
| Interactive Plots | Plotly | >= 5.0 |

### Development

| Component | Technology |
|-----------|-----------|
| Version Control | Git |
| Linting | flake8, black |
| Type Checking | mypy (optional) |
| Testing | pytest |

---

## 3. Project Directory Structure (Target)

```
DeepLab_NWPU-VHR-10_Dataset_coco/
├── configs/
│   └── default.yaml              # Default training configuration
│
├── data/
│   └── NWPU_VHR-10/
│       ├── positive_image_set/   # 650 satellite images (downloaded)
│       ├── annotations.json      # Full COCO annotations
│       ├── train.json            # Training split
│       ├── val.json              # Validation split
│       └── test.json             # Test split (NEW)
│
├── models/
│   ├── __init__.py               # Model registry
│   ├── deeplabv1.py              # DeepLab V1 (Large FOV)
│   ├── deeplabv2.py              # DeepLab V2 (ResNet101 + ASPP)
│   ├── deeplabv3.py              # DeepLab V3 (Improved ASPP)
│   ├── deeplabv3plus.py          # DeepLab V3+ (Encoder-Decoder) [RENAMED]
│   └── backbones.py              # Backbone configurations (NEW)
│
├── utils/
│   ├── __init__.py
│   ├── dataset.py                # CocoDataset class (MOVED)
│   ├── transforms.py             # Data augmentation pipeline (NEW)
│   ├── losses.py                 # Loss functions (REFACTORED)
│   ├── metrics.py                # IoU, confusion matrix (REFACTORED)
│   ├── visualization.py          # Mask overlay, plotting (REFACTORED)
│   └── checkpoint.py             # Model save/load utilities (NEW)
│
├── scripts/
│   ├── train.py                  # CLI training script (REFACTORED)
│   ├── predict.py                # Single/batch inference (NEW)
│   ├── evaluate.py               # Full evaluation pipeline (NEW)
│   ├── export.py                 # ONNX/TorchScript export (NEW)
│   └── split_dataset.py          # Train/val/test split (REFACTORED)
│
├── app/
│   ├── app.py                    # Streamlit main entry (NEW)
│   ├── pages/
│   │   ├── predict.py            # Prediction page
│   │   ├── compare.py            # Model comparison page
│   │   └── results.py            # Benchmark results page
│   ├── components/
│   │   ├── sidebar.py            # Sidebar controls
│   │   └── visualizer.py         # Visualization components
│   └── assets/
│       ├── samples/              # Sample satellite images
│       └── style.css             # Custom styling
│
├── checkpoints/                  # Saved model weights (.pth)
├── runs/                         # TensorBoard logs
├── results/                      # Evaluation outputs
│   ├── confusion_matrices/
│   ├── prediction_samples/
│   └── comparison_table.csv
│
├── docs/                         # Project documentation
│   ├── 01_PRD.md
│   ├── 02_TRD.md
│   ├── 03_APP_FLOW.md
│   ├── 04_DESIGN.md
│   ├── 05_SCHEMA.md
│   ├── 06_IMPLEMENTATION_PLAN.md
│   └── 07_TODO.md
│
├── notebooks/
│   └── Train_Visualize.ipynb     # Updated Colab notebook
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup (optional)
├── README.md                     # Project README (REWRITTEN)
├── LICENSE                       # MIT License
└── .gitignore                    # Updated gitignore
```

---

## 4. Data Pipeline

### 4.1 Dataset Specifications

| Property | Value |
|----------|-------|
| Total Images | 650 (positive set) |
| Image Source | Google Earth (715 color) + Vaihingen (85 pansharpened) |
| Spatial Resolution | 0.08m to 2.0m |
| Image Format | JPEG |
| Annotation Format | COCO JSON (polygons + RLE) |
| Number of Classes | 10 (+ background) |

### 4.2 Data Split Strategy

| Split | Ratio | Images | Purpose |
|-------|-------|--------|---------|
| Train | 70% | ~455 | Model training |
| Validation | 15% | ~98 | Hyperparameter tuning, early stopping |
| Test | 15% | ~97 | Final evaluation (never seen during training) |

> **Change from original:** Original used 80/20 train/val with no test set. We add a proper held-out test set.

### 4.3 Data Augmentation Pipeline

```python
# Training transforms
train_transforms = Compose([
    RandomResizedCrop(size=(512, 512), scale=(0.5, 2.0)),
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=[0, 90, 180, 270]),
    ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    GaussianBlur(kernel_size=3, p=0.1),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor()
])

# Validation/Test transforms
val_transforms = Compose([
    Resize(size=(512, 512)),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor()
])
```

> **IMPORTANT:** Augmentations must be applied jointly to both image and mask.

---

## 5. Model Architecture Details

### 5.1 DeepLab V1 (Large FOV)

| Property | Value |
|----------|-------|
| Backbone | VGG-16 (pretrained on ImageNet) |
| Key Technique | Atrous (Dilated) Convolutions |
| Atrous Rates | [1, 2, 4, 12] |
| Output Stride | 8 |
| Classifier | 512 -> 1024 -> 1024 -> n_classes |
| Upsampling | Bilinear interpolation to input size |

### 5.2 DeepLab V2

| Property | Value |
|----------|-------|
| Backbone | ResNet-101 (pretrained) |
| Key Technique | Atrous Spatial Pyramid Pooling (ASPP) |
| ASPP Rates | [6, 12, 18, 24] |
| Output Stride | 8 |
| Upsampling | Bilinear interpolation to input size |

### 5.3 DeepLab V3

| Property | Value |
|----------|-------|
| Backbone | ResNet-101 (pretrained) |
| Key Technique | Improved ASPP + Image-level Pooling + BatchNorm |
| ASPP Rates | [6, 12, 18] |
| Multi-grid | [1, 2, 4] |
| Output Stride | 8 or 16 (configurable) |
| Post-ASPP | Concat -> 1x1 Conv -> n_classes |

### 5.4 DeepLab V3+

| Property | Value |
|----------|-------|
| Backbone | ResNet-101 (pretrained) |
| Key Technique | Encoder-Decoder + Atrous Separable Convolutions |
| ASPP Rates | [6, 12, 18] |
| Decoder | Low-level features (layer1) -> 48ch, concat with ASPP output |
| Separable Conv | Depthwise + Pointwise convolutions |
| Output Stride | 16 (default) |

---

## 6. Training Configuration

### 6.1 Default Hyperparameters

```yaml
# configs/default.yaml
model:
  name: deeplabv3plus        # deeplabv1, deeplabv2, deeplabv3, deeplabv3plus
  backbone: resnet101
  output_stride: 16
  num_classes: 10

training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.007
  weight_decay: 0.0005
  momentum: 0.9
  optimizer: sgd              # sgd, adam, adamw
  scheduler: poly             # poly, step, cosine
  scheduler_power: 0.9
  early_stopping_patience: 10
  use_amp: true

loss:
  type: bce_dice              # bce_dice, focal, cross_entropy
  bce_weight: 0.5
  focal_gamma: 2.0

data:
  image_size: 512
  train_split: train.json
  val_split: val.json
  test_split: test.json
  num_workers: 4
  augmentation: true

logging:
  tensorboard: true
  log_interval: 10            # Log every N batches
  save_predictions: true
  save_interval: 5            # Save sample predictions every N epochs
```

### 6.2 Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| BCE + Dice | `0.5 * BCE + 0.5 * Dice` | Default, good for balanced datasets |
| Focal Loss | `-alpha * (1-p)^gamma * log(p)` | Class imbalance (bridges: 124 instances) |
| Cross Entropy | `-sum(y * log(p))` | Standard multi-class segmentation |

### 6.3 Evaluation Metrics

| Metric | Description | Scope |
|--------|-------------|-------|
| mIoU | Mean Intersection over Union | Primary metric |
| Per-class IoU | IoU for each of 10 classes | Class-level analysis |
| Pixel Accuracy | Correct pixels / total pixels | Secondary metric |
| Confusion Matrix | N x N matrix of class predictions | Detailed error analysis |
| Dice Coefficient | 2*TP / (2*TP + FP + FN) | Additional segmentation metric |

---

## 7. Inference Pipeline

### 7.1 Single Image Prediction

```
Input Image (any size) 
    -> Resize to 512x512 
    -> Normalize (ImageNet stats) 
    -> Model Forward Pass 
    -> Sigmoid activation 
    -> Argmax across class channels 
    -> Resize back to original size 
    -> Color-coded mask overlay
```

### 7.2 Performance Targets

| Metric | Target |
|--------|--------|
| GPU Inference (single image) | < 500ms |
| CPU Inference (single image) | < 3 seconds |
| Batch Inference (8 images) | < 2 seconds (GPU) |
| ONNX Inference | < 300ms (GPU) |
| Model Size (ResNet-101) | ~250 MB (.pth) |
| ONNX Model Size | ~200 MB |

---

## 8. Demo Application Architecture

### 8.1 Streamlit App Structure

```
Streamlit App (app.py)
├── Sidebar
│   ├── Model Selector (Dropdown)
│   ├── Confidence Threshold (Slider)
│   ├── Class Toggle (Checkboxes)
│   └── Settings
│
├── Page: Predict (default)
│   ├── Image Upload / Sample Gallery
│   ├── Prediction Display (side-by-side)
│   ├── Class Legend
│   └── Download Button
│
├── Page: Compare
│   ├── Run All 4 Models
│   ├── Side-by-Side Grid (2x2)
│   └── Metrics Comparison Table
│
└── Page: Results Dashboard
    ├── Training Loss Curves
    ├── mIoU Progress
    ├── Confusion Matrix Heatmap
    ├── Per-Class IoU Bar Chart
    └── Model Comparison Table
```

### 8.2 App Dependencies

```
streamlit >= 1.28
torch >= 2.0
torchvision >= 0.15
Pillow >= 9.0
plotly >= 5.0
matplotlib >= 3.7
numpy >= 1.24
```

---

## 9. Environment & System Requirements

### 9.1 Development

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB | 8 GB+ |
| Disk Space | 5 GB | 10 GB |
| OS | Windows 10, Linux, macOS | Any |

### 9.2 Training (Google Colab)

| Requirement | Value |
|-------------|-------|
| Runtime | GPU (T4 or better) |
| RAM | 12 GB (Colab default) |
| Training Time (per model) | ~1-2 hours |

---

## 10. API Contracts

### 10.1 Model Interface

All models must implement:

```python
class DeepLabModel(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        ...
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Input tensor [B, 3, H, W], float32, normalized
        Returns:
            logits: Output tensor [B, num_classes, H, W], float32
        """
        ...
```

### 10.2 Checkpoint Format

```python
checkpoint = {
    'epoch': int,
    'model_name': str,           # e.g., 'deeplabv3plus'
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'scheduler_state_dict': dict,
    'best_miou': float,
    'config': dict,              # Full training config
    'class_names': list,
    'metrics_history': {
        'train_loss': list,
        'val_loss': list,
        'train_miou': list,
        'val_miou': list,
        'class_ious': dict
    }
}
```

### 10.3 Prediction Output Format

```python
prediction = {
    'mask': np.ndarray,          # [H, W] int, class IDs 0-10
    'class_masks': np.ndarray,   # [num_classes, H, W] float, probabilities
    'class_names': list,
    'confidence': np.ndarray,    # [H, W] float, max probability per pixel
    'inference_time_ms': float
}
```
