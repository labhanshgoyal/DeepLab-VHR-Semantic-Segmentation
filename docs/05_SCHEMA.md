# 📐 Schema Document

## Data Formats, Configuration, & Storage Schemas

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Date** | August 16, 2026 |
| **Status** | Draft |

---

## 1. COCO Annotation Schema (Input Data)

The NWPU VHR-10 dataset uses standard COCO annotation format.

### 1.1 Top-Level Structure

```json
{
    "info": {
        "description": "NWPU VHR-10 Dataset",
        "version": "1.0",
        "year": 2019,
        "contributor": "NWPU",
        "date_created": "2019-01-01"
    },
    "licenses": [
        {
            "id": 1,
            "name": "Attribution License",
            "url": ""
        }
    ],
    "categories": [
        {"id": 1, "name": "airplane", "supercategory": "vehicle"},
        {"id": 2, "name": "ship", "supercategory": "vehicle"},
        {"id": 3, "name": "storage-tank", "supercategory": "building"},
        {"id": 4, "name": "baseball-diamond", "supercategory": "field"},
        {"id": 5, "name": "tennis-court", "supercategory": "field"},
        {"id": 6, "name": "basketball-court", "supercategory": "field"},
        {"id": 7, "name": "ground-track-field", "supercategory": "field"},
        {"id": 8, "name": "harbor", "supercategory": "waterfront"},
        {"id": 9, "name": "bridge", "supercategory": "infrastructure"},
        {"id": 10, "name": "vehicle", "supercategory": "vehicle"}
    ],
    "images": ["<ImageInfo>"],
    "annotations": ["<AnnotationInfo>"]
}
```

### 1.2 Image Info Schema

```json
{
    "id": 1,
    "file_name": "001.jpg",
    "width": 757,
    "height": 561,
    "date_captured": "",
    "license": 1,
    "coco_url": "",
    "flickr_url": ""
}
```

### 1.3 Annotation Info Schema

```json
{
    "id": 1,
    "image_id": 1,
    "category_id": 4,
    "segmentation": [[x1, y1, x2, y2, x3, y3, ...]],
    "area": 12345.67,
    "bbox": [x_min, y_min, width, height],
    "iscrowd": 0
}
```

---

## 2. Training Configuration Schema (YAML)

### 2.1 Full Config Schema

```yaml
# configs/default.yaml

# ===== Model Configuration =====
model:
  name: deeplabv3plus           # Options: deeplabv1, deeplabv2, deeplabv3, deeplabv3plus
  backbone: resnet101           # Options: resnet101, resnet50 (for V2/V3/V3+)
  num_classes: 10               # Number of segmentation classes (excluding background)
  output_stride: 16             # Options: 8, 16 (for V3/V3+)
  pretrained: true              # Use ImageNet pretrained backbone
  
  # V1-specific
  atrous_rates_v1: [1, 2, 4, 12]
  
  # V2-specific  
  n_blocks: [3, 4, 23, 3]       # ResNet-101 block configuration
  atrous_rates_v2: [6, 12, 18, 24]
  
  # V3/V3+-specific
  atrous_rates_v3: [6, 12, 18]
  multi_grids: [1, 2, 4]
  atr_sep_conv: true             # V3+ only: use atrous separable convolutions

# ===== Training Configuration =====
training:
  epochs: 100
  batch_size: 8
  learning_rate: 0.007
  weight_decay: 0.0005
  momentum: 0.9
  optimizer: sgd                 # Options: sgd, adam, adamw
  
  # Learning Rate Scheduler
  scheduler: poly                # Options: poly, step, cosine, reduce_on_plateau
  scheduler_power: 0.9           # For poly scheduler
  step_size: 30                  # For step scheduler
  step_gamma: 0.1                # For step scheduler
  
  # Early Stopping
  early_stopping: true
  patience: 10                   # Stop after N epochs without improvement
  min_delta: 0.001               # Minimum improvement threshold
  
  # Mixed Precision
  use_amp: true                  # Automatic Mixed Precision training
  
  # Gradient
  gradient_clip: 1.0             # Max gradient norm (0 to disable)

# ===== Loss Configuration =====
loss:
  type: bce_dice                 # Options: bce_dice, focal, cross_entropy
  bce_weight: 0.5                # Weight for BCE in combined loss
  dice_smooth: 1.0               # Smoothing for Dice loss
  focal_alpha: 0.25              # Focal loss alpha
  focal_gamma: 2.0               # Focal loss gamma

# ===== Data Configuration =====
data:
  dataset_dir: "data/NWPU_VHR-10"
  images_dir: "positive_image_set"
  train_annotations: "train.json"
  val_annotations: "val.json"
  test_annotations: "test.json"
  image_size: 512                # Resize images to this size
  num_workers: 4                 # DataLoader workers
  pin_memory: true
  
  # Augmentation
  augmentation:
    enabled: true
    horizontal_flip: 0.5
    vertical_flip: 0.5
    rotation_degrees: [0, 90, 180, 270]
    color_jitter:
      brightness: 0.3
      contrast: 0.3
      saturation: 0.2
      hue: 0.1
    random_scale: [0.5, 2.0]
    gaussian_blur: 0.1

# ===== Logging Configuration =====
logging:
  tensorboard: true
  log_dir: "runs/"
  log_interval: 10               # Log every N batches
  save_predictions: true         # Save sample predictions during training
  prediction_interval: 5         # Save predictions every N epochs
  num_prediction_samples: 4      # Number of samples to visualize

# ===== Checkpoint Configuration =====
checkpoint:
  save_dir: "checkpoints/"
  save_best: true                # Save best model by validation mIoU
  save_last: true                # Always save latest model
  save_interval: 10              # Save checkpoint every N epochs
  resume: null                   # Path to checkpoint to resume from

# ===== Evaluation Configuration =====
evaluation:
  output_dir: "results/"
  generate_confusion_matrix: true
  generate_per_class_iou: true
  generate_predictions: true
  num_visualization_samples: 10
  iou_threshold: 0.5
```

---

## 3. Model Checkpoint Schema (.pth)

### 3.1 Checkpoint Contents

```python
checkpoint = {
    # === Core Model State ===
    "model_state_dict": OrderedDict,      # Model weights
    "optimizer_state_dict": OrderedDict,   # Optimizer state
    "scheduler_state_dict": OrderedDict,   # LR scheduler state (if used)
    "scaler_state_dict": OrderedDict,      # AMP scaler state (if use_amp=True)
    
    # === Training Metadata ===
    "epoch": int,                          # Current epoch number
    "global_step": int,                    # Total training steps
    "best_miou": float,                    # Best validation mIoU achieved
    
    # === Configuration ===
    "config": dict,                        # Full training config (YAML as dict)
    "model_name": str,                     # e.g., "deeplabv3plus"
    "backbone": str,                       # e.g., "resnet101"
    "num_classes": int,                    # e.g., 10
    
    # === Dataset Info ===
    "class_names": [                       # Ordered list of class names
        "airplane", "ship", "storage-tank", "baseball-diamond",
        "tennis-court", "basketball-court", "ground-track-field",
        "harbor", "bridge", "vehicle"
    ],
    "train_size": int,                     # Number of training images
    "val_size": int,                       # Number of validation images
    
    # === Metrics History ===
    "metrics_history": {
        "train_loss": [float],             # Per-epoch training loss
        "val_loss": [float],               # Per-epoch validation loss
        "train_miou": [float],             # Per-epoch training mIoU
        "val_miou": [float],               # Per-epoch validation mIoU
        "learning_rates": [float],         # LR at each epoch
        "class_ious": {                    # Per-class IoU history
            "train": {
                "airplane": [float],
                "ship": [float],
                # ... all 10 classes
            },
            "val": {
                "airplane": [float],
                "ship": [float],
                # ... all 10 classes
            }
        }
    },
    
    # === System Info ===
    "pytorch_version": str,                # e.g., "2.0.1"
    "torchvision_version": str,
    "cuda_version": str,                   # or "cpu"
    "timestamp": str                       # ISO 8601 format
}
```

### 3.2 Checkpoint File Naming Convention

```
checkpoints/
├── deeplabv1_best.pth            # Best model by val mIoU
├── deeplabv1_last.pth            # Latest model
├── deeplabv1_epoch_050.pth       # Periodic save
├── deeplabv2_best.pth
├── deeplabv3_best.pth
├── deeplabv3plus_best.pth
└── deeplabv3plus_last.pth
```

---

## 4. Evaluation Results Schema

### 4.1 Comparison Table (CSV)

```csv
# results/comparison_table.csv
model,backbone,output_stride,mIoU,pixel_accuracy,params_M,flops_G,inference_ms,airplane,ship,storage_tank,baseball_diamond,tennis_court,basketball_court,ground_track_field,harbor,bridge,vehicle
deeplabv1,vgg16,8,35.2,78.5,12.5,18.3,180,42.1,28.3,45.6,38.9,52.3,22.1,35.8,30.2,18.4,38.5
deeplabv2,resnet101,8,42.1,83.2,44.8,62.1,320,50.3,35.6,52.8,45.2,58.7,30.5,42.3,38.9,25.1,42.0
deeplabv3,resnet101,8,45.8,85.1,58.6,85.4,350,54.2,40.1,55.3,48.7,62.1,35.8,46.5,42.3,28.7,44.6
deeplabv3plus,resnet101,16,48.3,87.3,54.7,78.2,380,57.8,43.5,58.1,51.2,65.3,38.2,49.1,45.6,31.2,47.3
```

### 4.2 Confusion Matrix (JSON)

```json
{
    "model": "deeplabv3plus",
    "num_classes": 11,
    "class_names": ["background", "airplane", "ship", "..."],
    "matrix": [
        [98234, 12, 5, "..."],
        [45, 892, 3, "..."],
        ["..."]
    ],
    "normalized_matrix": [
        [0.95, 0.001, 0.0005, "..."],
        ["..."]
    ]
}
```

### 4.3 Training History (JSON)

```json
{
    "model": "deeplabv3plus",
    "total_epochs": 100,
    "best_epoch": 87,
    "best_val_miou": 0.483,
    "total_training_time_hours": 2.5,
    "history": {
        "epochs": [1, 2, 3, "..."],
        "train_loss": [0.85, 0.72, 0.65, "..."],
        "val_loss": [0.90, 0.78, 0.70, "..."],
        "train_miou": [0.12, 0.18, 0.23, "..."],
        "val_miou": [0.10, 0.15, 0.20, "..."],
        "learning_rate": [0.007, 0.007, 0.006, "..."]
    }
}
```

---

## 5. Prediction Output Schema

### 5.1 Single Prediction

```python
prediction = {
    "image_path": str,                    # Input image path
    "image_size": (int, int),             # Original (H, W)
    "model_name": str,                    # Model used
    "inference_time_ms": float,           # Time in milliseconds
    
    "segmentation_mask": np.ndarray,      # Shape: [H, W], dtype: int
                                          # Values: 0 (background) to 10 (vehicle)
    
    "class_probabilities": np.ndarray,    # Shape: [10, H, W], dtype: float32
                                          # Sigmoid probabilities per class
    
    "confidence_map": np.ndarray,         # Shape: [H, W], dtype: float32
                                          # Max probability at each pixel
    
    "detected_classes": [                 # Classes with non-zero mask area
        {
            "class_id": int,
            "class_name": str,
            "pixel_count": int,
            "area_percentage": float,
            "mean_confidence": float
        }
    ]
}
```

### 5.2 Batch Prediction Output

```
results/predictions/
├── image_001/
│   ├── overlay.png              # Image with mask overlay
│   ├── mask.png                 # Color-coded segmentation mask
│   ├── raw_mask.npy             # Raw numpy mask [H, W]
│   └── metadata.json            # Prediction metadata
├── image_002/
│   └── ...
└── summary.csv                  # Batch summary
```

---

## 6. TensorBoard Log Schema

### 6.1 Scalar Logs

```
runs/<model_name>_<timestamp>/
├── events.out.tfevents.*
│
│   Scalars logged:
│   ├── Loss/train                    # Training loss per epoch
│   ├── Loss/val                      # Validation loss per epoch
│   ├── Loss/train_bce                # BCE component
│   ├── Loss/train_dice               # Dice component
│   ├── mIoU/train                    # Training mIoU per epoch
│   ├── mIoU/val                      # Validation mIoU per epoch
│   ├── IoU/airplane                  # Per-class IoU (val)
│   ├── IoU/ship
│   ├── IoU/...                       # (all 10 classes)
│   ├── LearningRate                  # Current learning rate
│   └── Epoch_Time_sec                # Time per epoch
│
│   Images logged:
│   ├── Predictions/sample_0          # Sample prediction overlays
│   ├── Predictions/sample_1
│   ├── Predictions/sample_2
│   └── Predictions/sample_3
```

---

## 7. Streamlit Session State Schema

```python
st.session_state = {
    # === Image State ===
    "uploaded_image": PIL.Image | None,        # User-uploaded image
    "selected_sample": str | None,             # Path to selected sample image
    "current_image": np.ndarray | None,        # Active image (uploaded or sample)
    
    # === Prediction State ===
    "prediction": dict | None,                 # Latest prediction result
    "comparison_results": dict | None,         # Comparison page results
    
    # === Model State ===
    "loaded_models": dict,                     # Cached loaded models
    "selected_model": str,                     # Currently selected model name
    "device": str,                             # "cuda" or "cpu"
    
    # === UI State ===
    "active_classes": list[bool],              # 10 toggles for class visibility
    "confidence_threshold": float,             # 0.1 to 0.9
    "overlay_opacity": float,                  # 0.0 to 1.0
    "current_page": str,                       # "predict", "compare", "results"
    
    # === Results State ===
    "training_history": dict | None,           # Loaded from checkpoint
    "comparison_table": pd.DataFrame | None    # Loaded from CSV
}
```

---

## 8. ONNX Export Schema

```python
# ONNX model specification
onnx_model = {
    "opset_version": 11,
    "input": {
        "name": "input",
        "shape": [1, 3, 512, 512],       # Batch, Channels, Height, Width
        "dtype": "float32"
    },
    "output": {
        "name": "output",
        "shape": [1, 10, 512, 512],       # Batch, Classes, Height, Width
        "dtype": "float32"                 # Logits (pre-sigmoid)
    },
    "dynamic_axes": {
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"}
    },
    "metadata": {
        "model_name": str,
        "num_classes": 10,
        "class_names": list[str],
        "input_normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
}
```
