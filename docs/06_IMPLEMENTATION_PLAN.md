# 🗺 Implementation Plan

## Phased Execution Plan

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Date** | August 16, 2026 |
| **Status** | Draft |

---

## Overview

The implementation is divided into **5 phases**, ordered by dependency. Each phase builds on the previous one.

```
Phase 0 ──> Phase 1 ──> Phase 2 ──> Phase 3 ──> Phase 4
 Setup      Core ML     Advanced    Demo App    Polish
 & Fixes    Pipeline    Features                & Results
```

---

## Phase 0: Project Setup & Bug Fixes

**Goal:** Get the existing codebase running without errors on Python 3.10+

### 0.1 Project Restructuring

| Task | Details |
|------|---------|
| Create new directory structure | `configs/`, `utils/`, `scripts/`, `app/`, `checkpoints/`, `runs/`, `results/` |
| Move `dataset.py` to `utils/dataset.py` | Update imports |
| Move `loss_metrics.py` to `utils/losses.py` + `utils/metrics.py` | Split into separate files |
| Rename `deeplabv3+.py` to `deeplabv3plus.py` | Python-friendly filename |
| Add `__init__.py` to `models/` and `utils/` | Make them proper packages |
| Create `models/__init__.py` with model registry | Factory pattern: `get_model(name, **config)` |

### 0.2 Bug Fixes

| File | Fix |
|------|-----|
| `lib/utils.py` | Remove `import tensorflow as tf` (line 16) and TF-dependent functions |
| `lib/utils.py` | Replace `from distutils.version import LooseVersion` with `from packaging.version import Version` |
| `dataset.py` | Replace `np.bool` with `bool` (line 205) |
| `lib/show_coco.py` | Replace `np.bool` with `bool` (line 155) |
| All model files | Replace `pretrained=True` with `weights=ResNet101_Weights.DEFAULT` |
| `train.py` | Replace `from tqdm.notebook import tqdm` with `from tqdm import tqdm` (auto-detect) |
| `deeplabv2.py`, `deeplabv3.py`, `deeplabv3plus.py` | Fix typo: `self.assp` to `self.aspp` |

### 0.3 Dependencies

| Task | Details |
|------|---------|
| Create `requirements.txt` | Pin all dependency versions |
| Create `setup.py` or `pyproject.toml` | Optional: installable package |
| Update `.gitignore` | Add `checkpoints/`, `runs/`, `results/`, `data/`, `*.pth` |

### 0.4 Configuration System

| Task | Details |
|------|---------|
| Create `configs/default.yaml` | Default training configuration |
| Create `utils/config.py` | YAML config loader with CLI override support |

**Deliverables:** Working imports, no crashes, `requirements.txt`, clean directory structure.

---

## Phase 1: Core ML Pipeline (Tier 1)

**Goal:** Complete training, inference, and evaluation pipeline with CLI

### 1.1 CLI Training Interface

#### [NEW] `scripts/train.py`

```
python scripts/train.py \
    --model deeplabv3plus \
    --config configs/default.yaml \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.007 \
    --device cuda \
    --resume checkpoints/deeplabv3plus_last.pth
```

Key arguments:
- `--model`: Model name (deeplabv1, deeplabv2, deeplabv3, deeplabv3plus)
- `--config`: Path to YAML config file
- `--epochs`, `--batch-size`, `--lr`: Override config values
- `--device`: cuda/cpu
- `--resume`: Resume from checkpoint
- `--experiment-name`: TensorBoard experiment name

### 1.2 Model Checkpointing

#### [NEW] `utils/checkpoint.py`

Functions:
- `save_checkpoint(model, optimizer, scheduler, epoch, metrics, config, path)`
- `load_checkpoint(path, model, optimizer=None, scheduler=None)`
- `load_model_for_inference(path, device)` — lightweight, model-only load

### 1.3 Data Augmentation

#### [NEW] `utils/transforms.py`

- Joint image+mask augmentation (both must be transformed identically)
- Training: RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, ColorJitter, GaussianBlur, Normalize
- Validation: Resize, Normalize
- Use `torchvision.transforms.v2` or custom implementation

### 1.4 Dataset Refactoring

#### [MODIFY] `utils/dataset.py`

- Fix mask binarization bug (`mask[mask!=0] = 1` removes multi-class info)
- Add proper multi-class semantic mask loading
- Support train/val/test splits
- Add `num_workers` and `pin_memory` to DataLoader
- Remove commented-out dead code

### 1.5 Inference Script

#### [NEW] `scripts/predict.py`

```
# Single image
python scripts/predict.py \
    --checkpoint checkpoints/deeplabv3plus_best.pth \
    --input path/to/image.jpg \
    --output results/prediction.png

# Batch
python scripts/predict.py \
    --checkpoint checkpoints/deeplabv3plus_best.pth \
    --input-dir path/to/images/ \
    --output-dir results/predictions/
```

### 1.6 Evaluation Script

#### [NEW] `scripts/evaluate.py`

```
python scripts/evaluate.py \
    --checkpoint checkpoints/deeplabv3plus_best.pth \
    --split test \
    --output-dir results/deeplabv3plus/
```

Outputs:
- Per-class IoU table (printed + saved as CSV)
- Overall mIoU, pixel accuracy
- Sample prediction visualizations (input | GT | prediction)

### 1.7 Train/Val/Test Split

#### [MODIFY] `scripts/split_dataset.py`

- Change from 80/20 to 70/15/15 split
- Add `--seed` argument for reproducibility
- Generate `train.json`, `val.json`, `test.json`

**Deliverables:** Working CLI training, model saving/loading, inference on new images, evaluation with metrics.

---

## Phase 2: Advanced Features (Tier 2)

**Goal:** Add TensorBoard, AMP, Focal Loss, confusion matrix, ONNX export

### 2.1 TensorBoard Integration

#### [MODIFY] `scripts/train.py`

- Add `SummaryWriter` for TensorBoard
- Log scalars: loss (train/val), mIoU (train/val), LR, per-class IoU
- Log images: sample predictions every N epochs
- Log hparams: hyperparameter summary

### 2.2 Mixed Precision Training (AMP)

#### [MODIFY] `scripts/train.py`

- Add `torch.cuda.amp.GradScaler` and `autocast`
- Configurable via `--use-amp` flag
- Save scaler state in checkpoints

### 2.3 Focal Loss

#### [NEW] `utils/losses.py`

- Implement Focal Loss: `-alpha * (1-p)^gamma * log(p)`
- Configurable alpha and gamma
- Selectable via `--loss focal` CLI argument

### 2.4 Confusion Matrix & Advanced Metrics

#### [MODIFY] `scripts/evaluate.py`

- Generate NxN confusion matrix (11 classes including background)
- Save as interactive Plotly heatmap (HTML) and matplotlib PNG
- Compute precision, recall, F1 per class
- Generate per-class IoU grouped bar chart

### 2.5 Prediction Visualization Grid

#### [NEW] `utils/visualization.py` (refactored)

- `generate_comparison_grid(image, gt_mask, pred_mask, class_colors)` — 3-panel visualization
- `generate_model_comparison(image, predictions_dict)` — 4-model comparison grid
- `plot_training_curves(history)` — loss and mIoU plots
- `plot_confusion_matrix(matrix, class_names)` — heatmap

### 2.6 ONNX Export

#### [NEW] `scripts/export.py`

```
python scripts/export.py \
    --checkpoint checkpoints/deeplabv3plus_best.pth \
    --format onnx \
    --output exports/deeplabv3plus.onnx
```

- Export to ONNX with dynamic batch size
- Validate exported model with onnxruntime
- Benchmark inference speed comparison (PyTorch vs ONNX)

**Deliverables:** TensorBoard dashboards, faster training (AMP), confusion matrices, ONNX models.

---

## Phase 3: Demo Application

**Goal:** Build a Streamlit web app with 3 pages

### 3.1 App Foundation

#### [NEW] `app/app.py`

- Multi-page Streamlit app
- Page navigation in sidebar
- Model loading and caching (`@st.cache_resource`)
- Custom CSS injection
- Session state initialization

### 3.2 Predict Page

#### [NEW] `app/pages/predict.py`

- Image upload with drag-and-drop
- Sample image gallery (6-8 pre-loaded images)
- Model selection dropdown
- Confidence threshold slider
- Three-tab result display (Original, Mask, Overlay)
- Class legend with colors
- Detection summary
- Download buttons (mask PNG, overlay PNG)

### 3.3 Compare Page

#### [NEW] `app/pages/compare.py`

- Same image input as Predict page
- "Compare All Models" button
- 2x2 prediction grid
- Metrics comparison table
- Best model highlighting

### 3.4 Results Dashboard

#### [NEW] `app/pages/results.py`

- Load training history from checkpoints or saved JSON
- Model comparison table
- Interactive Plotly charts:
  - Training/validation loss curves
  - mIoU progress curves
  - Confusion matrix heatmap
  - Per-class IoU grouped bar chart
- Sample predictions gallery

### 3.5 Sidebar & Components

#### [NEW] `app/components/sidebar.py`

- App branding (logo, title)
- Navigation radio buttons
- Model selector
- Settings controls (confidence, opacity, device)
- Class visibility toggles
- About section (GitHub, paper links)

#### [NEW] `app/components/visualizer.py`

- `render_segmentation(image, mask, class_colors, opacity)`
- `render_class_legend(class_names, class_colors)`
- `render_metric_card(value, label, color)`

### 3.6 Styling

#### [NEW] `app/assets/style.css`

- Dark theme customization
- Metric card styles
- Image gallery hover effects
- Responsive grid layout

### 3.7 Sample Images

- Copy 6-8 representative satellite images to `app/assets/samples/`
- Mix of different object types for demo variety

**Deliverables:** Working Streamlit app with prediction, comparison, and dashboard pages.

---

## Phase 4: Polish & Results

**Goal:** Train all models, collect results, write README

### 4.1 Training Runs

| Model | Config | Expected Time |
|-------|--------|---------------|
| DeepLabV1 | VGG-16, rates=[1,2,4,12] | ~1 hour |
| DeepLabV2 | ResNet-101, ASPP=[6,12,18,24] | ~1.5 hours |
| DeepLabV3 | ResNet-101, ASPP=[6,12,18], OS=8 | ~2 hours |
| DeepLabV3+ | ResNet-101, ASPP=[6,12,18], OS=16 | ~2 hours |

### 4.2 Evaluation

- Run `evaluate.py` on test set for all 4 models
- Generate confusion matrices
- Generate comparison table CSV
- Save sample prediction grids

### 4.3 README Rewrite

#### [MODIFY] `README.md`

Sections:
1. Project Title + Badges (Python, PyTorch, License)
2. Project Overview (2-3 sentences)
3. Demo Screenshot / GIF
4. Results Table (all 4 models)
5. Architecture Diagram
6. Installation Instructions
7. Quick Start (train, predict, demo)
8. Project Structure
9. Dataset Setup
10. Training Guide
11. Evaluation
12. Demo App Usage
13. Technical Details
14. Citations
15. License

### 4.4 Final Code Cleanup

- Add docstrings to all functions
- Remove all dead/commented code
- Run flake8/black formatting
- Verify all scripts work end-to-end

**Deliverables:** Trained models, results table, polished README, clean code.

---

## Dependency Graph

```
Phase 0 (Setup)
    ├── 0.1 Directory Structure
    ├── 0.2 Bug Fixes ──────────────────────┐
    ├── 0.3 Dependencies                    │
    └── 0.4 Config System ─────────────┐    │
                                       │    │
Phase 1 (Core Pipeline)               │    │
    ├── 1.1 CLI Training ◄─────────────┘    │
    ├── 1.2 Checkpointing ◄────────────────┘
    ├── 1.3 Data Augmentation
    ├── 1.4 Dataset Refactoring
    ├── 1.5 Inference Script (depends on 1.2)
    ├── 1.6 Evaluation Script (depends on 1.2, 1.5)
    └── 1.7 Dataset Split
                │
Phase 2 (Advanced)
    ├── 2.1 TensorBoard (depends on 1.1)
    ├── 2.2 AMP (depends on 1.1)
    ├── 2.3 Focal Loss (depends on 1.1)
    ├── 2.4 Confusion Matrix (depends on 1.6)
    ├── 2.5 Visualization (depends on 1.5)
    └── 2.6 ONNX Export (depends on 1.2)
                │
Phase 3 (Demo App)
    ├── 3.1 App Foundation (depends on 1.2)
    ├── 3.2 Predict Page (depends on 1.5)
    ├── 3.3 Compare Page (depends on 3.2)
    ├── 3.4 Results Dashboard (depends on 2.4, 2.5)
    ├── 3.5 Sidebar
    ├── 3.6 Styling
    └── 3.7 Sample Images
                │
Phase 4 (Polish)
    ├── 4.1 Training Runs (depends on Phase 1 + 2)
    ├── 4.2 Evaluation (depends on 4.1)
    ├── 4.3 README (depends on 4.2)
    └── 4.4 Code Cleanup
```

---

## File Change Summary

### New Files (22)

| File | Phase |
|------|-------|
| `configs/default.yaml` | 0 |
| `requirements.txt` | 0 |
| `models/__init__.py` | 0 |
| `utils/__init__.py` | 0 |
| `utils/config.py` | 0 |
| `utils/transforms.py` | 1 |
| `utils/checkpoint.py` | 1 |
| `utils/losses.py` | 1 |
| `utils/metrics.py` | 1 |
| `utils/visualization.py` | 2 |
| `scripts/train.py` | 1 |
| `scripts/predict.py` | 1 |
| `scripts/evaluate.py` | 1 |
| `scripts/split_dataset.py` | 1 |
| `scripts/export.py` | 2 |
| `app/app.py` | 3 |
| `app/pages/predict.py` | 3 |
| `app/pages/compare.py` | 3 |
| `app/pages/results.py` | 3 |
| `app/components/sidebar.py` | 3 |
| `app/components/visualizer.py` | 3 |
| `app/assets/style.css` | 3 |

### Modified Files (6)

| File | Phase | Changes |
|------|-------|---------|
| `models/deeplabv1.py` | 0 | Fix deprecated API |
| `models/deeplabv2.py` | 0 | Fix deprecated API, typo |
| `models/deeplabv3.py` | 0 | Fix deprecated API, typo |
| `models/deeplabv3plus.py` (renamed) | 0 | Fix deprecated API, typo |
| `README.md` | 4 | Complete rewrite |
| `.gitignore` | 0 | Add new directories |

### Deprecated Files (kept for reference)

| File | Reason |
|------|--------|
| `train.py` (root) | Replaced by `scripts/train.py` |
| `dataset.py` (root) | Replaced by `utils/dataset.py` |
| `loss_metrics.py` (root) | Split into `utils/losses.py` + `utils/metrics.py` |
| `lib/` directory | Functionality moved to `utils/` |
