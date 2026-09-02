# ✅ Project TODO List

## DeepLab Semantic Segmentation — Task Tracker

> **Last Updated:** August 16, 2026
> **Progress:** 0 / 75 tasks completed (0%)

---

## Phase 0: Project Setup & Bug Fixes (0/14)

### Directory Structure
- [ ] Create `configs/` directory
- [ ] Create `utils/` directory with `__init__.py`
- [ ] Create `scripts/` directory
- [ ] Create `app/` directory structure (`pages/`, `components/`, `assets/`)
- [ ] Create `checkpoints/`, `runs/`, `results/` directories
- [ ] Add `__init__.py` to `models/` with model registry

### Bug Fixes
- [ ] Remove TensorFlow import from `lib/utils.py`
- [ ] Replace `distutils.version.LooseVersion` with `packaging.version.Version`
- [ ] Fix `np.bool` deprecation in `dataset.py` and `show_coco.py`
- [ ] Update `pretrained=True` to `weights=` API in all 4 model files
- [ ] Fix `tqdm.notebook` import in `train.py` (use auto-detect)
- [ ] Fix `assp` typo to `aspp` in model files

### Dependencies & Config
- [ ] Create `requirements.txt` with pinned versions
- [ ] Create `configs/default.yaml` with default training config

---

## Phase 1: Core ML Pipeline — Tier 1 (0/19)

### Config System
- [ ] Create `utils/config.py` — YAML config loader with CLI overrides

### Data Pipeline
- [ ] Create `utils/transforms.py` — joint image+mask augmentation pipeline
- [ ] Refactor `utils/dataset.py` — fix mask binarization, support train/val/test
- [ ] Refactor `scripts/split_dataset.py` — 70/15/15 split with seed

### Loss & Metrics (Refactored)
- [ ] Create `utils/losses.py` — BCE+Dice loss (extracted from `loss_metrics.py`)
- [ ] Create `utils/metrics.py` — mIoU computation (extracted from `loss_metrics.py`)

### Model Updates
- [ ] Rename `deeplabv3+.py` to `deeplabv3plus.py`
- [ ] Create `models/__init__.py` with `get_model()` factory function
- [ ] Add docstrings to all model classes

### Training Script
- [ ] Create `scripts/train.py` with argparse CLI
- [ ] Add early stopping with configurable patience
- [ ] Add learning rate scheduler support (poly, step, cosine)

### Checkpointing
- [ ] Create `utils/checkpoint.py` — save/load with full metadata

### Inference
- [ ] Create `scripts/predict.py` — single image and batch prediction
- [ ] Support output formats: overlay PNG, raw mask, metadata JSON

### Evaluation
- [ ] Create `scripts/evaluate.py` — per-class IoU table, pixel accuracy
- [ ] Generate sample prediction grids (input | GT | prediction)

### Dataset Split
- [ ] Update split script to create 70/15/15 train/val/test with seed

---

## Phase 2: Advanced Features — Tier 2 (0/15)

### TensorBoard
- [ ] Add TensorBoard `SummaryWriter` to training loop
- [ ] Log scalars: loss, mIoU, per-class IoU, learning rate
- [ ] Log images: sample predictions every N epochs
- [ ] Log hyperparameters summary

### Mixed Precision Training
- [ ] Add `torch.cuda.amp` (GradScaler + autocast) to training loop
- [ ] Add `--use-amp` CLI flag
- [ ] Save/restore scaler state in checkpoints

### Focal Loss
- [ ] Implement Focal Loss in `utils/losses.py`
- [ ] Add `--loss focal` CLI option

### Confusion Matrix & Advanced Evaluation
- [ ] Generate NxN confusion matrix in `evaluate.py`
- [ ] Create interactive Plotly heatmap + matplotlib PNG
- [ ] Compute precision, recall, F1 per class

### Visualization
- [ ] Create `utils/visualization.py` with comparison grid functions
- [ ] Generate per-class IoU grouped bar chart (all models)

### ONNX Export
- [ ] Create `scripts/export.py` — export to ONNX
- [ ] Validate exported model with onnxruntime

---

## Phase 3: Demo Application (0/17)

### App Foundation
- [ ] Create `app/app.py` — multi-page Streamlit entry point
- [ ] Implement model loading with `@st.cache_resource`
- [ ] Set up session state management
- [ ] Create `app/assets/style.css` — custom dark theme styling

### Predict Page
- [ ] Create `app/pages/predict.py`
- [ ] Image upload with drag-and-drop
- [ ] Sample image gallery (6-8 pre-loaded images)
- [ ] Three-tab result display (Original, Mask, Overlay)
- [ ] Class legend with color-coded labels
- [ ] Download buttons for mask and overlay

### Compare Page
- [ ] Create `app/pages/compare.py`
- [ ] 2x2 model comparison grid
- [ ] Metrics comparison table

### Results Dashboard
- [ ] Create `app/pages/results.py`
- [ ] Training curves (loss + mIoU) with Plotly
- [ ] Confusion matrix heatmap
- [ ] Per-class IoU bar chart

### Components
- [ ] Create `app/components/sidebar.py` — navigation, settings, class toggles
- [ ] Create `app/components/visualizer.py` — mask rendering, metric cards

---

## Phase 4: Polish & Results (0/10)

### Training Runs
- [ ] Train DeepLabV1 (VGG-16 backbone)
- [ ] Train DeepLabV2 (ResNet-101 + ASPP)
- [ ] Train DeepLabV3 (ResNet-101 + improved ASPP)
- [ ] Train DeepLabV3+ (ResNet-101 + encoder-decoder)

### Evaluation & Results
- [ ] Run evaluation on test set for all 4 models
- [ ] Generate final comparison table CSV
- [ ] Generate final confusion matrices and charts

### Documentation
- [ ] Rewrite `README.md` with results, setup guide, screenshots
- [ ] Add docstrings to all remaining functions
- [ ] Final code cleanup — remove dead code, format with black

---

## Summary

| Phase | Tasks | Done | Status |
|-------|-------|------|--------|
| Phase 0: Setup & Bug Fixes | 14 | 0 | Not Started |
| Phase 1: Core ML Pipeline | 19 | 0 | Not Started |
| Phase 2: Advanced Features | 15 | 0 | Not Started |
| Phase 3: Demo Application | 17 | 0 | Not Started |
| Phase 4: Polish & Results | 10 | 0 | Not Started |
| **Total** | **75** | **0** | **0%** |
