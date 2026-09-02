# 📋 Product Requirements Document (PRD)

## DeepLab Semantic Segmentation for VHR Remote Sensing Imagery

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Date** | August 16, 2026 |
| **Author** | Labha |
| **Status** | Draft |

---

## 1. Executive Summary

Build a production-quality **semantic segmentation pipeline** that trains, evaluates, and compares all four DeepLab architectures (V1, V2, V3, V3+) on the NWPU VHR-10 satellite imagery dataset. Deliver an interactive **Streamlit web demo** allowing users to upload satellite images and visualize pixel-level predictions in real-time.

---

## 2. Problem Statement

Remote sensing imagery analysis is critical for urban planning, disaster response, and environmental monitoring. Manual pixel-level annotation of satellite images is expensive and time-consuming. This project automates the identification and segmentation of 10 object categories in Very High Resolution (VHR) satellite images using state-of-the-art deep learning.

**Current pain points with the existing codebase:**
- Broken dependencies (TensorFlow in a PyTorch project, deprecated APIs)
- No CLI interface, no reproducibility
- No model saving, no inference pipeline
- No comparative analysis across architectures
- No way to demonstrate results interactively

---

## 3. Goals & Objectives

### Primary Goals
| # | Goal | Success Metric |
|---|------|---------------|
| G1 | Fix all critical bugs and modernize the codebase | Zero import errors, runs on Python 3.10+ |
| G2 | Train & benchmark all 4 DeepLab variants | Documented mIoU scores for each model |
| G3 | Build a complete inference pipeline | < 2 seconds per image prediction |
| G4 | Create interactive web demo | Functional Streamlit app with upload & predict |
| G5 | Generate comprehensive evaluation artifacts | Confusion matrix, per-class IoU, loss curves |

### Stretch Goals
| # | Goal | Success Metric |
|---|------|---------------|
| S1 | Mixed precision training (AMP) | Training speed improvement >= 1.5x |
| S2 | ONNX model export | Exported .onnx file passes validation |
| S3 | GradCAM visualizations | Heatmaps generated for sample predictions |

---

## 4. Target Users

| User | Use Case |
|------|----------|
| **Recruiters / Hiring Managers** | Review project quality, code structure, demo app |
| **Technical Interviewers** | Evaluate deep learning expertise, engineering skills |
| **Fellow Researchers** | Reproduce results, extend work |
| **The Developer (You)** | Portfolio showcase, learning, future reference |

---

## 5. Features

### 5.1 Core ML Pipeline

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F1 | CLI Training Interface | P0 | argparse-based CLI: `python train.py --model deeplabv3plus --epochs 50 --lr 0.001` |
| F2 | Model Checkpointing | P0 | Save/load .pth files with metadata (epoch, mIoU, hyperparams) |
| F3 | Data Augmentation | P0 | Random horizontal/vertical flip, rotation (90 deg), color jitter, random resize crop |
| F4 | Inference Script | P0 | `python predict.py --model best.pth --input image.jpg --output result.png` |
| F5 | Evaluation Script | P0 | Confusion matrix, per-class IoU table, precision/recall |
| F6 | TensorBoard Logging | P1 | Loss curves, mIoU plots, sample predictions per epoch |
| F7 | Mixed Precision Training | P1 | torch.cuda.amp for faster training with less memory |
| F8 | Focal Loss Option | P1 | Alternative loss function for class imbalance |
| F9 | ONNX Export | P1 | `python export.py --model best.pth --format onnx` |
| F10 | Multi-backbone Support | P2 | ResNet-50, MobileNetV2 options beyond ResNet-101 |

### 5.2 Demo Web Application (Streamlit)

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| D1 | Image Upload | P0 | Drag-and-drop satellite image upload |
| D2 | Model Selection | P0 | Dropdown to pick DeepLabV1/V2/V3/V3+ |
| D3 | Segmentation Visualization | P0 | Overlay predicted masks on input image with color legend |
| D4 | Side-by-Side Comparison | P0 | Input to Prediction to Ground Truth (if available) |
| D5 | Per-Class Toggle | P1 | Toggle visibility of individual class masks |
| D6 | Confidence Heatmap | P1 | Show prediction confidence per pixel |
| D7 | Model Comparison View | P1 | Run all 4 models and compare outputs side by side |
| D8 | Benchmark Results Page | P0 | Display training results table, loss curves, confusion matrix |
| D9 | Sample Gallery | P0 | Pre-loaded sample satellite images for quick demo |
| D10 | Download Results | P1 | Export predicted mask as PNG |

---

## 6. Scope

### In Scope
- All 4 DeepLab architectures (V1, V2, V3, V3+)
- NWPU VHR-10 dataset (10 classes, 650 images)
- Training, evaluation, inference pipeline
- Streamlit demo application
- Comparative analysis and visualization
- TensorBoard experiment tracking
- Model checkpointing and export

### Out of Scope
- Real-time video segmentation
- Training on multiple datasets
- Distributed/multi-GPU training
- Cloud deployment (AWS/GCP)
- Mobile deployment
- Custom dataset annotation tools

---

## 7. Success Criteria

| Criterion | Target |
|-----------|--------|
| All 4 models train without errors | Yes |
| Validation mIoU >= 40% (DeepLabV3+) | >= 40% |
| Inference time per image | < 2 seconds (GPU) |
| Demo app loads and predicts | < 5 seconds end-to-end |
| README has complete results table | All 4 models compared |
| Code passes linting (flake8) | No critical errors |
| All functions have docstrings | 100% coverage |

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Dataset images not available | Cannot train | Provide clear download instructions + automated setup script |
| GPU not available locally | Cannot train | Support Google Colab with updated notebook |
| Low mIoU scores | Weak resume impact | Tune hyperparameters, add augmentation, try different backbones |
| Streamlit deployment issues | Cannot demo | Provide Docker container + local run instructions |

---

## 9. Timeline (Estimated)

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Phase 0: Setup & Bug Fixes | 1 day | Fixed codebase, requirements.txt, working imports |
| Phase 1: Core Pipeline (Tier 1) | 2-3 days | CLI, checkpointing, augmentation, inference, evaluation |
| Phase 2: Advanced Features (Tier 2) | 2-3 days | TensorBoard, AMP, Focal Loss, ONNX, confusion matrix |
| Phase 3: Demo App | 2-3 days | Streamlit app with all features |
| Phase 4: Training & Results | 1-2 days | Train all models, collect results, update README |
| Phase 5: Polish | 1 day | Documentation, code cleanup, final README |
