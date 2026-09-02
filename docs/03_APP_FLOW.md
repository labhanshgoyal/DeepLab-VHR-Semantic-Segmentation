# 🔄 Web / App Flow Document

## Streamlit Demo Application — User Flow

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Date** | August 16, 2026 |
| **Status** | Draft |

---

## 1. Application Overview

The Streamlit demo app provides **three main pages**:
1. **Predict** — Upload or select an image, run segmentation, view results
2. **Compare** — Compare predictions from all 4 DeepLab models
3. **Results Dashboard** — View training metrics, confusion matrices, benchmarks

---

## 2. Overall App Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         APP ENTRY                                   │
│                    streamlit run app.py                              │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LANDING / PREDICT PAGE                            │
│                                                                     │
│  ┌─────────────┐  ┌────────────────────────────────────────────┐    │
│  │  SIDEBAR     │  │  MAIN CONTENT                              │    │
│  │             │  │                                            │    │
│  │ App Title   │  │  ┌─────────────────────────────────────┐   │    │
│  │ Page Nav    │  │  │  Image Input Section                 │   │    │
│  │             │  │  │                                     │   │    │
│  │ Model:      │  │  │  [Upload Image] or [Select Sample]  │   │    │
│  │ [Dropdown]  │  │  │                                     │   │    │
│  │             │  │  └─────────────┬───────────────────────┘   │    │
│  │ Confidence: │  │                │                           │    │
│  │ [===|====]  │  │                ▼                           │    │
│  │             │  │  ┌─────────────────────────────────────┐   │    │
│  │ Classes:    │  │  │  [Run Segmentation] Button           │   │    │
│  │ [x] Airplane│  │  └─────────────┬───────────────────────┘   │    │
│  │ [x] Ship   │  │                │                           │    │
│  │ [x] Tank   │  │                ▼                           │    │
│  │ [x] ...    │  │  ┌─────────────────────────────────────┐   │    │
│  │             │  │  │  Results Display                     │   │    │
│  │ About       │  │  │  [Original] [Predicted] [Overlay]    │   │    │
│  │             │  │  │                                     │   │    │
│  │             │  │  │  + Class Legend + Download Button    │   │    │
│  └─────────────┘  │  └─────────────────────────────────────┘   │    │
│                   └────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Page-by-Page Flow

### 3.1 Predict Page (Default)

```
User Opens App
    │
    ├── Sees landing page with project title and description
    │
    ├── OPTION A: Upload Image
    │   └── Drag & drop or browse for satellite image (.jpg, .png, .tif)
    │
    ├── OPTION B: Select from Gallery
    │   └── Click one of 6-8 pre-loaded sample satellite images
    │
    ├── Select Model from sidebar dropdown
    │   └── DeepLabV1 / V2 / V3 / V3+ (default: V3+)
    │
    ├── Adjust confidence threshold (slider: 0.1 to 0.9)
    │
    ├── Click "Run Segmentation"
    │   │
    │   ├── Loading spinner shown
    │   ├── Model runs inference (< 2 sec GPU, < 5 sec CPU)
    │   │
    │   └── Results appear:
    │       │
    │       ├── Three-tab view:
    │       │   ├── Tab 1: Original Image
    │       │   ├── Tab 2: Segmentation Mask (color-coded)
    │       │   └── Tab 3: Overlay (mask on image, adjustable opacity)
    │       │
    │       ├── Class Legend (color-coded list with detected classes)
    │       │
    │       ├── Detection Summary:
    │       │   └── "Found: 3 Airplanes, 2 Storage Tanks, 1 Bridge"
    │       │
    │       ├── Inference Time: "245ms (GPU)"
    │       │
    │       └── Download Buttons:
    │           ├── Download Mask (PNG)
    │           └── Download Overlay (PNG)
    │
    └── User can toggle individual classes on/off in sidebar
        └── Mask updates in real-time
```

### 3.2 Compare Page

```
User navigates to Compare page
    │
    ├── Same image input (upload or sample selection)
    │
    ├── Click "Compare All Models"
    │   │
    │   ├── Progress bar: "Running DeepLabV1... V2... V3... V3+"
    │   │
    │   └── 2x2 Grid Display:
    │       ┌─────────────────┬─────────────────┐
    │       │   DeepLabV1     │   DeepLabV2      │
    │       │   mIoU: 35.2%   │   mIoU: 42.1%   │
    │       │   Time: 180ms   │   Time: 320ms    │
    │       │   [prediction]  │   [prediction]   │
    │       ├─────────────────┼─────────────────┤
    │       │   DeepLabV3     │   DeepLabV3+     │
    │       │   mIoU: 45.8%   │   mIoU: 48.3%   │
    │       │   Time: 350ms   │   Time: 380ms    │
    │       │   [prediction]  │   [prediction]   │
    │       └─────────────────┴─────────────────┘
    │
    └── Below Grid: Comparison Metrics Table
        │
        ├── Model | mIoU | Inference Time | Params | FLOPs
        └── Highlighted best values
```

### 3.3 Results Dashboard Page

```
User navigates to Results Dashboard
    │
    ├── Section 1: Model Comparison Table
    │   └── All 4 models: mIoU, per-class IoU, training time, params
    │
    ├── Section 2: Training Curves
    │   ├── Loss Curves (train vs val) for selected model
    │   └── mIoU Progress (train vs val) for selected model
    │
    ├── Section 3: Confusion Matrix
    │   └── Interactive heatmap (Plotly) for selected model
    │
    ├── Section 4: Per-Class IoU Bar Chart
    │   └── Grouped bar chart comparing all 4 models across 10 classes
    │
    └── Section 5: Sample Predictions Gallery
        └── Grid of 4-6 examples: Input | GT | V1 | V2 | V3 | V3+
```

---

## 4. Sidebar Controls (Persistent across pages)

```
┌─────────────────────┐
│  🛰 DeepLab Demo    │
│  VHR Segmentation   │
│                     │
│ ─── Navigation ───  │
│ ( ) Predict         │
│ ( ) Compare         │
│ ( ) Results         │
│                     │
│ ─── Settings ─────  │
│ Model: [V3+    ▼]   │
│ Conf:  [===|====]   │
│ Opacity:[====|==]   │
│ Device: [GPU ▼]     │
│                     │
│ ─── Classes ──────  │
│ [x] Airplane        │
│ [x] Ship            │
│ [x] Storage Tank    │
│ [x] Baseball Dia.   │
│ [x] Tennis Court    │
│ [x] Basketball Ct.  │
│ [x] Ground Track    │
│ [x] Harbor          │
│ [x] Bridge          │
│ [x] Vehicle         │
│                     │
│ ─── About ────────  │
│ GitHub | Paper      │
│ Made by Labha       │
└─────────────────────┘
```

---

## 5. User Interaction Flow Diagram

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │  Select  │
                    │   Page   │
                    └────┬─────┘
                         │
            ┌────────────┼────────────┐
            │            │            │
       ┌────▼────┐  ┌────▼────┐  ┌───▼─────┐
       │ Predict │  │ Compare │  │ Results │
       └────┬────┘  └────┬────┘  └────┬────┘
            │            │            │
       ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
       │ Upload  │  │ Upload  │  │ Select  │
       │ or Pick │  │ or Pick │  │ Model   │
       │ Sample  │  │ Sample  │  └────┬────┘
       └────┬────┘  └────┬────┘       │
            │            │       ┌────▼─────────┐
       ┌────▼────┐  ┌────▼────┐  │ View Metrics │
       │ Select  │  │ Compare │  │ Loss Curves  │
       │ Model   │  │ All 4   │  │ Confusion Mx │
       └────┬────┘  │ Models  │  │ IoU Charts   │
            │       └────┬────┘  └──────────────┘
       ┌────▼────┐       │
       │   Run   │  ┌────▼────┐
       │ Predict │  │  View   │
       └────┬────┘  │  2x2    │
            │       │  Grid   │
       ┌────▼────┐  └────┬────┘
       │  View   │       │
       │ Results │  ┌────▼────┐
       │ 3 Tabs  │  │  Table  │
       └────┬────┘  │ Compare │
            │       └─────────┘
       ┌────▼────┐
       │Download │
       │  Mask   │
       └─────────┘
```

---

## 6. State Management

| State | Stored In | Lifecycle |
|-------|-----------|-----------|
| Uploaded Image | `st.session_state.uploaded_image` | Until page refresh or new upload |
| Selected Sample | `st.session_state.selected_sample` | Until page refresh |
| Prediction Result | `st.session_state.prediction` | Until new prediction |
| Selected Model | Sidebar dropdown (reactive) | Persistent across pages |
| Active Classes | `st.session_state.active_classes` | Persistent across pages |
| Comparison Results | `st.session_state.comparison` | Until new comparison |

---

## 7. Error Handling Flows

| Scenario | User Sees | Recovery |
|----------|-----------|----------|
| No model checkpoint found | Warning: "Model not trained yet. Please train first." | Link to training instructions |
| Image too large (> 10MB) | Error: "Image too large. Max 10MB." | Auto-resize option |
| CUDA out of memory | Warning: "GPU memory full. Switching to CPU." | Auto-fallback to CPU |
| Invalid image format | Error: "Unsupported format. Use JPG, PNG, or TIF." | List supported formats |
| No objects detected | Info: "No objects detected above confidence threshold." | Suggest lowering threshold |

---

## 8. Responsive Behavior

| Screen Width | Layout |
|-------------|--------|
| >= 1200px | Sidebar visible + full main content |
| 768px - 1199px | Collapsible sidebar + main content |
| < 768px | Sidebar collapsed by default + stacked layout |

> **Note:** Streamlit handles most responsive layout automatically. Custom CSS will be used for the prediction grid and comparison views.
