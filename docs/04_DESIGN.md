# 🎨 Design / UI-UX Document

## Streamlit Demo App — Visual Design Specification

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Date** | August 16, 2026 |
| **Status** | Draft |

---

## 1. Design Philosophy

| Principle | Implementation |
|-----------|---------------|
| **Professional & Clean** | Dark theme with accent colors; no visual clutter |
| **Data-First** | Large visualization areas; compact controls |
| **Satellite Imagery Aesthetic** | Deep space dark tones with vibrant neon mask colors |
| **Minimal Friction** | Sample images pre-loaded; one-click prediction |

---

## 2. Color Palette

### 2.1 App Theme Colors

| Role | Color | Hex | Usage |
|------|-------|-----|-------|
| Background (Primary) | Dark Navy | #0E1117 | Main background (Streamlit dark theme) |
| Background (Secondary) | Dark Slate | #1A1D23 | Cards, containers |
| Surface | Charcoal | #262730 | Sidebar, input areas |
| Primary Accent | Electric Blue | #4A9EFF | Buttons, links, highlights |
| Secondary Accent | Teal | #00D4AA | Success states, mIoU display |
| Warning | Amber | #FFA726 | Warnings, medium confidence |
| Error | Coral | #FF5252 | Errors, low confidence |
| Text (Primary) | White | #FAFAFA | Headlines, values |
| Text (Secondary) | Gray | #B0B0B0 | Labels, descriptions |

### 2.2 Segmentation Class Colors

Each of the 10 classes has a unique, visually distinct color for mask overlays:

| Class | Color Name | RGB | Hex | Rationale |
|-------|-----------|-----|-----|-----------|
| Airplane | Bright Red | (255, 50, 50) | #FF3232 | High contrast on green backgrounds |
| Ship | Ocean Blue | (50, 130, 255) | #3282FF | Intuitive for water-related objects |
| Storage Tank | Lime Green | (80, 255, 80) | #50FF50 | Stands out on building areas |
| Baseball Diamond | Gold | (255, 215, 0) | #FFD700 | Diamond shape association |
| Tennis Court | Cyan | (0, 255, 255) | #00FFFF | Court sport color coding |
| Basketball Court | Orange | (255, 165, 0) | #FFA500 | Warm sport color |
| Ground Track Field | Magenta | (255, 50, 255) | #FF32FF | Distinct from courts |
| Harbor | Deep Purple | (150, 50, 255) | #9632FF | Water infrastructure |
| Bridge | Hot Pink | (255, 105, 180) | #FF69B4 | Linear structure highlight |
| Vehicle | Yellow Green | (180, 255, 50) | #B4FF32 | Small object emphasis |
| Background | Transparent | — | — | Not rendered |

---

## 3. Typography

| Element | Font | Size | Weight |
|---------|------|------|--------|
| App Title | Source Sans Pro | 36px | 700 (Bold) |
| Page Headers (H1) | Source Sans Pro | 28px | 700 |
| Section Headers (H2) | Source Sans Pro | 22px | 600 |
| Card Titles (H3) | Source Sans Pro | 18px | 600 |
| Body Text | Source Sans Pro | 16px | 400 |
| Metric Values | Roboto Mono | 32px | 700 |
| Small Labels | Source Sans Pro | 12px | 400 |
| Code / Technical | Fira Code | 14px | 400 |

> **Note:** Source Sans Pro is Streamlit's default font. We use Roboto Mono for metric values.

---

## 4. Page Layouts

### 4.1 Predict Page Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  🛰 DeepLab VHR Segmentation                              [⚙]  │
│  Semantic segmentation of satellite imagery using DeepLab models │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  📷 Input Image                                            │  │
│  │  ┌─────────────────────────────────────────────────────┐   │  │
│  │  │                                                     │   │  │
│  │  │          Drag & Drop Image Here                     │   │  │
│  │  │          or click to browse                         │   │  │
│  │  │          (JPG, PNG, TIF - Max 10MB)                 │   │  │
│  │  │                                                     │   │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  │                                                            │  │
│  │  Or try a sample: [img1] [img2] [img3] [img4] [img5]      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│              [ 🔍 Run Segmentation ]  (Primary Button)           │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │  Metric Card     │  │  Metric Card     │  │  Metric Card  │  │
│  │  ┌──────────┐    │  │  ┌──────────┐    │  │  ┌────────┐   │  │
│  │  │  48.3%   │    │  │  │  245ms   │    │  │  │   7    │   │  │
│  │  └──────────┘    │  │  └──────────┘    │  │  └────────┘   │  │
│  │  mIoU Score      │  │  Inference Time  │  │  Classes      │  │
│  └──────────────────┘  └──────────────────┘  │  Detected     │  │
│                                              └───────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Results Tabs: [Original] [Mask] [Overlay]                 │  │
│  │  ┌─────────────────────────────────────────────────────┐   │  │
│  │  │                                                     │   │  │
│  │  │                                                     │   │  │
│  │  │          Segmentation Result                        │   │  │
│  │  │          (Large visualization area)                 │   │  │
│  │  │                                                     │   │  │
│  │  │                                                     │   │  │
│  │  └─────────────────────────────────────────────────────┘   │  │
│  │                                                            │  │
│  │  Class Legend:                                             │  │
│  │  [■ Airplane] [■ Ship] [■ Tank] [■ Baseball] [■ Tennis]   │  │
│  │  [■ Basketball] [■ Track] [■ Harbor] [■ Bridge] [■ Car]   │  │
│  │                                                            │  │
│  │  [ 📥 Download Mask ] [ 📥 Download Overlay ]              │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Compare Page Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  🔬 Model Comparison                                             │
│  Compare predictions from all 4 DeepLab architectures            │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Image Input Area - same as Predict page]                       │
│                                                                  │
│              [ 🔄 Compare All Models ]  (Primary Button)         │
│                                                                  │
│  ┌────────────────────────┐  ┌────────────────────────┐          │
│  │  DeepLabV1             │  │  DeepLabV2             │          │
│  │  mIoU: 35.2%  180ms   │  │  mIoU: 42.1%  320ms   │          │
│  │  ┌──────────────────┐  │  │  ┌──────────────────┐  │          │
│  │  │                  │  │  │  │                  │  │          │
│  │  │  Prediction      │  │  │  │  Prediction      │  │          │
│  │  │                  │  │  │  │                  │  │          │
│  │  └──────────────────┘  │  │  └──────────────────┘  │          │
│  └────────────────────────┘  └────────────────────────┘          │
│  ┌────────────────────────┐  ┌────────────────────────┐          │
│  │  DeepLabV3             │  │  DeepLabV3+ ⭐ Best     │          │
│  │  mIoU: 45.8%  350ms   │  │  mIoU: 48.3%  380ms   │          │
│  │  ┌──────────────────┐  │  │  ┌──────────────────┐  │          │
│  │  │                  │  │  │  │                  │  │          │
│  │  │  Prediction      │  │  │  │  Prediction      │  │          │
│  │  │                  │  │  │  │                  │  │          │
│  │  └──────────────────┘  │  │  └──────────────────┘  │          │
│  └────────────────────────┘  └────────────────────────┘          │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Comparison Summary Table                                  │  │
│  │  Model       mIoU    Time    Params     Best Class         │  │
│  │  V1          35.2%   180ms   12.5M      Tennis Court       │  │
│  │  V2          42.1%   320ms   44.8M      Storage Tank       │  │
│  │  V3          45.8%   350ms   58.6M      Airplane           │  │
│  │  V3+ (best)  48.3%   380ms   54.7M      Airplane           │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 4.3 Results Dashboard Layout

```
┌──────────────────────────────────────────────────────────────────┐
│  📊 Training Results & Benchmarks                                │
│  Comprehensive evaluation of all trained models                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Model Selector: [All Models ▼]                                  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Overall Comparison Table                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐  │  │
│  │  │ Model  │ mIoU │ Params │ FLOPs │ Speed │ Backbone  │  │  │
│  │  │ V1     │ 35%  │ 12M   │ 18G   │ 180ms │ VGG-16    │  │  │
│  │  │ V2     │ 42%  │ 44M   │ 62G   │ 320ms │ Res-101   │  │  │
│  │  │ V3     │ 46%  │ 58M   │ 85G   │ 350ms │ Res-101   │  │  │
│  │  │ V3+    │ 48%  │ 54M   │ 78G   │ 380ms │ Res-101   │  │  │
│  │  └──────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐  │
│  │  Training Loss Curves   │  │  mIoU Progress               │  │
│  │  ┌──────────────────┐   │  │  ┌──────────────────────┐    │  │
│  │  │     ╲             │   │  │  │              ╱──────│    │  │
│  │  │      ╲____        │   │  │  │           ╱        │    │  │
│  │  │           ╲___    │   │  │  │        ╱           │    │  │
│  │  │               ╲__ │   │  │  │     ╱              │    │  │
│  │  │  ── train ── val  │   │  │  │  ╱   ── train ─ val│    │  │
│  │  └──────────────────┘   │  │  └──────────────────────┘    │  │
│  └──────────────────────────┘  └──────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐  │
│  │  Confusion Matrix       │  │  Per-Class IoU Bar Chart      │  │
│  │  ┌──────────────────┐   │  │  ┌──────────────────────┐    │  │
│  │  │  [Interactive    │   │  │  │  ██ ██ ██ ██ ██      │    │  │
│  │  │   Plotly         │   │  │  │  ██ ██ ██ ██ ██      │    │  │
│  │  │   Heatmap]       │   │  │  │  ██ ██ ██    ██      │    │  │
│  │  │                  │   │  │  │  ██ ██          ██    │    │  │
│  │  │                  │   │  │  │  AP SH TK BD TC ...  │    │  │
│  │  └──────────────────┘   │  │  └──────────────────────┘    │  │
│  └──────────────────────────┘  └──────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Sample Predictions Gallery                                │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │  │
│  │  │Input │ │ GT   │ │ V1   │ │ V2   │ │ V3   │ │ V3+  │  │  │
│  │  │      │ │      │ │      │ │      │ │      │ │      │  │  │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘  │  │
│  │  ... (4-6 sample rows)                                    │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Component Specifications

### 5.1 Metric Card

```
┌──────────────────────┐
│                      │
│      48.3%           │   <- Roboto Mono, 32px, Bold, Teal (#00D4AA)
│                      │
│   mIoU Score         │   <- Source Sans Pro, 12px, Gray (#B0B0B0)
│                      │
└──────────────────────┘
Background: #1A1D23
Border: 1px solid #262730
Border-radius: 8px
Padding: 20px
```

### 5.2 Class Legend Item

```
 ■ Airplane (23.5%)    <- Color swatch (12x12) + Class name + IoU
```

### 5.3 Primary Button

```
┌──────────────────────────────┐
│   🔍 Run Segmentation        │
└──────────────────────────────┘
Background: #4A9EFF (Electric Blue)
Text: White, 16px, 600 weight
Border-radius: 6px
Padding: 12px 24px
Hover: #3A8EEF (slightly darker)
Active: scale(0.98)
```

### 5.4 Image Display Card

```
┌────────────────────────────────────┐
│  Tab Navigation:                   │
│  [Original] [Mask] [Overlay]       │
├────────────────────────────────────┤
│                                    │
│  ┌──────────────────────────────┐  │
│  │                              │  │
│  │     Image Content            │  │
│  │     (maintains aspect ratio) │  │
│  │                              │  │
│  └──────────────────────────────┘  │
│                                    │
│  Caption: "DeepLabV3+ Prediction"  │
└────────────────────────────────────┘
```

---

## 6. Animations & Transitions

| Element | Animation | Duration |
|---------|-----------|----------|
| Page load | Fade in | 300ms |
| Prediction loading | Streamlit spinner + progress bar | Duration of inference |
| Tab switching | Instant (Streamlit native) | — |
| Metric card values | Count up animation (CSS) | 500ms |
| Hover on sample images | Scale(1.05) + shadow | 200ms |
| Class toggle | Mask fades in/out | 200ms |

---

## 7. Custom CSS Overrides

```css
/* Main app styling */
.stApp {
    background-color: #0E1117;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1A1D23 0%, #262730 100%);
    border: 1px solid #333;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
}

.metric-value {
    font-family: 'Roboto Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    color: #00D4AA;
}

.metric-label {
    font-size: 14px;
    color: #808080;
    margin-top: 8px;
}

/* Sample image gallery */
.sample-image {
    border-radius: 8px;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 2px solid transparent;
}

.sample-image:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 20px rgba(74, 158, 255, 0.3);
    border-color: #4A9EFF;
}

/* Prediction grid */
.prediction-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    padding: 16px;
}

/* Class legend */
.class-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    padding: 12px;
}

.class-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 14px;
}

.color-swatch {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.2);
}
```

---

## 8. Accessibility

| Requirement | Implementation |
|-------------|---------------|
| Color contrast | All text meets WCAG AA (4.5:1 ratio) |
| Keyboard navigation | Streamlit native support |
| Screen reader | Alt text on all images |
| Color blind safe | Class colors chosen for deuteranopia compatibility |
| Loading states | Spinner with text description |
| Error messages | Clear, actionable text with suggestions |

---

## 9. Design Assets Needed

| Asset | Format | Size | Purpose |
|-------|--------|------|---------|
| App logo/icon | SVG/PNG | 64x64 | Sidebar header |
| Sample satellite images | JPG | 512x512 | Gallery (6-8 images) |
| Color legend SVGs | SVG | 14x14 | Class color swatches |
| Favicon | ICO | 32x32 | Browser tab |
| Social preview | PNG | 1200x630 | GitHub/social sharing |
