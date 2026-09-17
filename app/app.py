import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import streamlit as st
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for streamlit

from models import get_model
from utils.checkpoint import load_model_for_inference

CLASS_NAMES = [
    "Airplane", "Ship", "Storage Tank", "Baseball Diamond",
    "Tennis Court", "Basketball Court", "Ground Track Field",
    "Harbor", "Bridge", "Vehicle"
]

COLORS = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
    [0, 0, 128], [128, 128, 0]
]

CLASS_EMOJIS = ["✈️", "🚢", "🛢️", "⚾", "🎾", "🏀", "🏟️", "⚓", "🌉", "🚗"]


# ---------- model loading (cached so it only loads once) ----------
@st.cache_resource
def load_model(checkpoint_path, model_name="deeplabv3plus", num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, n_classes=num_classes)
    model, metadata = load_model_for_inference(checkpoint_path, model, device)
    return model, device, metadata

def preprocess(image, image_size=512):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def predict(model, tensor, device, threshold=0.5):
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output)
        mask = (probs > threshold).cpu().numpy()[0]
    return mask, probs.cpu().numpy()[0]

def create_overlay(image_np, mask, alpha=0.5):
    overlay = image_np.copy()
    for cls_idx in range(mask.shape[0]):
        if mask[cls_idx].any():
            color = COLORS[cls_idx % len(COLORS)]
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask[cls_idx],
                    overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                    overlay[:, :, c]
                )
    return overlay.astype(np.uint8)

def main():
    st.set_page_config(
        page_title="DeepLab Satellite Segmentation",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #2a2a4a;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 4px;
    }
    .detection-chip {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # header
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ DeepLab Satellite Segmentation</h1>
        <p style="color: #888; font-size: 1.1rem;">
            Multi-class semantic segmentation on VHR remote sensing imagery
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        checkpoint_path = st.text_input(
            "Checkpoint path",
            value="checkpoints/deeplabv3plus_best.pth"
        )
        model_name = st.selectbox(
            "Model", ["deeplabv3plus", "deeplabv3", "deeplabv2", "deeplabv1"]
        )
        threshold = st.slider(
            "Confidence threshold", 0.1, 0.9, 0.5, 0.05,
            help="Lower = more detections, Higher = more confident"
        )
        alpha = st.slider(
            "Overlay opacity", 0.1, 0.9, 0.5, 0.05,
            help="Transparency of the color overlay"
        )

        st.markdown("---")
        st.markdown("### 🎨 Class Colors")
        for i, (name, emoji) in enumerate(zip(CLASS_NAMES, CLASS_EMOJIS)):
            color = COLORS[i]
            st.markdown(
                f'<span style="color: rgb({color[0]},{color[1]},{color[2]})">'
                f'■</span> {emoji} {name}',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(
            "Built with [DeepLabV3+](https://arxiv.org/abs/1802.02611) "
            "on NWPU VHR-10 dataset"
        )

    # ---------- main content ----------
    # check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        st.warning(
            f"⚠️ Checkpoint not found: `{checkpoint_path}`\n\n"
            "Please train the model first or provide a valid checkpoint path."
        )
        st.info(
            "**To train:** `python scripts/train.py`\n\n"
            "**Or download:** Place your trained `.pth` file in `checkpoints/`"
        )
        return

    # load model
    with st.spinner("Loading model..."):
        model, device, metadata = load_model(checkpoint_path, model_name)
    st.sidebar.success(f"✅ Model loaded on **{device}**")

    # file upload
    uploaded = st.file_uploader(
        "Upload a satellite image",
        type=["jpg", "jpeg", "png", "tif"],
        help="Upload a VHR satellite or aerial image"
    )

    # sample images option
    use_sample = st.checkbox("Use sample image from dataset")

    if use_sample:
        sample_dir = "NWPU VHR-10 dataset/positive image set"
        if os.path.exists(sample_dir):
            samples = sorted([f for f in os.listdir(sample_dir) if f.endswith(".jpg")])
            selected = st.selectbox("Select sample", samples[:20])
            image = Image.open(os.path.join(sample_dir, selected)).convert("RGB")
        else:
            st.error("Sample dataset not found")
            return
    elif uploaded:
        image = Image.open(uploaded).convert("RGB")
    else:
        st.info("👆 Upload an image or select a sample to get started")
        return

    # run prediction
    with st.spinner("🔍 Running prediction..."):
        tensor = preprocess(image)
        mask, probs = predict(model, tensor, device, threshold)

    # resize image to match mask
    image_resized = np.array(image.resize((mask.shape[2], mask.shape[1])))

    # detected classes
    detected = []
    for i in range(mask.shape[0]):
        if mask[i].any():
            conf = probs[i][mask[i]].mean()
            pixel_pct = mask[i].sum() / mask[i].size * 100
            detected.append({
                "class": CLASS_NAMES[i],
                "emoji": CLASS_EMOJIS[i],
                "confidence": conf,
                "pixel_pct": pixel_pct,
                "idx": i,
                "color": COLORS[i]
            })

    # ---------- results display ----------

    # metric cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(detected)}</div>
            <div class="metric-label">Classes Detected</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        avg_conf = np.mean([d["confidence"] for d in detected]) if detected else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_conf:.1%}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        total_pct = sum(d["pixel_pct"] for d in detected)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_pct:.1f}%</div>
            <div class="metric-label">Pixels Segmented</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # images side by side
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 📷 Original Image")
        st.image(image_resized, use_container_width=True)
    with col_right:
        st.markdown("### 🎯 Segmentation Result")
        overlay = create_overlay(image_resized, mask, alpha)
        st.image(overlay, use_container_width=True)

    # detected classes chips
    if detected:
        st.markdown("### 🏷️ Detected Objects")
        chips_html = ""
        for d in sorted(detected, key=lambda x: x["confidence"], reverse=True):
            r, g, b = d["color"]
            chips_html += (
                f'<span class="detection-chip" '
                f'style="background: rgba({r},{g},{b},0.2); '
                f'border: 2px solid rgb({r},{g},{b}); '
                f'color: rgb({r},{g},{b});">'
                f'{d["emoji"]} {d["class"]} ({d["confidence"]:.0%})'
                f'</span>'
            )
        st.markdown(chips_html, unsafe_allow_html=True)

    # individual class masks
    if detected:
        st.markdown("### 🔬 Per-Class Masks")
        cols = st.columns(min(len(detected), 4))
        for i, d in enumerate(detected[:4]):
            with cols[i]:
                r, g, b = d["color"]
                # create colored mask
                colored = np.zeros((*mask[d["idx"]].shape, 3), dtype=np.uint8)
                colored[mask[d["idx"]] > 0] = d["color"]
                st.image(colored, caption=f'{d["emoji"]} {d["class"]}',
                         use_container_width=True)
                st.caption(f"Confidence: {d['confidence']:.1%} | Coverage: {d['pixel_pct']:.1f}%")


if __name__ == "__main__":
    main()