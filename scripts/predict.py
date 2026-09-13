import sys
import os 
import argparse
import glob
import torch
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.checkpoint import load_model_for_reference

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

def preprocess(image_path, image_size=512):
    image = Image.open(image_path).convert("RGB")
    original_size = image.image_size

    transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(image).unsqueeze(0)
    return tensor, image, original_size

def predict(model, tensor, device, threshold=0.5):
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output)
        mask = (probs > threshold).cpu().numpy()[0]
    return mask, probs.cpu(), numpy()[0]

def create_overlay(image, mask, alpha=0.5):
    image_np = np.array(image.resize((mask.shape[2], mask.shape[1])))
    overlay = image_np.copy()

    for cls_idx in range(mask.shape[0]):
        if mask[cls_idx].any():
            color = COLORS[cls_idx % len(COLORS)]
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask[cls_idx],
                    overlay[:, :, c] * (1-alpha) + color[c] * alpha,
                    overlay[:, :, c]
                )
    return overlay

def save_result(image, mask, probs, output_path, class_names=CLASS_NAMES):
    detected = []
    for i in range(mask.shape[0]):
        if mask[i].any():
            confidence = probs[i][mask[i]].mean()
            detected.append((class_names[i], confidence, i))
    n_panels = 2 + min(len(detected), 4)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    axes[0].imshow(image.resize((mask.shape[2], mask.shape[1])))
    axes[0].set_title("Original")
    axes[0].axis("off")
    overlay = create_overlay(image, mask)
    axes[1].imshow(overlay)
    axes[1].set_title(f"Prediction ({len(detected)} classes)")
    axes[1].axis("off")
    for idx, (name, conf, cls_idx) in enumerate(detected[:4]):
        axes[idx + 2].imshow(mask[cls_idx], cmap="gray")
        axes[idx + 2].set_title(f"{name} ({conf:.2f})")
        axes[idx + 2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")
def main():
    parser = argparse.ArgumentParser(description="DeepLab Prediction")
    parser.add_argument("--checkpoint", required=True, help="path to .pth checkpoint")
    parser.add_argument("--input", required=True, help="image path or directory")
    parser.add_argument("--output-dir", default="results/predictions", help="output directory")
    parser.add_argument("--model", default=None, help="model name (auto-detected from checkpoint)")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = args.model or ckpt.get("model_name", "deeplabv3plus")
    model = get_model(model_name, n_classes=args.num_classes)
    model, _ = load_model_for_inference(args.checkpoint, model, device)
    if os.path.isdir(args.input):
        image_paths = glob.glob(os.path.join(args.input, "*.jpg")) + \
                      glob.glob(os.path.join(args.input, "*.png"))
    else:
        image_paths = [args.input]
    print(f"Predicting {len(image_paths)} images with {model_name}...")
    for img_path in image_paths:
        tensor, image, original_size = preprocess(img_path, args.image_size)
        mask, probs = predict(model, tensor, device, args.threshold)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(args.output_dir, f"{basename}_pred.png")
        save_result(image, mask, probs, output_path)
        detected = [CLASS_NAMES[i] for i in range(mask.shape[0]) if mask[i].any()]
        print(f"  {os.path.basename(img_path)}: {detected}")
    print(f"\nDone! Results saved to {args.output_dir}")
if __name__ == "__main__":
    main()