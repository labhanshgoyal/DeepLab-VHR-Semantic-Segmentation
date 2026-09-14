import sys
import os
import json
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.config import get_config
from utils.dataset import get_dataloaders
from utils.losses import calc_loss
from utils.metrics import mIoU
from utils.checkpoint import load_model_for_inference

CLASS_NAMES = [
    "Airplane", "Ship", "Storage Tank", "Baseball Diamond",
    "Tennis Court", "Basketball Court", "Ground Track Field",
    "Harbor", "Bridge", "Vehicle"
]


def compute_confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for t, p in zip(target[mask], pred[mask]):
        cm[t.long(), p.long()] += 1
    return cm


def plot_confusion_matrix(cm, class_names, output_path):
    cm_norm = cm.float() / cm.sum(dim=1, keepdim=True).clamp(min=1)
    cm_norm = cm_norm.numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Normalized)")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.1%}", ha="center", va="center",
                    color=color, fontsize=8)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix: {output_path}")


def plot_per_class_iou(class_ious, class_names, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(class_names)))
    bars = ax.barh(class_names, class_ious, color=colors)

    for bar, iou in zip(bars, class_ious):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{iou:.1%}", va="center", fontsize=10)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("IoU")
    ax.set_title(f"Per-Class IoU (mIoU: {np.mean(class_ious):.1%})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved per-class IoU chart: {output_path}")


def evaluate(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    output_dir = config.get("eval_output_dir", "results/evaluation")
    os.makedirs(output_dir, exist_ok=True)

    # load model
    checkpoint_path = config.get("checkpoint", {}).get("resume")
    if not checkpoint_path:
        print("Error: pass --resume with checkpoint path")
        return

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_name = ckpt.get("model_name", config["model"]["name"])
    num_classes = config["model"]["num_classes"]

    model = get_model(model_name, n_classes=num_classes)
    model, _ = load_model_for_inference(checkpoint_path, model, device)

    # load validation data
    _, val_loader = get_dataloaders(config)

    # tracking
    metrics = defaultdict(float)
    IOUs = defaultdict(float)
    NUM_IMGS = defaultdict(int)
    confusion = torch.zeros(num_classes + 1, num_classes + 1, dtype=torch.long)

    print(f"Evaluating {model_name} on {len(val_loader.dataset)} images...")

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            calc_loss(outputs, masks, metrics,
                      loss_type=config["loss"]["type"],
                      bce_weight=config["loss"].get("bce_weight", 0.5))
            mIoU(outputs, masks, metrics, IOUs, NUM_IMGS, device)

            # build confusion matrix
            N, C, H, W = masks.shape
            bg = torch.zeros((N, 1, H, W)).to(device)
            pred_cls = torch.cat((bg, outputs), dim=1).argmax(1)
            true_cls = torch.cat((bg, masks), dim=1).argmax(1)
            confusion += compute_confusion_matrix(pred_cls.cpu(), true_cls.cpu(), num_classes + 1)

    n = len(val_loader.dataset)
    avg_metrics = {k: v / n for k, v in metrics.items()}

    # per-class IoU
    class_ious = []
    print(f"\n{'='*50}")
    print(f"{'Class':<25} {'IoU':>8} {'Images':>8}")
    print(f"{'='*50}")
    for cls in range(1, num_classes + 1):
        iou = IOUs[cls] / max(NUM_IMGS[cls], 1)
        class_ious.append(iou)
        print(f"{CLASS_NAMES[cls-1]:<25} {iou:>7.1%} {int(NUM_IMGS[cls]):>8}")
    print(f"{'='*50}")
    print(f"{'mIoU':<25} {avg_metrics['mIoU']:>7.4f}")
    print(f"{'Loss':<25} {avg_metrics['loss']:>7.4f}")

    plot_confusion_matrix(confusion[1:, 1:], CLASS_NAMES,
                          os.path.join(output_dir, "confusion_matrix.png"))
    plot_per_class_iou(class_ious, CLASS_NAMES,
                       os.path.join(output_dir, "per_class_iou.png"))

    results = {
        "model": model_name,
        "mIoU": avg_metrics["mIoU"],
        "loss": avg_metrics["loss"],
        "per_class_iou": {name: iou for name, iou in zip(CLASS_NAMES, class_ious)},
        "num_images": n,
    }
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    config = get_config()
    evaluate(config)