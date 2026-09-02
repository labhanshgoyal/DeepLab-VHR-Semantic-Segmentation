import torch
import os
from datetime import datetime

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, best_miou, config, metrics_history, save_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_miou": best_miou,
        "config": config,
        "model_name": config["model"]["name"],
        "num_classes": config["model"]["num_classes"],
        "metrics_history": metrics_history,
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu", weight_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (mIoU: {checkpoint['best_miou']:.4f})")
    return checkpoint

def load_model_for_inference(path, model, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded: {checkpoint['model_name']} (mIoU: {checkpoint['best_miou']:.4f})")
    return model, checkpoint
