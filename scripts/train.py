import os
import sys
import time
from collections import defaultdict
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import get_model
from utils.config import get_config
from utils.dataset import get_dataloaders
from utils.losses import calc_loss
from utils.metrics import mIoU
from utils.checkpoint import save_checkpoint, load_checkpoint

def get_optimizer(model, config):
    cfg = config["training"]
    if cfg["optimizer"] == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=cfg["learning_rate"],
            momentum=cfg["momentum"],
            weight_decay = cfg["weight_decay"],
        )
    elif cfg["optimizer"] == "adam":
        return optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "adamw":
        return optim.Adamw(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])

def get_scheduler(optimizer, config):
    cfg = config["training"]
    if cfg["scheduler"] == "poly":
        return optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (1 - epoch / cfg["epochs"]) ** cfg["scheduler_power"]
        )
    elif cfg["scheduler"] == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif cfg["scheduler"] == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

def train_one_epoch(model, loader, optimizer, scaler, config, device):
    model.train()  # enable dropout + batchnorm training mode
    metrics = defaultdict(float)  # tracks loss components
    IOUs = defaultdict(float)     # per-class IoU sums
    NUM_IMGS = defaultdict(int)   # images per class count
    loop = tqdm(loader, desc="Train", leave=False)
    for images, masks in loop:
        # Move data to GPU (or CPU)
        images = images.to(device)
        masks = masks.to(device)
        # autocast: automatically uses float16 where safe, float32 where needed
        # reduces GPU memory by ~50% and speeds up training ~1.5x
        with autocast(enabled=config["training"].get("use_amp", False)):
            outputs = model(images)           # forward pass — model predicts mask
            loss = calc_loss(
                outputs, masks, metrics,
                loss_type=config["loss"]["type"],
                bce_weight=config["loss"].get("bce_weight", 0.5),
            )
        # Backward pass — compute gradients
        optimizer.zero_grad()  # clear old gradients (they accumulate by default)
        if scaler:
            # AMP: scaler prevents underflow in float16 gradients
            scaler.scale(loss).backward()  # scale loss up, compute gradients
            scaler.step(optimizer)         # unscale gradients, update weights
            scaler.update()                # adjust scale factor for next iteration
        else:
            loss.backward()    # compute gradients
            optimizer.step()   # update weights
        # Compute mIoU for this batch
        with torch.no_grad():
            mIoU(outputs, masks, metrics, IOUs, NUM_IMGS, device)
        # Update progress bar with current loss
        loop.set_postfix(loss=loss.item())
    # Average metrics over all samples
    n = len(loader.dataset)
    epoch_metrics = {k: v / n for k, v in metrics.items()}
    return epoch_metrics

def validate(model, loader, config, device):
    model.eval()  # disable dropout + use fixed batchnorm stats
    metrics = defaultdict(float)
    IOUs = defaultdict(float)
    NUM_IMGS = defaultdict(int)
    with torch.no_grad():  # no gradients needed — saves memory
        loop = tqdm(loader, desc="Val", leave=False)
        for images, masks in loop:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            calc_loss(outputs, masks, metrics,
                      loss_type=config["loss"]["type"],
                      bce_weight=config["loss"].get("bce_weight", 0.5))
            mIoU(outputs, masks, metrics, IOUs, NUM_IMGS, device)
    n = len(loader.dataset)
    epoch_metrics = {k: v / n for k, v in metrics.items()}
    return epoch_metrics

def train(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    # Create model from registry
    model = get_model(
        config["model"]["name"],
        n_classes=config["model"]["num_classes"],
    )
    model = model.to(device)
    print(f"Model: {config['model']['name']} | Params: {sum(p.numel() for p in model.parameters()):,}")
    # Create data loaders
    train_loader, val_loader = get_dataloaders(config)
    # Setup optimizer, scheduler, AMP scaler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    scaler = GradScaler() if config["training"].get("use_amp", False) else None
    # Tracking variables
    best_miou = 0.0
    patience_counter = 0
    patience = config["training"].get("early_stopping_patience", 10)
    history = {"train_loss": [], "val_loss": [], "train_miou": [], "val_miou": []}
    # Resume from checkpoint if specified
    start_epoch = 0
    if config.get("checkpoint", {}).get("resume"):
        ckpt = load_checkpoint(config["checkpoint"]["resume"], model, optimizer, scheduler, scaler)
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt["best_miou"]
        history = ckpt.get("metrics_history", history)
    # Training loop
    print(f"\nTraining for {config['training']['epochs']} epochs...")
    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, config, device)
        val_metrics = validate(model, val_loader, config, device)
        # Step the learning rate scheduler
        if scheduler:
            scheduler.step()
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        # Log results
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train mIoU: {train_metrics['mIoU']:.4f}")
        print(f"  Val   Loss: {val_metrics['loss']:.4f} | Val   mIoU: {val_metrics['mIoU']:.4f}")
        print(f"  LR: {current_lr:.6f} | Time: {elapsed:.1f}s")
        # Save history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_miou"].append(train_metrics["mIoU"])
        history["val_miou"].append(val_metrics["mIoU"])
        # Save best model
        save_dir = config.get("checkpoint", {}).get("save_dir", "checkpoints/")
        model_name = config["model"]["name"]
        if val_metrics["mIoU"] > best_miou:
            best_miou = val_metrics["mIoU"]
            patience_counter = 0
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            best_miou, config, history,
                            os.path.join(save_dir, f"{model_name}_best.pth"))
            print(f"  ★ New best mIoU: {best_miou:.4f}")
        else:
            patience_counter += 1
        # Save latest checkpoint
        if config.get("checkpoint", {}).get("save_last", True):
            save_checkpoint(model, optimizer, scheduler, scaler, epoch,
                            best_miou, config, history,
                            os.path.join(save_dir, f"{model_name}_last.pth"))
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping — no improvement for {patience} epochs")
            break
    print(f"\nTraining complete! Best mIoU: {best_miou:.4f}")
    return model

if __name__ == "__main__":
    config = get_config()
    train(config)