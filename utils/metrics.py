import torch

SMOOTH = 1e-6

def iou_pytorch(outputs, labels):
    with torch.no_grad():
        intersection = (outputs & labels).float().sum((1, 2))
        union = (outputs | labels).float().sum((1, 2))
        iou = intersection / (union + SMOOTH)
    return iou

def mIoU(pred, target, metrics, IOUs, NUM_IMGS, device):
    N, C, H, W = target.shape
    # Background threshold channel at 0.5:
    # If all class probabilities < 0.5, background (index 0) wins in argmax.
    # If any class probability >= 0.5, the highest-probability class wins.
    bg_channel = torch.full((N, 1, H, W), 0.5, device=device)

    pred_probs = torch.sigmoid(pred)

    pred_cls = torch.cat((bg_channel, pred_probs), dim=1).argmax(1)
    target_cls = torch.cat((bg_channel, target), dim=1).argmax(1)

    class_iou = torch.zeros((N, C), device=device)
    target_present = torch.zeros((N, C), dtype=torch.bool, device=device)

    for cls in range(1, C + 1):
        pred_inds = pred_cls == cls
        target_inds = target_cls == cls
        iou = iou_pytorch(pred_inds, target_inds)

        # Check which samples in the batch actually contain this class in ground truth
        present = target_inds.view(N, -1).any(dim=1)
        target_present[:, cls - 1] = present
        class_iou[:, cls - 1] = iou * present

        img_in_cls = present.count_nonzero()
        if img_in_cls != 0:
            IOUs[cls] += class_iou[:, cls - 1].sum().item()
            NUM_IMGS[cls] += img_in_cls.item()

    # Per-image mIoU averaged over classes actually present in that image
    num_classes_in_img = target_present.sum(dim=1).clamp(min=1)
    miou_imgs = class_iou.sum(dim=1) / num_classes_in_img

    metrics["mIoU"] += miou_imgs.sum().item()
    return miou_imgs.mean()