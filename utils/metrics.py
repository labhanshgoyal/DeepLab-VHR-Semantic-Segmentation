import torch

SMOOTH = 1e-6

def iou_pytorch(outputs, labels):
    with torch.no_grad():
        intersection = (outputs & labels).float().sum((1,2))
        union = (outputs | labels).float().sum((1,2))
        iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou

def mIoU(pred, target, metrics, IOUs, NUM_IMGS, device):
    N, C, H, W = target.shape
    bg_channel = torch.zeros((N, 1, H, W)).to(device)

    # apply sigmoid to convert raw logits to probabilities [0-1]
    # without this, negative logits always lose to bg=0 → mIoU stays 0
    pred_probs = torch.sigmoid(pred)

    pred_cls = torch.cat((bg_channel, pred_probs), dim=1).argmax(1)
    target_cls = torch.cat((bg_channel, target), dim=1).argmax(1)

    class_iou = torch.zeros((N, C)).to(device)

    for cls in range(1, C + 1):
        pred_inds = pred_cls == cls
        target_inds = target_cls == cls
        iou = iou_pytorch(pred_inds, target_inds)

        a = target_inds.view(N, -1).any(dim=1)
        class_iou[:, cls - 1] = iou * a

        img_in_cls = class_iou[:, cls - 1].count_nonzero()
        if img_in_cls != 0:
            IOUs[cls] += class_iou[:, cls - 1].sum().item()
            NUM_IMGS[cls] += img_in_cls.item()

    cls_in_img = class_iou.count_nonzero(dim=1)
    cls_in_img = cls_in_img.clamp(min=1)
    miou_imgs = class_iou.sum(dim=1) / cls_in_img

    metrics["mIoU"] += miou_imgs.sum().item()
    return miou_imgs.mean()