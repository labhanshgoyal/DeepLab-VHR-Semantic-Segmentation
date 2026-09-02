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
    bg_channel = torch.zeroes((N, 1, H, W)).to(device)
    pred=torch.cat((bg_channel, pred), dim=1).argmax(1)
    target=torch.cat((bg_channel, target), dim=1).argmax(1)

    class_iou=torch.zeroes((N, C)).to(device)

    for cls in range(1, C+1):
        pred_inds=pred==cls
        target_inds=target==cls
        iou=iou_pytorch(pred_inds, target_inds)

        a=target_inds.view(N, -1).any(dim=1)
        class_iou[:, cls -1]=iou*a

        img_in_cls = class_iou[:, cls-1].count_nonzero()
        if img_in_cls != 0:
            IOUs[cls] += class_iou[:, cls-1].sum().item()
            NUM_IMGS[cls] += img_in_cls.item()

    cls_in_img = class_iou.count_nonzero(dim=1)
    cls_in_img = cls_in_img.clamp(min=1)
    miou_imgs = class_iou.sum(dim=1) / cls_in_img

    metrics["mIoU"] = miou_imgs.mean().item()
    return miou_imgs.mean()