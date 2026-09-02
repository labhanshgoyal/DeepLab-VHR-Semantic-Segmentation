import torch
import torch.nn as nn
import torch.nn.functional as F

#Dice Loss
def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous() #.contiguous () ensures tensor is stores in continuous memory 
    target=target.contiguous() # which is required for efficient element wise multiplication
    intersection=(pred*target).sum(dim=(2,3))

    dice = 1-(2.0*intersection+smooth)/(pred.sum(dim=(2,3))+target.sum(dim=(2,3))+smooth)

    return dice.mean()

#Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma = 2.0):
        super().__init__()
        self.alpha = alpha      #balances +ve vs -ve classes
        self.gamma = gamma      #controls focus on hard examples (0 = same as BCE)

    def forward(self, pred, target):
        bce=F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t= torch.exp(-bce)
        focal_weight = self.alpha*(1-p_t)**self.gamma
        loss = focal_weight*bce

        return loss.mean()

#Combined Loss Function
def calc_loss(pred, target, metrics, loss_type="bce_dice", bce_weight=  0.5, focal_alpha = 0.25, focal_gamma=2.0):
    if loss_type=="bce_dice":
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred_sig = torch.sigmoid(pred)
        dice = dice_loss(pred_sig, target)
        loss = bce * bce_weight + dice * (1 - bce_weight)
        metrics["bce"] += bce.item() * target.size(0)
        metrics["dice"] += dice.item() * target.size(0)
    elif loss_type == "focal":
        focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        loss = focal(pred, target)
        metrics["focal"] += loss.item() * target.size(0)

    metrics["loss"] += loss.item() * target.size(0)
    return loss