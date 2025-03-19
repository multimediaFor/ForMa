import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP: known incorrect sRGB profile.*")

import torch
import torch.nn as nn
import torch.nn.functional as F


class MyLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(MyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def Softmax_Focal_Loss(self, pred, target):
        """
        pred : [B, C, H, W]
        target : [B, H, W]
        """
        target = target.long()
        target = target.squeeze()
        b, c, h, w = pred.size()
        bt, ht, wt = target.size()
        if h != ht or w != wt:
            pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)

        p = F.softmax(pred, dim=1)

        ce_loss = F.cross_entropy(pred, target, reduction="none")

        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)

        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

            loss = alpha_t * loss

        loss = loss.mean()
        return loss

    #
    def Dice_loss(self, pred, mask):
        """
        pred:[b,2,h,w],
        mask:[b,h,w]
        """
        smooth = 1e-6
        mask = mask.squeeze(dim=1)
        b, c, h, w = pred.size()
        bt, ht, wt = mask.size()
        if h != ht or w != wt:
            pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)

        pred = torch.softmax(pred, dim=1)[:, 1]
        intersection = torch.sum(pred * mask, dim=(1, 2))
        union = torch.sum(pred, dim=(1, 2)) + torch.sum(mask, dim=(1, 2))

        dice_coefficient = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_coefficient.mean()
        return dice_loss

    def forward(self, pred, mask):
        return self.Dice_loss(pred, mask) + self.Softmax_Focal_Loss(pred, mask)
