import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*iCCP: known incorrect sRGB profile.*")
import math
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm
# class MyLoss(nn.Module):#使用pytorch框架，调用该类会自动运行forward方法
#     def __init__(self,alpha=0.5, gamma=2):
#         super(MyLoss, self).__init__()
#         self.alpha=alpha
#         self.gamma=gamma
#
#     def Softmax_Focal_Loss(self,pred, target, epoch=None, threshod=0, alpha=0.5, gamma=2):
#         """
#         pred : [B, 2, H, W]
#         target : [B, 1, H, W]
#         """
#
#         # 调整pred的大小
#         if pred.shape[2:] != target.shape[2:]:
#             pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=True)
#
#         target = target.squeeze(1)
#         target = target.long()
#
#         # 对预测输出应用softmax函数
#         p = F.softmax(pred, dim=1)
#
#         # 计算交叉熵损失
#         ce_loss = F.cross_entropy(pred, target, reduction="none")
#
#         if epoch is not None and epoch >= threshod:
#             prob = p[:, 1:2, :, :].squeeze(1)
#             mask = torch.abs(prob - 0.5) > (0.008 * epoch)
#             ce_loss = ce_loss * mask
#
#         # 计算p_t，即softmax输出和真实目标的加权和
#         p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
#         # 计算focal loss
#         loss = ce_loss * ((1 - p_t) ** gamma)
#
#         if alpha >= 0:
#             # 计算alpha_t，即正负样本的加权和
#             alpha_t = alpha * target + (1 - alpha) * (1 - target)
#             # 对loss进行加权
#             loss = alpha_t * loss
#
#         loss = loss.mean()
#
#         # 返回计算得到的损失值
#         return loss
#
#     def Dice_loss(self, pred, target, epoch=None, threshod=0):
#         """输入pred为[b,2,h,w],target为[b,1,h,w]"""
#         smooth = 1e-6
#         # 调整pred的大小
#         if pred.shape[2:] != target.shape[2:]:
#             pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=True)
#         target = target.squeeze(1)
#         target = target.long()
#         pred = torch.softmax(pred, dim=1)[:, 1]  # 将预测值转换为类别为1的概率
#         if epoch is not None and epoch >= threshod:
#             mask = torch.abs(pred - 0.5) > (0.008 * epoch)
#             pred = pred * mask
#             target = target * mask
#
#         intersection = torch.sum(pred * target, dim=(1, 2))  # 计算交集
#         union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))  # 计算联合集
#         dice_coefficient = (2. * intersection + smooth) / (union + smooth)  # 计算Dice系数
#         dice_loss = 1 - dice_coefficient.mean()  # 计算Dice损失，即1 - Dice系数的平均值
#
#         return dice_loss
#
#     def forward(self, pred, target, epoch=None, threshod=8):
#         loss1 = self.Softmax_Focal_Loss(pred=pred, target=target, epoch=epoch, threshod=threshod, alpha=0.5, gamma=2)
#         loss2 = self.Dice_loss(pred=pred, target=target, epoch=epoch, threshod=threshod)
#         return loss1 + loss2



def Softmax_Focal_Loss(pred, target, epoch=None, threshod=0, alpha=0.5, gamma=2):
    """
    pred : [B, 2, H, W]
    target : [B, 1, H, W]
    """

    # 调整pred的大小
    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=True)

    target = target.squeeze(1)
    target = target.long()

    # 对预测输出应用softmax函数
    p = F.softmax(pred, dim=1)

    # 计算交叉熵损失
    ce_loss = F.cross_entropy(pred, target, reduction="none")

    if epoch is not None and epoch >= threshod:
        prob = p[:, 1:2, :, :].squeeze(1)
        mask = torch.abs(prob - 0.5) > (0.008 * epoch)
        ce_loss = ce_loss * mask

    # 计算p_t，即softmax输出和真实目标的加权和
    p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
    # 计算focal loss
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        # 计算alpha_t，即正负样本的加权和
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        # 对loss进行加权
        loss = alpha_t * loss

    loss = loss.mean()

    # 返回计算得到的损失值
    return loss

def Dice_loss( pred, target, epoch=None, threshod=0):
    """输入pred为[b,2,h,w],target为[b,1,h,w]"""
    smooth = 1e-6
    # 调整pred的大小
    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=True)
    target = target.squeeze(1)
    target = target.long()
    pred = torch.softmax(pred, dim=1)[:, 1]  # 将预测值转换为类别为1的概率
    if epoch is not None and epoch >= threshod:
        mask = torch.abs(pred - 0.5) > (0.008 * epoch)
        pred = pred * mask
        target = target * mask

    intersection = torch.sum(pred * target, dim=(1, 2))  # 计算交集
    union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))  # 计算联合集
    dice_coefficient = (2. * intersection + smooth) / (union + smooth)  # 计算Dice系数
    dice_loss = 1 - dice_coefficient.mean()  # 计算Dice损失，即1 - Dice系数的平均值

    return dice_loss

def MyLoss(pred, target, epoch=None, threshod=8):
    loss1 = Softmax_Focal_Loss(pred=pred, target=target, epoch=epoch, threshod=threshod, alpha=0.5, gamma=2)
    loss2 = Dice_loss(pred=pred, target=target, epoch=epoch, threshod=threshod)
    return loss1 + loss2

# pred = torch.rand(5, 2, 128, 128)
# target = torch.randint(0, 2, (5, 1, 128, 128))
# loss = Focal_Dice_Loss(pred, target, epoch=5)
# print(loss)