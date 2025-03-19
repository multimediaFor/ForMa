import warnings

warnings.filterwarnings("ignore")
import numpy as np
import cv2
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F


def F1_IoU(pred, target):
    # 调整pred的大小
    pred = pred.unsqueeze(0).unsqueeze(0)
    target = target.unsqueeze(0).unsqueeze(0)
    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode='nearest')

    target = target.view(-1)  # 将target展平为一维数组
    pred = pred.view(-1)  # 将pred展平为一维数组
    tp = (target * pred).sum().float()  # 计算TP
    fp = ((1 - target.float()) * pred.float()).sum().float()  # 计算FP
    fn = (target.float() * (1 - pred.float())).sum().float()  # 计算FN
    precision = tp / (tp + fp + 1e-8)  # 计算精确率
    recall = tp / (tp + fn + 1e-8)  # 计算召回率
    f1 = 2 * precision * recall / (precision + recall + 1e-8)  # 计算F1值
    iou = tp / (tp + fp + fn + 1e-8)  # iou
    return f1.item(), iou.item()


def thresholding(x, thres=0.5):
    x[x <= int(thres * 255)] = 0
    x[x > int(thres * 255)] = 255
    return x


def metrics(pred_dir, mask_dir, dataset_name):
    f1_list, iou_list = [], []
    img_list = os.listdir(pred_dir)
    gt_list = os.listdir(mask_dir)

    for i in tqdm(range(len(img_list))):
        pred_path = os.path.join(pred_dir, img_list[i])
        gt_path = os.path.join(mask_dir, gt_list[i])#gt_list[i]
        print(pred_path, gt_path)

        image = thresholding(cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE))
        image = image.astype('float') / 255.
        image = torch.from_numpy(image).float()

        mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        mask = mask.astype('float') / 255.
        mask = torch.from_numpy(mask).float()

        f1, iou = F1_IoU(image, mask)
        f1_list.append(f1)
        iou_list.append(iou)
    print(dataset_name)
    print("f1:", np.mean(np.array(f1_list)), "iou:", np.mean(np.array(iou_list)))

    return np.mean(np.array(f1_list)), np.mean(np.array(iou_list))


def calculate_F1_IoU(pred_path, gt_path):
    # 读取预测图像并进行二值化处理
    pred_image = thresholding(cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE))
    pred_image = pred_image.astype('float') / 255.0  # 归一化到0~1
    pred_image = torch.from_numpy(pred_image).float()

    # 读取真实标签（GT）图像
    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    gt_image = gt_image.astype('float') / 255.0  # 归一化到0~1
    gt_image = torch.from_numpy(gt_image).float()

    # 计算F1和IoU
    f1, iou = F1_IoU(pred_image, gt_image)

    return f1, iou

if __name__ == '__main__':

    # dataset_name="AutoSplice"
    # pred_dir=fr"/home/zxmmd/ViFoNet/Test_Result_01172130/AutoSplice"
    # mask_dir=fr"/mnt/h/Academic/image_forgery/datasets/{dataset_name}/gt"
    # metrics(pred_dir,mask_dir,dataset_name)
    #
    dataset_name = "DEFACTO12K-test"
    pred_dir = fr"/home/zxmmd/ViFoNet/Test_Result_01172130/DEFACTO12K-test_6000"
    mask_dir = fr"/mnt/h/Academic/image_forgery/datasets/{dataset_name}/gt"
    metrics(pred_dir, mask_dir, dataset_name)


    #
