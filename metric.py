import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,precision_recall_curve

def calc_fixed_f1_iou(pred, target):
    pred=pred.unsqueeze(dim=0)
    target=target.squeeze().unsqueeze(dim=0)
    b, n, h, w = pred.size()
    bt, ht, wt = target.size()
    if h != ht or w != wt:
        pred = F.interpolate(pred, size=(ht, wt), mode="bilinear", align_corners=True)
    pred = torch.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred, dim=1)
    # F1
    tp = torch.sum(pred_labels[target == 1] == 1).float()
    fp = torch.sum(pred_labels[target == 0] == 1).float()
    fn = torch.sum(pred_labels[target == 1] == 0).float()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)

    # IoU
    intersection = torch.sum(pred_labels[target == 1] == 1).float()
    union = torch.sum(pred_labels + target >= 1).float()

    iou = intersection / (union + 1e-6)

    return f1_score, iou


def calc_best_f1_auc(y_pred, y_true):
    '''
    y_pred:[b,2,w,h]
    y_true:[b,w,h]
    '''
    b, n, h, w = y_pred.size()
    bt, ht, wt = y_true.size()
    if h != ht or w != wt:
        y_pred = F.interpolate(y_pred, size=(ht, wt), mode="bilinear", align_corners=True)

    y_pred = torch.softmax(y_pred, dim=1)[:, 1]
    batchsize = y_true.shape[0]
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        f1_best, auc = 0, 0

        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        for i in range(batchsize):

            true = y_true[i].flatten()
            true = true.astype(int)
            pred = y_pred[i].flatten()
            # F1
            precision, recall, thresholds = precision_recall_curve(true, pred)
            # auc、F1
            try:
                auc += roc_auc_score(true, pred)
            except ValueError:
                pass

            f1_best += max([(2 * p * r) / (p + r + 1e-10) for p, r in
                            zip(precision, recall)])

    return f1_best / batchsize, auc / batchsize