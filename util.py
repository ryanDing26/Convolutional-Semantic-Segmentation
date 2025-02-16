import numpy as np
import torch

def iou(pred, target, n_classes=21):
    '''
    Computes mean Intersection over Union (mIoU) over n_classes.
    :param pred: Tensor of predicted classes.
    :param target: Tensor of ground-truth classes.
    :param n_classes: number of classes that can be predicted; optional.
    :returns: mIOU score over all classes.
    '''
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(n_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        ious.append(np.nan if not union else intersection / union)

    # Average only over classes with non-zero union.
    ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(ious) if ious else 0.0

def pixel_acc(pred, target):
    '''
    Computes pixel-wise accuracy.
    :param pred: Tensor of predicted classes.
    :param target: Tensor of ground-truth classes.
    :returns: Pixel-wise accuracy.
    '''
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total