import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import save_location, best_model_path

def init_weights(m):
    '''
    Xavier weight initialization.
    :param m: The layer of weights to initialize.
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias)

def exportModel(model, inputs):
    '''
    Exports the model for inference.
    :param inputs: 
    '''
    model.eval()
    model.load_state_dict(torch.load(best_model_path))
    inputs = inputs.to(model.device)
    output_image = model(inputs)
    model.train()
    return output_image

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

def plots(train_epoch_loss, val_epoch_loss, early_stop=None):
    '''
    Helper function for creating the plots.
    :param train_epoch_loss: a list of training cross-entropy losses for each epoch of training
    :param val_epoch_loss: a list of validataion cross-entropy losses for each epoch of training
    :param early_stop: Early stopping epoch; optional.
    '''
    if not os.path.exists(save_location): os.makedirs(save_location)
        
    fig, ax = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(train_epoch_loss)+1,1)
    ax.plot(epochs, train_epoch_loss, 'r', label='Training Loss')
    ax.plot(epochs, val_epoch_loss, 'g', label='Validation Loss')
    if early_stop: plt.scatter(epochs[early_stop],val_epoch_loss[early_stop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax.set_title('Loss Plots', fontsize=35.0)
    ax.set_xlabel('Epochs', fontsize=35.0)
    ax.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax.legend(loc="upper right", fontsize=35.0)
    plt.savefig(save_location+'loss.png', dpi=300)

    #Save the losses for further offline use
    pd.DataFrame(train_epoch_loss).to_csv(save_location+'trainEpochLoss.csv')
    pd.DataFrame(val_epoch_loss).to_csv(save_location+'valEpochLoss.csv')