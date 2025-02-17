import gc
import os
import time
import util
import torch
import data.dataset
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
from models import resnet34, unet, custom_fcn
from constants import USE_COSINE_ANNEALING, USE_DATA_AUGMENTATION, USE_WEIGHTED_LOSS
from constants import best_model_path, num_classes, model_name, num_workers, patience, epochs, mean_std

epochs_without_improvement = 0

def train(model, criterion, optimizer, epochs, train_loader, val_loader, scheduler=None):
    '''
    Trains the model over a certain number of epochs.
    :param train_loader: DataLoader for the training data.
    :param val_loader: DataLoader for the validation data.
    '''
    global epochs_without_improvement
    best_loss = 100.0
    best_iou = 0.0
    train_losses = [] # for plotting
    val_losses = [] # for plotting
    stopped = False
    for epoch in range(epochs):
        epoch_loss = 0
        batches = 0
        model.train()
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss
            batches += 1
            loss.backward()
            optimizer.step()
            
            if iter % 20 == 0:
                print(f"Epoch {epoch}, Iter {iter}, Loss: {loss.item():.4f}")
        print(f"Finished epoch {epoch}, time elapsed: {time.time() - ts:.2f} sec")
        
        if USE_COSINE_ANNEALING: scheduler.step()
            
        train_losses.append((epoch_loss / batches).cpu().item())
        
        val_loss, val_miou = val(model, criterion, epoch, val_loader)
        val_losses.append(val_loss.item())

        # Checkpoint model if mIoU improves
        if val_miou > best_iou:
            best_ioue = val_miou
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch}: Improved mIOU to {val_miou:.4f}; model saved (best IOU-based).")
             
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            print(f"Epoch {epoch}: Loss improved.")
        else:
            epochs_without_improvement += 1
            print(f"Epoch {epoch}: No improvement. {epochs_without_improvement} epochs without improvement.")
        
        if epochs_without_improvement >= patience:
            stopped = True
            print("Early stopping triggered. Training terminated.")
            break
    
    # Plot the training and validation losses across epochs
    util.plots(train_losses, val_losses, len(train_losses) - 1 if not stopped else len(train_losses) - 3)

def val(model, criterion, epoch, val_loader):
    '''
    Evaluate model performance on the validation dataset.
    :param epoch: number epoch of training model is on.
    :param val_loader: DataLoader for the validation data.
    :returns: average Cross-Entropy loss and mIOU across batches.
    '''
    model.eval()
    losses = []
    mean_iou_scores = []
    accuracies = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            mean_iou = util.iou(preds, labels)
            acc = util.pixel_acc(preds, labels)
            mean_iou_scores.append(mean_iou)
            accuracies.append(acc)
    avg_loss = np.mean(losses)
    avg_iou = np.mean(mean_iou_scores)
    avg_acc = np.mean(accuracies)
    print(f'Epoch {epoch}: Validation Loss: {avg_loss:.4f}, Mean IoU: {avg_iou:.4f}, Pixel Acc: {avg_acc:.4f}')
    model.train()
    return avg_loss, avg_iou

def modelTest(model, criterion, test_loader):
    '''
    Evaluates model performance on the testing dataset.
    '''
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    losses = []
    mean_iou_scores = []
    accuracies = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(model.device)
            labels = labels.to(model.device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds = torch.argmax(outputs, dim=1)
            mean_iou = util.iou(preds, labels)
            acc = util.pixel_acc(preds, labels)
            mean_iou_scores.append(mean_iou)
            accuracies.append(acc)
    avg_loss = np.mean(losses)
    avg_iou = np.mean(mean_iou_scores)
    avg_acc = np.mean(accuracies)
    print(f"Test Loss: {avg_loss:.4f}, Mean IoU: {avg_iou:.4f}, Pixel Acc: {avg_acc:.4f}")
    model.train()

def main():
    train_dataset = data.dataset.VOC('train', transform=input_transform, target_transform=target_transform)
    val_dataset   = data.dataset.VOC('val', transform=input_transform, target_transform=target_transform)
    test_dataset  = data.dataset.VOC('test', transform=input_transform, target_transform=target_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

    if USE_DATA_AUGMENTATION:
        input_transform = standard_transforms.Compose([
            data.dataset.RandomTransformSync(),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        target_transform = data.dataset.MaskToTensorRandom()
    else:
        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
        target_transform = data.dataset.MaskToTensorDefault()

    # Set a Cosine Annealing LR scheduler if enabled.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Use class-balanced loss weighting for Cross-Entropy Loss if enabled.
    if USE_WEIGHTED_LOSS:
        beta = 0.99
        class_counts = np.zeros(num_classes)
        for imgs, masks in train_loader:
            masks_np = masks.numpy().flatten()
            for cls in range(num_classes):
                class_counts[cls] += np.sum(masks_np == cls)
        effective_num = torch.tensor([(1 - beta ** c) / (1 - beta) for c in class_counts], dtype=torch.float32)
        weights = 1 / effective_num
        class_weights = weights / weights.sum()
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Choose a model; by default the custom FCN is chosen.
    if model_name == 'resnet34':
        model = resnet34.ResNet34(n_class=num_classes)
    elif model_name == 'unet':
        model = unet.UNet(n_class=num_classes)
    elif model_name == 'fcn':
        model = custom_fcn.CustomFCN(n_class=num_classes)
    else:
        model = custom_fcn.CustomFCN(n_class=num_classes)
    
    # Initialize weights with Xavier initialization
    model.apply(util.init_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Set an optimizer to backpropagate model with
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Measure initial model validation loss and mIOU
    init_val_loss, init_val_iou = val(model, criterion, 0)
    print(f'Initial validation IOU Before Training: {init_val_iou} | Initial validation loss: {init_val_loss}')

    # Train and evaluate model on the testing set
    train(model, criterion, optimizer, epochs, device, train_loader, val_loader, scheduler=scheduler if USE_COSINE_ANNEALING else None)
    modelTest(model, criterion, test_loader)

    # Reclaim all unused memory and empty cache.
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__': main()