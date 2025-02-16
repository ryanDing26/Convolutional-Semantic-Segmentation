import gc
import os
import time
import util
import torch
import random
import dataset
import multiprocessing
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
from models import resnet34, unet, custom_fcn
from constants import best_model_path, num_classes, model_name, num_workers, save_location, patience, epochs, mean_std, USE_COSINE_ANNEALING, USE_DATA_AUGMENTATION, USE_WEIGHTED_LOSS

def plots(trainEpochLoss, valEpochLoss, early_stop=None):
    '''
    Helper function for creating the plots.
    :param trainEpochLoss:
    :param valEpochLoss:
    :param early_stop: Early stopping epoch; optional.
    '''
    if not os.path.exists(save_location): os.makedirs(save_location)
        
    fig, ax = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax.plot(epochs, trainEpochLoss, 'r', label='Training Loss')
    ax.plot(epochs, valEpochLoss, 'g', label='Validation Loss')
    if early_stop: plt.scatter(epochs[early_stop],valEpochLoss[early_stop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax.set_title('Loss Plots', fontsize=35.0)
    ax.set_xlabel('Epochs', fontsize=35.0)
    ax.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax.legend(loc="upper right", fontsize=35.0)
    plt.savefig(save_location+'loss.png', dpi=300)

    #Save the losses for further offline use
    pd.DataFrame(trainEpochLoss).to_csv(save_location+'trainEpochLoss.csv')
    pd.DataFrame(valEpochLoss).to_csv(save_location+'valEpochLoss.csv')

def init_weights(m):
    '''
    Xavier weight initialization.
    :param m:
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias)

if USE_WEIGHTED_LOSS:
    # -------------------------------
    # Part 4c: Weighted Loss for Class Imbalance
    # Compute class frequencies over the training set and define weights.
    def getClassWeights():
        beta = 0.99
        class_counts = np.zeros(num_classes)
        for imgs, masks in train_loader:
            masks_np = masks.numpy().flatten()
            for cls in range(num_classes):
                class_counts[cls] += np.sum(masks_np == cls)
        # Avoid division by zero; compute inverse frequency.
        # weights = 1.0 / (class_counts + 1e-6)
        # Normalize weights so that sum equals n_class (optional)
        # weights = torch.tensor([class_counts.sum() / (len(class_counts) * c) for c in class_counts], dtype=torch.float32)
        effective_num = torch.tensor([(1 - beta ** c) / (1 - beta) for c in class_counts], dtype=torch.float32)
        weights = 1 / effective_num
        return weights / weights.sum()
    class_weights = getClassWeights()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
else:
    criterion = nn.CrossEntropyLoss()

if USE_COSINE_ANNEALING: scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# ====================================
# Early Stopping Configuration
# ====================================
patience = 3  # Stop training if no improvement in validation mIoU for 5 consecutive epochs
epochs_without_improvement = 0

def train(model, optimizer, epochs, device, train_loader, val_loader):
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
            inputs = inputs.to(device)
            labels = labels.to(device)
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
        
        val_loss, val_miou = val(epoch, val_loader)
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
    plots(train_losses, val_losses, len(train_losses) - 1 if not stopped else len(train_losses) - 3)

def val(epoch, val_loader):
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
            inputs = inputs.to(device)
            labels = labels.to(device)
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

def modelTest(test_loader):
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
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = fcn_model(inputs)
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

def exportModel(inputs):
    '''
    Exports the model for inference.
    :param inputs: 
    '''
    model.eval()
    model.load_state_dict(torch.load(best_model_path))
    inputs = inputs.to(device)
    output_image = model(inputs)
    model.train()
    return output_image

def main():
    train_dataset = dataset.VOC('train', transform=input_transform, target_transform=target_transform)
    val_dataset   = dataset.VOC('val', transform=input_transform, target_transform=target_transform)
    test_dataset  = dataset.VOC('test', transform=input_transform, target_transform=target_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=num_workers)

    if USE_DATA_AUGMENTATION:
        AUG_PARAMS = {}
        class RandomTransformSync:
            '''
            Applies the same random transformation to both the image and mask.
            '''
            def __init__(self, apply_hflip=True, apply_rotation=True, apply_crop=True, image_size=(224, 224), rotation_range=10, crop_scale=(0.8, 1.0)):
                self.apply_hflip = apply_hflip
                self.apply_rotation = apply_rotation
                self.apply_crop = apply_crop
                self.image_size = image_size
                self.rotation_range = rotation_range
                self.crop_scale = crop_scale
            
            def __call__(self, img):
                '''
                Applies the same random transformations to the image.
                '''
                global AUG_PARAMS
        
                if img.mode == "RGB":  # Image transform (input)
                    # Sample new random transformations
                    AUG_PARAMS["hflip"] = random.random() > 0.5 if self.apply_hflip else False
                    AUG_PARAMS["rotation"] = random.uniform(-self.rotation_range, self.rotation_range) if self.apply_rotation else 0
                    AUG_PARAMS["crop_params"] = transforms.RandomResizedCrop.get_params(img, scale=self.crop_scale, ratio=(0.75, 1.33)) if self.apply_crop else None
        
                    # Apply transformations to the image
                    if AUG_PARAMS["hflip"]:
                        img = TF.hflip(img)
                    if self.apply_rotation:
                        img = TF.rotate(img, AUG_PARAMS["rotation"])
                    if self.apply_crop:
                        i, j, h, w = AUG_PARAMS["crop_params"]
                        img = TF.resized_crop(img, i, j, h, w, self.image_size)
                
                else:  # Mask transform (target)
                    # Ensure stored params are used for mask
                    if AUG_PARAMS.get("hflip", False):
                        img = TF.hflip(img)
                    if self.apply_rotation:
                        img = TF.rotate(img, AUG_PARAMS["rotation"], interpolation=TF.InterpolationMode.NEAREST)
                    if self.apply_crop:
                        i, j, h, w = AUG_PARAMS["crop_params"]
                        img = TF.resized_crop(img, i, j, h, w, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
        
                return img
            
        class MaskToTensor(object):
            '''
            A simple transform for segmentation masks. Converts mask to tensor while ensuring it uses the same augmentations as the image.
            '''
            def __call__(self, img):
                '''
                Applies the same stored transformations as the input transform, then converts to tensor.
                :param img:
                :returns: 
                '''
                img = RandomTransformSync()(img)  # Apply stored augmentation parameters
                return torch.from_numpy(np.array(img, dtype=np.int32)).long()
            
        input_transform = standard_transforms.Compose([
            RandomTransformSync(),
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
    else:
        class MaskToTensor(object):
            '''
            Converts mask to tensor while ensuring it uses the same augmentations as the image.
            '''
            def __call__(self, img):
                return torch.from_numpy(np.array(img, dtype=np.int32)).long()
        
        input_transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(*mean_std)
        ])
    
    target_transform = MaskToTensor()

    if model_name == 'resnet34':
        model = resnet34.ResNet34(n_class=num_classes)
    elif model_name == 'unet':
        model = unet.UNet(n_class=num_classes)
    elif model_name == 'fcn':
        model = custom_fcn.CustomFCN(n_class=num_classes)
    else:
        model = custom_fcn.CustomFCN(n_class=num_classes)
    
    model.apply(init_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    init_val_loss, init_val_iou = val(0)
    print(f'Initial validation IOU Before Training: {init_val_iou} | Initial validation loss: {init_val_loss}')

    train(model, optimizer, epochs, device, train_loader, val_loader)
    modelTest(test_loader)

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__': main()