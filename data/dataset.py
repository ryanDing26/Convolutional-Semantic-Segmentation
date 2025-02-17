import os
import torch
import random
import numpy as np
import torchvision.transforms as transforms

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data
from constants import num_classes, ignore_label, root

'''
Color Map:
0: background, 1: aeroplane, 2: bicycle, 3: bird, 4: boat, 5: bottle, 6: bus, 7: car, 
8: cat, 9: chair, 10: cow, 11: diningtable, 12: dog, 13: horse, 14: motorbike, 
15: person, 16: potted plant, 17: sheep, 18: sofa, 19: train, 20: tv/monitor
'''

# Color palette for visualization (if needed)
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]

def make_dataset(mode):
    '''
    Creates a list of tuples (image_path, mask_path) for a given dataset mode.
    :param mode: The mode of the dataset, either 'train', 'val', or 'test'.
    :returns: list of tuples, each containing paths (image_path, mask_path).
    '''
    items = []
    img_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    mask_path = os.path.join(root, 'VOCdevkit', 'VOC2012', 'SegmentationClass')\
    
    if mode == 'train':
        data_list = [l.strip('\n') for l in open(os.path.join(root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'train.txt')).readlines()]
    elif mode == 'val':
        data_list = [l.strip() for l in open(os.path.join(root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'trainval.txt')).readlines()]
    else:
        data_list = [l.strip() for l in open(os.path.join(root, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation', 'val.txt')).readlines()]

    for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    return items

class VOC(data.Dataset):
    '''
    A custom dataset class for the VOC Dataset.
    '''
    def __init__(self, mode, transform=None, target_transform=None):
        '''
        Initializes VOC Dataset.
        :param mode: Mode of the dataset ('train', 'val', etc.).
        :param transform: callable Transform to be applied to the images; optional.
        :param target_transform: callable Transform to be applied to the masks; optional.
        '''
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0: raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.width = 224
        self.height = 224

    def __getitem__(self, index):
        '''
        Gets the index-th item of the dataset.
        :param index: index of the item in the dataset.
        :returns: index-th item of the dataset.
        '''
        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB').resize((self.width, self.height))
        mask = Image.open(mask_path).resize((self.width, self.height))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        mask[mask==ignore_label]=0
        return img, mask

    def __len__(self):
        '''
        Returns the length of the dataset.
        :return: length of the dataset.
        '''
        return len(self.imgs)

class MaskToTensorDefault(object):
    '''
    Converts mask to tensor while ensuring it uses the same augmentations as the image.
    '''
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class MaskToTensorRandom(object):
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

class RandomTransformSync:
    '''
    Applies a random transformation to images in the dataset, ensuring that the same random transformation is applied to both the image and mask.
    '''
    def __init__(self, apply_hflip=True, apply_rotation=True, apply_crop=True, image_size=(224, 224), rotation_range=10, crop_scale=(0.8, 1.0)):
        self.apply_hflip = apply_hflip
        self.apply_rotation = apply_rotation
        self.apply_crop = apply_crop
        self.image_size = image_size
        self.rotation_range = rotation_range
        self.crop_scale = crop_scale
        self.AUG_PARAMS = {}
    
    def __call__(self, img):
        '''
        Applies the same random transformations to the image.
        '''
        if img.mode == "RGB":  # Image transform (input)
            # Sample new random transformations
            self.AUG_PARAMS["hflip"] = random.random() > 0.5 if self.apply_hflip else False
            self.AUG_PARAMS["rotation"] = random.uniform(-self.rotation_range, self.rotation_range) if self.apply_rotation else 0
            self.AUG_PARAMS["crop_params"] = transforms.RandomResizedCrop.get_params(img, scale=self.crop_scale, ratio=(0.75, 1.33)) if self.apply_crop else None

            # Apply transformations to the image
            if self.AUG_PARAMS["hflip"]:
                img = TF.hflip(img)
            if self.apply_rotation:
                img = TF.rotate(img, self.AUG_PARAMS["rotation"])
            if self.apply_crop:
                i, j, h, w = self.AUG_PARAMS["crop_params"]
                img = TF.resized_crop(img, i, j, h, w, self.image_size)
        
        else:  # Mask transform (target)
            # Ensure stored params are used for mask
            if self.AUG_PARAMS.get("hflip", False):
                img = TF.hflip(img)
            if self.apply_rotation:
                img = TF.rotate(img, self.AUG_PARAMS["rotation"], interpolation=TF.InterpolationMode.NEAREST)
            if self.apply_crop:
                i, j, h, w = self.AUG_PARAMS["crop_params"]
                img = TF.resized_crop(img, i, j, h, w, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        return img