import os
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