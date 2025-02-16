import torchvision

train_dataset =torchvision.datasets.VOCSegmentation(root='./data',year='2012',download=True,image_set='train')
val_dataset = torchvision.datasets.VOCSegmentation(root='./data',year='2012',download=True,image_set='trainval')
test_dataset = torchvision.datasets.VOCSegmentation(root='./data',year='2012',download=True,image_set='val')