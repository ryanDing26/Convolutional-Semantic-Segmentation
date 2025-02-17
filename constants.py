# Paths
save_location = './plots/'
root = './data'
best_model_path='./checkpoints/best_fcn_model.pth'
# Constant
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# Model and corresponding hyperparameters
model_name='fcn' # set to fcn, resnet34, or unet to get different models
num_classes=21
ignore_label=255
num_workers=16
epochs=30
patience=3
# Other related augmentations to the dataset and model parameters and hyperparameters
USE_COSINE_ANNEALING=False
USE_DATA_AUGMENTATION=False
USE_WEIGHTED_LOSS=False