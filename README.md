# Convolutional Semantic Segmentation

Here, we use three different models:

- A custom fully convolutional network
- The UNet architecture, commonly used for medical image segmentation
- The ResNet34 pre-trained architecture, originally used for the ImageNet dataset

in order to perform image segmentation with the PASCAL-VOC 2012 dataset. We achieve a pixel accuracy of **85%** with our best-trained model using just 20 epochs of training!

## Running the Model

To train your own model, one must first run `python3 download.py` to download the necessary datafiles needed for the training phase. Then, it is as easy as running `python3 train.py` and waiting for your model to be saved in `./checkpoints/best_model.pth` as well as different plots of the training process to be located in `./plots/`.

In order to change the model used, one can change the `model` variable located in `constants.py` by following the guidance of the comments in the file.

## Fine Tuning

Many of the hyperparameters of the model can be found in `constant.py`, and can be tweaked to the constraints of your application.

## Visualizing Predictions

The `visualize.ipynb` contains many of the utility methods and workflow needed to evaluate any trained models by taking a sample of your predicted masks on a particular dataset and comparing it side-by-side with the ground-truth masks semantically segmented in the dataset.