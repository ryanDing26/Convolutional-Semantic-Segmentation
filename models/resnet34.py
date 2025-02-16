import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet34(nn.Module):
    '''
    A custom ResNet34-based architecture for image segmentation using transfer learning.
    
    This model uses a pretrained ResNet34 backbone, freezes its layers, and adds a 
    decoder part consisting of transposed convolutions for upsampling. The final 
    segmentation output is produced through a 1x1 convolution to predict the desired 
    number of output classes.
    '''
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # Load pretrained ResNet34 model
        self.resnet34 = models.resnet34(pretrained=True)

        # Freeze parameters of actual ResNet34
        for param in self.resnet34.parameters(): param.requires_grad = False

        # Encoder portion (last 2 layers of ResNet34 removed to adhere to decoder input dimensions)
        self.encoder = nn.Sequential(*list(self.resnet34.children())[:-2])

        # Decoder portion
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, 
                                          padding=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, 
                                          padding=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, 
                                          padding=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, 
                                          padding=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, 
                                          padding=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)

        # Final classifier
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

        # 1x1 convolution to reduce channels before concatenation
        self.conv1x1 = nn.Conv2d(64, 32, kernel_size=1)

        # ReLU activation function
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        Performs a forward pass of the ResNet34-based segmentation model.
        :param x: Input tensor of shape (batch_size, in_channels, height, width).
        :returns: Predicted segmentation map of shape (batch_size, n_class, height, width).
        '''
        # Encoder portion (downsampling)
        x1 = self.resnet34.maxpool(self.resnet34.relu(self.resnet34.bn1(self.resnet34.conv1(x))))
        x2 = self.resnet34.layer1(x1)
        x3 = self.resnet34.layer2(x2)
        x4 = self.resnet34.layer3(x3)
        x5 = self.resnet34.layer4(x4)
        
        # Decoder portion (upsampling)
        y1 = self.relu(self.bn1(self.deconv1(x5)))
        y1_cat = torch.cat([y1, x4], dim=1)
        y2 = self.relu(self.bn2(self.deconv2(y1_cat)))
        y2_cat = torch.cat([y2, x3], dim=1)
        y3 = self.relu(self.bn3(self.deconv3(y2_cat)))
        y3_cat = torch.cat([y3, x2], dim=1)
        y4 = self.relu(self.bn4(self.deconv4(y3_cat)))

        # x1 needs to be upsampled and have its channels halved to be concatenated to y4
        x1_upsampled = F.interpolate(x1, size=y4.shape[2:], mode='bilinear', align_corners=True)
        x1_reduced = self.conv1x1(x1_upsampled)
        y4_cat = torch.cat([y4, x1_reduced], dim=1)
        y5 = self.relu(self.bn5(self.deconv5(y4_cat)))

        # Final classifier
        score = self.classifier(y5)
        return score