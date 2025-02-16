import torch.nn as nn
import torch

class CustomFCN(nn.Module):
    '''
    A custom fully convolutional network (FCN) for image segmentation.
    '''
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class

        # Encoder Portion
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd7 = nn.BatchNorm2d(2048)
        self.conv8 = nn.Conv2d(2048, 4096, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd8 = nn.BatchNorm2d(4096)
        
        # Decoder Portion
        self.deconv8 = nn.ConvTranspose2d(4096, 2048, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn8 = nn.BatchNorm2d(2048)
        self.deconv7 = nn.ConvTranspose2d(4096, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(1024)
        self.deconv6 = nn.ConvTranspose2d(2048, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=0)
        self.bn6 = nn.BatchNorm2d(512)
        self.deconv5 = nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(128, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Final classifier
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        '''
        Forward pass of the CustomFCN model.
        :param x: Input tensor of shape (batch_size, in_channels, height, width).
        :return: Predicted segmentation map of shape (batch_size, n_class, height, width).
        '''
        self.batch_size = x.shape[0]

        # Encoding portion (downsampling)
        x1 = self.bnd1(self.relu(self.conv1(x)))
        x2 = self.bnd2(self.relu(self.conv2(x1)))
        x3 = self.bnd3(self.relu(self.conv3(x2)))
        x4 = self.bnd4(self.relu(self.conv4(x3)))
        x5 = self.bnd5(self.relu(self.conv5(x4)))
        x6 = self.bnd6(self.relu(self.conv6(x5)))
        x7 = self.bnd7(self.relu(self.conv7(x6)))
        x8 = self.bnd8(self.relu(self.conv8(x7)))

        # Decoder portion (upsampling)
        y8 = self.relu(self.bn8(self.deconv8(x8)))
        y8_cat = torch.cat([y8, x7], dim=1)        
        y7 = self.relu(self.bn7(self.deconv7(y8_cat)))
        y7_cat = torch.cat([y7, x6], dim=1)
        y6 = self.relu(self.bn6(self.deconv6(y7_cat))) 
        y6_cat = torch.cat([y6, x5], dim=1)
        y5 = self.relu(self.bn5(self.deconv5(y6_cat)))
        y5_cat = torch.cat([y5, x4], dim=1)
        y4 = self.relu(self.bn4(self.deconv4(y5_cat)))
        y4_cat = torch.cat([y4, x3], dim=1) 
        y3 = self.relu(self.bn3(self.deconv3(y4_cat)))
        y3_cat = torch.cat([y3, x2], dim=1) 
        y2 = self.relu(self.bn2(self.deconv2(y3_cat)))
        y2_cat = torch.cat([y2, x1], dim=1) 
        y1 = self.relu(self.bn1(self.deconv1(y2_cat)))

        # Final classifier
        score = self.classifier(y1)
        return score