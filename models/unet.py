import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    '''
    DoubleConv layer, consisting of (Conv2d => BN => ReLU) * 2 with padding=1 to preserve spatial dimensions.
    '''
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        '''
        Performs forward pass of a DoubleConv layer.
        '''
        return self.double_conv(x)

class Down(nn.Module):
    '''
    Performs downscaling with maxpooling, then a DoubleConv layer.
    '''
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
        
    def forward(self, x):
        '''
        Performs forward pass of downscaling with maxpooling, then a DoubleConv layer.
        '''
        return self.maxpool_conv(x)

class Up(nn.Module):
    '''
    Upscaling then double conv. Uses transposed convolution to upsample, then concatenates with the corresponding
    encoder feature map. A center-crop is applied to the skip connection if necessary.
    '''
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up, self).__init__()
        # If bilinear, use the normal upsampling; otherwise, use a transposed convolution to cut channels in half
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if bilinear else nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
            
    def forward(self, x1, x2):
        '''
        Performs forward pass of an upscaling then double convolution layer.
        :param x1: The feature map from the decoder that needs upsampling.
        :param x2: The corresponding feature map from the encoder for the skip connection.
        :returns: The output feature map after concatenation and double convolution.
        '''
        x1 = self.up(x1)

        # Compute differences in spatial dimensions (if any)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Center-crop x2 to match x1 size if needed
        x2 = x2[:, :, diffY // 2 : x2.size()[2] - diffY // 2, diffX // 2 : x2.size()[3] - diffX // 2]

        # Concatenate along the channels dimension
        x = torch.cat([x2, x1], dim=1)
        
        # Apply a double convolution
        return self.conv(x)

class OutConv(nn.Module):
    '''
    Final 1×1 convolution to map features to n_class channels.
    '''
    def __init__(self, in_channels, n_class):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_class, kernel_size=1)
        
    def forward(self, x):
        '''
        Performs forward pass of a 1x1 convolution.
        :param x: The input feature map with shape (batch_size, in_channels, height, width).
        :returns: The output feature map with shape (batch_size, n_class, height, width).
        '''
        return self.conv(x)

class UNet(nn.Module):
    '''
    UNet architecture for image segmentation.
    '''
    def __init__(self, n_class, in_channels=3, bilinear=False):
        super(UNet, self).__init__()
        self.n_class = n_class
        self.in_channels = in_channels
        self.bilinear = bilinear
        
        self.inc   = DoubleConv(self.in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_class)

    def forward(self, x):
        '''
        Performs a forward pass of a UNet architecture.
        :param x: input into UNet input tensor of shape (batch_size, in_channels, height, width).
        :returns: Predicted segmentation map of shape (batch_size, n_class, height, width).
        '''
        # Encoder portion (downsampling)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder portion (upsampling)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # 1x1 convolution to map to n_class channels
        logits = self.outc(x)
        return logits