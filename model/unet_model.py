"""
Full assembly of the parts to form the complete network
Original Source: https://github.com/milesial/Pytorch-UNet
"""

import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = BasicConv(n_channels, 16)

        self.down1 = Down(  16,   32)
        self.down2 = Down(  32,   64)
        self.down3 = Down(  64,  128)
        self.down4 = Down( 128,  256)
        self.down5 = Down( 256,  512)
        self.down6 = Down( 512, 1024)
        factor = 2 if bilinear else 1
        self.down7 = Down(1024, 2048 // factor)

        self.brdige = BasicConv(2048 // factor, 2048 // factor)

        self.up7 = Up(2048, 1024 // factor, bilinear)
        self.up6 = Up(1024,  512 // factor, bilinear)
        self.up5 = Up( 512,  256 // factor, bilinear)
        self.up4 = Up( 256,  128 // factor, bilinear)
        self.up3 = Up( 128,   64 // factor, bilinear)
        self.up2 = Up(  64,   32 // factor, bilinear)
        self.up1 = Up(  32,   16, bilinear)

        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x0 = self.inc(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)

        x = self.brdige(x7)

        x = self.up7(x, x6)
        x = self.up6(x, x5)
        x = self.up5(x, x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)

        logits = self.outc(x)

        return logits