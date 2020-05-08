"""
Full assembly of the parts to form the complete network
Original Source: https://github.com/milesial/Pytorch-UNet
"""

import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,
                 start_channels=16, bilinear=True):
        super(UNet, self).__init__()

        self.inc = BasicConv(n_channels, start_channels)

        self.down1 = Down(start_channels, start_channels * (2 ** 1))
        self.down2 = Down(start_channels * (2 ** 1), start_channels * (2 ** 2))
        self.down3 = Down(start_channels * (2 ** 2), start_channels * (2 ** 3))
        self.down4 = Down(start_channels * (2 ** 3), start_channels * (2 ** 4))
        factor = 2 if bilinear else 1
        self.down5 = Down(start_channels * (2 ** 4),
                          start_channels * (2 ** 5))
        self.down6 = Down(start_channels * (2 ** 5),
                          start_channels * (2 ** 6))
        self.down7 = Down(start_channels * (2 ** 6),
                          start_channels * (2 ** 7) // factor)

        self.bridge = BasicConv((start_channels * (2 ** 7)) // factor,
                                (start_channels * (2 ** 7)) // factor)

        self.up7 = Up(start_channels * (2 ** 7),
                      start_channels * (2 ** 6) // factor,
                      bilinear)
        self.up6 = Up(start_channels * (2 ** 6),
                      start_channels * (2 ** 5) // factor,
                      bilinear)
        self.up5 = Up(start_channels * (2 ** 5),
                      start_channels * (2 ** 4) // factor,
                      bilinear)
        self.up4 = Up(start_channels * (2 ** 4),
                      start_channels * (2 ** 3) // factor,
                      bilinear)
        self.up3 = Up(start_channels * (2 ** 3),
                      start_channels * (2 ** 2) // factor,
                      bilinear)
        self.up2 = Up(start_channels * (2 ** 2),
                      start_channels * (2 ** 1) // factor,
                      bilinear)
        self.up1 = Up(start_channels * (2 ** 1),
                      start_channels,
                      bilinear)

        self.outc = OutConv(start_channels, n_classes)

    def forward(self, x):
        x0 = self.inc(x)

        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)

        x = self.bridge(x7)

        x = self.up7(x, x6)
        x = self.up6(x, x5)
        x = self.up5(x, x4)
        x = self.up4(x, x3)
        x = self.up3(x, x2)
        x = self.up2(x, x1)
        x = self.up1(x, x0)

        logits = self.outc(x)

        return logits