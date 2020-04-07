"""
Parts of the U-Net model
Original Source: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BasicConv', 'DoubleConv', 'Down', 'Up', 'OutConv']

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.basic_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual.add_module("res_conv",
                                     nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size=1,
                                               padding=0))

    def forward(self, x):
        out = self.basic_conv(x)

        # Residual connection
        x = self.residual(x)
        out = torch.add(x, out)

        return out

class DoubleConv(nn.Module):
    """
    ([BN] => LReLU => convolution) * 2
    In downsampling, first convolution is changed to MaxPool
    """
    def __init__(self, in_channels, out_channels,
                 mid_channels=None, down=True):
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential()
        self.double_conv.add_module("bn1", nn.BatchNorm2d(in_channels))
        self.double_conv.add_module("lrelu1", nn.LeakyReLU(inplace=True))

        if down:
            self.double_conv.add_module("maxpool1", nn.MaxPool2d(2))
            self.double_conv.add_module("bn2", nn.BatchNorm2d(in_channels))
            self.double_conv.add_module("lrelu2", nn.LeakyReLU(inplace=True))
            self.double_conv.add_module("conv2",
                                        nn.Conv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=3,
                                                  padding=1))
        else:
            self.double_conv.add_module("conv1",
                                        nn.Conv2d(in_channels,
                                                  mid_channels,
                                                  kernel_size=3,
                                                  padding=1))
            self.double_conv.add_module("bn2", nn.BatchNorm2d(mid_channels))
            self.double_conv.add_module("lrelu2", nn.LeakyReLU(inplace=True))
            self.double_conv.add_module("conv2",
                                        nn.Conv2d(mid_channels,
                                                  out_channels,
                                                  kernel_size=3,
                                                  padding=1))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels, down=True)

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            self.residual.add_module("res_conv",
                                     nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size=1,
                                               padding=0))
            self.residual.add_module("res_maxpool", nn.MaxPool2d(2))

    def forward(self, x):
        out = self.conv(x)
        to_upsample = out

        # Residual connection
        x = self.residual(x)
        out = torch.add(x, out)

        return out, to_upsample


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions
        # to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels,
                                   out_channels // 2,
                                   mid_channels=(in_channels // 2),
                                   down=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels, out_channels, down=False)

        self.residual = nn.Sequential()
        if in_channels != out_channels:
            if bilinear:
                self.residual.add_module("res_conv",
                                         nn.Conv2d(in_channels,
                                                   out_channels // 2,
                                                   kernel_size=1,
                                                   padding=0))
            else:
                self.residual.add_module("res_conv",
                                         nn.Conv2d(in_channels,
                                                   out_channels,
                                                   kernel_size=1,
                                                   padding=0))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)

        out = self.conv(x)

        # Residual connection
        x = self.residual(x)
        out = torch.add(x, out)

        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)