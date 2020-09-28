"""
Parts of the U-Net model
Original Source: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BasicConv', 'Down', 'Up', 'OutConv', 'LearnablePosition']

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConv, self).__init__()

        # Main block
        self.basic_conv = nn.Sequential()
        self.basic_conv.add_module('basic_conv1',
                                   nn.Conv2d(in_channels, out_channels,
                                             kernel_size=3, padding=1, bias=False))
        self.basic_conv.add_module('basic_bn', nn.BatchNorm2d(out_channels))
        self.basic_conv.add_module('basic_relu', nn.LeakyReLU(inplace=True))
        self.basic_conv.add_module('basic_conv2',
                                   nn.Conv2d(out_channels, out_channels,
                                             kernel_size=3, padding=1, bias=False))

        # Residual block
        self.residual = nn.Sequential()
        self.residual.add_module("res_conv",
                                 nn.Conv2d(in_channels, out_channels,
                                           kernel_size=1, padding=0, bias=False))
        self.residual.add_module("res_bn", nn.BatchNorm2d(out_channels))

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
        super(DoubleConv, self).__init__()

        # Main block
        self.double_conv = nn.Sequential()
        self.residual = nn.Sequential()

        if down:
            self.double_conv.add_module("down_conv1",
                                        nn.Conv2d(in_channels, out_channels,
                                                  kernel_size=3, padding=1, bias=False))
            self.double_conv.add_module("down_bn1",
                                        nn.BatchNorm2d(out_channels))
            self.double_conv.add_module("down_relu1",
                                        nn.ReLU(inplace=True))
            self.double_conv.add_module("down_conv2",
                                        nn.Conv2d(out_channels, out_channels,
                                                  kernel_size=3, padding=1, bias=False))
            self.double_conv.add_module("down_bn2",
                                        nn.BatchNorm2d(out_channels))
            self.double_conv.add_module("down_relu2",
                                        nn.ReLU(inplace=True))
            self.double_conv.add_module("down_maxpool1", nn.MaxPool2d(2))

            # Residual connection
            self.residual.add_module("down_res_conv",
                                    nn.Conv2d(in_channels, out_channels,
                                            kernel_size=1, stride=2,
                                            padding=0, bias=False))
        else:
            if not mid_channels:
                mid_channels = out_channels

            self.double_conv.add_module("up_conv1",
                                        nn.Conv2d(in_channels, mid_channels,
                                                  kernel_size=3, padding=1, bias=False))
            self.double_conv.add_module("up_bn1", nn.BatchNorm2d(mid_channels))
            self.double_conv.add_module("up_relu1",
                                        nn.ReLU(inplace=True))
            self.double_conv.add_module("up_conv2",
                                        nn.Conv2d(mid_channels, out_channels,
                                                  kernel_size=3, padding=1, bias=False))
            self.double_conv.add_module("up_bn2",
                                        nn.BatchNorm2d(out_channels))
            self.double_conv.add_module("up_relu2",
                                        nn.ReLU(inplace=True))

            # Residual connection
            self.residual.add_module("up_res_conv",
                                     nn.Conv2d(in_channels, out_channels,
                                               kernel_size=1, padding=0, bias=False))

    def forward(self, x):
        out = self.double_conv(x)

        # Residual connection
        x = self.residual(x)
        out = torch.add(x, out)

        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels, down=True)

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,
                 learnable_pos=None, bilinear=True, attention=False):
        super(Up, self).__init__()

        # Learnable position setting
        pos_dim = in_channels - (out_channels * 2)
        if pos_dim == 0:
            assert learnable_pos == None
        
        self.learnable_pos = learnable_pos

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.upsample = nn.Sequential()
        if bilinear:
            self.upsample.add_module('upsample', nn.Upsample(scale_factor=2,
                                                             mode='bilinear',
                                                             align_corners=True))
            self.upsample.add_module('upsample_conv', nn.Conv2d(in_channels - pos_dim, out_channels,
                                                                kernel_size=3, stride=1,
                                                                padding=1, bias=False))
        else:
            self.upsample.add_module('upsample', nn.ConvTranspose2d(in_channels - pos_dim,
                                                                    out_channels,
                                                                    kernel_size=2,
                                                                    stride=2))

        self.attn_gate = None
        if attention:
            self.attn_gate = AttentionGate(out_channels, out_channels, out_channels // 2)

        self.double_conv = DoubleConv(in_channels, out_channels,
                                      mid_channels=(in_channels // 2), down=False)

    def forward(self, x, x_res):
        # Upsample
        x = self.upsample(x)

        # input is CHW
        diffY = torch.tensor([x_res.size()[2] - x.size()[2]])
        diffX = torch.tensor([x_res.size()[3] - x.size()[3]])

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        # Attention gate if any
        if self.attn_gate:
            x_res = self.attn_gate(x, x_res)

        # Concat skip connection with previous feature map
        out = torch.cat([x_res, x], dim=1)

        # Concat with learnalble position if any
        if self.learnable_pos:
            out = self.learnable_pos(out)

        out = self.double_conv(out)

        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LearnablePosition(nn.Module):
    def __init__(self, batch_size, pos_dim, height, width):
        super(LearnablePosition, self).__init__()
        self.learnable_pos = nn.Parameter(torch.zeros(batch_size, pos_dim, height, width), requires_grad=True)

    def forward(self, x):
        x = torch.cat([x, self.learnable_pos], dim=1)

        return x

class AttentionGate(nn.Module):
    def __init__(self, feature_x, feature_g, inter):
        super(AttentionGate, self).__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(feature_g, inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter)
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(feature_x, inter, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(inter)
        )

        self.w_psi = nn.Sequential(
            nn.Conv2d(inter, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.w_psi(psi)

        return x * psi