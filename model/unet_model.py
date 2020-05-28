"""
Full assembly of the parts to form the complete network
Original Source: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks=7,
                 start_channels=16, target_num=1, bilinear=True):
        super(UNet, self).__init__()

        self.target_num = target_num

        factor = 2 if bilinear else 1
        self.inc = BasicConv(n_channels, start_channels)

        # Create down blocks
        self.down = nn.ModuleList([])
        for i in range(n_blocks - 1):
            cur_in_ch = start_channels * (2 ** i)
            self.down.append(Down(cur_in_ch, cur_in_ch * 2))

        bridge_channels = start_channels * (2 ** n_blocks)
        self.down.append(Down(bridge_channels // 2,
                              bridge_channels // factor))

        # Create bridge block
        self.bridge = BasicConv(bridge_channels // factor,
                                bridge_channels // factor)

        # Create up blocks
        self.up = nn.ModuleList([])
        for i in range(n_blocks, 1, -1):
            cur_in_ch = start_channels * (2 ** i)
            self.up.append(Up(cur_in_ch, (cur_in_ch // 2) // factor,
                              bilinear))
        self.up.append(Up(start_channels * 2, start_channels, bilinear))

        # Create out convolution block
        self.outc = OutConv(start_channels, n_classes)

    @property
    def name(self):
        return 'unet'

    def forward(self, x):
        logits = []

        for _ in range(self.target_num):
            out = self.inc(x)

            # Long residual list for Up phase
            long_residual = []
            long_residual.append(out.clone())

            # Down blocks
            for down_block in self.down:
                out = down_block(out)
                long_residual.append(out.clone())

            # Bridge block
            out = self.bridge(out)

            # Up blocks
            for i, up_block in enumerate(self.up):
                out = up_block(out, long_residual[-1 * (i + 2)])

            logit = self.outc(out)
            logits.append(logit)

        # Change logits shape to NS'CHW
        # S': # of targets, C: # of class for each target
        logits = torch.stack(logits).permute(1, 0, 2, 3, 4)

        return logits