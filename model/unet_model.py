"""
Full assembly of the parts to form the complete network
Original Source: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
from .unet_parts import *

__all__ = ['UNet', 'AttentionUNet']

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks=7,
                 start_channels=16, target_num=1, bilinear=True):
        super(UNet, self).__init__()

        self.n_blocks = n_blocks
        self.target_num = target_num

        # Add padding when n_blocks == 7 so that image size becomes 256 x 149
        if n_blocks == 7:
            self.pad_size = 3
            self.zero_pad = nn.ZeroPad2d((0, 0, self.pad_size, 0))

        self.inc = BasicConv(n_channels, start_channels)

        # Create down blocks
        self.down = nn.ModuleList([])
        for i in range(n_blocks):
            cur_in_ch = start_channels * (2 ** i)
            self.down.append(Down(cur_in_ch, cur_in_ch * 2))

        # Create bridge block
        bridge_channels = start_channels * (2 ** n_blocks)
        self.bridge = BasicConv(bridge_channels, bridge_channels)

        # Create up blocks
        self.up = nn.ModuleList([])
        for i in range(n_blocks, 0, -1):
            cur_in_ch = start_channels * (2 ** i)
            self.up.append(Up(cur_in_ch, (cur_in_ch // 2), bilinear))

        # Create out convolution block
        self.outc = OutConv(start_channels, n_classes)

    @property
    def name(self):
        return 'unet'

    def forward(self, x):
        logits = []

        # If n_blocks == 7, the input tensor becomes 1 by 1 images
        # after last down block, so there is an error when batch_size = 1.
        # Therefore, we do zero padding for H(height) dimension in this case.
        if self.n_blocks == 7:
            x = self.zero_pad(x)

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
        
        # Return to original size when n_blocks == 7
        if self.n_blocks == 7:
            logit = logit[:, :, self.pad_size:, :]

        return logit

class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks=7,
                 start_channels=16, target_num=1, bilinear=True):
        super(AttentionUNet, self).__init__()

        self.n_blocks = n_blocks
        self.target_num = target_num

        if n_blocks == 7:
            self.pad_size = 3
            self.zero_pad = nn.ZeroPad2d((0, 0, self.pad_size, 0))

        factor = 2 if bilinear else 1
        self.inc = BasicConv(n_channels, start_channels)

        # Create down blocks
        self.down = nn.ModuleList([])
        for i in range(n_blocks):
            cur_in_ch = start_channels * (2 ** i)
            self.down.append(Down(cur_in_ch, cur_in_ch * 2))

        # Create bridge block
        bridge_channels = start_channels * (2 ** n_blocks)
        self.bridge = BasicConv(bridge_channels, bridge_channels)

        # Create up blocks
        self.up = nn.ModuleList([])
        for i in range(n_blocks, 0, -1):
            cur_in_ch = start_channels * (2 ** i)
            self.up.append(Up(cur_in_ch, (cur_in_ch // 2), bilinear, attention=True))
        # self.up.append(Up(start_channels * 2, start_channels, bilinear))

        # Create out convolution block
        self.outc = OutConv(start_channels, n_classes)

    @property
    def name(self):
        return 'attn_unet'

    def forward(self, x):
        logits = []

        # If n_blocks == 7, the input tensor becomes 1 by 1 images
        # after last down block, so there is an error when batch_size = 1.
        # Therefore, we do zero padding for H(height) dimension in this case.
        if self.n_blocks == 7:
            x = self.zero_pad(x)

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

        # Return to original size when n_blocks == 7
        if self.n_blocks == 7:
            logits = logits[:, :, :, self.pad_size:, :]

        return logits