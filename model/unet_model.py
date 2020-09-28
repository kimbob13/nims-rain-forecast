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
                 start_channels=16, pos_dim=0, bilinear=False,
                 batch_size=1):
        super(UNet, self).__init__()

        self.n_blocks = n_blocks

        self.inc = BasicConv(n_channels + pos_dim, start_channels)

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
            self.up.append(Up(cur_in_ch, (cur_in_ch // 2), bilinear=bilinear))

        # Create out convolution block
        self.outc = OutConv(start_channels, n_classes)

        self.learnable_pos = None
        if pos_dim > 0:
            self.learnable_pos = nn.Parameter(torch.zeros(batch_size, pos_dim, 512, 512), requires_grad=True)

    @property
    def name(self):
        return 'unet'

    def forward(self, x):
        logits = []

        if self.learnable_pos != None:
            x = torch.cat([x, self.learnable_pos], dim=1)

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

        return logit

class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks=7,
                 start_channels=16, pos_dim=0, bilinear=False,
                 batch_size=1):
        super(AttentionUNet, self).__init__()

        self.n_blocks = n_blocks

        factor = 2 if bilinear else 1
        self.inc = BasicConv(n_channels + pos_dim, start_channels)

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

        # Create out convolution block
        self.outc = OutConv(start_channels, n_classes)

        self.learnable_pos = None
        if pos_dim > 0:
            self.learnable_pos = nn.Parameter(torch.zeros(batch_size, pos_dim, 512, 512), requires_grad=True)

    @property
    def name(self):
        return 'attn_unet'

    def forward(self, x):
        logits = []

        if self.learnable_pos != None:
            x = torch.cat([x, self.learnable_pos], dim=1)

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

        return logit
