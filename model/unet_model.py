"""
Full assembly of the parts to form the complete network
Original Source: https://github.com/milesial/Pytorch-UNet
"""

import torch
import torch.nn as nn
from .unet_parts import *
import math

__all__ = ['UNet', 'SuccessiveUNet', 'AttentionUNet']

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_blocks, start_channels,
                 pos_loc=0, pos_dim=0, bilinear=False, batch_size=1, use_lcn=False):
        super(UNet, self).__init__()

        # Learnable position related
        pos_loc_max = (2 * n_blocks) + 3
        assert (pos_loc >= 0) and (pos_loc <= pos_loc_max)
        if pos_loc == 0:
            assert pos_dim == 0
        if pos_dim == 0:
            assert pos_loc == 0

        # Model entrance block
        self.inc = nn.Sequential()
        if pos_loc == 1:
            n_channels += pos_dim
            self.inc.add_module('inc_pos', LearnablePosition(batch_size, pos_dim, 781, 602))
        self.inc.add_module('inc', BasicConv(n_channels, start_channels))

        # Create down blocks
        self.down = nn.ModuleList([])
        for i in range(n_blocks):
            cur_in_ch = start_channels * (2 ** i)
            if pos_loc == i + 2:
                cur_in_ch_pos = cur_in_ch + pos_dim
                down_with_pos = nn.Sequential()
                down_with_pos.add_module('down{}_pos'.format(i),
                                         LearnablePosition(batch_size, pos_dim, math.ceil(781 / (2 ** i)), math.ceil(602 / (2 ** i))))
                down_with_pos.add_module('down{}'.format(i), Down(cur_in_ch_pos, cur_in_ch * 2))

                self.down.append(down_with_pos)
            else:
                self.down.append(Down(cur_in_ch, cur_in_ch * 2))

        # Create bridge block
        self.bridge = nn.Sequential()
        bridge_channels = start_channels * (2 ** n_blocks)
        if pos_loc == n_blocks + 2:
            bridge_channels_pos = bridge_channels + pos_dim
            self.bridge.add_module('bridge_pos',
                                   LearnablePosition(batch_size, pos_dim, math.ceil(781 / (2 ** n_blocks)), math.ceil(602 / (2 ** n_blocks))))
            self.bridge.add_module('bridge_conv', BasicConv(bridge_channels_pos, bridge_channels))
        else:
            self.bridge.add_module('bridge_conv', BasicConv(bridge_channels, bridge_channels))

        # Create up blocks
        self.up = nn.ModuleList([])
        for i in range(n_blocks, 0, -1):
            cur_in_ch = start_channels * (2 ** i)
            if pos_loc + i == pos_loc_max:
                cur_in_ch_pos = cur_in_ch + pos_dim
                self.up.append(Up(cur_in_ch_pos, (cur_in_ch // 2),
                                  learnable_pos=LearnablePosition(batch_size,
                                                                  pos_dim,
                                                                  math.ceil(781 / (2 ** (i - 1))),
                                                                  math.ceil(602 / (2 ** (i - 1)))),
                                  bilinear=bilinear))
            else:
                self.up.append(Up(cur_in_ch, (cur_in_ch // 2), bilinear=bilinear))

        # Create out convolution block
        self.outc = nn.Sequential()
        if pos_loc == pos_loc_max:
            start_channels_pos = start_channels + pos_dim
            self.outc.add_module('out_pos', LearnablePosition(batch_size, pos_dim, 781, 602))
            self.outc.add_module('out_conv', OutConv(start_channels_pos, n_classes))
        elif use_lcn:
            self.outc.add_module('out_lcn', LCN2DLayer(in_channels=start_channels, out_channels=n_classes, width=781, height=602))
        else:
            self.outc.add_module('out_conv', OutConv(start_channels, n_classes))

    def forward(self, x):
        logits = []
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

class SuccessiveUNet(UNet):
    def __init__(self, n_channels, n_classes, n_blocks, start_channels,
                 pos_loc=0, pos_dim=0, bilinear=False, batch_size=1):
        super(SuccessiveUNet, self).__init__(n_channels=n_channels, n_classes=n_classes,
                                             n_blocks=n_blocks, start_channels=start_channels,
                                             pos_loc=pos_loc, pos_dim=pos_dim,
                                             bilinear=bilinear, batch_size=batch_size)

        # Learnable position related
        pos_loc_max = (2 * n_blocks) + 3
        assert (pos_loc >= 0) and (pos_loc <= pos_loc_max)
        if pos_loc == 0:
            assert pos_dim == 0
        if pos_dim == 0:
            assert pos_loc == 0
            
        # successive (2->2), parallel (s->2), parallel + previous logit (s+2->2)
        self.outc2 = nn.Sequential()
        self.outc2.add_module('out_conv2', OutConv(start_channels, 2)) # Parallel
        # self.outc2.add_module('out_conv2', OutConv(2, 2)) # Successive
        # self.outc2.add_module('out_conv2', OutConv(start_channels + 2, 2)) # Parallel + Successive

    def forward(self, x):
        logits = []
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
        logit2 = self.outc2(out) # Parallel
        # logit2 = self.outc2(logit) # Successive
        # logit2 = self.outc2(torch.cat([out, logit], axis=1)) # Parallel + Successive

        return [logit, logit2]
    
class AttentionUNet(UNet):
    def __init__(self, n_channels, n_classes, n_blocks, start_channels,
                 pos_loc=0, pos_dim=0, bilinear=False, batch_size=1):
        super(AttentionUNet, self).__init__(n_channels=n_channels, n_classes=n_classes,
                                            n_blocks=n_blocks, start_channels=start_channels,
                                            pos_loc=pos_loc, pos_dim=pos_dim,
                                            bilinear=bilinear, batch_size=batch_size)

        # Learnable position related
        pos_loc_max = (2 * n_blocks) + 3
        assert (pos_loc >= 0) and (pos_loc <= pos_loc_max)
        if pos_loc == 0:
            assert pos_dim == 0
        if pos_dim == 0:
            assert pos_loc == 0

        # Create up blocks
        self.up = nn.ModuleList([])
        for i in range(n_blocks, 0, -1):
            cur_in_ch = start_channels * (2 ** i)
            if pos_loc + i == pos_loc_max:
                cur_in_ch_pos = cur_in_ch + pos_dim
                self.up.append(Up(cur_in_ch_pos, (cur_in_ch // 2),
                                  learnable_pos=LearnablePosition(batch_size,
                                                                  pos_dim,
                                                                  512 // (2 ** (i - 1)),
                                                                  512 // (2 ** (i - 1))),
                                  bilinear=bilinear,
                                  attention=True))
            else:
                self.up.append(Up(cur_in_ch, (cur_in_ch // 2),
                                  bilinear=bilinear, attention=True))