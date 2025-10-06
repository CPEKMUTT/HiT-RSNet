"""
common_utils.py
----------------
Core utility modules and helper layers for HiT-RSNet and related
super-resolution architectures.

This module defines:
- Normalized convolution wrappers
- MeanShift for color normalization
- Basic and residual building blocks
- Multi-scale pixel shuffle upsampling module
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_basic(in_ch, out_ch, k_size=3, stride=1, bias=True):
    """Standard 2D convolution with same padding."""
    padding = k_size // 2
    return nn.Conv2d(in_ch, out_ch, k_size, stride=stride, padding=padding, bias=bias)

class MeanShift(nn.Conv2d):
    """
    Normalization layer that performs per-channel mean subtraction
    and standard deviation scaling (non-trainable).

    Typically used for dataset mean adjustment before or after network inference.
    """

    def __init__(self, rgb_range, rgb_mean=(0.4488, 0.4371, 0.4040),
                 rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super().__init__(3, 3, kernel_size=1)

        mean = torch.tensor(rgb_mean)
        std = torch.tensor(rgb_std)

        # Initialize fixed parameters
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * mean / std

        # Freeze parameters (non-trainable)
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Module):
    """
    A simple convolution + BN + activation module.
    Acts as a reusable feature extraction unit.
    """

    def __init__(self, conv_fn, in_ch, out_ch, k_size=3, stride=1,
                 bias=False, use_bn=True, activation=nn.ReLU(True)):
        super().__init__()
        layers = [conv_fn(in_ch, out_ch, k_size, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if activation is not None:
            layers.append(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual feature refinement block with optional scaling.
    """

    def __init__(self, conv_fn, n_feats, k_size=3, bias=True,
                 use_bn=False, activation=nn.ReLU(True), res_scale=1.0):
        super().__init__()
        layers = []
        for i in range(2):
            layers.append(conv_fn(n_feats, n_feats, k_size, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm2d(n_feats))
            if i == 0:  # activation after first conv
                layers.append(activation)
        self.body = nn.Sequential(*layers)
        self.scale = res_scale

    def forward(self, x):
        res = self.body(x) * self.scale
        return x + res

class Upsampler(nn.Module):
    """
    Multi-factor upsampling module based on sub-pixel convolution.

    Supports:
        - Scale = 2^n (e.g. 2, 4, 8)
        - Scale = 3
    """

    def __init__(self, conv_fn, scale, n_feats, use_bn=False, activation=None, bias=True):
        super().__init__()
        layers = []

        if (scale & (scale - 1)) == 0:  # power of 2
            for _ in range(int(math.log2(scale))):
                layers.append(conv_fn(n_feats, n_feats * 4, 3, bias=bias))
                layers.append(nn.PixelShuffle(2))

                if use_bn:
                    layers.append(nn.BatchNorm2d(n_feats))
                if activation == 'relu':
                    layers.append(nn.ReLU(True))
                elif activation == 'prelu':
                    layers.append(nn.PReLU(n_feats))

        elif scale == 3:
            layers.append(conv_fn(n_feats, n_feats * 9, 3, bias=bias))
            layers.append(nn.PixelShuffle(3))

            if use_bn:
                layers.append(nn.BatchNorm2d(n_feats))
            if activation == 'relu':
                layers.append(nn.ReLU(True))
            elif activation == 'prelu':
                layers.append(nn.PReLU(n_feats))

        else:
            raise ValueError(f"Unsupported upscaling factor: {scale}")

        self.upsample = nn.Sequential(*layers)

    def forward(self, x):
        return self.upsample(x)