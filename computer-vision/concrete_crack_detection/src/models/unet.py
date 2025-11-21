# src/models/unet.py
import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x_before_pool = self.conv(x)
        x_after_pool = self.pool(x_before_pool)
        return x_before_pool, x_after_pool


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_copy):
        x = self.up(x)
        x = torch.cat([x_copy, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Down path
        for feature in features:
            self.down_blocks.append(DownBlock(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Up path
        for feature in reversed(features):
            self.up_blocks.append(UpBlock(feature * 2, feature))

        # Final convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down path
        for down_block in self.down_blocks:
            x_before_pool, x = down_block(x)
            skip_connections.append(x_before_pool)

        # Bottleneck
        x = self.bottleneck(x)

        # Up path
        for up_block, skip_connection in zip(self.up_blocks, reversed(skip_connections)):
            x = up_block(x, skip_connection)

        return torch.sigmoid(self.final_conv(x))