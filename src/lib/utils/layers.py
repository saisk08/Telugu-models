from torch import nn
import torch


def conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_block(channels, channels, stride=1)
        self.conv2 = conv_block(channels, channels, stride=1)

    def forward(self, x): return x + self.conv2(self.conv1(x))


class DenseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = conv_block(channels, channels, stride=1)
        self.conv2 = conv_block(channels, channels, stride=1)

    def forward(self, x): return torch.cat(x, self.conv2(self.conv1(x)))


def res_block(channels, dense=False):
    if not dense:
        return ResBlock(channels)
    else:
        return DenseBlock(channels)
