from torch import nn
from utils.layers import conv_block


def conv(ic, oc): return conv_block(ic, oc, kernel_size=3, stride=2, padding=1)


class Telnet(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.net = nn.Sequential(
            conv(1, 8),  # 16
            conv(8, 16),  # 8
            conv(16, 32),  # 4
            conv(32, 64),  # 2
            conv(64, 24),  # 1
            nn.Flatten()
        )

    def forward(self, x): return self.net(x)


class Siameserdm(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.feats = nn.Sequential(
            conv(1, 8),  # 16
            conv(8, 16),  # 8
            conv(16, 32),  # 4
            conv(32, 64)  # 2
        )
        self.classifier = nn.Sequential(
            conv(64, 1),  # 1
            nn.Flatten()
        )

    def forward(self, x):
        x = self.feats(x)
        x = self.classifier(x)
        return x


class Siamesecat(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.net = self.create(out)

    def conv(self, ic, oc): return conv_block(
        ic, oc, kernel_szie=3, stride=2, padding=1)

    def create(self, out):
        return nn.Sequential(
            conv(1, 8),  # 16
            conv(8, 16),  # 8
            conv(16, 64),  # 4
            conv(64, 128),  # 2
            conv(256, 300),  # 1
            nn.Flatten()
        )

    def forward(self, x): return self.net(x)
