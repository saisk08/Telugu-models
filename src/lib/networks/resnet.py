from torch import nn
from utils.layers import res_block, conv_block


def conv(ic, oc, **kwargs): return conv_block(ic, oc, **kwargs)


def res(c): return res_block(c)


def conv_and_res(ic, oc, **kwargs):
    return nn.Sequential(
        conv(ic, oc, **kwargs),
        res(oc)
    )


class Telnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv_and_res(1, 8, kernel_size=5, stride=2, padding=2),  # 16
            conv_and_res(8, 16),  # 8
            conv_and_res(16, 32),  # 4
            conv_and_res(32, 24),  # 2
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x): return self.net(x)


class Siameserdm(nn.Module):
    def __init__(self):
        super().__init__()
        self.feats = nn.Sequential(
            conv_and_res(1, 8),  # 16
            conv_and_res(8, 16),  # 8
            conv_and_res(16, 32),  # 4
            conv_and_res(32, 64),  # 2
            conv(64, 24)
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
    def __init__(self):
        super().__init__()
        self.net = self.create()

    def conv(self, ic, oc): return conv_block(
        ic, oc, kernel_szie=3, stride=2, padding=1)

    def create(self):
        return nn.Sequential(
            conv_and_res(1, 8),  # 16
            conv_and_res(8, 16),  # 8
            conv_and_res(16, 64),  # 4
            conv_and_res(64, 128),  # 2
            conv_and_res(256, 300),  # 1
            nn.Flatten()
        )

    def forward(self, x): return self.net(x)
